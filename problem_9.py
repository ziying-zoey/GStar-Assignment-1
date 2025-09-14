import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids
    q_block_idx = tl.program_id(axis=0)
    bh_idx = tl.program_id(axis=1)

    b = bh_idx // N_Q_HEADS
    hq = bh_idx % N_Q_HEADS

    # GQA 映射
    group = N_Q_HEADS // N_KV_HEADS
    hkv = hq // group

    # 累加器
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 取 Q tile
    q_rows = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)         # [BM]
    d = tl.arange(0, HEAD_DIM)                                     # [D]
    tl.multiple_of(d, 16)
    tl.max_contiguous(d, 16)

    q_ptrs = (Q_ptr + b * q_stride_b + hq * q_stride_h
              + (q_rows[:, None] * q_stride_s + d[None, :]))
    q = tl.load(q_ptrs, mask=(q_rows[:, None] < SEQ_LEN), other=0.0)

    log2e = 1.4426950408889634
    qk_scale2 = softmax_scale * log2e

    q0 = q_block_idx * BLOCK_M

    # -------- Phase 0: Sink [0, SINK_SIZE) --------
    sink_blocks = (SINK_SIZE + BLOCK_N - 1) // BLOCK_N
    for ib in range(sink_blocks):
        start_n = ib * BLOCK_N
        k_cols = start_n + tl.arange(0, BLOCK_N)                   # [BN]

        k_ptrs = (K_ptr + b * k_stride_b + hkv * k_stride_h
                  + (k_cols[None, :] * k_stride_s + d[:, None]))
        v_ptrs = (V_ptr + b * v_stride_b + hkv * v_stride_h
                  + (k_cols[:, None] * v_stride_s + d[None, :]))
        k = tl.load(k_ptrs, mask=(k_cols[None, :] < SEQ_LEN), other=0.0)
        v = tl.load(v_ptrs, mask=(k_cols[:, None] < SEQ_LEN), other=0.0)

        s = tl.dot(q, k) * qk_scale2                                 # [BM,BN]
        keep = (k_cols[None, :] < SEQ_LEN) & (k_cols[None, :] <= q_rows[:, None]) & (k_cols[None, :] < SINK_SIZE)
        s = tl.where(keep, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p = tl.where(keep, tl.exp2(s - m_new[:, None]), 0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_i += tl.sum(p, axis=1)
        m_i = m_new

    # 计算窗口起点（与块对齐，且不落到 sink 里）
    k_min = tl.maximum(0, q0 - (WINDOW_SIZE - 1))
    k_min = tl.maximum(k_min, SINK_SIZE)
    window_start = (k_min // BLOCK_N) * BLOCK_N
    window_start = tl.minimum(window_start, q0)

    # -------- Phase 1: 窗口内非对角（排除 sink） --------
    for start_n in range(window_start, q0, BLOCK_N):
        k_cols = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K_ptr + b * k_stride_b + hkv * k_stride_h
                  + (k_cols[None, :] * k_stride_s + d[:, None]))
        v_ptrs = (V_ptr + b * v_stride_b + hkv * v_stride_h
                  + (k_cols[:, None] * v_stride_s + d[None, :]))
        k = tl.load(k_ptrs, mask=(k_cols[None, :] < SEQ_LEN), other=0.0)
        v = tl.load(v_ptrs, mask=(k_cols[:, None] < SEQ_LEN), other=0.0)

        s = tl.dot(q, k) * qk_scale2
        causal = k_cols[None, :] <= q_rows[:, None]
        in_window = k_cols[None, :] >= (q_rows[:, None] - (WINDOW_SIZE - 1))
        non_sink = k_cols[None, :] >= SINK_SIZE
        keep = (k_cols[None, :] < SEQ_LEN) & causal & in_window & non_sink
        s = tl.where(keep, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p = tl.where(keep, tl.exp2(s - m_new[:, None]), 0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_i += tl.sum(p, axis=1)
        m_i = m_new

    # -------- Phase 2: 对角带（同样排除 sink，限制窗口） --------
    max_k_span = WINDOW_SIZE if WINDOW_SIZE < BLOCK_M else BLOCK_M
    n_diag_iters = (max_k_span + BLOCK_N - 1) // BLOCK_N
    for it in range(n_diag_iters):
        start_n = q0 + it * BLOCK_N
        k_cols = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K_ptr + b * k_stride_b + hkv * k_stride_h
                  + (k_cols[None, :] * k_stride_s + d[:, None]))
        v_ptrs = (V_ptr + b * v_stride_b + hkv * v_stride_h
                  + (k_cols[:, None] * v_stride_s + d[None, :]))
        k = tl.load(k_ptrs, mask=(k_cols[None, :] < SEQ_LEN), other=0.0)
        v = tl.load(v_ptrs, mask=(k_cols[:, None] < SEQ_LEN), other=0.0)

        s = tl.dot(q, k) * qk_scale2
        causal = k_cols[None, :] <= q_rows[:, None]
        in_window = k_cols[None, :] >= (q_rows[:, None] - (WINDOW_SIZE - 1))
        non_sink = k_cols[None, :] >= SINK_SIZE
        keep = (k_cols[None, :] < SEQ_LEN) & causal & in_window & non_sink
        s = tl.where(keep, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p = tl.where(keep, tl.exp2(s - m_new[:, None]), 0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_i += tl.sum(p, axis=1)
        m_i = m_new

    # 写 O
    o = acc / (l_i[:, None] + 1e-6)
    o_ptrs = (O_ptr + b * o_stride_b + hq * o_stride_h
              + (q_rows[:, None] * o_stride_s + d[None, :]))
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=(q_rows[:, None] < SEQ_LEN))
    # 写 M（行最大）
    m_ptrs = M_ptr + b * m_stride_b + hq * m_stride_h + q_rows * m_stride_s
    tl.store(m_ptrs, m_i, mask=(q_rows < SEQ_LEN))


class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"
        B, Hq, S, D = q.shape
        Hkv = k.shape[1]
        assert Hq % Hkv == 0
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        o = torch.empty_like(q)
        M = torch.empty((B, Hq, S), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(S, BLOCK_M), B * Hq)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            S,
            Hq,
            Hkv,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        B, Hq, S, D = q.shape
        Hkv = k.shape[1]
        group = Hq // Hkv
        scale = ctx.softmax_scale
        scale2 = scale * 1.4426950408889634

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        W = ctx.window_size
        SINK = ctx.sink_size

        BN = 128  # 列块

        device = q.device
        f32 = torch.float32

        rows = torch.arange(S, device=device)[:, None]  # [S,1]

        for b in range(B):
            for hq in range(Hq):
                hkv = hq // group

                Q = q[b, hq].to(f32)       # [S,D]
                K = k[b, hkv].to(f32)      # [S,D]
                V = v[b, hkv].to(f32)      # [S,D]
                dO = do[b, hq].to(f32)     # [S,D]
                O = o[b, hq].to(f32)       # [S,D]
                m = M[b, hq].to(f32)       # [S]

                delta = (dO * O).sum(dim=-1)        # [S]
                denom = torch.zeros(S, device=device, dtype=f32)

                # ---- pass 1: 计算分母 ----
                for start in range(0, S, BN):
                    end = min(S, start + BN)
                    cols = torch.arange(start, end, device=device)[None, :]       # [1,BN]
                    S_blk = (Q @ K[start:end].T) * scale2                         # [S,BN]

                    causal = cols <= rows
                    in_window = cols >= (rows - (W - 1))
                    sink = cols < SINK
                    keep = causal & (in_window | sink)

                    S_blk = torch.where(keep, S_blk, torch.full_like(S_blk, -float("inf")))
                    E_blk = torch.exp2(S_blk - m[:, None])
                    E_blk = torch.where(keep, E_blk, torch.zeros_like(E_blk))
                    denom += E_blk.sum(dim=1)

                denom = torch.clamp(denom, min=1e-12)

                # ---- pass 2: 累积梯度 ----
                dq_blk = torch.zeros_like(Q)
                dk_blk_total = torch.zeros_like(K)
                dv_blk_total = torch.zeros_like(V)

                # ---- pass 2: 累积梯度 ----
                for start in range(0, S, BN):
                    end = min(S, start + BN)
                    cols = torch.arange(start, end, device=device)[None, :]  # [1,BN]

                    K_blk = K[start:end]                       # [BN, D]
                    V_blk = V[start:end]                       # [BN, D]

                    S_blk = (Q @ K_blk.T) * scale2             # [S, BN]
                    causal = cols <= rows
                    in_window = cols >= (rows - (W - 1))
                    sink = cols < SINK
                    keep = causal & (in_window | sink)

                    S_blk = torch.where(keep, S_blk, torch.full_like(S_blk, -float("inf")))
                    E_blk = torch.exp2(S_blk - m[:, None])
                    E_blk = torch.where(keep, E_blk, torch.zeros_like(E_blk))
                    P_blk = E_blk / denom[:, None]             # [S, BN]

                    # dV
                    dv_blk = P_blk.T @ dO
                    dv_blk_total[start:end] += dv_blk

                    # dS（**加掩码**）
                    t1 = dO @ V_blk.T                          # [S, BN]
                    dS_blk = P_blk * (t1 - delta[:, None])         # [S, BN]
                    dS_blk = torch.where(keep, dS_blk, 0.0)

                    # dQ / dK（只乘自然底 softmax_scale）
                    dq_blk += dS_blk @ K_blk * scale
                    dk_blk_total[start:end] += (dS_blk.T @ Q) * scale

                dq[b, hq] = dq_blk.to(q.dtype)
                dk[b, hkv] += dk_blk_total.to(k.dtype)   # GQA：同一 kv head 累加
                dv[b, hkv] += dv_blk_total.to(v.dtype)

        return dq, dk, dv, None, None, None, None


def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)