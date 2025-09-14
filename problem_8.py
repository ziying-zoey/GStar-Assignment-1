# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _fa2_forward_gqa_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Strides (B, H, S, D) for Q/K/V/O
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    # Stride (B, H_q, S) for M
    m_stride_b, m_stride_h, m_stride_s,
    # Params
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids
    q_block_idx = tl.program_id(axis=0)
    bh_idx = tl.program_id(axis=1)

    b = bh_idx // N_Q_HEADS
    hq = bh_idx % N_Q_HEADS

    group = N_Q_HEADS // N_KV_HEADS
    hkv = hq // group

    # accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load Q block
    q_rows = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    d = tl.arange(0, HEAD_DIM)                                 # [D]
    q_ptrs = (Q_ptr + b * q_stride_b + hq * q_stride_h
              + (q_rows[:, None] * q_stride_s + d[None, :]))
    q = tl.load(q_ptrs, mask=(q_rows[:, None] < SEQ_LEN), other=0.0)

    qk_scale = softmax_scale * 1.4426950408889634  # log2(e)

    # Phase 1: strictly-off-diagonal blocks (no elementwise causal mask)
    q0 = q_block_idx * BLOCK_M
    for start_n in range(0, q0, BLOCK_N):
        k_cols = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K_ptr + b * k_stride_b + hkv * k_stride_h
                  + (k_cols[None, :] * k_stride_s + d[:, None]))
        v_ptrs = (V_ptr + b * v_stride_b + hkv * v_stride_h
                  + (k_cols[:, None] * v_stride_s + d[None, :]))
        k = tl.load(k_ptrs, mask=(k_cols[None, :] < SEQ_LEN), other=0.0)
        v = tl.load(v_ptrs, mask=(k_cols[:, None] < SEQ_LEN), other=0.0)

        s = tl.dot(q, k) * qk_scale  # [BM, BN]
        keep = (k_cols[None, :] < SEQ_LEN)
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

    # Phase 2: diagonal band (elementwise causal)
    for start_n in range(q0, q0 + BLOCK_M, BLOCK_N):
        k_cols = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K_ptr + b * k_stride_b + hkv * k_stride_h
                  + (k_cols[None, :] * k_stride_s + d[:, None]))
        v_ptrs = (V_ptr + b * v_stride_b + hkv * v_stride_h
                  + (k_cols[:, None] * v_stride_s + d[None, :]))
        k = tl.load(k_ptrs, mask=(k_cols[None, :] < SEQ_LEN), other=0.0)
        v = tl.load(v_ptrs, mask=(k_cols[:, None] < SEQ_LEN), other=0.0)

        s = tl.dot(q, k) * qk_scale
        keep = (k_cols[None, :] < SEQ_LEN) & (k_cols[None, :] <= q_rows[:, None])
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

    # write O
    l_safe = l_i[:, None] + 1e-6
    o = acc / l_safe
    o_ptrs = (O_ptr + b * o_stride_b + hq * o_stride_h
              + (q_rows[:, None] * o_stride_s + d[None, :]))
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=(q_rows[:, None] < SEQ_LEN))

    # write M
    m_ptrs = M_ptr + b * m_stride_b + hq * m_stride_h + q_rows * m_stride_s
    tl.store(m_ptrs, m_i, mask=(q_rows < SEQ_LEN))


class FlashAttention2Function(torch.autograd.Function):
    """
    FlashAttention-2: forward in Triton (causal + GQA), backward recompute in PyTorch (blockwise).
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        B, Hq, S, D = q.shape
        Hkv = k.shape[1]
        assert is_causal, "This kernel only supports causal attention"
        assert Hq % Hkv == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        o = torch.empty_like(q)
        M = torch.empty((B, Hq, S), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(S, BLOCK_M), B * Hq)

        _fa2_forward_gqa_kernel[grid](
            # ptrs
            q, k, v, o, M,
            # strides Q/K/V/O
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            # strides M
            M.stride(0), M.stride(1), M.stride(2),
            # params
            softmax_scale, S, Hq, Hkv,
            # constexpr
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = Hq
        ctx.num_kv_heads = Hkv
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        B, Hq, S, D = q.shape
        Hkv = ctx.num_kv_heads
        scale = ctx.softmax_scale
        scale2 = ctx.softmax_scale * 1.4426950408889634  # 与前向一致（exp2 域）

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        BN = 128  # 列块宽度
        group = Hq // Hkv
        device = q.device
        f32 = torch.float32

        for b in range(B):
            for hq in range(Hq):
                hkv = hq // group

                Q  = q[b, hq].to(f32)      # [S, D]
                K  = k[b, hkv].to(f32)     # [S, D]
                V  = v[b, hkv].to(f32)     # [S, D]
                dO = do[b, hq].to(f32)     # [S, D]
                O  = o[b, hq].to(f32)      # [S, D]
                m  = M[b, hq].to(f32)      # [S]

                # delta_i = <dO_i, O_i>
                delta = (dO * O).sum(dim=-1)  # [S]

                # 先累分母：Z_i = Σ_j exp2(S_ij - m_i)
                denom = torch.zeros(S, device=device, dtype=f32)
                rows = torch.arange(S, device=device)[:, None]  # [S,1]
                for start in range(0, S, BN):
                    end = min(S, start + BN)
                    cols = torch.arange(start, end, device=device)[None, :]  # [1,BN]

                    S_blk = (Q @ K[start:end].T) * scale2                     # [S,BN]
                    keep = (cols <= rows)                                     # 因果
                    S_blk = torch.where(keep, S_blk, torch.full_like(S_blk, -float("inf")))
                    E_blk = torch.exp2(S_blk - m[:, None])
                    E_blk = torch.where(keep, E_blk, torch.zeros_like(E_blk))
                    denom += E_blk.sum(dim=1)

                denom = torch.clamp(denom, min=1e-12)

                # 再累梯度
                dq_blk = torch.zeros_like(Q)
                dk_blk_total = torch.zeros_like(K)
                dv_blk_total = torch.zeros_like(V)

                for start in range(0, S, BN):
                    end = min(S, start + BN)
                    cols = torch.arange(start, end, device=device)[None, :]  # [1,BN]

                    K_blk = K[start:end]                                     # [BN,D]
                    V_blk = V[start:end]                                     # [BN,D]

                    S_blk = (Q @ K_blk.T) * scale2
                    keep = (cols <= rows)
                    S_blk = torch.where(keep, S_blk, torch.full_like(S_blk, -float("inf")))
                    E_blk = torch.exp2(S_blk - m[:, None])
                    E_blk = torch.where(keep, E_blk, torch.zeros_like(E_blk))

                    P_blk = E_blk / denom[:, None]                           # [S,BN]

                    # dV += P^T @ dO
                    dv_blk_total[start:end] += P_blk.T @ dO

                    # **关键修正**：dS = P ⊙ (dO @ V^T - delta)
                    t1 = dO @ V_blk.T                                        # [S,BN]
                    dS_blk = P_blk * (t1 - delta[:, None])                   # [S,BN]

                    # dQ += dS @ K * scale2
                    dq_blk += dS_blk @ K_blk * scale
                    # dK += dS^T @ Q * scale2
                    dk_blk_total[start:end] += (dS_blk.T @ Q) * scale

                dq[b, hq] = dq_blk.to(q.dtype)
                dk[b, hkv] += dk_blk_total.to(k.dtype)  # GQA: 多个 q-head 累加到同一 kv-head
                dv[b, hkv] += dv_blk_total.to(v.dtype)

        return dq, dk, dv, None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)