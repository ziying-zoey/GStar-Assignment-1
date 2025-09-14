import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
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
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    d = tl.arange(0, HEAD_DIM)
    tl.multiple_of(d, 16)
    tl.max_contiguous(d, 16)

    q0 = q_block_idx * BLOCK_M

    # 滑窗起点：q0-(W-1)，并至少从 SINK_SIZE 后开始（避免和 sink 重复）
    k_min = tl.maximum(0, q0 - (WINDOW_SIZE - 1))
    k_min = tl.maximum(k_min, SINK_SIZE)                    # 窗口不覆盖 sink 段
    window_start = (k_min // BLOCK_N) * BLOCK_N             # 块对齐
    window_start = tl.minimum(window_start, q0)

    # =============== Phase 0: SINK（只处理 [0, SINK_SIZE)） ===============
    sink_blocks = (SINK_SIZE + BLOCK_N - 1) // BLOCK_N
    for sink_block_idx in range(sink_blocks):
        start_n = sink_block_idx * BLOCK_N
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d[None, :])

        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0).to(tl.float32)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        # sink 仅做：越界 + 因果（k <= q）+ k < SINK_SIZE
        keep = (k_offsets[None, :] < SEQ_LEN) & \
               (k_offsets[None, :] <= q_offsets[:, None]) & \
               (k_offsets[None, :] < SINK_SIZE)
        s_ij = tl.where(keep, s_ij, -float("inf"))

        neg_inf = -float("inf")
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # 空 tile 特判（避免 -inf - -inf）
        no_valid = (m_ij == neg_inf) & (m_i == neg_inf)
        alpha = tl.where(no_valid, 1.0, tl.exp2(m_i - m_new))
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_ij = tl.where(no_valid[:, None], 0.0, tl.exp2(s_ij - m_new[:, None]))
        acc += tl.dot(p_ij.to(tl.float16), v_block.to(tl.float16)).to(tl.float32) # [BLOCK_M, HEAD_DIM]
        l_i += tl.sum(p_ij, axis=1)
        m_i = tl.where(no_valid, m_i, m_new)

    # ===== Phase 1: 窗口内的非对角块（范围 [window_start, q0)；明确排除 sink） =====
    for start_n in range(window_start, q0, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d[None, :])

        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0).to(tl.float32)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        causal = (k_offsets[None, :] <= q_offsets[:, None])
        in_window = (k_offsets[None, :] >= (q_offsets[:, None] - (WINDOW_SIZE - 1)))
        non_sink = (k_offsets[None, :] >= SINK_SIZE)
        keep = (k_offsets[None, :] < SEQ_LEN) & causal & in_window & non_sink
        s_ij = tl.where(keep, s_ij, -float("inf"))

        neg_inf = -float("inf")
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        no_valid = (m_ij == neg_inf) & (m_i == neg_inf)
        alpha = tl.where(no_valid, 1.0, tl.exp2(m_i - m_new))
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_ij = tl.where(no_valid[:, None], 0.0, tl.exp2(s_ij - m_new[:, None]))
        acc += tl.dot(p_ij.to(tl.float16), v_block.to(tl.float16)).to(tl.float32) # [BLOCK_M, HEAD_DIM]
        l_i += tl.sum(p_ij, axis=1)
        m_i = tl.where(no_valid, m_i, m_new)

    # =============== Phase 2: 对角块（宽度 <= min(BLOCK_M, WINDOW_SIZE)） ===============
    diag_start = q0
    max_k_span = WINDOW_SIZE if WINDOW_SIZE < BLOCK_M else BLOCK_M
    n_diag_iters = (max_k_span + BLOCK_N - 1) // BLOCK_N

    for it in range(n_diag_iters):
        start_n = diag_start + it * BLOCK_N
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d[None, :])

        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0).to(tl.float32)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        causal = (k_offsets[None, :] <= q_offsets[:, None])
        in_window = (k_offsets[None, :] >= (q_offsets[:, None] - (WINDOW_SIZE - 1)))
        non_sink = (k_offsets[None, :] >= SINK_SIZE)  # sink 已在 Phase 0 处理
        keep = (k_offsets[None, :] < SEQ_LEN) & causal & in_window & non_sink
        s_ij = tl.where(keep, s_ij, -float("inf"))

        neg_inf = -float("inf")
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        no_valid = (m_ij == neg_inf) & (m_i == neg_inf)
        alpha = tl.where(no_valid, 1.0, tl.exp2(m_i - m_new))
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_ij = tl.where(no_valid[:, None], 0.0, tl.exp2(s_ij - m_new[:, None]))
        acc += tl.dot(p_ij.to(tl.float16), v_block.to(tl.float16)).to(tl.float32)
        l_i += tl.sum(p_ij, axis=1)
        m_i = tl.where(no_valid, m_i, m_new)
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o