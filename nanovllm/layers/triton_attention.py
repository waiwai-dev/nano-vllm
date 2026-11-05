"""Portable causal attention fallback, including paged KV on consumer Blackwell."""
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(Q, K, V, O, CUQ, CUK, LENGTHS, TABLE,
                     QS: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
                     OS: tl.constexpr, TABLE_S: tl.constexpr,
                     HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
                     PAGE: tl.constexpr, SCALE: tl.constexpr,
                     PAGED: tl.constexpr, DECODE: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if DECODE:
        q_start, q_len = batch, 1
        k_len = tl.load(LENGTHS + batch)
        k_start = 0
    else:
        q_start = tl.load(CUQ + batch)
        q_len = tl.load(CUQ + batch + 1) - q_start
        k_start = tl.load(CUK + batch)
        k_len = tl.load(CUK + batch + 1) - k_start
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    q = tl.load(Q + (q_start + rows[:, None]) * QS + head * DIM + dims[None, :],
                (rows[:, None] < q_len) & (dims[None, :] < DIM), 0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    kv_head = head // (HEADS // KV_HEADS)
    # Causal bottom-right alignment handles both chunk continuation and decode.
    last_key = tl.minimum(k_len, k_len - q_len + (tile + 1) * BM)
    for start in range(0, tl.cdiv(last_key, BN)):
        cols = start * BN + tl.arange(0, BN)
        if PAGED:
            page_id = tl.load(TABLE + batch * TABLE_S + cols // PAGE, cols < k_len, 0)
            token_base = page_id * KS0 + (cols % PAGE) * KS1
        else:
            token_base = (k_start + cols) * KS0
        ptr = token_base[:, None] + kv_head * DIM + dims[None, :]
        valid = (cols[:, None] < k_len) & (dims[None, :] < DIM)
        k = tl.load(K + ptr, valid, 0)
        v = tl.load(V + ptr, valid, 0)
        score = tl.dot(q, tl.trans(k)) * SCALE
        causal = cols[None, :] <= (k_len - q_len + rows[:, None])
        score = tl.where((cols[None, :] < k_len) & causal, score, -float('inf'))
        new_m = tl.maximum(m, tl.max(score, 1))
        p = tl.exp(score - new_m[:, None])
        alpha = tl.exp(m - new_m)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        denom = denom * alpha + tl.sum(p, 1)
        m = new_m
    out = acc / denom[:, None]
    tl.store(O + (q_start + rows[:, None]) * OS + head * DIM + dims[None, :], out,
             (rows[:, None] < q_len) & (dims[None, :] < DIM))


def attention(q, k, v, *, cu_seqlens_q=None, cu_seqlens_k=None,
              max_seqlen_q=1, block_table=None, cache_seqlens=None,
              softmax_scale=None, **kwargs):
    decode = cache_seqlens is not None
    if decode:
        q = q.squeeze(1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    paged = block_table is not None
    batch = cache_seqlens.numel() if decode else cu_seqlens_q.numel() - 1
    heads, dim = q.shape[-2:]
    kv_heads = k.shape[-2]
    attention_kernel[(triton.cdiv(max_seqlen_q, 16), heads, batch)](
        q, k, v, out, cu_seqlens_q, cu_seqlens_k, cache_seqlens, block_table,
        q.stride(0), k.stride(0), k.stride(1), out.stride(0),
        block_table.stride(0) if paged else 0, heads, kv_heads, dim,
        k.shape[1] if paged else 0, softmax_scale or dim ** -0.5,
        paged, decode, 16, 32, triton.next_power_of_2(dim), num_warps=4)
    return out.unsqueeze(1) if decode else out
