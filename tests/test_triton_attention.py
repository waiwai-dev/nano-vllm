import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.triton_attention import attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonAttentionTests(unittest.TestCase):
    @torch.inference_mode()
    def check_attention(self, q_lengths, k_lengths, dim, paged=True, decode=False):
        torch.manual_seed(123)
        heads, kv_heads, page = 8, 2, 256
        q = torch.randn(sum(q_lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[2, 0], [1, 3]], device="cuda", dtype=torch.int32)
        if paged:
            k = torch.randn(4, page, kv_heads, dim, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
        else:
            k = torch.randn(sum(k_lengths), kv_heads, dim, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
        cuq = torch.tensor([0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, k_lengths[0], sum(k_lengths)], device="cuda", dtype=torch.int32)
        if decode:
            actual = attention(q[:, None], k, v, block_table=table,
                               cache_seqlens=torch.tensor(k_lengths, device="cuda", dtype=torch.int32))[:, 0]
        else:
            actual = attention(q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                               max_seqlen_q=max(q_lengths), block_table=table if paged else None)
        offset_q = offset_k = 0
        for i, (qlen, klen) in enumerate(zip(q_lengths, k_lengths)):
            if paged:
                ki = k[table[i].long()].reshape(-1, kv_heads, dim)[:klen]
                vi = v[table[i].long()].reshape(-1, kv_heads, dim)[:klen]
            else:
                ki, vi = k[offset_k:offset_k + klen], v[offset_k:offset_k + klen]
            qi = q[offset_q:offset_q + qlen]
            mask = torch.arange(klen, device="cuda")[None, :] <= (klen - qlen + torch.arange(qlen, device="cuda")[:, None])
            expected = F.scaled_dot_product_attention(qi.transpose(0, 1).float(),
                ki.repeat_interleave(heads // kv_heads, 1).transpose(0, 1).float(),
                vi.repeat_interleave(heads // kv_heads, 1).transpose(0, 1).float(), attn_mask=mask).transpose(0, 1)
            torch.testing.assert_close(actual[offset_q:offset_q + qlen].float(), expected, atol=0.02, rtol=0.02)
            offset_q += qlen
            offset_k += klen

    def test_paged_chunk_crosses_block_boundary_hd256(self):
        self.check_attention([17, 5], [289, 271], 256)

    def test_paged_decode(self):
        self.check_attention([1, 1], [257, 19], 128, decode=True)

    def test_dense_varlen_prefill(self):
        self.check_attention([33, 7], [33, 7], 32, paged=False)


if __name__ == "__main__":
    unittest.main()
