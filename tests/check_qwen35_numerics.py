"""GPU integration check against Transformers, with random tiny hybrid weights.

Run from the checkout: python -m tests.check_qwen35_numerics
"""
import tempfile
import argparse

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import reset_context


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3", action="store_true", help="Check the original Qwen3 path for regressions")
    args = parser.parse_args()
    torch.manual_seed(42)
    cfg = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=256,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, max_position_embeddings=1024,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2]},
        dtype="bfloat16", tie_word_embeddings=True)
    reference_class = Qwen3_5ForCausalLM
    if args.qwen3:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=256,
            max_position_embeddings=1024, rope_theta=1000000, dtype="bfloat16", tie_word_embeddings=True)
        reference_class = Qwen3ForCausalLM
    cfg._attn_implementation = "eager"
    reference = reference_class(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    with tempfile.TemporaryDirectory() as tmp:
        reference.save_pretrained(tmp)
        config = Config(tmp, max_num_seqs=2, max_model_len=768, max_num_batched_tokens=17,
                        gpu_memory_utilization=0.2, enforce_eager=True)
        runner = ModelRunner(config, 0, [])
        try:
            scheduler = Scheduler(config)
            prompts = [[5, 17, 23, 40, 8] * 5, [6, 14, 37] * 7, [5, 17, 23, 40, 8] * 5]
            sequences = [Sequence(p, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)) for p in prompts]
            for seq in sequences:
                scheduler.add(seq)
            comparisons = 0
            max_error = 0
            while not scheduler.is_finished():
                seqs, prefill = scheduler.schedule()
                captured = []
                hook = runner.model.lm_head.register_forward_hook(lambda module, args, output: captured.append(output.clone()))
                try:
                    token_ids = runner.run(seqs, prefill)
                finally:
                    hook.remove()
                logits = captured[0]
                for i, seq in enumerate(seqs):
                    end = seq.num_cached_tokens + seq.num_scheduled_tokens if prefill else len(seq)
                    expected = reference(torch.tensor([seq.token_ids[:end]], device="cuda"), use_cache=False).logits[0, -1]
                    actual = logits[i]
                    error = (actual.float() - expected.float()).abs().max().item()
                    max_error = max(max_error, error)
                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
                    assert actual.argmax().item() == expected.argmax().item(), (i, end, error)
                    comparisons += 1
                scheduler.postprocess(seqs, token_ids, prefill)
            if config.is_hybrid:
                assert len(scheduler.free_state_slots) == 2
            assert sequences[0].completion_token_ids == sequences[2].completion_token_ids
            print(f"PASS: {comparisons} prefill/chunk/decode logits vs Transformers, max_abs_error={max_error:.6f}")
            print("PASS: batched unequal lengths, state-slot reuse, repeated-prompt isolation")
            if config.is_hybrid:
                # Two pages force a decode eviction at the 256-token boundary.
                config.num_kvcache_blocks = 2
                scheduler = Scheduler(config)
                sequences = [Sequence([5, 17, 23, 40, 8] * 51,
                    SamplingParams(temperature=0, max_tokens=5, ignore_eos=True)) for _ in range(2)]
                for seq in sequences:
                    scheduler.add(seq)
                evictions = []
                preempt = scheduler.preempt
                def record_preempt(seq):
                    evictions.append(seq.seq_id)
                    preempt(seq)
                scheduler.preempt = record_preempt
                while not scheduler.is_finished():
                    seqs, prefill = scheduler.schedule()
                    token_ids = runner.run(seqs, prefill)
                    scheduler.postprocess(seqs, token_ids, prefill)
                assert evictions, "Pressure test did not exercise eviction"
                assert sequences[0].completion_token_ids == sequences[1].completion_token_ids
                expected = reference.generate(torch.tensor([sequences[0].prompt_token_ids], device="cuda"),
                    do_sample=False, max_new_tokens=5, eos_token_id=None, pad_token_id=0)[0, 255:].tolist()
                assert sequences[0].completion_token_ids == expected
                print(f"PASS: {len(evictions)} real KV-pressure preemption(s), full replay restores GDN state")
        finally:
            reset_context()
            runner.exit()


if __name__ == "__main__":
    main()
