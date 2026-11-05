"""Real-checkpoint smoke and greedy comparison with Transformers on GPU.

python -m tests.check_qwen35_checkpoint --model /path/to/Qwen3.5-4B
"""
import argparse
import json
from time import perf_counter

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions = ["你好，请用一句话介绍你自己。", "计算 17 × 23，只回答结果。",
                 "用 Python 写一个判断整数是否为偶数的函数。",
                 "这是一段用于验证分块预填充的测试文本。" * 12 + "请只回答：完成。"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False) for text in questions]
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_num_seqs=2,
              max_model_len=2048, max_num_batched_tokens=32)
    prefill_logits = {}
    def capture_prefill(module, inputs, output):
        context = get_context()
        if not context.is_prefill:
            return
        active = {seq.state_slot: seq for seq in list(llm.scheduler.running) + list(llm.scheduler.waiting) if seq.state_slot >= 0}
        for row, slot in enumerate(context.state_slots.tolist()):
            seq = active[slot]
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_prompt_tokens:
                prefill_logits[tuple(seq.prompt_token_ids)] = output[row].float().cpu()
    hook = llm.model_runner.model.lm_head.register_forward_hook(capture_prefill)
    try:
        start = perf_counter()
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
        elapsed = perf_counter() - start
        # A fresh call with the first prompt exercises freed GDN slots and disabled prefix reuse.
        repeated = llm.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0]
        assert repeated["token_ids"] == outputs[0]["token_ids"], "State leaked across generate calls"
        assert all(output["token_ids"] for output in outputs)
        print("NATIVE_OUTPUTS", json.dumps(outputs, ensure_ascii=False), flush=True)
    finally:
        hook.remove()
        llm.exit()
    torch.cuda.empty_cache()
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="eager").eval()
    expected = []
    numerical = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = torch.tensor([prompt], device="cuda:0")
            ref_logits = reference(ids, use_cache=False).logits[0, -1].float().cpu()
            native_logits = prefill_logits[tuple(prompt)]
            diff = native_logits - ref_logits
            numerical.append({"max_abs_error": diff.abs().max().item(), "rms_error": diff.square().mean().sqrt().item(),
                "native_top5": native_logits.topk(5).indices.tolist(), "reference_top5": ref_logits.topk(5).indices.tolist(),
                "native_top5_values": native_logits.topk(5).values.tolist(), "reference_top5_values": ref_logits.topk(5).values.tolist()})
            result = reference.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                max_new_tokens=32, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
            expected.append(result[0, len(prompt):].tolist())
    matches = [actual["token_ids"] == ref for actual, ref in zip(outputs, expected)]
    report = {"tp": args.tp, "prompts": questions, "outputs": outputs,
              "reference_token_ids": expected, "reference_text": [tokenizer.decode(ids) for ids in expected],
              "exact_matches": matches, "state_reuse_exact": True, "native_seconds": elapsed,
              "prefill_logits": numerical, "prompt_lengths": [len(p) for p in prompts],
              "scope": "BF16, eager, text-only, chunk=32, max_num_seqs=2, 32 generated tokens"}
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    # A bounded-logit smoke check, not a proof of quality equivalence. Preserve
    # both RMS error and exact sequence matches in the report, including ties.
    assert all(m["max_abs_error"] <= 0.5 for m in numerical), "Real-checkpoint logit smoke bound exceeded"
    print("PASS: real checkpoint, max logit error <= 0.5 and state isolation; exact greedy matches:", matches, flush=True)


if __name__ == "__main__":
    main()
