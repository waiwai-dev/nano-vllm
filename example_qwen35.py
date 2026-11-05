"""Text-only Qwen3.5-4B example (TP1 by default; --tp 2 for two GPUs)."""
import argparse

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/waiwai/models/Qwen3.5-4B")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
        for text in ["你好，请用一句话介绍你自己。", "计算 17 × 23，只回答结果。"]]
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_num_seqs=4,
              max_model_len=2048, max_num_batched_tokens=256)
    try:
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64))
        for output in outputs:
            print(output["text"])
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
