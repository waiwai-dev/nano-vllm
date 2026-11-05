<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Qwen3.5-4B text inference

This branch supports the dense Qwen3.5 text backbone: Gated DeltaNet, gated
full attention, partial RoPE, zero-centered RMSNorm, and official checkpoint
names. It runs inside Nano-vLLM with paged KV, chunked prefill, continuous
batching, request-local recurrent/conv states, and tensor parallelism.

```bash
pip install -e '.[qwen35]'
python example_qwen35.py --model /mnt/waiwai/models/Qwen3.5-4B
python example_qwen35.py --model /mnt/waiwai/models/Qwen3.5-4B --tp 2
```

Qwen3.5 currently runs in eager mode with prefix caching disabled. The default
`max_num_seqs` is 16 for hybrid models (512 for Qwen3); choose a smaller value
to reduce the fp32 recurrent-state allocation. Vision, video, MoE, MTP, and
hybrid CUDA graphs / prefix-state snapshots are not implemented. BF16 kernel
rounding can change greedy tokens near ties; this is not bitwise equivalent
to Transformers. See [QWEN35.md](QWEN35.md) for setup and validation scope.

FlashAttention 2 is optional (`pip install -e '.[flash]'`). Without its API,
the built-in Triton attention path handles dense and paged attention, including
consumer Blackwell GPUs where FA4 paged KV is unavailable.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
