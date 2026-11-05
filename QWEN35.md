# Qwen3.5-4B 文本推理

本分支将官方 `Qwen/Qwen3.5-4B` 的文本骨干接入 nano-vllm 引擎，
不是调用 Transformers 的 `generate` 代替引擎执行。Transformers 仅用于
配置、分词，以及独立的验证基线。

## 运行

```bash
cd /mnt/waiwai/work/Code_repo/nano-vllm
/mnt/waiwai/work/nano-vllm-qwen35-venv/bin/python example_qwen35.py
/mnt/waiwai/work/nano-vllm-qwen35-venv/bin/python example_qwen35.py --tp 2
```

其他环境安装：

```bash
pip install -e '.[qwen35]'
```

官方完整模型位于 `/mnt/waiwai/models/Qwen3.5-4B`，下载 revision：
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。包含两个权重分片、
分词器和官方多模态配置文件，约 8.8 GiB。加载时只读取文本骨干，
跳过视觉和 MTP 权重。

## 实现与限制

- 按 `text_config.layer_types` 构造 24 层 Gated DeltaNet 和 8 层门控全注意力。
- 支持 GDN 卷积、fp32 循环状态、门控 RMSNorm、零中心 RMSNorm、部分 RoPE、共享词嵌入。
- GDN 计算使用 FLA Triton kernels；模型权重加载、逐层执行、调度和状态池由本项目管理。
- 全注意力使用分页 KV，仅为 8 个全注意力层分配 KV。
- 支持不同长度请求、分块预填充、连续批处理、状态槽复用和 KV 抢占后的完整重算。
- TP 按注意力头和 GDN 头切分权重与状态；本次验证 TP1 和 TP2。
- 没有 FlashAttention 2 API 时，自动使用内置 Triton 注意力；不依赖 FA4 的分页支持。
- `temperature=0` 为贪心解码，保留正温度采样。
- 混合模型默认 `max_num_seqs=16`，原 Qwen3 保持 512。
  4B 的 TP1 状态池每个槽约 49 MiB，按配置槽数预分配并计入 KV 容量预算。
- 当前为文本支持；没有图像、视频、MoE、MTP 推理。
- Qwen3.5 自动启用 eager，关闭 prefix cache；尚未实现 GDN 前缀状态快照与 CUDA Graph。
- 默认上下文长度 4096；不得将配置中的 262144 视为本分支已经完成的长上下文验证。

## 验证方法

```bash
python -m unittest tests.test_hybrid_scheduler tests.test_triton_attention -v
python -m tests.check_qwen35_numerics
python -m tests.check_qwen35_numerics --qwen3
python -m tests.check_qwen35_checkpoint --model /mnt/waiwai/models/Qwen3.5-4B --output /mnt/waiwai/work/qwen35-tp1-validation.json
python -m tests.check_qwen35_checkpoint --model /mnt/waiwai/models/Qwen3.5-4B --tp 2 --output /mnt/waiwai/work/qwen35-tp2-validation.json
```

本次环境：2×RTX 5090，PyTorch 2.14.1+cu130、Transformers 5.19.0、
flash-linear-attention / fla-core 0.6.0，BF16 权重、fp32 GDN 状态，Triton 全注意力。

小模型验证将同一份随机权重交给本引擎与 Transformers，比较 15 次
预填充、分块续算和解码 logits，使用 `atol=0.04, rtol=0.04`，并要求
每次 argmax 一致。另用只有两个 KV block 的真实 GPU 执行路径强制抢占，
核对重算后的生成 token。

真实 4B 验证包含中文介绍、整数乘法、代码和 162-token 分块输入；每条
最多生成 32 token，状态槽为 2、prefill chunk 为 32。记录完整输出 ID、
与 Transformers 的 exact match、首个生成 token 的 logits 误差和 top-5。
其中 `max_abs_error <= 0.5` 是运行检查边界，不代表质量等价。

BF16 算子融合、分块方式和 TP 归约会产生数值差异，尤其在候选 token
并列附近可能改变生成序列。本次测试不是生产性能、完整任务质量、
最大并发容量或长上下文能力的证明。具体运行结果见 `VALIDATION_QWEN35.md`。
