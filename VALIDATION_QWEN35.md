# Qwen3.5-4B 验证记录（2026-11-05）

服务器：`waiwai_5090`；仓库：`/mnt/waiwai/work/Code_repo/nano-vllm`；
基线 commit：`576772550f9379ad2eaa3897f673d5c23c523a40`。
分支：`codex/qwen3.5-4b`。环境与运行方法见 `QWEN35.md`。

## 已通过的检查

| 检查 | 结果 |
| --- | --- |
| 分块预填充的状态槽连续性、回收、重复提示隔离 | 通过 |
| 抢占后状态重置、TP 序列传输携带状态槽 | 通过 |
| Triton 注意力 vs FP32 SDPA | 3 项通过：变长 dense、分页 decode、HD256 分块跨页 |
| 随机小混合模型 vs Transformers | 15 次 logits 比较通过；最大绝对误差 0.007812；每次 argmax 一致 |
| GPU KV 压力抢占 | 强制 1 次抢占；完整重算后的生成 ID 与未被抢占请求及 Transformers 一致 |
| 原 Qwen3 路径回归 | 15 次 logits 比较通过；最大绝对误差 0.007812 |
| 真实 4B TP1、TP2 | 均通过非空生成、重复调用隔离及首 token logits 的绝对误差上界检查 |
| 安装、语法和 diff | editable extras 安装通过；compileall、git diff --check 通过 |

小模型 logits 使用 `atol=0.04, rtol=0.04`；Triton 对 FP32 SDPA 使用
`atol=0.02, rtol=0.02`。实际 checkpoint 的 `max_abs_error <= 0.5`
仅为运行检查边界，不作为质量等价标准；RMS 误差仍完整保留。
最初试用的 RMS<=0.05 条件在 162-token 输入上不满足，最终没有采用该
任意绝对 RMS 阈值。下表公开报告观测值，未将该项改称通过。

## 真实权重结果

BF16、eager、Triton 全注意力、FP32 GDN 状态，2 个状态槽，prefill chunk=32。
四个输入长度分别为 19、25、24、162 token；每个最多生成 32 token。
Transformers 使用 eager attention，并与本引擎采用相同的 tokenizer EOS 停止规则。

| 指标 | TP1 | TP2 |
| --- | --- | --- |
| 完整输出 ID 与 Transformers 完全一致 | 3/4 | 4/4 |
| 四个提示首 token logits 的最大绝对误差 | 0.328125 | 0.359375 |
| 四个提示首 token logits 的最大 RMS 误差 | 0.0627511 | 0.0609491 |
| 再次调用第一条提示的输出 ID 与本引擎上次结果一致 | 是 | 是 |

TP1 中文自我介绍首 token 出现分歧。Transformers 的“我是”与“你好”
候选 logits 同为 23.375；本引擎“你好”为 23.5，“我是”为 23.375。
因此 TP1 从“你好，我是 …”开始，基线从“我是 …”开始。这支持数值
舍入在并列候选附近改变 argmax 的解释，但不是所有差异都已被完全归因的证明。

两种 TP 的乘法回答均为 `391`；代码提示均生成偶数判断函数开头；
162-token 分块输入均回答“完成”。代码生成在 32-token 上限处截断，
不能把它当作完整函数测试。

完整证据保留在服务器 `/mnt/waiwai/work/`：

- `qwen35-tp1-validation.json`、`qwen35-tp2-validation.json`：输出 ID、参考 ID、logits 误差及 top-5。
- `qwen35-tp1-validation.log`、`qwen35-tp2-validation.log`：真实模型执行日志。
- `qwen35-unit-validation.log`、`qwen35-numerics-validation.log`、`qwen3-regression-validation.log`：专项测试日志。
- `qwen35-install-validation.log`：安装记录。
- `nano-vllm-download-qwen35.log`：完整模型下载完成记录。

这些是功能与小规模数值验证，不是完整质量基准、生产性能或容量结论。
最长真实模型验证输入为 162 token；小模型抢占验证输入为 255 token，
解码跨过 256-token 分页边界。没有完成 Qwen3.5-4B 的 4K/长上下文测试。
