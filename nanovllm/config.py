import os
import warnings
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_cache: bool = True
    is_hybrid: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        model_type = self.hf_config.model_type
        self.is_hybrid = model_type in ("qwen3_5", "qwen3_5_text")
        if self.is_hybrid:
            self.hf_config = getattr(self.hf_config, "text_config", self.hf_config)
            if self.tensor_parallel_size > self.hf_config.num_key_value_heads:
                raise ValueError("Qwen3.5 TP must divide the number of KV heads")
            if not self.enforce_eager or self.enable_prefix_cache:
                warnings.warn("Qwen3.5 uses eager execution with prefix caching disabled; GDN state snapshots are not implemented.")
            self.enforce_eager = True
            self.enable_prefix_cache = False
        if self.max_num_seqs is None:
            self.max_num_seqs = 16 if self.is_hybrid else 512
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1 or self.max_model_len < 1:
            raise ValueError("Sequence and token limits must be positive")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
