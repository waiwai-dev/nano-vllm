from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative (0 selects greedy decoding)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
