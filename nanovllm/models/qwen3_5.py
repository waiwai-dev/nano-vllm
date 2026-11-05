"""Text-only Qwen3.5 with paged full attention and request-local GDN states.

Equations and checkpoint layout follow the Apache-2.0 Qwen/Transformers
implementation. Gated DeltaNet uses flash-linear-attention's Triton kernels.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear, divide
from nanovllm.models.qwen3 import Qwen3MLP
from nanovllm.utils.context import get_context


class RMSNorm(nn.Module):
    def __init__(self, size, eps, zero_centered=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size))
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, x, gate=None):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if self.zero_centered:
            y = (y * (1 + self.weight.float())).to(x.dtype)
        else:
            y = y.to(x.dtype) * self.weight
        if gate is not None:
            y = (y.float() * F.silu(gate.float())).to(x.dtype)
        return y


class FullAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        tp = dist.get_world_size()
        self.num_heads = divide(config.num_attention_heads, tp)
        self.num_kv_heads = divide(config.num_key_value_heads, tp)
        self.head_dim = config.head_dim
        self.q_proj = ColumnParallelLinear(config.hidden_size, config.num_attention_heads * self.head_dim * 2)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        rope = config.rope_parameters
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Qwen3.5 currently supports default RoPE only")
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        self.register_buffer("inv_freq", 1.0 / (rope["rope_theta"] ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        )), persistent=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def rotate(self, x, cos, sin):
        rotary, rest = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        a, b = rotary.chunk(2, -1)
        # HF applies RoPE in the input dtype, including each multiplication.
        rotated = rotary * cos + torch.cat((-b, a), -1) * sin
        return torch.cat((rotated, rest), -1)

    def forward(self, positions, x):
        q, gate = self.q_proj(x).view(-1, self.num_heads, self.head_dim * 2).chunk(2, -1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        freq = positions.float()[:, None] * self.inv_freq[None, :]
        freq = torch.cat((freq, freq), -1)[:, None, :]
        cos, sin = freq.cos().to(x.dtype), freq.sin().to(x.dtype)
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
        output = self.attn(q, k, v).reshape(-1, self.num_heads, self.head_dim)
        return self.o_proj((output * gate.sigmoid()).flatten(1))


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        except ImportError as exc:
            raise ImportError("Qwen3.5 requires pip install 'nano-vllm[qwen35]'") from exc
        self.chunk_rule = chunk_gated_delta_rule
        self.recurrent_rule = fused_recurrent_gated_delta_rule
        self.tp = dist.get_world_size()
        self.rank = dist.get_rank()
        self.nk = divide(config.linear_num_key_heads, self.tp)
        self.nv = divide(config.linear_num_value_heads, self.tp)
        divide(self.nv, self.nk)
        self.dk = config.linear_key_head_dim
        self.dv = config.linear_value_head_dim
        self.kdim, self.vdim = self.nk * self.dk, self.nv * self.dv
        self.kernel_size = config.linear_conv_kernel_dim
        self.global_dims = [config.linear_num_key_heads * self.dk] * 2 + [config.linear_num_value_heads * self.dv]
        self.in_proj_qkv = nn.Linear(config.hidden_size, 2 * self.kdim + self.vdim, bias=False)
        self.in_proj_qkv.weight.weight_loader = self.load_segmented
        self.in_proj_z = ColumnParallelLinear(config.hidden_size, config.linear_num_value_heads * self.dv)
        self.in_proj_a = ColumnParallelLinear(config.hidden_size, config.linear_num_value_heads)
        self.in_proj_b = ColumnParallelLinear(config.hidden_size, config.linear_num_value_heads)
        self.conv1d = nn.Conv1d(2 * self.kdim + self.vdim, 2 * self.kdim + self.vdim,
                                self.kernel_size, groups=2 * self.kdim + self.vdim, bias=False)
        self.conv1d.weight.weight_loader = self.load_segmented
        self.dt_bias = nn.Parameter(torch.empty(self.nv))
        self.A_log = nn.Parameter(torch.empty(self.nv, dtype=torch.float32))
        self.dt_bias.weight_loader = self.load_heads
        self.A_log.weight_loader = self.load_heads
        self.norm = RMSNorm(self.dv, config.rms_norm_eps, zero_centered=False)
        self.out_proj = RowParallelLinear(config.linear_num_value_heads * self.dv, config.hidden_size)

    def load_segmented(self, param, weight):
        param.data.copy_(torch.cat([w.chunk(self.tp, 0)[self.rank] for w in weight.split(self.global_dims, 0)], 0))

    def load_heads(self, param, weight):
        param.data.copy_(weight.chunk(self.tp, 0)[self.rank])

    def allocate_states(self, slots):
        device, dtype = self.conv1d.weight.device, self.conv1d.weight.dtype
        self.register_buffer("conv_states", torch.zeros(slots, 2 * self.kdim + self.vdim,
                             self.kernel_size - 1, device=device, dtype=dtype), persistent=False)
        self.register_buffer("recurrent_states", torch.zeros(slots, self.nv, self.dk, self.dv,
                             device=device, dtype=torch.float32), persistent=False)

    def mix(self, mixed, a, b, conv_state, recurrent_state, is_prefill):
        batch, length, _ = mixed.shape
        conv_input = torch.cat((conv_state, mixed.transpose(1, 2)), -1)
        new_conv = conv_input[..., -(self.kernel_size - 1):].contiguous()
        mixed = F.silu(F.conv1d(conv_input, self.conv1d.weight, groups=2 * self.kdim + self.vdim)).transpose(1, 2)
        q, k, v = mixed.split([self.kdim, self.kdim, self.vdim], -1)
        q, k = (t.reshape(batch, length, self.nk, self.dk).contiguous() for t in (q, k))
        v = v.reshape(batch, length, self.nv, self.dv).contiguous()
        g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())).contiguous()
        beta = b.sigmoid().contiguous()
        # FLA supports grouped value heads directly; states stay in fp32.
        rule = self.chunk_rule if is_prefill else self.recurrent_rule
        out, state = rule(q, k, v, g=g, beta=beta, initial_state=recurrent_state,
                          output_final_state=True, use_qk_l2norm_in_kernel=True)
        return out.reshape(batch, length, self.vdim), new_conv, state

    def forward(self, x):
        ctx = get_context()
        mixed = self.in_proj_qkv(x)
        a, b, z = self.in_proj_a(x), self.in_proj_b(x), self.in_proj_z(x)
        slots = ctx.state_slots
        if not ctx.is_prefill:
            out, conv, state = self.mix(mixed[:, None], a[:, None], b[:, None],
                self.conv_states[slots], self.recurrent_states[slots], False)
            self.conv_states.index_copy_(0, slots, conv)
            self.recurrent_states.index_copy_(0, slots, state)
            out = out[:, 0]
        else:
            outputs = []
            for i, (start, end, reset) in enumerate(ctx.state_spans):
                slot = slots[i:i + 1]
                conv, state = self.conv_states[slot], self.recurrent_states[slot]
                if reset:
                    conv.zero_()
                    state.zero_()
                out, conv, state = self.mix(mixed[None, start:end], a[None, start:end], b[None, start:end], conv, state, True)
                self.conv_states.index_copy_(0, slot, conv)
                self.recurrent_states.index_copy_(0, slot, state)
                outputs.append(out[0])
            out = torch.cat(outputs, 0)
        out = self.norm(out.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(-1, self.vdim)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        elif layer_type == "full_attention":
            self.self_attn = FullAttention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(self, positions, x):
        h = self.input_layernorm(x)
        x = x + (self.linear_attn(h) if self.layer_type == "linear_attention" else self.self_attn(positions, h))
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, kind) for kind in config.layer_types])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(positions, x)
        return self.norm(x)


class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}

    def __init__(self, config):
        super().__init__()
        self.model = TextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def checkpoint_name(self, name):
        # Official conditional-generation checkpoints include a vision tower and MTP.
        if name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")):
            return None
        if name.startswith("model.language_model."):
            return "model." + name.removeprefix("model.language_model.")
        if name.startswith("language_model."):
            return "model." + name.removeprefix("language_model.")
        return name

    def allocate_states(self, slots):
        for module in self.modules():
            if isinstance(module, GatedDeltaNet):
                module.allocate_states(slots)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
