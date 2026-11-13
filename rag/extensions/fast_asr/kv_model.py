"""Investigation variants of GLM-ASR-Nano, built around the supplied model object.

V1  generate_recompute  same control flow as GlmAsr.generate (no KV cache), with timing
V2  generate_kv         prefill once, then one position per token from a KV cache
V3  generate_kv(encoder_frames=T)  V2 with the encoder input trimmed to T mel frames

Every operation goes through glm_asr_cuda.ops, so ASR_CUDA / ops.set_cuda_groups picks the
student kernels or the PyTorch reference exactly as for the supplied model. The supplied
model.py is never modified; FastGlmAsr only reads base.c, base.w and calls its methods.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from . import _paths  # noqa: F401
from glm_asr_cuda import ops
from glm_asr_cuda.model import GlmAsr, load_model

from .rope_offset import rope_at_


@dataclass
class GenResult:
    tokens: list
    probs: list
    encode_ms: float
    decode_ms: float
    prompt_len: int
    enc_frames: int
    kv_bytes: int = 0
    prefill_ms: float = 0.0  # part of decode_ms: prompt splice, (cache alloc,) prompt pass, token 1


@dataclass
class KVCache:
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)
    s_max: int = 0

    @classmethod
    def allocate(cls, layers, s_max, kv_heads, head_dim, device):
        shape = (1, s_max, kv_heads, head_dim)
        return cls([torch.zeros(shape, device=device) for _ in range(layers)],
                   [torch.zeros(shape, device=device) for _ in range(layers)], s_max)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def prepare_inputs(processor, audio) -> dict:
    """Processor (CPU mel) + upload; feat_ms is the CPU feature time."""
    t0 = time.perf_counter()
    inputs = processor.apply_transcription_request(audio)
    feat_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "features": inputs["input_features"].to("cuda", torch.float32),
        "input_ids": inputs["input_ids"].to("cuda"),
        "features_mask": inputs["input_features_mask"].to("cuda"),
        "feat_ms": feat_ms,
    }


def real_frames(features_mask) -> int:
    return int(features_mask[0].sum())


class _Timer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()

    @property
    def ms(self) -> float:
        return self.start.elapsed_time(self.end)


class FastGlmAsr:
    def __init__(self, base: GlmAsr):
        self.base = base

    # ---------------- shared pieces ----------------

    def _prompt(self, audio, input_ids):
        """Embed the prompt and splice the audio embeddings in, exactly as GlmAsr.generate."""
        b = self.base
        embeds = ops.embedding(input_ids, b.w["language_model.model.embed_tokens.weight"])
        is_audio = (input_ids[0] == b.c.audio_token_id).nonzero().flatten()
        first, last = int(is_audio[0]), int(is_audio[-1])
        return torch.cat([embeds[:, :first], audio[None], embeds[:, last + 1:]], dim=1)

    def _embed_token(self, best):
        return ops.embedding(best[:, None], self.base.w["language_model.model.embed_tokens.weight"])

    def _head(self, h):
        """Final norm + lm_head + softmax on the last position (as GlmAsr.next_token_probs)."""
        b = self.base
        last = ops.rmsnorm(h[:, -1], b.w["language_model.model.norm.weight"], b.c.rms_eps)
        logits = ops.linear(last, b.w["language_model.lm_head.weight"])
        return ops.softmax(logits)

    # ---------------- V1 ----------------

    @torch.no_grad()
    def generate_recompute(self, features, input_ids, features_mask, max_new_tokens=100) -> GenResult:
        b = self.base
        with _Timer() as enc:
            audio = b.encode_audio(features, features_mask)
        with _Timer() as dec:
            with _Timer() as pre:
                embeds = self._prompt(audio, input_ids)
                prompt_len = embeds.shape[1]
                p = b.next_token_probs(embeds)
            tokens, probs = [], []
            for step in range(max_new_tokens):
                if step > 0:
                    p = b.next_token_probs(embeds)
                best = torch.argmax(p, dim=-1)
                tokens.append(int(best[0]))
                probs.append(float(p[0, best[0]]))
                if tokens[-1] in b.c.eos_token_ids:
                    break
                embeds = torch.cat([embeds, self._embed_token(best)], dim=1)
        torch.cuda.synchronize()
        return GenResult(tokens, probs, enc.ms, dec.ms, prompt_len, features.shape[-1],
                         prefill_ms=pre.ms)

    # ---------------- V2 / V3 ----------------

    def decoder_layer_kv(self, h, i, cache: KVCache, start: int):
        """GlmAsr.decoder_layer for positions start..start+S-1, reading earlier k/v from cache."""
        b = self.base
        c, w, p = b.c, b.w, f"language_model.model.layers.{i}."
        B, S, _ = h.shape
        x = ops.rmsnorm(h, w[p + "input_layernorm.weight"], c.rms_eps)
        q = ops.linear(x, w[p + "self_attn.q_proj.weight"]).view(B, S, c.text_heads, b.text_head_dim)
        k = ops.linear(x, w[p + "self_attn.k_proj.weight"]).view(B, S, c.text_kv_heads, b.text_head_dim)
        v = ops.linear(x, w[p + "self_attn.v_proj.weight"]).view(B, S, c.text_kv_heads, b.text_head_dim)
        rope_at_(q, k, b.text_inv_freq, b.text_rotary_dim, start)
        end = start + S
        if end > cache.s_max:
            raise RuntimeError(f"KV cache overflow: writing up to {end} > s_max {cache.s_max}")
        cache.k[i][:, start:end].copy_(k)
        cache.v[i][:, start:end].copy_(v)
        a = ops.attention(q, cache.k[i][:, :end], cache.v[i][:, :end], causal=True).view(B, S, c.text_hidden)
        h = h + ops.linear(a, w[p + "self_attn.o_proj.weight"])
        x = ops.rmsnorm(h, w[p + "post_attention_layernorm.weight"], c.rms_eps)
        gate = ops.linear(x, w[p + "mlp.gate_proj.weight"])
        up = ops.linear(x, w[p + "mlp.up_proj.weight"])
        return h + ops.linear(ops.silu_mul(gate, up), w[p + "mlp.down_proj.weight"])

    def _decoder_kv(self, h, cache, start):
        for i in range(self.base.c.text_layers):
            h = self.decoder_layer_kv(h, i, cache, start)
        return h

    @torch.no_grad()
    def generate_kv(self, features, input_ids, features_mask, max_new_tokens=100,
                    encoder_frames: int | None = None) -> GenResult:
        b = self.base
        if encoder_frames is not None:
            real = real_frames(features_mask)
            if encoder_frames % 8 or encoder_frames < real or encoder_frames > features.shape[-1]:
                raise ValueError(f"encoder_frames must be a multiple of 8 in [{real}, {features.shape[-1]}], "
                                 f"got {encoder_frames}")
            features = features[:, :, :encoder_frames].contiguous()
            features_mask = features_mask[:, :encoder_frames].contiguous()
        with _Timer() as enc:
            audio = b.encode_audio(features, features_mask)
        with _Timer() as dec:
            with _Timer() as pre:
                embeds = self._prompt(audio, input_ids)
                P = embeds.shape[1]
                cache = KVCache.allocate(b.c.text_layers, P + max_new_tokens, b.c.text_kv_heads,
                                         b.text_head_dim, embeds.device)
                p = self._head(self._decoder_kv(embeds, cache, 0))  # prefill
            tokens, probs = [], []
            for step in range(max_new_tokens):
                best = torch.argmax(p, dim=-1)
                tokens.append(int(best[0]))
                probs.append(float(p[0, best[0]]))
                if tokens[-1] in b.c.eos_token_ids or step == max_new_tokens - 1:
                    break
                p = self._head(self._decoder_kv(self._embed_token(best), cache, P + step))
        torch.cuda.synchronize()
        return GenResult(tokens, probs, enc.ms, dec.ms, P, features.shape[-1], cache.nbytes,
                         prefill_ms=pre.ms)


def use_cache_if_present():
    """GPU nodes may have no internet: stay offline when the model is already downloaded
    (same logic as hw1-asr/benchmark.py)."""
    import os
    from huggingface_hub import snapshot_download
    from glm_asr_cuda.model import MODEL_NAME
    try:
        snapshot_download(MODEL_NAME, local_files_only=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
    except Exception:
        pass


def load_fast(device="cuda"):
    use_cache_if_present()
    base, processor = load_model(device)
    return FastGlmAsr(base), processor
