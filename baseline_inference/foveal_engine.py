"""Serial cached generation for Foveal CPT checkpoints.

Decode gathers local tokens and selected remote pages. KV storage still grows
with the full prefix; no 512K/1M VRAM or flat-latency guarantee is established.
"""

from __future__ import annotations

import atexit
import gc
import math
import time
from dataclasses import fields
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from foveal_cpt.checkpoint import load_foveal_weights, load_pretrained
from foveal_cpt.config import FovealConfig
from inference.sampling_params import SamplingParams
from model.blocks import gated_delta_chunked, polar_reduce


class FovealLLM:
    """Autoregressive generation and serving engine for Foveal CPT models.

    Requests run serially, using the checkpoint's page size, window, and heads.
    CUDA checkpoint parity and serving performance require separate validation.
    """

    def __init__(self, model_path: str, **kwargs: Any):
        self.model_path = model_path
        self.device = torch.device(kwargs.pop("device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_model_len = kwargs.get("max_model_len")

        # Resolve weights and config
        from benchmarks.model import read_checkpoint_config, resolve_checkpoint

        self.weights_path, self.ckpt_dir = resolve_checkpoint(model_path)
        self.cfg_dict = read_checkpoint_config(model_path)
        foveal_raw = self.cfg_dict.get("foveal_config") or self.cfg_dict
        while isinstance(foveal_raw.get("foveal_config"), dict):
            foveal_raw = foveal_raw["foveal_config"]
        valid_fields = {f.name for f in fields(FovealConfig)}
        config_values = {k: v for k, v in foveal_raw.items() if k in valid_fields}
        if not config_values.get("checkpoint"):
            base_checkpoint = self.cfg_dict.get("base_checkpoint") or self.cfg_dict.get("checkpoint")
            if base_checkpoint:
                config_values["checkpoint"] = base_checkpoint
        self.foveal_config = FovealConfig(**config_values)

        self.foveal_config.validate()

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_dir, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

        # Load model
        base, self.atma_config, _ = load_pretrained(self.foveal_config, device=self.device)
        self.model = base
        load_foveal_weights(self.model, self.weights_path)
        self.model.eval()

        # Configure foveal layers for eval/serving
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "set_mode"):
                attn.set_mode("sparse")
                attn.teacher_query_blocks = 0
                attn.set_route(
                    self.foveal_config.top_p,
                    self.foveal_config.min_remote_pages,
                    self.foveal_config.max_remote_pages,
                )

        self._last_metrics: dict[str, Any] | None = None
        atexit.register(self.exit)

    @property
    def last_metrics(self) -> dict[str, Any] | None:
        return self._last_metrics

    def exit(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.exit()

    @staticmethod
    def _history(x: torch.Tensor, width: int) -> torch.Tensor:
        history = x.transpose(1, 2)
        if width == 0:
            return history[:, :, :0].clone()
        history = history[:, :, -width:]
        return F.pad(history, (max(0, width - history.shape[-1]), 0)).clone()

    @staticmethod
    def _rotate(x: torch.Tensor, rotary, pos: int) -> torch.Tensor:
        theta = pos * rotary.angular_freq
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat((left * theta.cos() + right * theta.sin(),
                          right * theta.cos() - left * theta.sin()), dim=-1).type_as(x)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        if temperature == 0:
            return int(logits.argmax(-1).item())
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probabilities.reshape(-1), 1).item())

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.model.proj(self.model.norm(hidden)).float()
        return 15.0 * logits * (logits.square() + 225.0).rsqrt()

    @staticmethod
    def _conv_step(full: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # The prefill CUDA convolution accumulates products in FP32 and rounds
        # once. BF16 multiply followed by sum rounds every product first.
        dtype = full.dtype
        compute = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
        return (full.to(compute) * weight.to(compute)[None]).sum(
            -1, keepdim=True).to(dtype)

    @staticmethod
    def _set_route(attn, qi, qb, cache, idx):
        """Route once from the block's first query, over earlier complete pages."""
        pages = cache["k_pages"][idx][:, :qb]
        scores = F.normalize(qi.float(), dim=-1) @ pages.transpose(1, 2)
        scores = scores / math.sqrt(attn.index_dim)
        cache["cached_scores"][idx] = scores
        cache["current_qb"][idx] = qb
        selected = torch.empty(0, dtype=torch.long, device=qi.device)
        first_local = max(0, qb - attn.local_window // attn.page_size)
        if attn.adaptation_mode != "local" and first_local:
            probs = scores[0, 0].softmax(-1)
            target = (attn.route_top_p - probs[first_local:].sum()).clamp_min(0)
            # Include invalid slots so tie ordering matches the full-prefix router.
            remote = F.pad(probs, (0, 1), value=-1)
            remote[first_local:] = -1
            sorted_prob, order = remote.sort(descending=True)
            cumulative = sorted_prob.clamp_min(0).cumsum(-1)
            needed = int(((cumulative < target).sum() + (target > 0)).item())
            count = min(max(needed, int(attn.route_min_remote_pages.item())),
                        int(attn.route_max_remote_pages.item()), first_local)
            selected = order[:count]
        cache["selected_remote"][idx] = selected

    @staticmethod
    def _memory_state(attn, x, q, k, v):
        """Return the unpadded prefix state in [B,H,K,V] layout."""
        mem = attn.base.mem
        groups = attn.num_heads // attn.num_kv_heads
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
        g_logit = mem.w_gamma(x).float() + mem.gamma_bias
        beta = torch.sigmoid(mem.w_beta(x).float() + mem.beta_bias)
        from model import blocks as memory_ops

        if x.is_cuda and mem.kernel in ("auto", "fla") and memory_ops._HAS_FLA:
            _, state = memory_ops.chunk_gated_delta_rule(
                q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                g=F.logsigmoid(g_logit).contiguous(), beta=beta.contiguous(),
                scale=1.0, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            return state.float()
        _, state = gated_delta_chunked(
            F.normalize(q.transpose(1, 2).float(), dim=-1),
            F.normalize(k.transpose(1, 2).float(), dim=-1),
            v.transpose(1, 2).float(), torch.sigmoid(g_logit).transpose(1, 2),
            beta.transpose(1, 2), chunk=mem.chunk,
        )
        # The torch reference uses [V,K]; FLA and this decoder use [K,V].
        return state.transpose(-1, -2).contiguous()

    @torch.no_grad()
    def _prefill(self, prompt_tokens: list[int], temperature: float = 0.0):
        if not prompt_tokens:
            raise ValueError("prefill requires at least one token")
        length = len(prompt_tokens)
        page_size = self.foveal_config.page_size
        eos = self.tokenizer.eos_token_id
        padded = prompt_tokens + [eos if eos is not None else 0] * (-length % page_size)
        x = self.model.embed(torch.tensor([padded], dtype=torch.long, device=self.device))
        cache = {name: {} for name in (
            "conv_states", "canon_q", "canon_k", "canon_v", "mem_states",
            "k_cache", "v_cache", "k_pages", "v_pages", "partial_k", "partial_v",
            "current_qb", "selected_remote", "cached_scores",
        )}
        cache["seq_len"] = length
        for idx, block in enumerate(self.model.blocks):
            attn = block.attn
            normed = block.norm1(x)
            if not hasattr(attn, "page_size"):
                b, _, value = attn.in_proj(normed).chunk(3, dim=-1)
                cache["conv_states"][idx] = self._history(
                    (b * value)[:, :length], attn.conv.weight.shape[-1] - 1)
            else:
                _, _, k, v, qm, km, vm = attn._project_qkv(normed)
                cache["k_cache"][idx] = k[:, :length].clone()
                cache["v_cache"][idx] = v[:, :length].clone()
                if attn.base.canon_q is not None:
                    heads, kv_heads, dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
                    q = attn.base.q(normed).view(1, -1, heads, 2 * dim).chunk(2, -1)[0]
                    raw_k = attn.base.k(normed).view(1, -1, kv_heads, dim)
                    raw_v = attn.base.v(normed).view(1, -1, kv_heads, dim)
                    for name, value in (("q", F.rms_norm(q, (dim,))),
                                        ("k", F.rms_norm(raw_k, (dim,))), ("v", raw_v)):
                        conv = getattr(attn.base, "canon_" + name)
                        cache["canon_" + name][idx] = self._history(
                            value[:, :length].flatten(2), conv.weight.shape[-1] - 1)
                if attn.base.mem is not None:
                    cache["mem_states"][idx] = self._memory_state(
                        attn, normed[:, :length], qm[:, :length], km[:, :length], vm[:, :length])

                qi = F.rms_norm(attn.index_q(normed), (attn.index_dim,))
                ki = F.rms_norm(attn.index_k(normed), (attn.index_dim,))
                vi = attn.index_v(normed)
                if attn.index_rotary is not None:
                    qi = attn.index_rotary(qi.unsqueeze(2)).squeeze(2)
                    ki = attn.index_rotary(ki.unsqueeze(2)).squeeze(2)
                complete = length // page_size
                end = complete * page_size
                kp = ki[:, :end].reshape(1, complete, page_size, attn.index_dim).mean(2)
                vp = vi[:, :end].reshape(1, complete, page_size, attn.index_dim).mean(2)
                cache["k_pages"][idx] = F.normalize(kp.float(), dim=-1)
                cache["v_pages"][idx] = vp
                cache["partial_k"][idx] = ki[:, end:length].clone()
                cache["partial_v"][idx] = vi[:, end:length].clone()
                cache["current_qb"][idx] = -1
                if length % page_size:
                    self._set_route(attn, qi[:, end:end+1], complete, cache, idx)
            x = block(x)[0]
        cache["logits"] = self._logits(x[:, length-1:length])
        return self._sample(cache["logits"], temperature), cache

    @torch.no_grad()
    def _decode_step(self, tok: int, cache: dict[str, Any], temperature: float = 0.0):
        pos = cache["seq_len"]
        x = self.model.embed(torch.tensor([[tok]], device=self.device))
        for idx, block in enumerate(self.model.blocks):
            attn = block.attn
            normed = block.norm1(x)
            if not hasattr(attn, "page_size"):
                b, c, value = attn.in_proj(normed).chunk(3, -1)
                full = torch.cat((cache["conv_states"][idx], (b * value).transpose(1, 2)), -1)
                weight = attn.conv.weight.squeeze(1).to(full.dtype)
                conv = self._conv_step(full, weight).transpose(1, 2)
                cache["conv_states"][idx] = full[:, :, 1:]
                out = attn.out_proj(c * conv)
            else:
                heads, kv_heads, dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
                groups = heads // kv_heads
                qb = pos // attn.page_size
                qi = F.rms_norm(attn.index_q(normed), (attn.index_dim,))
                ki = F.rms_norm(attn.index_k(normed), (attn.index_dim,))
                vi = attn.index_v(normed)
                if attn.index_rotary is not None:
                    qi = self._rotate(qi, attn.index_rotary, pos)
                    ki = self._rotate(ki, attn.index_rotary, pos)
                if cache["current_qb"][idx] != qb:
                    self._set_route(attn, qi, qb, cache, idx)

                q, gate = attn.base.q(normed).view(1, 1, heads, 2 * dim).chunk(2, -1)
                k = attn.base.k(normed).view(1, 1, kv_heads, dim)
                v = attn.base.v(normed).view(1, 1, kv_heads, dim)
                q, k = F.rms_norm(q, (dim,)), F.rms_norm(k, (dim,))
                qm, km, vm = q, k, v
                if getattr(attn.base, "pos", None) == "rope":
                    q = self._rotate(q, attn.base.rotary, pos)
                    k = self._rotate(k, attn.base.rotary, pos)
                else:
                    projected = []
                    for name, value in (("q", q), ("k", k), ("v", v)):
                        raw = value.flatten(2).transpose(1, 2)
                        full = torch.cat((cache["canon_" + name][idx], raw), -1)
                        weight = getattr(attn.base, "canon_" + name).weight.squeeze(1).to(value.dtype)
                        conv = self._conv_step(full, weight)
                        projected.append((raw + conv).transpose(1, 2).reshape_as(value))
                        cache["canon_" + name][idx] = full[:, :, 1:]
                    q, k, v = projected
                    qm, km, vm = q, k, v
                cache["k_cache"][idx] = torch.cat((cache["k_cache"][idx], k), 1)
                cache["v_cache"][idx] = torch.cat((cache["v_cache"][idx], v), 1)
                local = torch.arange(max(0, pos + 1 - attn.local_window), pos + 1, device=self.device)
                remote = cache["selected_remote"][idx]
                offsets = torch.arange(attn.page_size, device=self.device)
                remote_tokens = (remote[:, None] * attn.page_size + offsets).flatten()
                active = torch.cat((remote_tokens, local))
                kt = cache["k_cache"][idx][:, active].repeat_interleave(groups, 2).transpose(1, 2)
                vt = cache["v_cache"][idx][:, active].repeat_interleave(groups, 2).transpose(1, 2)
                qt = q.transpose(1, 2)
                if attn.is_polar:
                    # Match Triton's FP32 score accumulator; a BF16 matmul
                    # result would round scores before temperature/softmax.
                    scores = (qt.float() @ kt.float().transpose(-2, -1)) / math.sqrt(dim)
                    direction, mag = polar_reduce(
                        scores, vt, torch.tensor([pos + 1.0], device=self.device),
                        v_null=attn.base.v_null, null_base=attn.base.null_base,
                        null_slope_raw=attn.base.null_slope_raw,
                        len_gain_raw=attn.base.len_gain_raw, mag_beta_raw=attn.base.mag_beta_raw,
                    )
                    flat = direction.transpose(1, 2).flatten(2)
                    out = attn.base.proj(flat * torch.sigmoid(gate.flatten(2)))
                    out = out + attn.base.mu_proj(mag.transpose(1, 2))
                else:
                    attended = F.scaled_dot_product_attention(qt, kt, vt, scale=attn.base.sdpa_scale)
                    flat = attended.transpose(1, 2).flatten(2)
                    out = attn.base.proj(flat * torch.sigmoid(gate.flatten(2)))
                if attn.uses_lm_output and qb:
                    scores = cache["cached_scores"][idx]
                    values = cache["v_pages"][idx][:, :qb]
                    context = scores.softmax(-1).to(values.dtype) @ values
                    out = out + attn.index_out(context)

                if attn.base.mem is not None:
                    mem = attn.base.mem
                    g_logit = mem.w_gamma(normed).float() + mem.gamma_bias
                    beta = torch.sigmoid(mem.w_beta(normed).float() + mem.beta_bias)
                    from model import blocks as memory_ops

                    if x.is_cuda and mem.kernel in ("auto", "fla") and memory_ops._HAS_FLA:
                        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

                        read, state = fused_recurrent_gated_delta_rule(
                            q=qm.contiguous(), k=km.repeat_interleave(groups, 2).contiguous(),
                            v=vm.repeat_interleave(groups, 2).contiguous(),
                            g=F.logsigmoid(g_logit).contiguous(), beta=beta.contiguous(),
                            initial_state=cache["mem_states"][idx], output_final_state=True,
                            scale=1.0, use_qk_l2norm_in_kernel=True,
                        )
                        cache["mem_states"][idx] = state
                    else:
                        gamma = torch.sigmoid(g_logit)[0, 0]
                        qn = F.normalize(qm[0, 0].float(), dim=-1)
                        kn = F.normalize(km[0, 0].repeat_interleave(groups, 0).float(), dim=-1)
                        vn = vm[0, 0].repeat_interleave(groups, 0).float()
                        state = gamma[:, None, None] * cache["mem_states"][idx][0]
                        prediction = torch.einsum("hkv,hk->hv", state, kn)
                        update = beta[0, 0, :, None] * (vn - prediction)
                        state = state + kn[:, :, None] * update[:, None, :]
                        cache["mem_states"][idx] = state.unsqueeze(0)
                        read = torch.einsum("hkv,hk->hv", state, qn)
                    read = F.rms_norm(read, (dim,)).reshape(1, 1, -1).to(x.dtype)
                    out = out + mem.proj(read * torch.sigmoid(mem.gate(normed)))

                # The new page becomes eligible only for the NEXT query block.
                pk = torch.cat((cache["partial_k"][idx], ki), 1)
                pv = torch.cat((cache["partial_v"][idx], vi), 1)
                if pk.shape[1] == attn.page_size:
                    kp = F.normalize(pk.mean(1, keepdim=True).float(), dim=-1)
                    cache["k_pages"][idx] = torch.cat((cache["k_pages"][idx], kp), 1)
                    cache["v_pages"][idx] = torch.cat((cache["v_pages"][idx], pv.mean(1, keepdim=True)), 1)
                    pk, pv = pk[:, :0], pv[:, :0]
                cache["partial_k"][idx], cache["partial_v"][idx] = pk, pv
            x = x + out
            x = x + block.mlp(block.norm2(x))
        cache["seq_len"] = pos + 1
        cache["logits"] = self._logits(x)
        return self._sample(cache["logits"], temperature)

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        del use_tqdm
        params = sampling_params if sampling_params is not None else SamplingParams()
        params = params if isinstance(params, list) else [params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("sampling_params must contain one entry per prompt")
        for sp in params:
            if not isinstance(sp.max_tokens, int) or sp.max_tokens < 0:
                raise ValueError("max_tokens must be a nonnegative integer")
            if not math.isfinite(sp.temperature) or sp.temperature < 0:
                raise ValueError("temperature must be finite and nonnegative")
        outputs = []
        metrics = dict(prefill_tokens=0, decode_tokens=0, prefill_time=0.0, decode_time=0.0)

        def sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        for prompt, sp in zip(prompts, params):
            if sp.max_tokens == 0:
                outputs.append({"text": "", "token_ids": []})
                continue
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
            eos = self.tokenizer.eos_token_id
            if not tokens:
                if eos is None:
                    raise ValueError("an empty prompt requires a tokenizer EOS token")
                tokens = [eos]
            limit = getattr(self, "max_model_len", None)
            if limit is not None and len(tokens) + sp.max_tokens > limit:
                raise ValueError("prompt plus max_tokens exceeds max_model_len")
            sync()
            start = time.perf_counter()
            token, cache = self._prefill(tokens, sp.temperature)
            sync()
            metrics["prefill_time"] += time.perf_counter() - start
            metrics["prefill_tokens"] += len(tokens)
            generated = [token]
            while len(generated) < sp.max_tokens and (sp.ignore_eos or token != eos):
                sync()
                start = time.perf_counter()
                token = self._decode_step(token, cache, sp.temperature)
                sync()
                metrics["decode_time"] += time.perf_counter() - start
                metrics["decode_tokens"] += 1
                generated.append(token)
            outputs.append({"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                            "token_ids": generated})
            del cache
        for phase in ("prefill", "decode"):
            elapsed = metrics[phase + "_time"]
            metrics[phase + "_throughput"] = metrics[phase + "_tokens"] / elapsed if elapsed else 0.0
        self._last_metrics = metrics
        return outputs
