"""Inference and serving engine for Foveal CPT checkpoints with true sparse indexing.

Supports all four Foveal adaptation modes: local, lm_output, kl, and lm_output_kl
across polar, nope, and rope cores.

Uses a dynamic KV-cache with active-page gathering:
  - Local sliding window: 512 tokens
  - Remote pages: up to 32 pages (2,048 tokens) selected via the 16D MQA indexer
  - Decode attention attends only to active tokens (at most 2,560 tokens)
  - Memory scales dynamically with context, supporting 512K and 1M length within 48GB GPU VRAM
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
from model.blocks import polar_reduce, polar_temp_null


class FovealLLM:
    """Autoregressive generation and serving engine for Foveal CPT models.

    Implements genuine Foveal sparse attention with the 16D MQA page indexer,
    dynamic KV caching, and sub-second per-step decode.
    """

    def __init__(self, model_path: str, **kwargs: Any):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve weights and config
        from benchmarks.model import read_checkpoint_config, resolve_checkpoint

        self.weights_path, self.ckpt_dir = resolve_checkpoint(model_path)
        self.cfg_dict = read_checkpoint_config(model_path)
        foveal_raw = self.cfg_dict.get("foveal_config") or self.cfg_dict
        valid_fields = {f.name for f in fields(FovealConfig)}
        self.foveal_config = FovealConfig(
            **{k: v for k, v in foveal_raw.items() if k in valid_fields}
        )

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

        self.offsets_64 = torch.arange(64, device=self.device)
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

    def _prefill(self, prompt_tokens: list[int]) -> tuple[int, dict[str, Any]]:
        """Run prefill forward pass, extract KV/indexer cache, and return first token."""
        T = len(prompt_tokens)
        pad_len = (64 - (T % 64)) % 64
        full_len = T + pad_len
        eos_id = self.tokenizer.eos_token_id if getattr(self.tokenizer, "eos_token_id", None) is not None else 50256
        padded_ids = prompt_tokens + [eos_id] * pad_len if pad_len > 0 else prompt_tokens
        inp_tensor = torch.tensor([padded_ids], dtype=torch.int32, device=self.device)

        conv_states: dict[int, torch.Tensor] = {}
        canon_q: dict[int, torch.Tensor] = {}
        canon_k: dict[int, torch.Tensor] = {}
        canon_v: dict[int, torch.Tensor] = {}
        mem_states: dict[int, torch.Tensor] = {}
        k_cache: dict[int, torch.Tensor] = {}
        v_cache: dict[int, torch.Tensor] = {}
        k_pages: dict[int, torch.Tensor] = {}
        v_pages: dict[int, torch.Tensor] = {}
        current_qb: dict[int, int] = {}
        selected_remote: dict[int, torch.Tensor] = {}
        cached_scores: dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x = self.model.embed(inp_tensor)
            for idx, block in enumerate(self.model.blocks):
                if idx % 4 != 2:
                    # LFM2Conv block
                    normed = block.norm1(x)
                    proj = block.attn.in_proj(normed)
                    B, C, x_proj = proj.chunk(3, dim=-1)
                    x_gated = B * x_proj
                    conv_states[idx] = x_gated[:, :T, :].transpose(1, 2)[:, :, -2:].clone()
                    x = block(x)[0]
                else:
                    # FovealAttention block
                    attn = block.attn
                    normed = block.norm1(x)
                    gate, q_attn, k_attn, v_attn, q_mem, k_mem, v_mem = attn._project_qkv(normed)
                    k_cache[idx] = k_attn[:, :T].contiguous().clone()
                    v_cache[idx] = v_attn[:, :T].contiguous().clone()

                    # Indexer pages
                    qi = F.rms_norm(attn.index_q(normed), (16,))
                    ki = F.rms_norm(attn.index_k(normed), (16,))
                    vi = attn.index_v(normed)
                    P = T // 64
                    if P > 0:
                        kp = ki[:, : P * 64].view(1, P, 64, 16).mean(dim=2)
                        vp = vi[:, : P * 64].view(1, P, 64, 16).mean(dim=2)
                        k_pages[idx] = F.normalize(kp.float(), dim=-1).to(k_attn.dtype)
                        v_pages[idx] = vp
                    else:
                        k_pages[idx] = torch.empty((1, 0, 16), device=self.device, dtype=k_attn.dtype)
                        v_pages[idx] = torch.empty((1, 0, 16), device=self.device, dtype=v_attn.dtype)

                    # Canon conv states for NoPE / Polar
                    if hasattr(attn.base, "canon_q") and attn.base.canon_q is not None:
                        q_raw = attn.base.q(normed).view(1, full_len, 8, 256).chunk(2, dim=-1)[0]
                        q_flat = F.rms_norm(q_raw, (128,)).reshape(1, full_len, -1)
                        k_raw = attn.base.k(normed).view(1, full_len, 2, 128)
                        k_flat = F.rms_norm(k_raw, (128,)).reshape(1, full_len, -1)
                        v_flat = attn.base.v(normed).view(1, full_len, 2, 128).reshape(1, full_len, -1)
                        canon_q[idx] = q_flat[:, :T, :].transpose(1, 2)[:, :, -3:].clone()
                        canon_k[idx] = k_flat[:, :T, :].transpose(1, 2)[:, :, -3:].clone()
                        canon_v[idx] = v_flat[:, :T, :].transpose(1, 2)[:, :, -3:].clone()

                    current_qb[idx] = -1
                    selected_remote[idx] = torch.empty(0, dtype=torch.long, device=self.device)
                    cached_scores[idx] = torch.zeros((1, 1, 0), device=self.device)

                    # Titans memory state
                    mem_states[idx] = torch.zeros(
                        1, attn.num_heads, attn.head_dim, attn.head_dim,
                        device=self.device, dtype=torch.float32,
                    )

                    x = block(x)[0]

            logits = self.model.proj(self.model.norm(x[:, T - 1 : T])).float()
            logits = 15.0 * logits * (logits.square() + 225.0).rsqrt()
            tok0 = int(logits[0, 0].argmax().item())

        cache = {
            "conv_states": conv_states,
            "canon_q": canon_q,
            "canon_k": canon_k,
            "canon_v": canon_v,
            "mem_states": mem_states,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "k_pages": k_pages,
            "v_pages": v_pages,
            "current_qb": current_qb,
            "selected_remote": selected_remote,
            "cached_scores": cached_scores,
            "seq_len": T,
        }
        return tok0, cache

    def _decode_step(self, tok: int, cache: dict[str, Any]) -> int:
        """Run a single-token autoregressive decode step with sparse index gathering."""
        pos = cache["seq_len"]
        qb = pos // 64
        tok_in = torch.tensor([[tok]], device=self.device)

        conv_states = cache["conv_states"]
        canon_q = cache["canon_q"]
        canon_k = cache["canon_k"]
        canon_v = cache["canon_v"]
        mem_states = cache["mem_states"]
        k_cache = cache["k_cache"]
        v_cache = cache["v_cache"]
        k_pages = cache["k_pages"]
        v_pages = cache["v_pages"]
        current_qb = cache["current_qb"]
        selected_remote = cache["selected_remote"]
        cached_scores = cache["cached_scores"]

        with torch.no_grad():
            xt = self.model.embed(tok_in)

            for idx, block in enumerate(self.model.blocks):
                if idx % 4 != 2:
                    # LFM2Conv
                    normed = block.norm1(xt)
                    proj = block.attn.in_proj(normed)
                    B, C, x_proj = proj.chunk(3, dim=-1)
                    x_gated = B * x_proj
                    st = conv_states[idx]
                    full_in = torch.cat([st, x_gated.transpose(1, 2)], dim=2)
                    w = block.attn.conv.weight.view(1024, 3).to(full_in.dtype)
                    conv_out = (full_in * w[None, :, :]).sum(dim=2, keepdim=True).transpose(1, 2)
                    conv_states[idx] = full_in[:, :, 1:]
                    xt = xt + block.attn.out_proj(C * conv_out)
                    xt = xt + block.mlp(block.norm2(xt))
                else:
                    # FovealAttention
                    attn = block.attn
                    normed = block.norm1(xt)

                    # 1. Indexer evaluation across all P pages
                    qi = F.rms_norm(attn.index_q(normed), (16,))
                    ki = F.rms_norm(attn.index_k(normed), (16,))
                    vi = attn.index_v(normed)
                    if attn.index_rotary is not None:
                        theta = (float(pos) * attn.index_rotary.angular_freq)[None, None, :]
                        cos, sin = theta.cos(), theta.sin()
                        x1, x2 = qi.float().chunk(2, dim=-1)
                        qi = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1).type_as(qi)
                        x1, x2 = ki.float().chunk(2, dim=-1)
                        ki = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1).type_as(ki)

                    P = k_pages[idx].shape[1]
                    if qb != current_qb[idx]:
                        current_qb[idx] = qb
                        q_block = F.normalize(qi.float(), dim=-1)
                        if P > 0:
                            scores = (q_block @ k_pages[idx].transpose(1, 2).float()) / 4.0
                            first_local = max(0, qb - 8)
                            probs = F.softmax(scores[0, 0].float(), dim=-1)
                            local_mass = probs[first_local:P].sum()
                            top_p_val = float(attn.route_top_p.detach().item())
                            target = max(0.0, top_p_val - float(local_mass))
                            remote_probs = probs[:first_local]
                            if len(remote_probs) > 0 and target > 0:
                                sorted_p, sorted_idx = remote_probs.sort(descending=True)
                                cum = sorted_p.cumsum(dim=-1)
                                needed = int((cum < target).sum().item()) + 1
                                min_rem = int(attn.route_min_remote_pages.detach().item())
                                max_rem = int(attn.route_max_remote_pages.detach().item())
                                k_rem = min(max(needed, min_rem), min(max_rem, len(remote_probs)))
                                selected_remote[idx] = sorted_idx[:k_rem]
                            else:
                                selected_remote[idx] = torch.empty(0, dtype=torch.long, device=self.device)
                            cached_scores[idx] = scores
                        else:
                            selected_remote[idx] = torch.empty(0, dtype=torch.long, device=self.device)
                            cached_scores[idx] = torch.zeros((1, 1, 0), device=self.device)

                    # 2. QKV projection
                    q_gate = attn.base.q(normed).view(1, 1, 8, 256)
                    q, gate = torch.chunk(q_gate, 2, dim=-1)
                    k = attn.base.k(normed).view(1, 1, 2, 128)
                    v = attn.base.v(normed).view(1, 1, 2, 128)
                    q = F.rms_norm(q, (128,))
                    k = F.rms_norm(k, (128,))

                    # Position encoding: RoPE or Canon conv step
                    if getattr(attn.base, "pos", None) == "rope":
                        theta = (float(pos) * attn.base.rotary.angular_freq)[None, None, None, :]
                        cos, sin = theta.cos(), theta.sin()
                        x1, x2 = q.float().chunk(2, dim=-1)
                        q = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1).type_as(q)
                        x1, x2 = k.float().chunk(2, dim=-1)
                        k = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1).type_as(k)
                    elif idx in canon_q:
                        wq = attn.base.canon_q.weight.squeeze(1).to(q.dtype)
                        wk = attn.base.canon_k.weight.squeeze(1).to(k.dtype)
                        wv = attn.base.canon_v.weight.squeeze(1).to(v.dtype)
                        qf = q.reshape(1, 1, 1024).transpose(1, 2)
                        kf = k.reshape(1, 1, 256).transpose(1, 2)
                        vf = v.reshape(1, 1, 256).transpose(1, 2)
                        fq = torch.cat([canon_q[idx], qf], dim=2)
                        fk = torch.cat([canon_k[idx], kf], dim=2)
                        fv = torch.cat([canon_v[idx], vf], dim=2)
                        q_conv = (fq * wq[None, :, :]).sum(dim=2, keepdim=True)
                        k_conv = (fk * wk[None, :, :]).sum(dim=2, keepdim=True)
                        v_conv = (fv * wv[None, :, :]).sum(dim=2, keepdim=True)
                        canon_q[idx] = fq[:, :, 1:]
                        canon_k[idx] = fk[:, :, 1:]
                        canon_v[idx] = fv[:, :, 1:]
                        q = (qf + q_conv).transpose(1, 2).reshape(1, 1, 8, 128)
                        k = (kf + k_conv).transpose(1, 2).reshape(1, 1, 2, 128)
                        v = (vf + v_conv).transpose(1, 2).reshape(1, 1, 2, 128)

                    # Append to dynamic KV cache
                    k_cache[idx] = torch.cat([k_cache[idx], k], dim=1)
                    v_cache[idx] = torch.cat([v_cache[idx], v], dim=1)

                    # 3. Gather active KV: local window (512) + selected remote pages (up to 32*64 = 2048)
                    k_local = k_cache[idx][:, max(0, pos - 511) : pos + 1]
                    v_local = v_cache[idx][:, max(0, pos - 511) : pos + 1]
                    rem = selected_remote[idx]
                    if len(rem) > 0:
                        rem_toks = (rem[:, None] * 64 + self.offsets_64[None, :]).flatten()
                        k_rem = k_cache[idx][:, rem_toks]
                        v_rem = v_cache[idx][:, rem_toks]
                        k_act = torch.cat([k_rem, k_local], dim=1)
                        v_act = torch.cat([v_rem, v_local], dim=1)
                    else:
                        k_act = k_local
                        v_act = v_local

                    # 4. Attention over active tokens (at most 2,560 tokens)
                    k_act_exp = k_act.repeat_interleave(4, dim=2)
                    v_act_exp = v_act.repeat_interleave(4, dim=2)
                    q_t = q.transpose(1, 2)
                    k_t = k_act_exp.transpose(1, 2)
                    v_t = v_act_exp.transpose(1, 2)

                    if attn.is_polar:
                        n_keys = torch.tensor([float(pos + 1)], device=self.device)
                        temp, null = polar_temp_null(
                            n_keys, attn.base.len_gain_raw, attn.base.null_base, attn.base.null_slope_raw
                        )
                        scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
                        c, mag = polar_reduce(
                            scores, v_t, n_keys,
                            v_null=attn.base.v_null,
                            null_base=attn.base.null_base,
                            null_slope_raw=attn.base.null_slope_raw,
                            len_gain_raw=attn.base.len_gain_raw,
                            mag_beta_raw=attn.base.mag_beta_raw,
                        )
                        flat = c.transpose(1, 2).reshape(1, 1, 1024)
                        content = attn.base.proj(flat * torch.sigmoid(gate.reshape(1, 1, -1)))
                        out = content + attn.base.mu_proj(mag.transpose(1, 2))
                    else:
                        scale = attn.base.sdpa_scale if getattr(attn.base, "sdpa_scale", None) is not None else None
                        attended = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
                        flat = attended.transpose(1, 2).reshape(1, 1, 1024)
                        out = attn.base.proj(flat * torch.sigmoid(gate.reshape(1, 1, -1)))

                    # 5. Soft page read for lm_output
                    if attn.uses_lm_output and P > 0:
                        vp = v_pages[idx]
                        sc = cached_scores[idx]
                        pr = F.softmax(sc[0, 0].float(), dim=-1).to(vp.dtype)
                        ctx_vec = pr.view(1, 1, -1) @ vp
                        out = out + attn.index_out(ctx_vec)

                    # 6. Titans Memory step
                    if attn.base.mem is not None:
                        mem = attn.base.mem
                        g_logit = mem.w_gamma(normed).float() + mem.gamma_bias
                        b_logit = mem.w_beta(normed).float() + mem.beta_bias
                        gamma = torch.sigmoid(g_logit)[0, 0]
                        beta = torch.sigmoid(b_logit)[0, 0]
                        qn = F.normalize(q.transpose(1, 2).float(), dim=-1)[0]
                        kn = F.normalize(k.repeat_interleave(4, dim=2).transpose(1, 2).float(), dim=-1)[0]
                        vn = v.repeat_interleave(4, dim=2).transpose(1, 2).float()[0]
                        S = mem_states[idx][0]
                        Sd = gamma[:, None, None] * S
                        pred = torch.einsum("hkv,hk->hv", Sd, kn[:, 0])
                        u = beta[:, None] * (vn[:, 0] - pred)
                        S_new = Sd + kn[:, 0, :, None] * u[:, None, :]
                        mem_states[idx][0] = S_new
                        r = torch.einsum("hkv,hk->hv", S_new, qn[:, 0])
                        r_norm = F.rms_norm(r.unsqueeze(1), (attn.head_dim,)).reshape(1, 1, 1024).to(xt.dtype)
                        out = out + mem.proj(r_norm * torch.sigmoid(mem.gate(normed)))

                    xt = xt + out
                    xt = xt + block.mlp(block.norm2(xt))

            logits = self.model.proj(self.model.norm(xt)).float()
            logits = 15.0 * logits * (logits.square() + 225.0).rsqrt()
            tok_next = int(logits[0, 0].argmax().item())

        cache["seq_len"] = pos + 1
        return tok_next

    def generate(
        self,
        prompts: list[Any],
        sampling_params: Any = None,
        use_tqdm: bool = False,
    ) -> list[dict[str, Any]]:
        del use_tqdm
        if sampling_params is None:
            sampling_params = SamplingParams()
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )

        outputs: list[dict[str, Any]] = []
        total_prefill_tokens = 0
        total_decode_tokens = 0
        total_prefill_time = 0.0
        total_decode_time = 0.0

        for prompt, sp in zip(prompts, params):
            if isinstance(prompt, str):
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            else:
                token_ids = list(prompt)

            max_new = getattr(sp, "max_tokens", 16) or 16
            eos = self.tokenizer.eos_token_id

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Prefill
            tok0, cache = self._prefill(token_ids)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            p_time = time.perf_counter() - t0
            total_prefill_time += p_time
            total_prefill_tokens += len(token_ids)

            # Decode
            generated = [tok0]
            cur_tok = tok0
            d_time = 0.0

            if cur_tok != eos:
                for _ in range(max_new - 1):
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t_step = time.perf_counter()

                    next_tok = self._decode_step(cur_tok, cache)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    d_time += time.perf_counter() - t_step

                    generated.append(next_tok)
                    cur_tok = next_tok
                    if next_tok == eos:
                        break

            total_decode_time += d_time
            total_decode_tokens += len(generated) - 1 if len(generated) > 1 else 0

            out_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            outputs.append({
                "text": out_text,
                "token_ids": generated,
            })

            # Clean cache between prompts
            del cache

        self._last_metrics = {
            "prefill_tokens": total_prefill_tokens,
            "decode_tokens": total_decode_tokens,
            "prefill_time": total_prefill_time,
            "decode_time": total_decode_time,
            "prefill_throughput": total_prefill_tokens / total_prefill_time if total_prefill_time else 0.0,
            "decode_throughput": total_decode_tokens / total_decode_time if total_decode_time else 0.0,
        }
        return outputs
