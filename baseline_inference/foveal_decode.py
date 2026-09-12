"""Fixed-address, graph-captured Foveal decode; the eager engine remains the oracle."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from kernel.causal_conv1d_triton import causal_conv1d_decode_step
from kernel.gated_delta_triton import gated_delta_decode_step
from kernel.inference_ops_triton import squared_relu_gate
from baseline_inference.foveal_triton import sparse_decode, update_index_page


class FovealGraphDecoder:
    def __init__(self, engine, capacity, output_capacity):
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.dtype = self.model.embed.weight.dtype
        self.page_size = engine.foveal_config.page_size
        self.capacity = (
            math.ceil(max(capacity, 2 * self.page_size) / self.page_size)
            * self.page_size
        )
        self.output_capacity = max(output_capacity, 2)
        self.pages = self.capacity // self.page_size
        self.token = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position = torch.full(
            (1,), self.page_size, dtype=torch.long, device=self.device
        )
        self.output_index = torch.ones(1, dtype=torch.long, device=self.device)
        self.generated = torch.empty(
            self.output_capacity, dtype=torch.long, device=self.device
        )
        self.slots = torch.zeros(1, dtype=torch.long, device=self.device)
        self.page_ids = torch.arange(self.pages, device=self.device)
        self.weights = {}
        # The training Linear casts FP32 master weights on every call. Keep the
        # identical rounded inference weights resident once, as the fast engine does.
        for module in self.model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                self.weights[id(module)] = (
                    module.weight.detach().to(self.dtype),
                    module.bias.detach().to(self.dtype)
                    if getattr(module, "bias", None) is not None
                    else None,
                )
        self.states = {}
        for idx, block in enumerate(self.model.blocks):
            a = block.attn
            if not hasattr(a, "page_size"):
                channels = a.conv.weight.shape[0]
                self.states[idx] = {
                    "conv": torch.zeros(
                        1,
                        channels,
                        a.conv.weight.shape[-1] - 1,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "conv_weight": a.conv.weight.detach()
                    .squeeze(1)
                    .to(self.dtype)
                    .float(),
                }
                continue
            h, kh, d = a.num_heads, a.num_kv_heads, a.head_dim
            limit = (
                int(a.route_max_remote_pages.item())
                if a.adaptation_mode != "local"
                else 0
            )
            state = dict(
                k=torch.empty(
                    1, self.capacity, kh, d, device=self.device, dtype=self.dtype
                ),
                v=torch.empty(
                    1, self.capacity, kh, d, device=self.device, dtype=self.dtype
                ),
                kp=torch.zeros(
                    1, self.pages, a.index_dim, device=self.device, dtype=torch.float32
                ),
                vp=torch.zeros(
                    1, self.pages, a.index_dim, device=self.device, dtype=self.dtype
                ),
                scores=torch.zeros(
                    1, self.pages, device=self.device, dtype=torch.float32
                ),
                partial_k=torch.zeros(
                    self.page_size, a.index_dim, device=self.device, dtype=self.dtype
                ),
                partial_v=torch.zeros(
                    self.page_size, a.index_dim, device=self.device, dtype=self.dtype
                ),
                remote=torch.full(
                    (max(1, limit),), -1, device=self.device, dtype=torch.long
                ),
                count=torch.zeros(1, device=self.device, dtype=torch.int32),
                index_out=torch.zeros(
                    1,
                    self.model.embed.weight.shape[1],
                    device=self.device,
                    dtype=self.dtype,
                ),
                limit=limit,
                minimum=int(a.route_min_remote_pages.item()),
                top_p=float(a.route_top_p.item()),
            )
            if a.base.canon_q is not None:
                for key, n in [("q", h * d), ("k", kh * d), ("v", kh * d)]:
                    conv = getattr(a.base, "canon_" + key)
                    state["canon_" + key] = torch.zeros(
                        1,
                        n,
                        conv.weight.shape[-1] - 1,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    state["weight_" + key] = (
                        conv.weight.detach().squeeze(1).to(self.dtype).float()
                    )
            if a.base.mem is not None:
                state["memory"] = torch.zeros(
                    1, h, d, d, device=self.device, dtype=torch.float32
                )
            if a.is_polar:
                state["polar"] = [
                    a.base.v_null.detach().float(),
                    a.base.null_base.detach().float(),
                    F.softplus(a.base.null_slope_raw.detach().float()),
                    F.softplus(a.base.len_gain_raw.detach().float()),
                    F.softplus(a.base.mag_beta_raw.detach().float()),
                ]
            self.states[idx] = state
        self.graphs = {}
        self.graph_logits = {}
        pool = None
        # Capture mutates dummy state only; reset() installs real prompt state later.
        with torch.no_grad():
            for boundary in (False, True):
                self.position.fill_(self.page_size)
                self.output_index.fill_(1)
                self.forward(boundary)
                self.position.fill_(self.page_size)
                self.output_index.fill_(1)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool):
                    self.forward(boundary)
                pool = graph.pool()
                self.graphs[boundary] = graph
                self.graph_logits[boundary] = self.logits
        torch.cuda.synchronize()

    def linear(self, module, x):
        w, b = self.weights[id(module)]
        return F.linear(x, w, b)

    def norm(self, module, x):
        return F.rms_norm(x, (x.shape[-1],), self.weights[id(module)][0], module.eps)

    def rotate(self, x, rotary):
        theta = self.position * rotary.angular_freq
        left, right = x.float().chunk(2, -1)
        return torch.cat(
            (
                left * theta.cos() + right * theta.sin(),
                right * theta.cos() - left * theta.sin(),
            ),
            -1,
        ).to(x.dtype)

    def route(self, a, s, qi):
        qb = self.position // self.page_size
        scores = (F.normalize(qi.float(), dim=-1) @ s["kp"][0].T) / math.sqrt(
            a.index_dim
        )
        s["scores"].copy_(scores)
        valid = self.page_ids < qb
        # Avoid all -inf softmax for the first block; its context is explicitly zero.
        masked = scores.masked_fill(~valid, -1e30)
        probs = masked.softmax(-1) * valid
        first_local = (qb - a.local_window // self.page_size).clamp_min(0)
        if s["limit"]:
            target = (
                s["top_p"] - (probs * (self.page_ids >= first_local)).sum()
            ).clamp_min(0)
            remote_probs = probs.masked_fill(self.page_ids >= first_local, -1)
            values, order = remote_probs.sort(dim=-1, descending=True)
            needed = (
                (values.clamp_min(0).cumsum(-1) < target).sum() + (target > 0)
            ).clamp(min=s["minimum"], max=s["limit"])
            count = torch.minimum(needed, first_local).to(torch.int32)
            s["count"].copy_(count)
            chosen = order[0, : s["limit"]]
            # Capacity can be smaller than the configured remote-page cap.
            s["remote"].fill_(-1)
            s["remote"][: chosen.numel()].copy_(chosen)
        if a.uses_lm_output:
            context = probs.to(self.dtype) @ s["vp"][0]
            projected = self.linear(a.index_out, context)
            # index_out has no bias, but preserve zero-block semantics explicitly.
            s["index_out"].copy_(projected * (qb > 0))

    def forward(self, boundary):
        x = F.embedding(self.token, self.weights[id(self.model.embed)][0])
        for idx, block in enumerate(self.model.blocks):
            a = block.attn
            s = self.states[idx]
            normed = self.norm(block.norm1, x)
            if not hasattr(a, "page_size"):
                b, c, v = self.linear(a.in_proj, normed).chunk(3, -1)
                conv = causal_conv1d_decode_step(
                    b * v, s["conv_weight"], self.slots, s["conv"]
                )
                out = self.linear(a.out_proj, c * conv)
            else:
                h, kh, d = a.num_heads, a.num_kv_heads, a.head_dim
                groups = h // kh
                if a.adaptation_mode != "local":
                    ki = F.rms_norm(self.linear(a.index_k, normed), (a.index_dim,))
                    vi = self.linear(a.index_v, normed)
                    if a.index_rotary is not None:
                        ki = self.rotate(ki, a.index_rotary)
                    if boundary:
                        qi = F.rms_norm(self.linear(a.index_q, normed), (a.index_dim,))
                        if a.index_rotary is not None:
                            qi = self.rotate(qi, a.index_rotary)
                        self.route(a, s, qi)
                    update_index_page(
                        ki,
                        vi,
                        s["partial_k"],
                        s["partial_v"],
                        s["kp"],
                        s["vp"],
                        self.position,
                        self.page_size,
                    )
                q, gate = self.linear(a.base.q, normed).view(1, h, 2 * d).chunk(2, -1)
                k = self.linear(a.base.k, normed).view(1, kh, d)
                v = self.linear(a.base.v, normed).view(1, kh, d)
                q, k = F.rms_norm(q, (d,)), F.rms_norm(k, (d,))
                qm, km, vm = q, k, v
                if getattr(a.base, "pos", None) == "rope":
                    q, k = self.rotate(q, a.base.rotary), self.rotate(k, a.base.rotary)
                else:
                    projected = []
                    for name, value in [("q", q), ("k", k), ("v", v)]:
                        raw = value.flatten(1)
                        conv = causal_conv1d_decode_step(
                            raw, s["weight_" + name], self.slots, s["canon_" + name]
                        )
                        projected.append((raw + conv).reshape_as(value))
                    q, k, v = projected
                    qm, km, vm = q, k, v
                s["k"][0].index_copy_(0, self.position, k)
                s["v"][0].index_copy_(0, self.position, v)
                attended, mag = sparse_decode(
                    q,
                    s["k"],
                    s["v"],
                    s["remote"],
                    s["count"],
                    self.position,
                    page_size=self.page_size,
                    window=a.local_window,
                    scale=(getattr(a.base, "sdpa_scale", None) or 1 / math.sqrt(d)),
                    polar=s.get("polar"),
                )
                out = self.linear(
                    a.base.proj, attended.flatten(1) * torch.sigmoid(gate.flatten(1))
                )
                if a.is_polar:
                    out = out + self.linear(a.base.mu_proj, mag)
                if a.uses_lm_output:
                    out = out + s["index_out"]
                if a.base.mem is not None:
                    mem = a.base.mem
                    gamma = torch.sigmoid(
                        self.linear(mem.w_gamma, normed).float() + mem.gamma_bias
                    )
                    beta = torch.sigmoid(
                        self.linear(mem.w_beta, normed).float() + mem.beta_bias
                    )
                    read = gated_delta_decode_step(
                        qm,
                        km.repeat_interleave(groups, 1),
                        vm.repeat_interleave(groups, 1),
                        gamma,
                        beta,
                        s["memory"],
                        self.slots,
                    )
                    # Prefill FLA rounds readout to activation dtype before RMSNorm.
                    read = F.rms_norm(read.to(self.dtype), (d,)).flatten(1)
                    out = out + self.linear(
                        mem.proj, read * torch.sigmoid(self.linear(mem.gate, normed))
                    )
            x = x + out
            z = self.linear(block.mlp.fc, self.norm(block.norm2, x))
            value, gate = z.chunk(2, -1)
            x = x + self.linear(block.mlp.proj, squared_relu_gate(value, gate))
        logits = self.linear(self.model.proj, self.norm(self.model.norm, x)).float()
        self.logits = 15 * logits * (logits.square() + 225).rsqrt()
        self.token.copy_(self.logits.argmax(-1))
        self.generated.index_copy_(0, self.output_index, self.token)
        self.position.add_(1)
        self.output_index.add_(1)

    @torch.no_grad()
    def reset(self, cache, token):
        self.length = cache["seq_len"]
        self.steps = 0
        self.position.fill_(self.length)
        self.token.fill_(token)
        self.output_index.fill_(1)
        self.generated[0] = token
        for idx, block in enumerate(self.model.blocks):
            a = block.attn
            s = self.states[idx]
            if not hasattr(a, "page_size"):
                s["conv"].copy_(cache["conv_states"][idx])
                continue
            s["k"][:, : self.length].copy_(cache["k_cache"][idx])
            s["v"][:, : self.length].copy_(cache["v_cache"][idx])
            if "memory" in s:
                s["memory"].copy_(cache["mem_states"][idx])
            for name in ("q", "k", "v"):
                if "canon_" + name in s:
                    s["canon_" + name].copy_(cache["canon_" + name][idx])
            complete = self.length // self.page_size
            partial = self.length % self.page_size
            s["kp"].zero_()
            s["vp"].zero_()
            s["partial_k"].zero_()
            s["partial_v"].zero_()
            s["kp"][:, :complete].copy_(cache["k_pages"][idx])
            s["vp"][:, :complete].copy_(cache["v_pages"][idx])
            s["partial_k"][:partial].copy_(cache["partial_k"][idx][0])
            s["partial_v"][:partial].copy_(cache["partial_v"][idx][0])
            remote = cache["selected_remote"].get(
                idx, torch.empty(0, device=self.device)
            )
            s["remote"].fill_(-1)
            s["remote"][: remote.numel()].copy_(remote)
            s["count"].fill_(remote.numel())
            s["index_out"].zero_()
            s["scores"].zero_()
            if idx in cache["cached_scores"]:
                previous = cache["cached_scores"][idx].flatten()
                s["scores"][0, : previous.numel()].copy_(previous)
            if a.uses_lm_output and partial and complete:
                context = (
                    cache["cached_scores"][idx].softmax(-1).to(self.dtype)
                    @ cache["v_pages"][idx][:, :complete]
                )
                s["index_out"].copy_(
                    self.linear(a.index_out, context).reshape_as(s["index_out"])
                )

    def replay(self):
        boundary = self.length % self.page_size == 0
        self.graphs[boundary].replay()
        self.length += 1
        self.steps += 1
        self.logits = self.graph_logits[boundary]
        # Each graph owns a distinct logits allocation.
        self.last_boundary = boundary

    @torch.no_grad()
    def export_cache(self, cache):
        cache["seq_len"] = self.length
        if self.steps:
            cache["logits"] = self.logits.unsqueeze(1)
        for idx, block in enumerate(self.model.blocks):
            a = block.attn
            s = self.states[idx]
            if not hasattr(a, "page_size"):
                cache["conv_states"][idx] = s["conv"]
                continue
            for key in ("k", "v"):
                cache[key + "_cache"][idx] = s[key][:, : self.length]
            if "memory" in s:
                cache["mem_states"][idx] = s["memory"]
            for key in ("q", "k", "v"):
                if "canon_" + key in s:
                    cache["canon_" + key][idx] = s["canon_" + key]
            count = int(s["count"].item())
            cache["selected_remote"][idx] = s["remote"][:count]
            qb = (self.length - 1) // self.page_size
            cache["current_qb"][idx] = qb
            cache["cached_scores"][idx] = s["scores"][:, :qb].unsqueeze(1)
            complete = self.length // self.page_size
            partial = self.length % self.page_size
            cache["k_pages"][idx] = s["kp"][:, :complete]
            cache["v_pages"][idx] = s["vp"][:, :complete]
            cache["partial_k"][idx] = s["partial_k"][:partial].unsqueeze(0)
            cache["partial_v"][idx] = s["partial_v"][:partial].unsqueeze(0)
