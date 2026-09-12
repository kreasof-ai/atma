"""Graph-safe selected-page attention and index-page completion for Foveal decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _index_page(K, V, PK, PV, KP, VP, POS, R: tl.constexpr, PAGE: tl.constexpr):
    pos = tl.load(POS)
    offset = pos % PAGE
    page = pos // PAGE
    d = tl.arange(0, R)
    ki = tl.load(K + d)
    vi = tl.load(V + d)
    if offset == PAGE - 1:
        rows = tl.arange(0, PAGE)
        ks = tl.load(PK + rows[:, None] * R + d[None, :])
        vs = tl.load(PV + rows[:, None] * R + d[None, :])
        ks = tl.where(rows[:, None] == offset, ki[None, :], ks).to(tl.float32)
        vs = tl.where(rows[:, None] == offset, vi[None, :], vs).to(tl.float32)
        # Match torch BF16 mean followed by FP32 page normalization.
        km = (tl.sum(ks, 0) / PAGE).to(K.dtype.element_ty).to(tl.float32)
        vm = tl.sum(vs, 0) / PAGE
        km = km / tl.maximum(tl.sqrt(tl.sum(km * km)), 1e-12)
        tl.store(KP + page * R + d, km)
        tl.store(VP + page * R + d, vm)
    tl.store(PK + offset * R + d, ki)
    tl.store(PV + offset * R + d, vi)


def update_index_page(k, v, partial_k, partial_v, kp, vp, position, page_size):
    _index_page[(1,)](
        k,
        v,
        partial_k,
        partial_v,
        kp,
        vp,
        position,
        R=k.shape[-1],
        PAGE=page_size,
        num_warps=4,
    )


@triton.jit
def _sparse_decode(
    Q,
    K,
    V,
    REMOTE,
    COUNT,
    POS,
    OUT,
    MAG,
    VNULL,
    NB,
    SPS,
    SPG,
    BETA,
    qh: tl.constexpr,
    kpos: tl.constexpr,
    kh: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BN: tl.constexpr,
    POLAR: tl.constexpr,
    DTYPE: tl.constexpr,
):
    kvh = tl.program_id(0)
    od = tl.arange(0, D)
    og = tl.arange(0, GP)
    hg = kvh * G + og
    q = tl.load(Q + hg[:, None] * qh + od[None, :], mask=og[:, None] < G, other=0).to(
        DTYPE
    )
    pos = tl.load(POS)
    nr = tl.load(COUNT) * PAGE
    local_start = tl.maximum(0, pos + 1 - WINDOW)
    nlocal = pos + 1 - local_start
    n = nr + nlocal
    m = tl.full([GP], -1e38, tl.float32)
    z = tl.zeros([GP], tl.float32)
    z2 = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    if POLAR:
        gain = tl.load(SPG + hg, mask=og < G, other=0)
        slope = tl.load(SPS + hg, mask=og < G, other=0)
        nb = tl.load(NB + hg, mask=og < G, other=0)
        beta = tl.load(BETA + hg, mask=og < G, other=0)
        # Foveal uses absolute causal length, including when sparse support is bounded.
        nf = (pos + 1).to(tl.float32)
        temp = 1 + gain * tl.log(nf)
        null = nb + slope * tl.sqrt(tl.log(nf + 1))
    for start in range(0, tl.cdiv(n, BN) * BN, BN):
        t = start + tl.arange(0, BN)
        valid = t < n
        page = tl.load(REMOTE + t // PAGE, mask=(t < nr) & valid, other=0)
        token = tl.where(t < nr, page * PAGE + t % PAGE, local_start + t - nr)
        k = tl.load(
            K + token[:, None] * kpos + kvh * kh + od[None, :],
            mask=valid[:, None],
            other=0,
        ).to(DTYPE)
        v = tl.load(
            V + token[:, None] * kpos + kvh * kh + od[None, :],
            mask=valid[:, None],
            other=0,
        ).to(DTYPE)
        score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        if POLAR:
            score = score * temp[:, None]
        score = tl.where(valid[None, :] & (og[:, None] < G), score, -1e38)
        new_m = tl.maximum(m, tl.max(score, 1))
        alpha = tl.exp(m - new_m)
        prob = tl.where(valid[None, :], tl.exp(score - new_m[:, None]), 0)
        z = z * alpha + tl.sum(prob, 1)
        z2 = z2 * alpha * alpha + tl.sum(prob * prob, 1)
        acc = acc * alpha[:, None] + tl.dot(prob.to(DTYPE), v, input_precision="ieee")
        m = new_m
    if POLAR:
        new_m = tl.maximum(m, temp * null)
        alpha = tl.exp(m - new_m)
        z = z * alpha
        z2 = z2 * alpha * alpha
        acc = acc * alpha[:, None]
        null_p = tl.exp(temp * null - new_m)
        den = z + null_p
        vn = tl.load(
            VNULL + hg[:, None] * D + od[None, :], mask=og[:, None] < G, other=0
        ).to(tl.float32)
        value = (acc + null_p[:, None] * vn) / den[:, None]
        value = value / tl.maximum(tl.sqrt(tl.sum(value * value, 1)), 1e-6)[:, None]
        # Preserve the full-forward oracle's clamp on normalized squared mass.
        mass_scale = tl.maximum(z, 1e-6 * den)
        effective = tl.minimum(mass_scale * mass_scale / tl.maximum(z2, 1e-30), 1e6) * (
            1 - tl.div_rn(null_p, den)
        )
        mag = 2 * tl.sigmoid(2 * beta * tl.log(1 + effective)) - 1
        tl.store(MAG + hg, mag, mask=og < G)
    else:
        value = acc / tl.maximum(z[:, None], 1e-9)
    tl.store(OUT + hg[:, None] * D + od[None, :], value, mask=og[:, None] < G)


@torch.no_grad()
def sparse_decode(
    q, k, v, remote, count, position, *, page_size, window, scale, polar=None
):
    _, h, d = q.shape
    groups = h // k.shape[2]
    out = torch.empty_like(q)
    mag = torch.empty((1, h), device=q.device, dtype=q.dtype)
    dtype = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[q.dtype]
    params = polar if polar is not None else [q] * 5
    _sparse_decode[(k.shape[2],)](
        q,
        k,
        v,
        remote,
        count,
        position,
        out,
        mag,
        *params,
        qh=q.stride(1),
        kpos=k.stride(1),
        kh=k.stride(2),
        SCALE=float(scale),
        PAGE=page_size,
        WINDOW=window,
        D=d,
        G=groups,
        GP=max(16, triton.next_power_of_2(groups)),
        BN=64,
        POLAR=polar is not None,
        DTYPE=dtype,
        num_warps=4,
        num_stages=2,
    )
    return out, mag
