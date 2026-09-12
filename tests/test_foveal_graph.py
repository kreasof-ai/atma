"""Graph decode must preserve sparse support, state transitions, and request isolation."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph tests"
)


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("position", [0, 15, 16, 31, 32, 63, 64])
def test_selected_decode_matches_materialized_support(polar, position):
    from baseline_inference.foveal_triton import sparse_decode
    from model.blocks import polar_reduce

    torch.manual_seed(41)
    q = torch.randn(1, 4, 32, device="cuda")
    k = torch.randn(1, 96, 2, 32, device="cuda")
    v = torch.randn_like(k)
    window, page = 16, 8
    selected = [0, 1] if position >= 32 else []
    remote = torch.tensor(selected or [-1], device="cuda", dtype=torch.long)
    count = torch.tensor([len(selected)], device="cuda", dtype=torch.int32)
    pos = torch.tensor([position], device="cuda")
    raw = [
        torch.randn(4, 32, device="cuda"),
        torch.randn(4, device="cuda"),
        torch.randn(4, device="cuda"),
        torch.randn(4, device="cuda"),
        torch.randn(4, device="cuda"),
    ]
    params = [
        raw[0],
        raw[1],
        F.softplus(raw[2]),
        F.softplus(raw[3]),
        F.softplus(raw[4]),
    ]
    out, mag = sparse_decode(
        q,
        k,
        v,
        remote,
        count,
        pos,
        page_size=page,
        window=window,
        scale=32**-0.5,
        polar=params if polar else None,
    )
    indices = [p * page + j for p in selected for j in range(page)] + list(
        range(max(0, position + 1 - window), position + 1)
    )
    kt = k[:, indices].repeat_interleave(2, 2).transpose(1, 2)
    vt = v[:, indices].repeat_interleave(2, 2).transpose(1, 2)
    if polar:
        expected, expected_mag = polar_reduce(
            q.unsqueeze(2) @ kt.transpose(-2, -1) * 32**-0.5,
            vt,
            torch.tensor([position + 1.0], device="cuda"),
            v_null=raw[0],
            null_base=raw[1],
            null_slope_raw=raw[2],
            len_gain_raw=raw[3],
            mag_beta_raw=raw[4],
        )
        torch.testing.assert_close(mag, expected_mag.squeeze(-1), atol=2e-5, rtol=2e-5)
    else:
        expected = F.scaled_dot_product_attention(q.unsqueeze(2), kt, vt)
    torch.testing.assert_close(out, expected.squeeze(2), atol=2e-5, rtol=2e-5)


def test_index_page_completion_rounding_and_invisibility():
    from baseline_inference.foveal_triton import update_index_page

    torch.manual_seed(17)
    pk = torch.randn(64, 16, device="cuda", dtype=torch.bfloat16)
    pv = torch.randn_like(pk)
    kp = torch.full((1, 3, 16), 7.0, device="cuda")
    vp = torch.full((1, 3, 16), 7.0, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    update_index_page(k, v, pk, pv, kp, vp, torch.tensor([62], device="cuda"), 64)
    assert torch.all(kp == 7)
    update_index_page(k, v, pk, pv, kp, vp, torch.tensor([63], device="cuda"), 64)
    torch.testing.assert_close(
        kp[0, 0], F.normalize(pk.mean(0).float(), dim=-1), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(vp[0, 0], pv.mean(0), atol=0, rtol=0)
    assert torch.all(kp[0, 1:] == 7)


@pytest.mark.parametrize("core", ["polar", "nope", "rope"])
def test_graph_generation_reset_and_page_boundaries(core):
    from tests.test_foveal_engine import make_engine
    from inference.sampling_params import SamplingParams

    e = make_engine(core, head_dim=32, hidden_size=128, page_size=16, local_window=32)
    e.model.cuda().bfloat16()
    e.device = torch.device("cuda")
    for block in e.model.blocks:
        if hasattr(block.attn, "base"):
            block.attn.base.mem.kernel = "fla"
    prompt = [3, 4, 5, 6, 7]
    # Exercise a partially prefilled page, fully generated remote pages, and reset.
    params = SamplingParams(temperature=0, max_tokens=80, ignore_eos=True)
    actual = e.generate([prompt], params)[0]["token_ids"]
    again = e.generate([prompt], params)[0]["token_ids"]
    assert actual == again
    short = e.generate(
        [[2, 8]], SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    )[0]["token_ids"]
    assert (
        short
        == e.generate(
            [[2, 8]], SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
        )[0]["token_ids"]
    )
    e.enforce_eager = True
    assert (
        short
        == e.generate(
            [[2, 8]], SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
        )[0]["token_ids"]
    )
    e.close()


@pytest.mark.parametrize("core", ["polar", "nope", "rope"])
def test_graph_fp32_state_and_routes_match_eager_across_generated_pages(core):
    import copy
    from tests.test_foveal_engine import make_engine

    e = make_engine(core, head_dim=32, hidden_size=128, page_size=16, local_window=32)
    e.model.cuda()
    e.device = torch.device("cuda")
    _, cache = e._prefill([3, 4, 5, 6, 7])
    reference = copy.deepcopy(cache)
    for i in range(80):
        token = 1 + (i * 17) % 63
        e._decode_step_graph(token, cache)
        e._decode_step(token, reference)
        torch.testing.assert_close(
            cache["logits"], reference["logits"], atol=1e-4, rtol=1e-4
        )
        for idx in cache["selected_remote"]:
            torch.testing.assert_close(
                cache["selected_remote"][idx],
                reference["selected_remote"][idx],
                atol=0,
                rtol=0,
            )
            for key in ["k_cache", "v_cache", "k_pages", "v_pages", "mem_states"]:
                torch.testing.assert_close(
                    cache[key][idx], reference[key][idx], atol=1e-4, rtol=1e-4
                )
    e.close()
