"""Cached Foveal parity against the full forward pass; no checkpoint downloads."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from baseline_inference.foveal_engine import FovealLLM
from foveal_cpt.checkpoint import wrap_foveal
from foveal_cpt.config import FovealConfig
from inference.sampling_params import SamplingParams
from model.config import AtmaConfig
from train.model import Model


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def make_engine(core="nope", mode="lm_output_kl", *, page_size=8, local_window=16,
                min_pages=0, top_p=0.95, hidden_size=32, head_dim=8, layers=7):
    torch.manual_seed(123)
    fc = FovealConfig(
        adaptation_mode=mode, sequence_length=128, batch_tokens=128,
        page_size=page_size, query_block_size=page_size, local_window=local_window,
        remote_capacity=4, max_remote_pages=0 if mode == "local" else 2,
        initial_max_remote_pages=0 if mode == "local" else 2,
        min_remote_pages=min_pages, initial_min_remote_pages=min_pages,
        top_p=top_p, initial_top_p=top_p, teacher_query_blocks=0, flex_compile=False,
    )
    cfg = AtmaConfig(vocab_size=64, num_hidden_layers=layers, hidden_size=hidden_size,
                     head_dim=head_dim, attn_type=core, attn_kernel="torch",
                     mem_enabled=True, mem_kernel="torch")
    engine = FovealLLM.__new__(FovealLLM)
    engine.model = wrap_foveal(Model(cfg), fc).float().eval()
    for block in engine.model.blocks:
        if hasattr(block.attn, "base"):
            torch.nn.init.normal_(block.attn.base.mem.proj.weight, std=0.05)
            torch.nn.init.normal_(block.attn.index_out.weight, std=0.1)
    engine.foveal_config = fc
    engine.device = torch.device("cpu")
    engine.max_model_len = None
    engine.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: str(ids))
    return engine


@torch.no_grad()
def full_logits(engine, tokens):
    padded = tokens + [0] * (-len(tokens) % engine.foveal_config.page_size)
    x = engine.model.embed(torch.tensor([padded], device=engine.device))
    for block in engine.model.blocks:
        x = block(x)[0]
    logits = engine.model.proj(engine.model.norm(x[:, len(tokens)-1:len(tokens)])).float()
    return 15 * logits * (logits.square() + 225).rsqrt()


@pytest.mark.parametrize("core", ["polar", "nope", "rope"])
@pytest.mark.parametrize("mode", ["local", "lm_output", "kl", "lm_output_kl"])
def test_cached_matches_full_across_pages(core, mode):
    engine = make_engine(core, mode)
    tokens = torch.randint(1, 64, (5,)).tolist()
    _, cache = engine._prefill(tokens)
    torch.testing.assert_close(cache["logits"], full_logits(engine, tokens), rtol=2e-5, atol=2e-5)
    saw_remote = False
    # Cross four page boundaries, reach remote support, and finalize a page
    # assembled partly from prompt tokens and partly from generated tokens.
    for tok in torch.randint(1, 64, (30,)).tolist():
        actual = engine._decode_step(tok, cache)
        tokens.append(tok)
        expected = full_logits(engine, tokens)
        torch.testing.assert_close(cache["logits"], expected, rtol=2e-5, atol=2e-5)
        assert actual == expected.argmax(-1).item()
        for idx in cache["k_pages"]:
            assert cache["k_pages"][idx].shape[1] == len(tokens) // 8
            assert cache["partial_k"][idx].shape[1] == len(tokens) % 8
            saw_remote |= cache["selected_remote"][idx].numel() > 0
    assert saw_remote == (mode != "local")


@pytest.mark.parametrize("length", [1, 2, 3, 63, 64, 65])
@pytest.mark.parametrize("core", ["polar", "nope", "rope"])
def test_prompt_boundaries(core, length):
    engine = make_engine(core, page_size=64, local_window=512, layers=3)
    tokens = torch.randint(1, 64, (length,)).tolist()
    _, cache = engine._prefill(tokens)
    for token in [7, 11]:
        engine._decode_step(token, cache)
        tokens.append(token)
        torch.testing.assert_close(cache["logits"], full_logits(engine, tokens), rtol=2e-5, atol=2e-5)


def test_checkpoint_dimensions_and_remote_page():
    engine = make_engine("nope", page_size=64, local_window=512,
                         hidden_size=1024, head_dim=128, layers=3)
    tokens = torch.randint(1, 64, (641,)).tolist()
    _, cache = engine._prefill(tokens)
    assert cache["selected_remote"][2].numel() > 0
    engine._decode_step(13, cache)
    torch.testing.assert_close(cache["logits"], full_logits(engine, tokens + [13]), rtol=3e-5, atol=3e-5)


def test_minimum_pages_even_when_local_mass_exceeds_top_p():
    engine = make_engine(min_pages=2, top_p=0.01)
    _, cache = engine._prefill(list(range(1, 35)))
    assert all(pages.numel() == 2 for pages in cache["selected_remote"].values())


def test_zero_budget_and_invalid_parameters_do_not_prefill():
    engine = make_engine()
    engine._prefill = Mock(side_effect=AssertionError("unexpected prefill"))
    assert engine.generate([[1]], SamplingParams(max_tokens=0)) == [{"text": "", "token_ids": []}]
    for params in (SamplingParams(max_tokens=-1), SamplingParams(temperature=-1),
                   SamplingParams(temperature=float("nan"))):
        with pytest.raises(ValueError):
            engine.generate([[1]], params)
    with pytest.raises(ValueError, match="one entry"):
        engine.generate([[1], [2]], [SamplingParams()])


def test_eos_and_temperature_are_respected():
    engine = make_engine()
    with torch.no_grad():
        engine.model.proj.weight.zero_()
        engine.model.proj.bias.zero_()
    assert len(engine.generate([[]], SamplingParams(temperature=0, max_tokens=3))[0]["token_ids"]) == 1
    assert len(engine.generate([[1]], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))[0]["token_ids"]) == 3
    with patch("torch.multinomial", return_value=torch.tensor([9])) as sample:
        assert engine.generate([[1]], SamplingParams(temperature=0.5, max_tokens=1))[0]["token_ids"] == [9]
        sample.assert_called_once()


def test_foveal_dispatch_and_loader_refuses_parameter_loss():
    from benchmarks.model import EvalModel
    from inference.utils.loader import load_model

    engine = EvalModel.__new__(EvalModel)
    engine._llm = None
    engine.cfg = {"is_foveal": True}
    engine.hf_config = SimpleNamespace(attn_type="polar")
    engine.weights_path = "checkpoint.pt"
    engine._llm_kwargs = {}
    with patch("baseline_inference.FovealLLM") as cls:
        engine.load()
        cls.assert_called_once_with("checkpoint.pt")
    for strict in [False, True]:
        with pytest.raises(ValueError, match="FovealLLM"):
            load_model(torch.nn.Linear(2, 2), {"blocks.2.attn.index_q.weight": torch.zeros(2, 2)}, strict=strict)


def test_checkpoint_verifier_on_cpu_fixture():
    from benchmarks.verify_foveal_cache import verify, verify_greedy

    engine = make_engine("rope", layers=3)
    engine.tokenizer.encode = lambda text, **kwargs: list(range(1, 64))
    cases = verify(engine, [5, 8, 9], 20, atol=2e-5, rtol=2e-5)
    assert all(c["within_tolerance"] and c["greedy_agreement"]
               for case in cases for c in case["comparisons"])
    assert all(case["identical"] for case in verify_greedy(engine, [5, 9], 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [3, 4])
def test_bf16_cached_convolution_matches_cuda_prefill(width):
    from train.model import causal_conv1d_fn

    torch.manual_seed(81)
    full = torch.randn(1, 1024, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, width, device="cuda", dtype=torch.bfloat16)
    expected = causal_conv1d_fn(full, weight)[..., -1:]
    actual = FovealLLM._conv_step(full, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Ensure the fixture detects the old per-product BF16 rounding.
    assert not torch.equal((full * weight[None]).sum(-1, keepdim=True), expected)


def test_serving_audit_retains_actual_cache_and_token_counts(monkeypatch):
    from benchmarks.audit_foveal_serving import measure

    engine = make_engine(layers=3)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2000)
    result = measure(engine, [1, 2], 3)
    assert result["prefill_tokens"] == 2
    assert result["generated_tokens"] == 3
    assert result["decode_tokens"] == 2
    assert result["cache_sequence_tokens"] == 4
    assert result["full_prefix_kv_bytes"] == 2 * 4 * 8 * 4
    assert result["peak_allocated_bytes"] == 1000
    assert result["peak_reserved_bytes"] == 2000


def test_serving_records_backend_and_excludes_warmup(monkeypatch):
    from benchmarks.serving import run_serving

    calls = []

    class FakeEngine:
        backend = "FovealLLM"

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompts, *, max_tokens, use_tqdm, ignore_eos):
            calls.append(len(prompts))
            assert ignore_eos
            self.last_call_metrics = {
                "prefill_tokens": sum(map(len, prompts)), "decode_tokens": len(prompts) * (max_tokens-1),
                "prefill_time": 1, "decode_time": 2, "prefill_throughput": 10, "decode_throughput": 20,
            }
            return ["answer"] * len(prompts)

        def close(self):
            pass

    monkeypatch.setattr("benchmarks.model.EvalModel", FakeEngine)
    monkeypatch.setattr("benchmarks.model.read_checkpoint_config", lambda path: {"is_foveal": True})
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_serving("unused", ["2k"], decode_tokens=3, samples=2, warmup_samples=1, log_fn=lambda _: None)
    assert calls == [1, 2]
    assert result["backend"] == "FovealLLM"
    assert result["protocol"] == "exact-token-prefill-v2"
    assert result["results"]["2k"]["prefill_tokens"] == 4096
    assert result["results"]["2k"]["decode_tokens"] == 4


def test_ordinary_loader_still_accepts_compiled_prefixes():
    from inference.utils.loader import load_model

    model = torch.nn.Linear(2, 2)
    source = {"_orig_mod.weight": torch.ones(2, 2), "_orig_mod.bias": torch.ones(2)}
    load_model(model, source, strict=True)
    assert torch.equal(model.weight, torch.ones(2, 2))
