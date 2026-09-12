"""Compare cached Foveal logits with full recomputation on one real checkpoint.

This is a correctness check, not a performance benchmark. Run each checkpoint
before collecting serving timings. A nonzero exit code means validation failed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch


@torch.no_grad()
def reference_logits(engine, tokens, *, start=None, extra_pages=0):
    size = engine.foveal_config.page_size
    eos = engine.tokenizer.eos_token_id
    padded = tokens + [eos if eos is not None else 0] * (-len(tokens) % size + extra_pages * size)
    x = engine.model.embed(torch.tensor([padded], device=engine.device))
    for block in engine.model.blocks:
        x = block(x)[0]
    start = len(tokens) - 1 if start is None else start
    logits = engine.model.proj(engine.model.norm(x[:, start:len(tokens)])).float()
    return 15.0 * logits * (logits.square() + 225.0).rsqrt()


@contextlib.contextmanager
def recurrent_reference():
    """Change only the full-forward FLA algorithm, preserving tokens/weights."""
    from model import blocks
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    previous = blocks._fla_gated_delta

    def recurrent(q, k, v, g, beta):
        return fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=1.0,
            use_qk_l2norm_in_kernel=True,
        )[0]

    blocks._fla_gated_delta = recurrent
    try:
        yield
    finally:
        blocks._fla_gated_delta = previous


@contextlib.contextmanager
def sequential_torch_reference(engine):
    """Same FP32 full forward with sequential, rather than chunked, memory."""
    memories = [block.attn.base.mem for block in engine.model.blocks
                if hasattr(block.attn, "base") and block.attn.base.mem is not None]
    previous = [mem.chunk for mem in memories]
    try:
        for mem in memories:
            mem.chunk = 1
        yield
    finally:
        for mem, chunk in zip(memories, previous):
            mem.chunk = chunk


def comparison(actual, expected, *, atol, rtol):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual - expected
    top = expected.flatten().topk(2)
    return {
        "max_abs_logit_error": float(difference.abs().max()) if finite else None,
        "rms_logit_error": float(difference.square().mean().sqrt()) if finite else None,
        "relative_l2_logit_error": float(difference.norm() / expected.norm().clamp_min(1e-20)) if finite else None,
        "within_tolerance": finite and torch.allclose(actual, expected, atol=atol, rtol=rtol),
        "greedy_agreement": finite and int(actual.argmax()) == int(expected.argmax()),
        "cached_greedy_token": int(actual.argmax()) if finite else None,
        "reference_greedy_token": int(expected.argmax()) if finite else None,
        "reference_greedy_margin": float(top.values[0] - top.values[1]) if finite else None,
    }


def verify(engine, lengths, decode_steps, *, atol, rtol, controls=False,
           workload="records", compare_every=1, fp32_controls=False):
    text = "\n".join(
        f"Record {i}: the parcel marked {i * 71 + 19} is in room {i % 17}. "
        f"The earlier record names a different room. Remember record {i}."
        for i in range(300)
    )
    if workload == "prose":
        text = (
            "Mira reached the old observatory before sunrise. The eastern door was open, "
            "and a notebook lay beside the telescope. Its pages described the weather, "
            "the positions of the planets, and the names of visitors from the valley. "
            "She read the final entry twice. The astronomer had left a brass key under "
            "the blue cup in the library, hoping that someone would finish his map. "
            "Outside, the clouds began to clear. Mira carried the notebook downstairs "
            "and asked the caretaker to help her find the library. He remembered that "
            "the blue cup had been moved to the north shelf during the previous winter. "
        ) * 100
    unit = engine.tokenizer.encode(text, add_special_tokens=False)
    count = max(lengths) + decode_steps
    stream = (unit * (count // len(unit) + 1))[:count]
    cases = []
    for length in lengths:
        tokens = stream[:length]
        # Causality makes these longer full forwards mathematically identical at
        # the observed positions. They control for GEMM shape/padding and for
        # chunked versus recurrent FLA, without using the cached decoder.
        control_logits = {}
        if controls:
            extended = stream[:length + decode_steps]
            control_logits["full_forward_shape_control"] = reference_logits(
                engine, extended, start=length-1, extra_pages=1)
            with recurrent_reference():
                control_logits["full_forward_recurrent_control"] = reference_logits(
                    engine, extended, start=length-1, extra_pages=1)
        _, cache = engine._prefill(tokens)
        comparisons = []
        for step in range(decode_steps + 1):
            if step:
                token = stream[length + step - 1]
                engine._decode_step(token, cache)
                tokens.append(token)
            if step % compare_every and step != decode_steps:
                continue
            expected = reference_logits(engine, tokens)
            actual = cache["logits"]
            record = {"position": len(tokens) - 1,
                      **comparison(actual, expected, atol=atol, rtol=rtol)}
            for name, logits in control_logits.items():
                record[name] = comparison(logits[:, step:step+1], expected, atol=atol, rtol=rtol)
            if fp32_controls and not record["within_tolerance"]:
                with sequential_torch_reference(engine):
                    sequential = reference_logits(engine, tokens)
                record["fp32_sequential_control"] = comparison(sequential, expected, atol=atol, rtol=rtol)
                record["cached_vs_fp32_sequential"] = comparison(actual, sequential, atol=atol, rtol=rtol)
            comparisons.append(record)
        # Compare state and routes to a fresh prefill after the whole continuation.
        _, fresh = engine._prefill(tokens)
        states = {}
        for name in ("mem_states", "k_cache", "v_cache", "k_pages", "v_pages"):
            states[name] = {
                str(i): float((value.float() - fresh[name][i].float()).norm()
                              / fresh[name][i].float().norm().clamp_min(1e-20))
                for i, value in cache[name].items()
            }
        routes = {
            str(i): {"cached": sorted(value.tolist()),
                     "reference": sorted(fresh["selected_remote"][i].tolist())}
            for i, value in cache["selected_remote"].items()
            # At an exact page boundary fresh prefill routes on the next token.
            if i in fresh["selected_remote"]
        }
        for i, value in cache["selected_remote"].items():
            if str(i) not in routes:
                continue
            actual_scores = cache["cached_scores"][i].flatten().float()
            expected_scores = fresh["cached_scores"][i].flatten().float()
            first_local = max(0, (len(tokens)-1) // engine.foveal_config.page_size
                              - engine.foveal_config.local_window // engine.foveal_config.page_size)
            selected = fresh["selected_remote"][i].long()
            unselected_mask = torch.ones(first_local, dtype=torch.bool, device=engine.device)
            unselected_mask[selected] = False
            unselected = expected_scores[:first_local][unselected_mask]
            margin = (float(expected_scores[selected].min() - unselected.max())
                      if selected.numel() and unselected.numel() else None)
            changed = sorted(set(value.tolist()) ^ set(selected.tolist()))
            routes[str(i)].update(
                max_abs_score_error=float((actual_scores - expected_scores).abs().max())
                if actual_scores.numel() else 0.0,
                reference_remote_cutoff_margin=margin,
                changed_pages=[{"page": page, "cached_score": float(actual_scores[page]),
                                "reference_score": float(expected_scores[page])}
                               for page in changed],
            )
        cases.append({"prompt_tokens": length, "comparisons": comparisons,
                      "final_state_relative_l2_errors": states, "final_routes": routes})
        print(f"checked {workload} prompt={length} steps={decode_steps} "
              f"max_error={max(c['max_abs_logit_error'] or 0 for c in comparisons):.6f} "
              f"greedy_disagreements={sum(not c['greedy_agreement'] for c in comparisons)}", flush=True)
        del cache
        del fresh
    return cases


@torch.no_grad()
def verify_greedy(engine, lengths, steps):
    unit = engine.tokenizer.encode(
        "The astronomer left a notebook in the library. Mira opened it and read "
        "the observations from the previous night. The sky was clear, and the "
        "moon had set behind the hills. She decided to continue the record. ",
        add_special_tokens=False,
    )
    stream = unit * (max(lengths) // len(unit) + 1)
    cases = []
    for length in lengths:
        prompt = stream[:length]
        cached_token, cache = engine._prefill(prompt)
        reference_tokens = list(prompt)
        cached_output, reference_output = [], []
        first_divergence = None
        for step in range(steps):
            expected = reference_logits(engine, reference_tokens)
            reference_token = int(expected.argmax())
            if cached_token != reference_token and first_divergence is None:
                first_divergence = {"step": step, **comparison(
                    cache["logits"], expected, atol=1e-4, rtol=1e-4)}
            cached_output.append(cached_token)
            reference_output.append(reference_token)
            reference_tokens.append(reference_token)
            if step + 1 < steps:
                cached_token = engine._decode_step(cached_token, cache)
        cases.append({"prompt_tokens": length, "generated_tokens": steps,
                      "cached_tokens": cached_output, "reference_tokens": reference_output,
                      "identical": cached_output == reference_output,
                      "first_divergence": first_divergence})
        print(f"greedy prompt={length} steps={steps} identical={cached_output == reference_output}", flush=True)
        del cache
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[1, 2, 63, 64, 65, 511, 512, 513, 639, 640, 641])
    parser.add_argument("--decode_steps", type=int, default=66)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--float32_oracle", action="store_true",
                        help="diagnostic eager FP32 attention/memory, not serving kernels")
    parser.add_argument("--workload", choices=("records", "prose"), default="records")
    parser.add_argument("--compare_every", type=int, default=1)
    parser.add_argument("--greedy_steps", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.lengths) < 1 or args.decode_steps < 1 or args.atol < 0 or args.rtol < 0 or args.compare_every < 1:
        parser.error("lengths/steps must be positive and tolerances nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; real-checkpoint GPU validation must run on another machine")
    from baseline_inference.foveal_engine import FovealLLM

    engine = None
    report = {"backend": "FovealLLM", "model": args.model, "device": args.device,
              "torch_version": torch.__version__, "atol": args.atol, "rtol": args.rtol,
              "passed": False, "workload": args.workload, "controls": args.controls,
              "float32_oracle": args.float32_oracle, "decode_steps": args.decode_steps,
              "compare_every": args.compare_every,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "decoder_sha256": hashlib.sha256(Path("baseline_inference/foveal_engine.py").read_bytes()).hexdigest()}
    started = time.perf_counter()
    try:
        engine = FovealLLM(args.model, device=args.device)
        if args.float32_oracle:
            if args.controls:
                raise ValueError("BF16 controls and the FP32 oracle are separate experiments")
            import foveal_cpt.attention as attention

            engine.model.float()
            attention.HAS_POLAR_TRITON = False
            attention.HAS_FLEX_ATTENTION = False
            for block in engine.model.blocks:
                if hasattr(block.attn, "base") and block.attn.base.mem is not None:
                    block.attn.base.mem.kernel = "torch"
        report["weights_path"] = engine.weights_path
        report["parameter_dtype"] = str(engine.model.embed.weight.dtype)
        report["gpu"] = torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None
        report["cases"] = verify(engine, args.lengths, args.decode_steps, atol=args.atol, rtol=args.rtol,
                                 controls=args.controls, workload=args.workload,
                                 compare_every=args.compare_every, fp32_controls=args.float32_oracle)
        report["passed"] = all(c["within_tolerance"] and c["greedy_agreement"]
                               for case in report["cases"] for c in case["comparisons"])
        if args.greedy_steps:
            report["greedy_cases"] = verify_greedy(engine, args.lengths, args.greedy_steps)
            report["passed"] &= all(case["identical"] for case in report["greedy_cases"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            engine.close()
        report["elapsed_s"] = time.perf_counter() - started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'} -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
