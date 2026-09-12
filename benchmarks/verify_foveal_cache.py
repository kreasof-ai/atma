"""Compare cached Foveal logits with full recomputation on one real checkpoint.

This is a correctness check, not a performance benchmark. Run each checkpoint
before collecting serving timings. A nonzero exit code means validation failed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


@torch.no_grad()
def reference_logits(engine, tokens):
    size = engine.foveal_config.page_size
    eos = engine.tokenizer.eos_token_id
    padded = tokens + [eos if eos is not None else 0] * (-len(tokens) % size)
    x = engine.model.embed(torch.tensor([padded], device=engine.device))
    for block in engine.model.blocks:
        x = block(x)[0]
    logits = engine.model.proj(engine.model.norm(x[:, len(tokens)-1:len(tokens)])).float()
    return 15.0 * logits * (logits.square() + 225.0).rsqrt()


def verify(engine, lengths, decode_steps, *, atol, rtol):
    text = "\n".join(
        f"Record {i}: the parcel marked {i * 71 + 19} is in room {i % 17}. "
        f"The earlier record names a different room. Remember record {i}."
        for i in range(300)
    )
    unit = engine.tokenizer.encode(text, add_special_tokens=False)
    count = max(lengths) + decode_steps
    stream = (unit * (count // len(unit) + 1))[:count]
    cases = []
    for length in lengths:
        tokens = stream[:length]
        _, cache = engine._prefill(tokens)
        comparisons = []
        for step in range(decode_steps + 1):
            if step:
                token = stream[length + step - 1]
                engine._decode_step(token, cache)
                tokens.append(token)
            expected = reference_logits(engine, tokens)
            actual = cache["logits"]
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            error = float((actual - expected).abs().max()) if finite else None
            comparisons.append({
                "position": len(tokens) - 1, "max_abs_logit_error": error,
                "within_tolerance": finite and torch.allclose(actual, expected, atol=atol, rtol=rtol),
                "greedy_agreement": finite and int(actual.argmax()) == int(expected.argmax()),
            })
        cases.append({"prompt_tokens": length, "comparisons": comparisons})
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.lengths) < 1 or args.decode_steps < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("lengths/steps must be positive and tolerances nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; real-checkpoint GPU validation must run on another machine")
    from baseline_inference.foveal_engine import FovealLLM

    engine = None
    report = {"backend": "FovealLLM", "model": args.model, "device": args.device,
              "torch_version": torch.__version__, "atol": args.atol, "rtol": args.rtol,
              "passed": False}
    started = time.perf_counter()
    try:
        engine = FovealLLM(args.model, device=args.device)
        report["weights_path"] = engine.weights_path
        report["parameter_dtype"] = str(engine.model.embed.weight.dtype)
        report["gpu"] = torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None
        report["cases"] = verify(engine, args.lengths, args.decode_steps, atol=args.atol, rtol=args.rtol)
        report["passed"] = all(c["within_tolerance"] and c["greedy_agreement"]
                               for case in report["cases"] for c in case["comparisons"])
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
