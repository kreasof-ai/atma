"""Repetition-level serial Foveal serving audit (one model/length per process).

Use only after checkpoint cache correctness has been assessed. Cold requests,
warm requests, actual token counts, and cache/allocator bytes remain separate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from benchmarks.serving import _prompt_ids
from inference.sampling_params import SamplingParams


def measure(engine, prompt, decode_tokens):
    cache_ref = {}
    prefill = engine._prefill

    def capture(*args, **kwargs):
        token, cache = prefill(*args, **kwargs)
        cache_ref['cache'] = cache
        return token, cache

    engine._prefill = capture
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        outputs = engine.generate([prompt], SamplingParams(
            temperature=0, max_tokens=decode_tokens, ignore_eos=True))
        torch.cuda.synchronize()
        result = dict(engine.last_metrics)
        result['wall_time_s'] = time.perf_counter() - started
        ids = outputs[0]['token_ids']
        result['generated_tokens'] = len(ids)
        result['output_tokens_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert len(ids) == decode_tokens
        assert result['prefill_tokens'] == len(prompt)
        assert result['decode_tokens'] == decode_tokens - 1
        cache = cache_ref['cache']
        result['cache_sequence_tokens'] = cache['seq_len']
        result['full_prefix_kv_bytes'] = sum(
            t.numel() * t.element_size()
            for name in ('k_cache', 'v_cache') for t in cache[name].values())
        result['remote_pages_per_layer'] = {
            str(i): t.numel() for i, t in cache['selected_remote'].items()}
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return result
    except Exception as exc:
        exc.foveal_phase = 'decode' if 'cache' in cache_ref else 'prefill'
        if 'cache' in cache_ref:
            cache = cache_ref['cache']
            exc.foveal_cache_bytes = sum(t.numel() * t.element_size()
                                        for name in ('k_cache', 'v_cache')
                                        for t in cache[name].values())
        raise
    finally:
        engine._prefill = prefill
        cache_ref.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--context_tokens', type=int, required=True)
    parser.add_argument('--decode_tokens', type=int, default=130)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmup_samples', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.context_tokens, args.decode_tokens, args.samples, args.warmup_samples) < 1:
        parser.error('context/decode/samples/warmup must be positive')
    from baseline_inference.foveal_engine import FovealLLM

    report = dict(backend='FovealLLM', suite='foveal_cached_generation',
                  protocol='exact-token-prefill-v2', ignore_eos=True,
                  model=args.model, context_tokens=args.context_tokens,
                  requested_decode_tokens=args.decode_tokens, max_num_seqs=1,
                  samples=args.samples, warmup_samples=args.warmup_samples,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  decoder_sources_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                      for name in ('baseline_inference/foveal_engine.py', 'baseline_inference/foveal_decode.py',
                                   'baseline_inference/foveal_triton.py')},
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  compiler_cache_policy='reuse persistent caches; first request excluded from warm statistics',
                  warmups=[], repetitions=[], oom=False, completed=False)
    started = time.perf_counter()
    engine = None
    phase = 'load'
    try:
        torch.cuda.reset_peak_memory_stats()
        engine = FovealLLM(args.model, device='cuda')
        torch.cuda.synchronize()
        report['load_time_s'] = time.perf_counter() - started
        report['load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['load_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['weights_path'] = engine.weights_path
        report['target_full_prefix_kv_bytes'] = (args.context_tokens + args.decode_tokens - 1) * sum(
            2 * block.attn.num_kv_heads * block.attn.head_dim * engine.model.embed.weight.element_size()
            for block in engine.model.blocks if hasattr(block.attn, 'page_size'))
        prompt = _prompt_ids(engine.tokenizer, args.context_tokens)
        for i in range(args.warmup_samples + args.samples):
            phase = 'warmup' if i < args.warmup_samples else 'measurement'
            result = measure(engine, prompt, args.decode_tokens)
            report['warmups' if phase == 'warmup' else 'repetitions'].append(result)
            print(f'{phase} {i}: prefill={result["prefill_time"]:.3f}s '
                  f'decode={result["decode_time"]:.3f}s '
                  f'peak={result["peak_allocated_bytes"]/2**30:.3f}GiB', flush=True)
        report['completed'] = True
    except torch.cuda.OutOfMemoryError as exc:
        report.update(oom=True, phase=phase, error=str(exc),
                      failure_operation=getattr(exc, 'foveal_phase', phase),
                      full_prefix_kv_bytes_at_failure=getattr(exc, 'foveal_cache_bytes', None),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        report.update(phase=phase, error=f'{type(exc).__name__}: {exc}')
    finally:
        if engine is not None:
            engine.close()
        gc.collect()
        torch.cuda.empty_cache()
        report['elapsed_s'] = time.perf_counter() - started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{"OK" if report["completed"] else "OOM" if report["oom"] else "ERROR"} -> {args.out}', flush=True)
    return 0 if report['completed'] or report['oom'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
