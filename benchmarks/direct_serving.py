"""Systems measurements through the exact training forward, without a KV cache.

This backend is explicitly distinct from paged/cached serving. Every decode step
recomputes the growing prefix; use its results only with that qualification.
"""
from __future__ import annotations

import time


def measure_sequence(scorer, prompt, decode_tokens):
    """Time prompt-to-first-token separately from subsequent full-prefix steps.

    Ignore EOS for a fixed output-token budget. Input construction, forward, and
    greedy selection are included; checkpoint loading and tokenization are not.
    """
    import torch

    if not prompt or decode_tokens < 1:
        raise ValueError('nonempty prompt and positive decode_tokens required')
    ids = list(prompt)
    prefill_time = decode_time = 0.0
    for step in range(decode_tokens):
        if scorer.device.type == 'cuda':
            torch.cuda.synchronize(scorer.device)
        started = time.perf_counter()
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.int32, device=scorer.device)
            hidden = scorer._forward_hidden(inputs)
            logits = scorer.model.proj(scorer.model.norm(hidden[:, -1:])).float()
            token = int(logits[0, 0].argmax().item())
        if scorer.device.type == 'cuda':
            torch.cuda.synchronize(scorer.device)
        elapsed = time.perf_counter() - started
        if step == 0:
            prefill_time = elapsed
        else:
            decode_time += elapsed
        ids.append(token)
        del inputs, hidden, logits
    return {
        'prefill_tokens': len(prompt),
        'decode_tokens': decode_tokens - 1,
        'generated_tokens': decode_tokens,
        'prefill_time_s': prefill_time,
        'decode_time_s': decode_time,
    }


def run_direct_serving(model_path, lengths, *, decode_tokens=32, samples=1,
                       max_num_seqs=1, log_fn=print):
    import torch
    from benchmarks.longdoc import _parse_length
    from benchmarks.retrieval import _is_cuda_oom
    from benchmarks.scoring import DirectScorer
    from benchmarks.serving import _clear_cuda, _prompt_ids
    from benchmarks.model import read_checkpoint_config

    if samples < 1 or decode_tokens < 1:
        raise ValueError('samples and decode_tokens must be positive')
    if max_num_seqs != 1:
        raise ValueError('direct serving supports batch size 1 only')
    results = {}
    max_success = None
    started = time.perf_counter()
    for label in lengths:
        scorer = None
        context = _parse_length(label)
        if context < 1:
            raise ValueError('context lengths must be positive')
        _clear_cuda()
        cell_started = time.perf_counter()
        try:
            scorer = DirectScorer(model_path, max_length=None, batch_size=1)
            prompt = _prompt_ids(scorer.tokenizer, context)
            # Warm the exact prefill/decode shapes once before measuring.
            measure_sequence(scorer, prompt, decode_tokens)
            _clear_cuda()
            if scorer.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(scorer.device)
            measurements = [measure_sequence(scorer, prompt, decode_tokens)
                            for _ in range(samples)]
            totals = {key: sum(row[key] for row in measurements)
                      for key in measurements[0]}
            prefill_time = totals['prefill_time_s']
            decode_time = totals['decode_time_s']
            gpu = scorer.device.type == 'cuda'
            cell = {
                'context_tokens': context, 'samples': samples,
                'requested_decode_tokens': decode_tokens,
                **totals,
                'prefill_tokens_per_s': totals['prefill_tokens'] / prefill_time,
                'decode_tokens_per_s': totals['decode_tokens'] / decode_time if decode_time else None,
                'time_to_first_token_s': prefill_time / samples,
                'decode_latency_per_token_s': decode_time / totals['decode_tokens'] if totals['decode_tokens'] else None,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(scorer.device) if gpu else 0,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(scorer.device) if gpu else 0,
                'measurements': measurements, 'oom': False,
            }
            max_success = max(max_success or 0, context)
            log_fn(f'[serving:direct-full-recompute] context={label} '
                   f'prefill={cell["prefill_tokens_per_s"]:.1f} tok/s '
                   f'decode={cell["decode_tokens_per_s"]} tok/s')
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            cell = {'context_tokens': context, 'samples': samples,
                    'requested_decode_tokens': decode_tokens,
                    'oom': True, 'error': str(exc)[:500]}
            log_fn(f'[serving:direct-full-recompute] context={label}: OOM')
        finally:
            if scorer is not None:
                scorer.close()
            _clear_cuda()
        cell['wall_time_s'] = round(time.perf_counter() - cell_started, 3)
        results[str(label)] = cell
    return {
        'benchmark': 'serving', 'protocol': 'direct-full-recompute-v1',
        'backend': 'direct-full-recompute', 'cached_decode': False,
        'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        'eos_policy': 'ignore-for-fixed-token-budget', 'warmup_sequences_per_length': 1,
        'lengths': list(lengths), 'decode_tokens': decode_tokens,
        'samples': samples, 'max_num_seqs': 1,
        'max_successful_context_tokens': max_success, 'results': results,
        'elapsed_s': round(time.perf_counter() - started, 1),
        'model_config': read_checkpoint_config(model_path),
    }
