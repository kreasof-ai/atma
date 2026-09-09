"""Run the completed TDA checkpoint's full benchmark suite, with resumable stages."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmarks.aggregate import _extract, aggregate, write_outputs
from benchmarks.run_pipeline import ROOT, _run_with_console_log

LENGTHS = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k']
BABI_REVISION = 'ee0d588794c7ac098062ee0d247c733d62e94fe2'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jobs(model, output, adapted):
    manifest = ROOT / 'benchmarks/logs/atma_10b/checkpoint_manifest.json'
    common = [sys.executable, '-m', 'benchmarks.run', '--model', str(model), '--strict']
    stages = [('verify', [sys.executable, '-m', 'external_baselines.verify_checkpoint', str(model)], None)]
    def bench(name, kind, options, checkpoint=None, gate=False):
        log = output / ('gate' if gate else 'results') / f'{name}.log'
        command = common.copy()
        if checkpoint is not None:
            command[command.index('--model') + 1] = str(checkpoint)
        stages.append((name, command + ['--benchmark', kind, *options, '--out', str(log)], log))
    bench('base', 'base', ['--batch_size', '8', '--scoring_max_length', '2048',
                         '--dataset_revisions', str(manifest)])
    retrieval = ['--tasks', 'passkey', 'niah', '--lengths', *LENGTHS,
                 '--depths', '0.1', '0.5', '0.9', '--samples', '50', '--seed', '1234',
                 '--retrieval_value_tokens', '5']
    bench('retrieval_synthetic', 'retrieval', retrieval)
    revision = json.loads(manifest.read_text())['datasets']['codelion/finepdfs-1B']['resolved_revision']
    bench('retrieval_real', 'retrieval', retrieval + ['--haystack', 'codelion/finepdfs-1B', '--haystack_revision', revision])
    bench('bpb', 'longdoc', ['--datasets', 'finepdfs', 'pg19', 'proof_pile',
                           '--lengths', *LENGTHS, '--target_tokens', '256', '--num_docs', '8',
                           '--max_scan', '100000', '--dataset_revisions', str(manifest)])
    bench('serving_direct', 'serving', ['--serving_backend', 'direct', '--lengths', *LENGTHS[:-1],
                                      '--decode_tokens', '32', '--serving_samples', '1', '--max_num_seqs', '1'])
    stages.append(('babilong_finetune', [sys.executable, '-m', 'benchmarks.finetune_babilong',
                   '--model', str(model), '--output_dir', str(adapted),
                   '--dataset_revision', BABI_REVISION, '--tasks', *[f'qa{i}' for i in range(1, 11)],
                   '--train_lengths', '0k', '1k', '2k', '--seq_len', '2048',
                   '--train_start', '0', '--train_end', '80', '--val_start', '80', '--val_end', '90',
                   '--epochs', '3', '--micro_batch_size', '1', '--grad_accum_steps', '8', '--seed', '1234'], None))
    babi = ['--row_start', '90', '--row_end', '100', '--babilong_backend', 'direct', '--max_tokens', '16']
    bench('babilong_gate', 'babilong', ['--tasks', 'qa1', '--lengths', '256k', '--samples', '1', *babi], adapted, gate=True)
    bench('babilong', 'babilong', ['--tasks', *[f'qa{i}' for i in range(1, 11)],
                                '--lengths', '0k', '1k', *LENGTHS, '--samples', '10', *babi], adapted)
    return stages


def gate_passed(log):
    records = [r for kind, r in _extract(log) if kind == 'babilong']
    return bool(records and not records[-1].get('oom_cells')
                and not records[-1].get('unsupported_checkpoint')
                and records[-1].get('counts', {}).get('qa1', {}).get('256k') == 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/supplementary_robustness/scaled_tda_hybrid')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'supplementary/robustness/work/evaluation/tda_pipeline')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    model, output = args.model.resolve(), args.output_dir.resolve()
    adapted = output / 'babilong_checkpoint'
    cfg = json.loads((model / 'config.json').read_text())
    if cfg.get('arch_type') != 'tda_hybrid':
        raise SystemExit('expected a tda_hybrid checkpoint')
    stages = jobs(model, output, adapted)
    if args.dry_run:
        import shlex
        for name, command, _ in stages:
            print(f'{name}: {shlex.join(command)}')
        return
    from benchmarks.babilong import _load_official_prompts
    if _load_official_prompts():
        raise SystemExit('This comparison requires the original builtin-v1 BABILong prompts.')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another runner is using this output directory')
        sources = [model / 'weights.pt', model / 'config.json', Path(__file__).resolve(),
                   ROOT / 'benchmarks/logs/atma_10b/checkpoint_manifest.json',
                   ROOT / 'supplementary/robustness/dependencies.json']
        for directory in ('benchmarks', 'external_baselines', 'model', 'train', 'raven_baseline'):
            sources.extend((ROOT / directory).glob('*.py'))
        signature = {str(p): sha256(p) for p in sorted(set(sources))}
        plan = {'sources': signature, 'stages': stages, 'gpu': args.gpu}
        fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
        state_path = output / 'progress.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'fingerprint': fingerprint, 'completed': {}}
        if state['fingerprint'] != fingerprint:
            raise SystemExit('Checkpoint/code/plan changed; use a new --output-dir to preserve previous results.')
        (output / 'plan.json').write_text(json.dumps(plan, indent=2, default=str) + '\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        try:
            for name, command, log in stages:
                done = state['completed'].get(name)
                if done and all(Path(p).is_file() and sha256(p) == digest for p, digest in done.items()):
                    print(f'[tda] resume: {name} already complete', flush=True)
                    continue
                if name == 'babilong_finetune' and adapted.exists():
                    raise SystemExit('Incomplete/changed BABILong checkpoint retained. Use a new --output-dir; no weights were overwritten.')
                if log is not None and log.exists():
                    # Preserve incomplete output outside the aggregate input tree.
                    archive = output / 'incomplete'
                    archive.mkdir(exist_ok=True)
                    import time
                    log.rename(archive / f'{name}.{time.time_ns()}.log')
                rc = _run_with_console_log(command, output / 'console' / f'{name}.console.log', env)
                if rc:
                    raise SystemExit(f'{name} failed with exit code {rc}; rerun this command to resume completed stages')
                if log is not None and not _extract(log):
                    raise SystemExit(f'{name} produced no complete result block')
                if name == 'babilong_gate' and not gate_passed(log):
                    raise SystemExit('BABILong 256K gate failed; prior results and adapted weights are preserved.')
                artifacts = [log] if log is not None else [output / 'console' / f'{name}.console.log']
                if name == 'babilong_finetune':
                    artifacts += [adapted / p for p in ('weights.pt', 'config.json', 'finetune_manifest.json', 'training_summary.json')]
                state['completed'][name] = {str(p): sha256(p) for p in artifacts}
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=2) + '\n')
                temporary.replace(state_path)
        finally:
            result = aggregate(output / 'results')
            write_outputs(result, output / 'benchmark_matrix.json', output / 'benchmark_matrix.csv')
        print(f'[tda] all stages complete; results: {output}')


if __name__ == '__main__':
    main()
