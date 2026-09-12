"""Run reproducible, serialized checkpoint audit subprocesses on one GPU.

Strict BF16 failures are evidence, not silently accepted serving gates. Review
cache reports and document numerical acceptance before invoking --phase serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CORES = ('polar', 'nope', 'rope')
MODES = ('local', 'lm_output', 'kl', 'lm_output_kl')
CACHE_CASES = {
    'boundaries': ['--controls'],
    'long': ['--controls', '--lengths', '2048', '8192', '32768', '--decode_steps', '2'],
    'continuation': ['--controls', '--lengths', '65', '--decode_steps', '514'],
    'prose': ['--controls', '--workload', 'prose', '--lengths', '65', '641', '--decode_steps', '66'],
    'fp32': ['--float32_oracle', '--decode_steps', '2'],
    'fp32_continuation': ['--float32_oracle', '--lengths', '65', '--decode_steps', '514', '--compare_every', '64'],
    'generated_pages': ['--controls', '--lengths', '1', '--decode_steps', '642', '--compare_every', '64'],
    'fp32_generated_pages': ['--float32_oracle', '--lengths', '1', '--decode_steps', '642', '--compare_every', '64'],
    'greedy': ['--controls', '--workload', 'prose', '--lengths', '65', '641', '--decode_steps', '2', '--greedy_steps', '130'],
    'routing': ['--controls', '--lengths', '8192', '32768', '--decode_steps', '2'],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_root', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--cores', nargs='+', choices=CORES, default=CORES)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--phase', choices=('cache', 'serving'), default='cache')
    parser.add_argument('--cases', nargs='+', choices=CACHE_CASES, default=list(CACHE_CASES))
    parser.add_argument('--lengths', nargs='+', type=int,
                        default=[2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576])
    parser.add_argument('--decode_tokens', nargs='+', type=int, default=[130,32])
    parser.add_argument('--timeout_s', type=int, default=600)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_DISABLE_PROGRESS_BARS='1')
    decoder_hash = hashlib.sha256(Path('baseline_inference/foveal_engine.py').read_bytes()).hexdigest()
    acceptance = None
    if args.phase == 'serving':
        summary_path = args.out_dir / 'cache_summary.json'
        if not summary_path.exists():
            parser.error('serving requires cache_summary.json from summarize_foveal_audit')
        acceptance = json.loads(summary_path.read_text())
        if acceptance.get('decoder_sha256') != decoder_hash:
            parser.error('cache_summary.json describes a different decoder')
    for core in args.cores:
        for mode in args.modes:
            if acceptance is not None:
                gate = acceptance['models'].get(core + '_' + mode, {})
                if not gate.get('serving_eligible') or len(gate.get('greedy', [])) != 2:
                    print('SKIP serving: incomplete/rejected cache audit', core, mode, flush=True)
                    continue
            model = args.checkpoint_root / core / mode / 'cpt'
            jobs = []
            if args.phase == 'cache':
                for name in args.cases:
                    jobs.append((f'{core}_{mode}_{name}', 'benchmarks.verify_foveal_cache', CACHE_CASES[name]))
            else:
                for length in args.lengths:
                    for budget in args.decode_tokens:
                        jobs.append((f'serving_{core}_{mode}_{length}_{budget}', 'benchmarks.audit_foveal_serving',
                                     ['--context_tokens', str(length), '--decode_tokens', str(budget)]))
            for name, module, extra in jobs:
                out = args.out_dir / (name + '.json')
                if out.exists():
                    previous = json.loads(out.read_text())
                    if previous.get('decoder_sha256') == decoder_hash and ('cases' in previous or previous.get('completed') or previous.get('oom')):
                        print('REUSE',name,flush=True)
                        continue
                command = [sys.executable,'-m',module,'--model',str(model),*extra,'--out',str(out)]
                print('START',name,flush=True)
                with (args.out_dir/(name+'.console.log')).open('w') as log:
                    try:
                        result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                timeout=args.timeout_s)
                        code = result.returncode
                    except subprocess.TimeoutExpired:
                        code = 'timeout'
                if out.exists():
                    data = json.loads(out.read_text())
                else:
                    data = {'error': f'subprocess exit {code}', 'model':str(model)}
                data.update(command=command, subprocess_returncode=code, decoder_sha256=decoder_hash)
                out.write_text(json.dumps(data,indent=2)+'\n')
                print('DONE',name,'exit',code,flush=True)


if __name__ == '__main__':
    main()
