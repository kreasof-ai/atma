"""Validate and summarize the independent L40S Foveal audit artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmarks.run_foveal_audit import CORES, MODES

BF16_CASES = {'boundaries': 737, 'long': 9, 'continuation': 515, 'prose': 134,
              'generated_pages': 12}
FP32_CASES = {'fp32': 33, 'fp32_continuation': 10, 'fp32_generated_pages': 12}
CONTROLS = ('full_forward_shape_control', 'full_forward_recurrent_control')


def stats(rows):
    return dict(comparisons=len(rows),
                max_abs_logit_error=max(r['max_abs_logit_error'] for r in rows),
                mean_rms_logit_error=statistics.mean(r['rms_logit_error'] for r in rows),
                max_relative_l2_logit_error=max(r['relative_l2_logit_error'] for r in rows),
                strict_failures=sum(not r['within_tolerance'] for r in rows),
                greedy_disagreements=sum(not r['greedy_agreement'] for r in rows))


def summarize(root):
    decoder_hash = hashlib.sha256(Path('baseline_inference/foveal_engine.py').read_bytes()).hexdigest()
    result = {'decoder_sha256': decoder_hash, 'models': {}}
    for core in CORES:
        for mode in MODES:
            name = core + '_' + mode
            reports, missing = {}, []
            for case, count in (BF16_CASES | FP32_CASES).items():
                path = root / f'{name}_{case}.json'
                if not path.exists():
                    missing.append(case)
                    continue
                data = json.loads(path.read_text())
                rows = [r for c in data.get('cases', []) for r in c['comparisons']]
                if len(rows) != count or data.get('decoder_sha256') != decoder_hash:
                    missing.append(case + ': incomplete or stale')
                    continue
                reports[case] = data
            if missing:
                result['models'][name] = {'complete': False, 'missing': missing, 'serving_eligible': False}
                continue
            bf = [r for case in BF16_CASES for c in reports[case]['cases'] for r in c['comparisons']]
            fp = [r for case in FP32_CASES for c in reports[case]['cases'] for r in c['comparisons']]
            finite = all(isinstance(r[key], (int, float)) and math.isfinite(r[key])
                         for row in bf for r in [row, *[row[c] for c in CONTROLS]]
                         for key in ('max_abs_logit_error', 'rms_logit_error', 'relative_l2_logit_error'))
            if not finite:
                result['models'][name] = {'complete': True, 'serving_eligible': False, 'error': 'nonfinite comparison'}
                continue
            cached = stats(bf)
            controls = {control: stats([r[control] for r in bf]) for control in CONTROLS}
            prefill_exact = all(c['comparisons'][0]['max_abs_logit_error'] == 0
                                for case in BF16_CASES for c in reports[case]['cases'])
            fp_routes_equal = all(v['cached'] == v['reference'] for case in FP32_CASES
                                  for c in reports[case]['cases'] for v in c['final_routes'].values())
            structural = all(reports[case]['passed'] for case in FP32_CASES) and fp_routes_equal
            envelope = all(cached[key] <= sum(control[key] for control in controls.values())
                           for key in ('mean_rms_logit_error', 'max_relative_l2_logit_error'))
            state_errors = {key: max((v for case in BF16_CASES for c in reports[case]['cases']
                                     for v in c['final_state_relative_l2_errors'][key].values()), default=0)
                            for key in ('mem_states', 'k_cache', 'v_cache', 'k_pages', 'v_pages')}
            routes = [v for case in BF16_CASES for c in reports[case]['cases'] for v in c['final_routes'].values()]
            generated_routes = reports['generated_pages']['cases'][0]['final_routes']
            generated_selected = bool(generated_routes) and all(
                not v['cached'] if mode == 'local' else any(page >= 1 for page in v['cached'])
                for v in generated_routes.values())
            model = dict(complete=True, serving_eligible=structural and prefill_exact and envelope and generated_selected,
                         finite=finite, structural_gate=structural, prefill_exact=prefill_exact,
                         numerical_envelope_gate=envelope, generated_pages_gate=generated_selected,
                         bf16=cached, fp32=stats(fp), controls=controls,
                         max_final_state_relative_l2_errors=state_errors,
                         bf16_final_route_sets_compared=len(routes),
                         bf16_final_route_set_disagreements=sum(v['cached'] != v['reference'] for v in routes),
                         generated_page_routes=generated_routes,
                         sources=[f'{name}_{case}.json' for case in reports])
            greedy_path = root / f'{name}_greedy.json'
            if greedy_path.exists():
                greedy = json.loads(greedy_path.read_text())
                if greedy.get('decoder_sha256') == decoder_hash and 'greedy_cases' in greedy:
                    model['greedy'] = greedy['greedy_cases']
            result['models'][name] = model
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.log_dir)
    (args.log_dir/'cache_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, model in result['models'].items():
        print(name, 'eligible' if model.get('serving_eligible') else model.get('missing', 'REJECTED'))


if __name__ == '__main__':
    main()
