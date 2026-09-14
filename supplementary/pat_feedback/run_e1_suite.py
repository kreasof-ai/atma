#!/usr/bin/env python3
"""Master E1 Orchestrator: Trains and evaluates the 3 arms across 3 seeds.

Arms:
  - nope_memory
  - temperature_softmax_memory
  - polar_memory

Seeds:
  - s1: 20270912
  - s2: 20270913
  - s3: 20270914

Total: 9 matched runs. Checkpoints persisted in checkpoints/pat_feedback/.
Evaluation persisted in work/evaluation/ and records/.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FLA_CUSTOM_OP"] = "1"

from supplementary.pat_feedback.train_e1 import train_single_run

CONFIGS_DIR = ROOT / "supplementary" / "pat_feedback" / "configs"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "pat_feedback"
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work"
EVAL_DIR = WORK_DIR / "evaluation"
RECORDS_DIR = ROOT / "supplementary" / "pat_feedback" / "records"

RUNS = [
    # Triplet 1 (Seed 20270912)
    {"run_id": "pat_e1_s1_nope_memory", "triplet": "s1", "arm": "nope_memory", "seed": 20270912},
    {"run_id": "pat_e1_s1_temperature_softmax_memory", "triplet": "s1", "arm": "temperature_softmax_memory", "seed": 20270912},
    {"run_id": "pat_e1_s1_polar_memory", "triplet": "s1", "arm": "polar_memory", "seed": 20270912},
    # Triplet 2 (Seed 20270913)
    {"run_id": "pat_e1_s2_nope_memory", "triplet": "s2", "arm": "nope_memory", "seed": 20270913},
    {"run_id": "pat_e1_s2_temperature_softmax_memory", "triplet": "s2", "arm": "temperature_softmax_memory", "seed": 20270913},
    {"run_id": "pat_e1_s2_polar_memory", "triplet": "s2", "arm": "polar_memory", "seed": 20270913},
    # Triplet 3 (Seed 20270914)
    {"run_id": "pat_e1_s3_nope_memory", "triplet": "s3", "arm": "nope_memory", "seed": 20270914},
    {"run_id": "pat_e1_s3_temperature_softmax_memory", "triplet": "s3", "arm": "temperature_softmax_memory", "seed": 20270914},
    {"run_id": "pat_e1_s3_polar_memory", "triplet": "s3", "arm": "polar_memory", "seed": 20270914},
]


def update_master_records():
    all_evals = {}
    triplet_results = {"s1": {}, "s2": {}, "s3": {}}

    for r in RUNS:
        eval_file = EVAL_DIR / f"{r['run_id']}_eval.json"
        if eval_file.is_file():
            try:
                data = json.loads(eval_file.read_text(encoding="utf-8"))
                all_evals[r["run_id"]] = data
                triplet_results[r["triplet"]][r["arm"]] = data["primary_natural_text_exact_mean_8k_to_64k"]
            except Exception:
                pass

    out_eval = EVAL_DIR / "e1_component_attribution.json"
    out_eval.write_text(json.dumps(all_evals, indent=2) + "\n", encoding="utf-8")

    # Build summary
    contrasts = {}
    for t_id, arms in triplet_results.items():
        if len(arms) == 3:
            p = arms["polar_memory"]
            t = arms["temperature_softmax_memory"]
            n = arms["nope_memory"]
            contrasts[t_id] = {
                "nope_memory": n,
                "temperature_softmax_memory": t,
                "polar_memory": p,
                "primary_contrast_polar_vs_temp": p - t,
                "secondary_contrast_temp_vs_nope": t - n,
                "total_polar_vs_nope": p - n,
            }

    summary_file = RECORDS_DIR / "e1_results_summary.json"
    summary_file.write_text(json.dumps({
        "status": "in_progress" if len(all_evals) < 9 else "completed",
        "completed_runs": len(all_evals),
        "total_planned_runs": 9,
        "triplet_contrasts": contrasts,
        "runs": list(all_evals.keys()),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"[Master E1] Updated records ({len(all_evals)}/9 runs completed).")
    if contrasts:
        print("[Master E1] Available Triplet Contrasts:")
        for t_id, c in contrasts.items():
            print(f"  Triplet {t_id}: Polar={c['polar_memory']:.2f}% | Temp={c['temperature_softmax_memory']:.2f}% | NoPE={c['nope_memory']:.2f}% -> Contrast={c['primary_contrast_polar_vs_temp']:+.2f}%")


def main():
    print("==========================================================================")
    print("Launching Full E1 Study (3 arms x 3 seeds = 9 fresh runs)")
    print("==========================================================================")

    for i, r in enumerate(RUNS, 1):
        run_id = r["run_id"]
        cfg_path = CONFIGS_DIR / f"{run_id}.json"
        ckpt_file = CHECKPOINTS_DIR / run_id / "weights.pt"
        eval_file = EVAL_DIR / f"{run_id}_eval.json"

        print(f"\n[{i}/9] Checking run {run_id}...")
        if ckpt_file.is_file() and eval_file.is_file():
            print(f"  Checkpoint and evaluation already exist for {run_id}. Skipping.")
            continue

        print(f"  Starting fresh training for {run_id}...")
        train_single_run(cfg_path)
        update_master_records()

    print("\nAll planned E1 runs executed successfully!")
    update_master_records()


if __name__ == "__main__":
    main()
