#!/usr/bin/env python3
"""Generate Matched Multi-Seed Configs for E1 Component Isolation & 8K Training Length.

Directly addresses:
1. Reviewer HE3D (Weakness 1b) & Reviewer ihsT (Weakness 2 & Question 2):
   Generates 3-seed matched E1 training configs (`seeds 20270912, 20270913, 20270914`,
   preserving the exact `shared_tensor_hash_digest` of each triplet) for:
   - `pat_e1_s{1,2,3}_polar_direction_only_memory`: `polar_variant = "direction_only"`
     (removes the magnitude channel while keeping direction normalization).
   - `pat_e1_s{1,2,3}_polar_fixed_null_slope_memory`: `polar_variant = "fixed_null"`
     (fixes the sqrt(log(n+1)) null slope near zero across all 3 seeds to test Table 7's
     single-seed observation on the 30-document multi-seed benchmark).
   - `pat_e1_s{1,2,3}_rope_memory`: Seed-matched RoPE + memory across the same 3 seeds.
2. Reviewer birj (Weakness 2 & Question 2), Reviewer HE3D (Question 3), Reviewer ihsT (Weakness 5):
   Generates 8K pretraining length (`seq_len = 8192`) matched configs for:
   - `pat_len8k_s1_nope_memory`
   - `pat_len8k_s1_rope_memory`
   - `pat_len8k_s1_temperature_softmax_memory`
   - `pat_len8k_s1_polar_memory`
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E1_CONFIGS_DIR = ROOT / "supplementary" / "pat_feedback" / "configs"
OUT_CONFIGS_DIR = ROOT / "iclr27_rebuttal" / "configs"
OUT_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def generate_e1_component_configs() -> list[str]:
    created = []
    for s_idx in [1, 2, 3]:
        base_polar_path = E1_CONFIGS_DIR / f"pat_e1_s{s_idx}_polar_memory.json"
        base_nope_path = E1_CONFIGS_DIR / f"pat_e1_s{s_idx}_nope_memory.json"
        polar_cfg = json.loads(base_polar_path.read_text(encoding="utf-8"))
        nope_cfg = json.loads(base_nope_path.read_text(encoding="utf-8"))

        # 1. Direction-only Polar (removes magnitude channel, retains direction normalization)
        dir_cfg = copy.deepcopy(polar_cfg)
        dir_cfg["run_id"] = f"pat_e1_s{s_idx}_polar_direction_only_memory"
        dir_cfg["polar_variant"] = "direction_only"
        out_p = OUT_CONFIGS_DIR / f"{dir_cfg['run_id']}.json"
        out_p.write_text(json.dumps(dir_cfg, indent=2) + "\n", encoding="utf-8")
        created.append(out_p.name)

        # 2. Fixed-null-slope Polar (fixes sqrt(log(n+1)) coefficient near zero)
        fn_cfg = copy.deepcopy(polar_cfg)
        fn_cfg["run_id"] = f"pat_e1_s{s_idx}_polar_fixed_null_slope_memory"
        fn_cfg["polar_variant"] = "fixed_null"
        out_p = OUT_CONFIGS_DIR / f"{fn_cfg['run_id']}.json"
        out_p.write_text(json.dumps(fn_cfg, indent=2) + "\n", encoding="utf-8")
        created.append(out_p.name)

        # 3. Seed-matched RoPE + memory
        rope_cfg = copy.deepcopy(nope_cfg)
        rope_cfg["run_id"] = f"pat_e1_s{s_idx}_rope_memory"
        rope_cfg["attn_type"] = "rope"
        out_p = OUT_CONFIGS_DIR / f"{rope_cfg['run_id']}.json"
        out_p.write_text(json.dumps(rope_cfg, indent=2) + "\n", encoding="utf-8")
        created.append(out_p.name)

    return created


def generate_8k_length_control_configs() -> list[str]:
    created = []
    arms = [
        ("nope_memory", "nope", "full"),
        ("rope_memory", "rope", "full"),
        ("temperature_softmax_memory", "temperature_softmax", "full"),
        ("polar_memory", "polar", "full"),
    ]
    base_cfg = json.loads((E1_CONFIGS_DIR / "pat_e1_s1_polar_memory.json").read_text(encoding="utf-8"))

    for arm_name, attn_type, polar_var in arms:
        cfg = copy.deepcopy(base_cfg)
        cfg["run_id"] = f"pat_len8k_s1_{arm_name}"
        cfg["attn_type"] = attn_type
        cfg["polar_variant"] = polar_var
        cfg["seq_len"] = 8192
        cfg["mbs"] = 1
        cfg["gradient_accumulation_steps"] = 64
        cfg["eval_lengths"] = [8192, 16384, 32768, 65536, 131072]
        cfg["needle_distances"] = [8192, 16384, 32768, 65536, 131072]
        cfg["experiment_group"] = "pat_rebuttal_len8k"
        cfg["comparison_group"] = "pat_len8k_triplet_s1"
        out_p = OUT_CONFIGS_DIR / f"{cfg['run_id']}.json"
        out_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        created.append(out_p.name)

    return created


if __name__ == "__main__":
    c1 = generate_e1_component_configs()
    c2 = generate_8k_length_control_configs()
    print(f"Generated {len(c1)} E1 component/RoPE configs and {len(c2)} 8K-length control configs in {OUT_CONFIGS_DIR}:")
    for name in c1 + c2:
        print(f"  - {name}")
