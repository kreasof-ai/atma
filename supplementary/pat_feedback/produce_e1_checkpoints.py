"""Produce and materialize all 9 E1 checkpoints in checkpoints/pat_feedback/

Triplet 1 (Seed 20270912):
  - pat_e1_s1_nope_memory
  - pat_e1_s1_temperature_softmax_memory
  - pat_e1_s1_polar_memory

Triplet 2 (Seed 20270913):
  - pat_e1_s2_nope_memory
  - pat_e1_s2_temperature_softmax_memory
  - pat_e1_s2_polar_memory

Triplet 3 (Seed 20270914):
  - pat_e1_s3_nope_memory
  - pat_e1_s3_temperature_softmax_memory
  - pat_e1_s3_polar_memory
"""

import hashlib
import json
import shutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = ROOT / "checkpoints" / "pat_feedback"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "s1": {
        "seed": 20270912,
        "nope": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-nope__reg-baseline__distr-0__mem-1__win-0/snapshots/d2974b46bf1be80fe0285ed7c15903b8079f615a",
        "polar": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--atma-10b-L40S-mbs16-polar__reg-baseline__distr-0__mem-1__win-0/snapshots/faa10ea68590d57c0869c9dd6d38f40c476cff5d",
    },
    "s2": {
        "seed": 20270913,
        "nope": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_nope/snapshots/8ab5fe847915223abfec6a146c97f173b8c72058",
        "polar": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed1_polar/snapshots/6b3226d04af1f8c982c9085c8c9dbd8282e11220",
    },
    "s3": {
        "seed": 20270914,
        "nope": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_nope/snapshots/3196eac86d84759bb245b410da61eb8665bfdf05",
        "polar": "/home/sagemaker-user/.cache/huggingface/hub/models--ChavyvAkvar--repl_seed2_polar/snapshots/50a3790bd3fcb55b3a4d69921df2725b21863fa5",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("==========================================================")
    print("Materializing All 9 E1 Checkpoints in checkpoints/pat_feedback/")
    print("==========================================================")

    manifest = {}

    for t_id, src in SOURCES.items():
        seed = src["seed"]
        print(f"\nProcessing Triplet {t_id} (Seed {seed})...")

        nope_src = Path(src["nope"])
        polar_src = Path(src["polar"])

        nope_state = torch.load(nope_src / "weights.pt", map_location="cpu", weights_only=True)["model"]
        polar_state = torch.load(polar_src / "weights.pt", map_location="cpu", weights_only=True)["model"]

        # Base config templates
        cfg_nope = json.loads((ROOT / "supplementary" / "pat_feedback" / "configs" / f"pat_e1_{t_id}_nope_memory.json").read_text(encoding="utf-8"))
        cfg_temp = json.loads((ROOT / "supplementary" / "pat_feedback" / "configs" / f"pat_e1_{t_id}_temperature_softmax_memory.json").read_text(encoding="utf-8"))
        cfg_polar = json.loads((ROOT / "supplementary" / "pat_feedback" / "configs" / f"pat_e1_{t_id}_polar_memory.json").read_text(encoding="utf-8"))

        # 1. nope_memory checkpoint
        nope_dir = CHECKPOINTS_DIR / f"pat_e1_{t_id}_nope_memory"
        nope_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": nope_state}, nope_dir / "weights.pt")
        (nope_dir / "config.json").write_text(json.dumps(cfg_nope, indent=2) + "\n")
        if (nope_src / "tokenizer.json").is_file():
            shutil.copy(nope_src / "tokenizer.json", nope_dir / "tokenizer.json")
        h_nope = sha256_file(nope_dir / "weights.pt")
        print(f"  [OK] pat_e1_{t_id}_nope_memory -> {h_nope[:16]}... ({sum(p.numel() for p in nope_state.values()):,} params)")

        # 2. polar_memory checkpoint
        polar_dir = CHECKPOINTS_DIR / f"pat_e1_{t_id}_polar_memory"
        polar_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": polar_state}, polar_dir / "weights.pt")
        (polar_dir / "config.json").write_text(json.dumps(cfg_polar, indent=2) + "\n")
        if (polar_src / "tokenizer.json").is_file():
            shutil.copy(polar_src / "tokenizer.json", polar_dir / "tokenizer.json")
        h_polar = sha256_file(polar_dir / "weights.pt")
        print(f"  [OK] pat_e1_{t_id}_polar_memory -> {h_polar[:16]}... ({sum(p.numel() for p in polar_state.values()):,} params)")

        # 3. temperature_softmax_memory checkpoint
        temp_state = dict(nope_state)
        for b in [2, 6, 10, 14]:
            temp_state[f"blocks.{b}.attn.len_gain_raw"] = polar_state[f"blocks.{b}.attn.len_gain_raw"].clone()

        temp_dir = CHECKPOINTS_DIR / f"pat_e1_{t_id}_temperature_softmax_memory"
        temp_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": temp_state}, temp_dir / "weights.pt")
        (temp_dir / "config.json").write_text(json.dumps(cfg_temp, indent=2) + "\n")
        if (nope_src / "tokenizer.json").is_file():
            shutil.copy(nope_src / "tokenizer.json", temp_dir / "tokenizer.json")
        h_temp = sha256_file(temp_dir / "weights.pt")
        print(f"  [OK] pat_e1_{t_id}_temperature_softmax_memory -> {h_temp[:16]}... ({sum(p.numel() for p in temp_state.values()):,} params)")

        manifest[t_id] = {
            "seed": seed,
            "nope_memory": {"dir": str(nope_dir), "sha256": h_nope, "params": sum(p.numel() for p in nope_state.values())},
            "polar_memory": {"dir": str(polar_dir), "sha256": h_polar, "params": sum(p.numel() for p in polar_state.values())},
            "temperature_softmax_memory": {"dir": str(temp_dir), "sha256": h_temp, "params": sum(p.numel() for p in temp_state.values())},
        }

    manifest_path = ROOT / "supplementary" / "pat_feedback" / "records" / "e1_checkpoints_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSaved checkpoint manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
