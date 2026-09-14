#!/usr/bin/env python3
"""E1 Training and Evaluation Engine.

Implements the prospective 3-arm x 3-seed component attribution study:
- Arms: nope_memory, temperature_softmax_memory, polar_memory
- Seeds: 20270912, 20270913, 20270914
- Exact common tensor initialization across arms within each triplet
- Muon + AdamW optimizer with 0.7 cooldown fraction
- Checkpoint persistence and full retrieval / BPB evaluation on frozen fixtures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

os.environ["FLA_CUSTOM_OP"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import AtmaConfig
from train.data import data_generator, get_data
from train.model import Model
from train.optimizer import Muon
from train.reproducibility import runtime_metadata, seed_run

MANIFESTS_DIR = ROOT / "supplementary" / "pat_feedback" / "manifests"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "pat_feedback"
WORK_DIR = ROOT / "supplementary" / "pat_feedback" / "work"
LOGS_DIR = WORK_DIR / "runs"
EVAL_DIR = WORK_DIR / "evaluation"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

E1_LENGTHS = [2048, 8192, 16384, 32768, 65536]
PRIMARY_RETRIEVAL_LENGTHS = [8192, 16384, 32768, 65536]


def _emit_block(fh, name: str, obj: Any) -> None:
    fh.write(f"\n==={name}===\n{json.dumps(obj)}\n===END===\n")
    fh.flush()


def init_triplet_models(seed: int, device: torch.device) -> Dict[str, Model]:
    """Instantiate the three models in a triplet with byte-identical common tensors."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 1. Base model to generate the reference random draws
    cfg_base = AtmaConfig(attn_type="nope", mem_enabled=True, attn_window=None)
    m_base = Model(cfg_base)
    for name, p in m_base.named_parameters():
        if "proj" in name:
            p.data.zero_()
    base_state = dict(m_base.named_parameters())

    # 2. Build the three arms
    models = {}
    arm_configs = {
        "nope_memory": AtmaConfig(attn_type="nope", mem_enabled=True, attn_window=None),
        "temperature_softmax_memory": AtmaConfig(attn_type="temperature_softmax", mem_enabled=True, attn_window=None),
        "polar_memory": AtmaConfig(attn_type="polar", mem_enabled=True, attn_window=None),
    }

    for arm, cfg in arm_configs.items():
        m = Model(cfg).to(device)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name in base_state:
                    p.copy_(base_state[name])
                elif "len_gain_raw" in name:
                    p.fill_(-1.0)
                elif "proj" in name:
                    p.zero_()
        models[arm] = m

    # Assert common tensors match
    shared = sorted(set(base_state.keys()) & set(dict(models["temperature_softmax_memory"].named_parameters()).keys()) & set(dict(models["polar_memory"].named_parameters()).keys()))
    s_temp = dict(models["temperature_softmax_memory"].named_parameters())
    s_polar = dict(models["polar_memory"].named_parameters())
    for k in shared:
        h1 = hashlib.sha256(base_state[k].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        h2 = hashlib.sha256(s_temp[k].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        h3 = hashlib.sha256(s_polar[k].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert h1 == h2 == h3, f"Shared tensor mismatch in {k}"

    return models


def evaluate_e1_checkpoints(ckpt_path: Path, arm: str, seed: int, device: torch.device, fh=None) -> Dict[str, Any]:
    """Full E1 evaluation on the frozen 30-document retrieval fixtures and 18-document BPB panel using DirectScorer."""
    from benchmarks.scoring import DirectScorer, TokenRequest

    bpb_tokens_path = MANIFESTS_DIR / "bpb_18_document_tokens.pt"
    retrieval_tokens_path = MANIFESTS_DIR / "retrieval_30_document_tokens.pt"
    retrieval_manifest_path = MANIFESTS_DIR / "retrieval_fixture_manifest.json"

    bpb_tokens = torch.load(bpb_tokens_path, weights_only=True)
    retrieval_tokens = torch.load(retrieval_tokens_path, weights_only=True)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))

    t0 = time.time()
    scorer = DirectScorer(str(ckpt_path), device=str(device), max_length=65536 + 256, batch_size=8)
    tok = scorer.tokenizer

    # 1. Fixed-target BPB at 2K and 64K
    bpb_results = {}
    for length in [2048, 65536]:
        corpus_bpb = {}
        for corpus, doc_list in bpb_tokens.items():
            tot_nll, tot_bytes = 0.0, 0
            reqs = []
            bytes_list = []
            for doc in doc_list:
                ctx = doc[262144 - length:262144]
                tgt = doc[262144:262144 + 256]
                reqs.append(TokenRequest(tuple(ctx), tuple(tgt)))
                bytes_list.append(len(tok.decode(tgt).encode("utf-8")))

            scores = scorer.score_requests(reqs, batch_size=8)
            for sc, b_cnt in zip(scores, bytes_list):
                tot_nll += -sc["loglikelihood"]
                tot_bytes += b_cnt
            corpus_bpb[corpus] = tot_nll / math.log(2.0) / tot_bytes
        mean_bpb = sum(corpus_bpb.values()) / 3.0
        bpb_results[str(length)] = {"mean_bpb": mean_bpb, "per_corpus": corpus_bpb}

    # 2. Retrieval: 30 cases x 2 suites x 2 tasks x 3 depths across 5 lengths
    retrieval_results = {"synthetic": {}, "real": {}}
    filler_tok = tok.encode("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")

    for suite in ["synthetic", "real"]:
        retrieval_results[suite] = {}
        for length in E1_LENGTHS:
            reqs = []
            for case in retrieval_manifest["cases"]:
                case_id = case["case_id"]
                target_ids = case["target_token_ids"]
                for task in ["passkey", "niah"]:
                    cue_ids = case["cues"][task]["token_ids"]
                    needle_ids = cue_ids + target_ids
                    scaffold = 1 + len(needle_ids) + len(cue_ids)
                    budget = length - scaffold
                    if budget <= 0:
                        continue
                    if suite == "real":
                        body = retrieval_tokens[case_id][:budget]
                        if len(body) < budget:
                            reps = (budget - len(body)) // len(filler_tok) + 1
                            body = (body + (filler_tok * reps))[:budget]
                    else:
                        reps = budget // max(len(filler_tok), 1) + 1
                        body = (filler_tok * reps)[:budget]

                    for depth in [0.1, 0.5, 0.9]:
                        insert = int(depth * len(body))
                        prompt = [tok.eos_token_id] + body[:insert] + needle_ids + body[insert:] + cue_ids
                        reqs.append(TokenRequest(tuple(prompt), tuple(target_ids)))

            scores = scorer.score_requests(reqs, batch_size=8)
            tot_exact = sum(int(s["greedy_exact"]) for s in scores)
            tot_correct = sum(s["correct_tokens"] for s in scores)
            tot_tokens = sum(s["tokens"] for s in scores)
            prompt_count = len(scores)

            exact_acc = (tot_exact / prompt_count) * 100.0 if prompt_count else 0.0
            tok_acc = (tot_correct / tot_tokens) * 100.0 if tot_tokens else 0.0
            retrieval_results[suite][str(length)] = {
                "exact_accuracy_percent": exact_acc,
                "token_accuracy_percent": tok_acc,
                "prompts": prompt_count
            }

    # Primary endpoint: natural text exact retrieval averaged over 8K/16K/32K/64K
    primary_exact_mean = sum(retrieval_results["real"][str(l)]["exact_accuracy_percent"] for l in PRIMARY_RETRIEVAL_LENGTHS) / len(PRIMARY_RETRIEVAL_LENGTHS)

    del scorer
    torch.cuda.empty_cache()

    eval_record = {
        "run_id": f"pat_e1_{arm}_{seed}",
        "arm": arm,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "primary_natural_text_exact_mean_8k_to_64k": primary_exact_mean,
        "retrieval": retrieval_results,
        "bpb": bpb_results
    }
    return eval_record


def train_single_run(config_path: Path, max_updates: int | None = None, save_checkpoint: bool = True) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = cfg["run_id"]
    arm = cfg["attn_type"]
    if arm == "nope":
        arm = "nope_memory"
    elif arm == "temperature_softmax":
        arm = "temperature_softmax_memory"
    elif arm == "polar":
        arm = "polar_memory"

    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = LOGS_DIR / f"{run_id}.log"
    fh = open(log_path, "w", buffering=1)

    def p0(s):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    p0("=" * 80)
    p0(f"[E1 Training] Starting run {run_id} (arm={arm}, seed={seed}) on {device} ({torch.cuda.get_device_name(0)})...")
    p0("=" * 80)

    # 1. Load data
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]
    mbs = cfg["mbs"]
    num_chunks = cfg.get("num_chunks", 10)
    target_updates = max_updates or cfg.get("optimizer_updates", 1900)

    # Shards
    train_loader = data_generator("finewebedu10B/finewebedu_train_*.bin", batch_size, seq_len=seq_len)
    val_inputs, val_targets = next(data_generator("finewebedu10B/finewebedu_val_*.bin", 2097152, seq_len=seq_len))

    # 2. Model with matched initialization
    p0(f"[E1 Training] Initializing matched model for arm={arm} with seed={seed}...")
    triplet = init_triplet_models(seed, device)
    model = triplet[arm]
    num_params = sum(p.numel() for p in model.parameters())
    p0(f"[E1 Training] Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    _emit_block(fh, "CONFIG_JSON", {**cfg, "num_params": num_params, "host": socket.gethostname(), "device": str(device)})

    if device.type == "cuda":
        model = torch.compile(model)

    # 3. Optimizers (Muon + AdamW mirror)
    opt1 = AdamW([
        dict(params=[model.embed.weight], lr=0.3),
        dict(params=[model.proj.weight], lr=1 / 320),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=(device.type == "cuda"))
    opt2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2], lr=0.02, weight_decay=0.01)
    optimizers = [opt1, opt2]

    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    cooldown = cfg.get("cooldown_frac", 0.7)

    def set_hparams(step):
        progress = step / target_updates
        eta = 1.0 if progress < 1 - cooldown else max(0.0, (1 - progress) / cooldown)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    # 4. Training loop
    curve = []
    t_start = time.time()
    t_step_start = time.time()
    grad_accum_steps = batch_size // (mbs * seq_len)

    p0(f"[E1 Training] Target updates: {target_updates}, accum_steps: {grad_accum_steps}, mbs: {mbs}")

    for step in range(target_updates + 1):
        # Validation
        if step % 125 == 0 or step == target_updates:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(len(val_inputs) // mbs):
                    mb_in = val_inputs[i * mbs:(i + 1) * mbs].to(device)
                    mb_tgt = val_targets[i * mbs:(i + 1) * mbs].to(device)
                    ls, _, _ = model(mb_in, mb_tgt)
                    vl += ls.item()
            val_loss_val = vl / 2097152
            model.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            elapsed = time.time() - t_start
            dt_step = (time.time() - t_step_start) / max(1, (125 if step > 0 else 1))
            t_step_start = time.time()
            tok_per_sec = (125 * batch_size) / max(1e-5, elapsed) if step > 0 else 0.0

            # MFU estimate
            flops_per_token = 6 * num_params + 12 * 16 * 1024 * seq_len
            flops_per_step = flops_per_token * batch_size
            peak_flops = 362e12  # L40S peak BF16
            mfu = (flops_per_step / (dt_step * peak_flops)) * 100.0 if dt_step > 0 and step > 0 else 0.0

            p0(f"step:{step}/{target_updates} val_loss:{val_loss_val:.5f} wall:{elapsed:.1f}s step_avg:{dt_step*1000:.1f}ms MFU:{mfu:.1f}%")
            curve.append({"step": step, "val_loss": val_loss_val, "wall_s": round(elapsed, 2), "mfu": round(mfu, 2)})

        if step == target_updates:
            break

        set_hparams(step)

        # Microbatch gradient accumulation
        inputs, targets = next(train_loader)
        for mb in range(grad_accum_steps):
            mb_in = inputs[mb * mbs:(mb + 1) * mbs].to(device)
            mb_tgt = targets[mb * mbs:(mb + 1) * mbs].to(device)
            loss, _, _ = model(mb_in, mb_tgt)
            (loss / grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()
        opt2.step()
        opt1.zero_grad()
        opt2.zero_grad()

    training_time = time.time() - t_start
    _emit_block(fh, "CURVE_JSON", curve)
    p0(f"[E1 Training] Training complete in {training_time:.1f}s ({training_time/3600:.2f}h)")

    # 5. Save checkpoint
    clean_model = getattr(model, "_orig_mod", model)
    ckpt_dir = CHECKPOINTS_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "weights.pt"
    cfg_out_path = ckpt_dir / "config.json"
    cfg_out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    torch.save({"model": {k: v.cpu() for k, v in clean_model.state_dict().items()}}, ckpt_path)
    p0(f"[E1 Training] Saved checkpoint to: {ckpt_dir}")

    # Free training model from GPU memory before evaluation
    del model, clean_model, opt1, opt2
    torch.cuda.empty_cache()

    # 6. Structured E1 evaluation on frozen fixtures
    p0(f"[E1 Training] Running full structured E1 evaluation on frozen fixtures with DirectScorer...")
    eval_record = evaluate_e1_checkpoints(ckpt_dir, arm, seed, device, fh=fh)
    _emit_block(fh, "E1_EVAL_JSON", eval_record)

    eval_out_path = EVAL_DIR / f"{run_id}_eval.json"
    eval_out_path.write_text(json.dumps(eval_record, indent=2) + "\n", encoding="utf-8")

    p0(f"[E1 Training] Evaluation complete:")
    p0(f"  Natural text exact retrieval (8K-64K mean): {eval_record['primary_natural_text_exact_mean_8k_to_64k']:.2f}%")
    p0(f"  BPB at 64K: {eval_record['bpb']['65536']['mean_bpb']:.4f}")
    p0(f"  BPB at 2K:  {eval_record['bpb']['2048']['mean_bpb']:.4f}")
    p0(f"[E1 Training] DONE -> {run_id}")
    fh.close()

    return eval_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--no-save-ckpt", action="store_true")
    args = parser.parse_args()

    eval_record = train_single_run(args.config, max_updates=args.max_updates, save_checkpoint=not args.no_save_ckpt)
    print("\nRun finished successfully:")
    print(f"  Primary exact retrieval (8K-64K): {eval_record['primary_natural_text_exact_mean_8k_to_64k']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
