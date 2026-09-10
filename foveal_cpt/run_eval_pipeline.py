"""Download and benchmark the twelve ATMA Foveal CPT checkpoints.

Evaluates all 12 checkpoints across:
  1. Downstream task quality (LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-E, ARC-C, OpenBookQA, BoolQ)
  2. Training-aligned retrieval (passkey, synthetic NIAH, real-text NIAH)
  3. BABILong long-context reasoning (qa1-qa10 over length grid)
  4. BPB long-document likelihood (PG-19, Proof-Pile, FinePDFs bits-per-byte)
  5. System inference performance (prefill tok/s, decode tok/s, peak memory, max context)

Each job runs in an isolated subprocess to prevent GPU memory or compiler leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "benchmarks" / "logs" / "foveal_cpt"
HF_REPO = "ChavyvAkvar/atma-foveal-cpt-all"

CORES = ("polar", "nope", "rope")
VARIANTS = ("local", "lm_output", "kl", "lm_output_kl")


@dataclass(frozen=True)
class FovealModelSpec:
    key: str
    core: str
    variant: str
    subfolder: str
    repo_id: str = HF_REPO
    step: int = 1908


def build_model_specs() -> dict[str, FovealModelSpec]:
    specs = {}
    for core in CORES:
        for variant in VARIANTS:
            key = f"{core}_{variant}"
            subfolder = f"{core}/{variant}/cpt"
            specs[key] = FovealModelSpec(
                key=key,
                core=core,
                variant=variant,
                subfolder=subfolder,
            )
    return specs


MODEL_SPECS = build_model_specs()


@dataclass(frozen=True)
class BenchmarkJob:
    stage: str
    benchmark: str
    suite: str
    model: str
    command_args: tuple[str, ...]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


RESULT_MARKERS = (
    "===RETRIEVAL_RESULTS_JSON===",
    "===BASE_RESULTS_JSON===",
    "===LONGDOC_RESULTS_JSON===",
    "===SERVING_RESULTS_JSON===",
    "===BABILONG_RESULTS_JSON===",
)


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    for marker in RESULT_MARKERS:
        start = text.rfind(marker)
        if start >= 0 and text.find("===END===", start) >= 0:
            return True
    return False


def _extract_result(path: Path) -> dict | None:
    if not _is_complete(path):
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    positions = [
        (text.rfind(marker), marker) for marker in RESULT_MARKERS if marker in text
    ]
    if not positions:
        return None
    _, marker = max(positions)
    block = text.rsplit(marker, 1)[1].split("===END===", 1)[0].strip()
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def _download_foveal_checkpoint(
    spec: FovealModelSpec,
    cache_dir: str | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    print(
        f"[foveal_pipeline] resolving {spec.key} ({spec.subfolder}) from {spec.repo_id}",
        flush=True,
    )
    files = {}
    for filename in ("config.json", "latest.json", "cpt-step-001908.pt"):
        remote_path = f"{spec.subfolder}/{filename}"
        p = hf_hub_download(
            repo_id=spec.repo_id,
            filename=remote_path,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        files[filename] = p

    ckpt_dir = os.path.dirname(files["config.json"])
    weights_path = files["cpt-step-001908.pt"]

    return {
        "key": spec.key,
        "core": spec.core,
        "variant": spec.variant,
        "subfolder": spec.subfolder,
        "repo_id": spec.repo_id,
        "step": spec.step,
        "checkpoint_dir": ckpt_dir,
        "weights_path": weights_path,
        "weights_bytes": os.path.getsize(weights_path),
    }


def _resolve_stages(stage_args: list[str] | str) -> set[str]:
    raw = set(stage_args) if isinstance(stage_args, list) else {stage_args}
    if "all" in raw:
        return {"base", "retrieval", "longdoc", "babilong_pipeline", "serving"}
    return raw


def _job_fingerprint(job: BenchmarkJob) -> str:
    payload = json.dumps(asdict(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _job_output(log_dir: Path, job: BenchmarkJob) -> Path:
    return log_dir / (
        f"{job.stage}_{job.benchmark}_{job.suite}_{job.model}_{_job_fingerprint(job)}.log"
    )


def _build_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    stages = _resolve_stages(args.stage)
    jobs = []

    # 1. Smoke Stage (quick retrieval gate)
    if "smoke" in stages:
        for model in args.models:
            jobs.append(
                BenchmarkJob(
                    stage="smoke",
                    benchmark="retrieval",
                    suite="synthetic",
                    model=model,
                    command_args=(
                        "--tasks", "passkey", "niah",
                        "--lengths", *args.smoke_lengths,
                        "--depths", "0.5",
                        "--samples", str(args.smoke_samples),
                        "--seed", str(args.seed),
                        "--retrieval_value_tokens", str(args.retrieval_value_tokens),
                    ),
                )
            )

    # 2. Retrieval Stage
    if "retrieval" in stages:
        for model in args.models:
            for suite in args.suites:
                extra = []
                if suite == "real":
                    extra = ["--haystack", args.haystack]
                jobs.append(
                    BenchmarkJob(
                        stage="retrieval",
                        benchmark="retrieval",
                        suite=suite,
                        model=model,
                        command_args=(
                            "--tasks", *args.tasks,
                            "--lengths", *args.lengths,
                            "--depths", *(str(d) for d in args.depths),
                            "--samples", str(args.samples),
                            "--seed", str(args.seed),
                            "--retrieval_value_tokens", str(args.retrieval_value_tokens),
                            *extra,
                        ),
                    )
                )

    # 3. Downstream LM Tasks (Base Benchmark)
    if "base" in stages:
        for model in args.models:
            cmd = [
                "--tasks", *args.base_tasks,
                "--batch_size", str(args.base_batch_size),
                "--scoring_max_length", str(args.base_max_length),
            ]
            if args.base_limit:
                cmd.extend(["--limit", str(args.base_limit)])
            jobs.append(
                BenchmarkJob(
                    stage="base",
                    benchmark="base",
                    suite="zero_shot",
                    model=model,
                    command_args=tuple(cmd),
                )
            )

    # 4. Longdoc (BPB)
    if "longdoc" in stages:
        for model in args.models:
            jobs.append(
                BenchmarkJob(
                    stage="longdoc",
                    benchmark="longdoc",
                    suite="fixed_target",
                    model=model,
                    command_args=(
                        "--datasets", *args.longdoc_datasets,
                        "--lengths", *args.longdoc_lengths,
                        "--target_tokens", str(args.target_tokens),
                        "--num_docs", str(args.num_docs),
                        "--max_scan", str(args.max_scan),
                    ),
                )
            )

    # 5. BABILong Adaptation Fine-Tuning Stage
    if "babilong_finetune" in stages or "babilong_pipeline" in stages:
        for model in args.models:
            jobs.append(
                BenchmarkJob(
                    stage="babilong_finetune",
                    benchmark="finetune_babilong",
                    suite="2k_ft",
                    model=model,
                    command_args=(
                        "--tasks", *args.finetune_tasks,
                        "--train_lengths", *args.finetune_train_lengths,
                        "--seq_len", str(args.finetune_seq_len),
                        "--train_start", str(args.finetune_train_start),
                        "--train_end", str(args.finetune_train_end),
                        "--val_start", str(args.finetune_val_start),
                        "--val_end", str(args.finetune_val_end),
                        "--epochs", str(args.finetune_epochs),
                        "--micro_batch_size", str(args.finetune_micro_batch_size),
                        "--grad_accum_steps", str(args.finetune_grad_accum_steps),
                        "--lr", str(args.finetune_lr),
                        "--seed", str(args.seed),
                    ),
                )
            )

    # 6. BABILong Evaluation Stage
    if "babilong" in stages or "babilong_pipeline" in stages:
        for model in args.models:
            cmd = [
                "--tasks", *args.babilong_tasks,
                "--lengths", *args.babilong_lengths,
                "--row_start", str(args.row_start),
                "--row_end", str(args.row_end),
                "--samples", str(args.babilong_samples),
                "--babilong_backend", args.babilong_backend,
                "--max_tokens", str(args.max_tokens),
            ]
            suite_name = (
                f"{args.babilong_backend}_ft"
                if (args.use_finetuned or "babilong_pipeline" in stages)
                else args.babilong_backend
            )
            jobs.append(
                BenchmarkJob(
                    stage="babilong",
                    benchmark="babilong",
                    suite=suite_name,
                    model=model,
                    command_args=tuple(cmd),
                )
            )

    # 7. Serving Performance (Prefill / Decode tok/s)
    if "serving" in stages:
        for model in args.models:
            cmd = [
                "--lengths", *args.serving_lengths,
                "--decode_tokens", str(args.decode_tokens),
                "--serving_samples", str(args.serving_samples),
                "--max_num_seqs", str(args.max_num_seqs),
                "--serving_backend", args.serving_backend,
            ]
            jobs.append(
                BenchmarkJob(
                    stage="serving",
                    benchmark="serving",
                    suite="generation",
                    model=model,
                    command_args=tuple(cmd),
                )
            )

    return jobs


def _run_with_console_log(cmd: list[str], console_path: Path, env: dict) -> int:
    console_path.parent.mkdir(parents=True, exist_ok=True)
    print("[pipeline] " + " ".join(cmd), flush=True)
    with console_path.open("w", encoding="utf-8", buffering=1) as console:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                console.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def main() -> int:
    from benchmarks.base_tasks import BASE_TASK_SPECS
    from benchmarks.longdoc import LONGDOC_SPECS

    parser = argparse.ArgumentParser(
        description="Foveal CPT comprehensive evaluation runner across all 12 checkpoints."
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=(
            "smoke",
            "retrieval",
            "base",
            "longdoc",
            "babilong",
            "babilong_finetune",
            "babilong_pipeline",
            "serving",
            "all",
        ),
        default=["smoke"],
        help="smoke, retrieval, base, longdoc, babilong, babilong_finetune, babilong_pipeline, serving, all",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        choices=CORES,
        default=list(CORES),
        help="filter by attention core (polar, nope, rope)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="filter by adaptation variant (local, lm_output, kl, lm_output_kl)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="explicit model keys (e.g. polar_local, nope_lm_output_kl)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Retrieval args
    parser.add_argument("--suites", nargs="+", choices=("synthetic", "real"), default=("synthetic", "real"))
    parser.add_argument("--tasks", nargs="+", choices=("passkey", "niah"), default=("passkey", "niah"))
    parser.add_argument("--smoke_lengths", nargs="+", default=("2k", "8k"))
    parser.add_argument("--smoke_samples", type=int, default=5)
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=("2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"),
    )
    parser.add_argument("--depths", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--retrieval_value_tokens", type=int, default=5)
    parser.add_argument("--haystack", default="codelion/finepdfs-1B")

    # Base task args
    parser.add_argument("--base_tasks", nargs="+", choices=tuple(BASE_TASK_SPECS), default=tuple(BASE_TASK_SPECS))
    parser.add_argument("--base_batch_size", type=int, default=8)
    parser.add_argument("--base_max_length", type=int, default=2048)
    parser.add_argument("--base_limit", type=int, default=None)

    # Longdoc args
    parser.add_argument("--longdoc_datasets", nargs="+", choices=tuple(LONGDOC_SPECS), default=tuple(LONGDOC_SPECS))
    parser.add_argument(
        "--longdoc_lengths",
        nargs="+",
        default=("2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"),
    )
    parser.add_argument("--target_tokens", type=int, default=256)
    parser.add_argument("--num_docs", type=int, default=4)
    parser.add_argument("--max_scan", type=int, default=100000)

    # BABILong args
    parser.add_argument(
        "--babilong_tasks",
        nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    parser.add_argument(
        "--babilong_lengths",
        nargs="+",
        default=["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"],
    )
    parser.add_argument("--babilong_samples", type=int, default=10)
    parser.add_argument("--babilong_backend", choices=("direct", "paged"), default="direct")
    parser.add_argument("--row_start", type=int, default=90)
    parser.add_argument("--row_end", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument(
        "--use_finetuned",
        action="store_true",
        help="evaluate the fine-tuned BABILong checkpoint instead of base CPT",
    )
    parser.add_argument(
        "--finetune_output_root",
        type=Path,
        default=ROOT / "checkpoints" / "foveal_babilong_2k_ft",
    )
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--finetune_micro_batch_size", type=int, default=1)
    parser.add_argument("--finetune_grad_accum_steps", type=int, default=8)
    parser.add_argument("--finetune_seq_len", type=int, default=2048)
    parser.add_argument(
        "--finetune_tasks",
        nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    parser.add_argument("--finetune_train_lengths", nargs="+", default=["0k", "1k", "2k"])
    parser.add_argument("--finetune_train_start", type=int, default=0)
    parser.add_argument("--finetune_train_end", type=int, default=80)
    parser.add_argument("--finetune_val_start", type=int, default=80)
    parser.add_argument("--finetune_val_end", type=int, default=90)

    # Serving args
    parser.add_argument(
        "--serving_lengths",
        nargs="+",
        default=("2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"),
    )
    parser.add_argument("--decode_tokens", type=int, default=32)
    parser.add_argument("--serving_samples", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--serving_backend", choices=("paged", "direct"), default="paged")

    args = parser.parse_args()

    # Determine model list
    if args.models is None:
        selected_models = [
            f"{c}_{v}" for c in args.cores for v in args.variants if f"{c}_{v}" in MODEL_SPECS
        ]
    else:
        selected_models = [m for m in args.models if m in MODEL_SPECS]
    args.models = selected_models

    print(
        f"[foveal_pipeline] stage={args.stage} models={len(args.models)} ({', '.join(args.models)})",
        flush=True,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.log_dir / "checkpoint_manifest.json"
    manifest = _read_json(manifest_path, {})

    # Download / resolve checkpoints
    resolved_checkpoints: dict[str, dict[str, Any]] = {}
    for key in args.models:
        spec = MODEL_SPECS[key]
        record = _download_foveal_checkpoint(spec, cache_dir=args.cache_dir, offline=args.offline)
        resolved_checkpoints[key] = record
        manifest[key] = record
    _write_json(manifest_path, manifest)

    # Build jobs
    jobs = _build_jobs(args)
    print(f"[foveal_pipeline] total jobs to run: {len(jobs)}", flush=True)

    summary_path = args.log_dir / "pipeline_summary.json"
    summary = _read_json(summary_path, {"jobs": {}, "updated_at": None})

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        output_path = _job_output(args.log_dir, job)
        console_path = output_path.with_suffix(".console.log")
        fingerprint = _job_fingerprint(job)

        if not args.rerun and _is_complete(output_path):
            print(
                f"[{idx}/{len(jobs)}] skip complete job: {job.stage} {job.benchmark} "
                f"{job.model} ({fingerprint})"
            )
            continue

        ckpt_info = resolved_checkpoints[job.model]
        model_path = ckpt_info["checkpoint_dir"]

        if job.benchmark == "finetune_babilong":
            out_ckpt = args.finetune_output_root / job.model
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            if not args.rerun and (out_ckpt / "weights.pt").is_file():
                print(
                    f"[{idx}/{len(jobs)}] skip complete fine-tune: {job.model} -> {out_ckpt}"
                )
                continue
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.finetune_babilong",
                "--model",
                str(model_path),
                "--output_dir",
                str(out_ckpt),
                *job.command_args,
            ]
            if args.rerun:
                cmd.append("--overwrite")
        else:
            stages = _resolve_stages(args.stage)
            eval_model_path = (
                str(args.finetune_output_root / job.model)
                if (
                    job.benchmark == "babilong"
                    and (args.use_finetuned or "babilong_pipeline" in stages)
                )
                else str(model_path)
            )
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.run",
                "--benchmark",
                job.benchmark,
                "--model",
                eval_model_path,
                *job.command_args,
                "--out",
                str(output_path),
                "--strict",
            ]

        t0 = time.time()
        print(
            f"\n[{idx}/{len(jobs)}] running: {job.stage} {job.benchmark} {job.model} ({fingerprint})",
            flush=True,
        )
        exit_code = _run_with_console_log(cmd, console_path, env)
        elapsed = round(time.time() - t0, 1)

        if job.benchmark == "finetune_babilong":
            out_ckpt = args.finetune_output_root / job.model
            success = exit_code == 0 and (out_ckpt / "weights.pt").is_file()
        else:
            res = _extract_result(output_path)
            success = exit_code == 0 and res is not None

        summary["jobs"][fingerprint] = {
            "stage": job.stage,
            "benchmark": job.benchmark,
            "suite": job.suite,
            "model": job.model,
            "success": success,
            "exit_code": exit_code,
            "elapsed_s": elapsed,
            "log_path": str(output_path),
            "console_path": str(console_path),
        }
        summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(summary_path, summary)

        if not success:
            failures += 1
            print(f"[foveal_pipeline] job FAILED (exit {exit_code})", flush=True)
            if args.fail_fast:
                print("[foveal_pipeline] aborting on failure (--fail_fast)", flush=True)
                return 1
        else:
            print(f"[foveal_pipeline] job SUCCESS in {elapsed}s", flush=True)

    # Run aggregation
    try:
        from benchmarks.aggregate import aggregate, write_outputs

        print("[foveal_pipeline] generating aggregate benchmark matrix...", flush=True)
        res_matrix = aggregate(args.log_dir)
        write_outputs(
            res_matrix,
            args.log_dir / "benchmark_matrix.json",
            args.log_dir / "benchmark_matrix.csv",
        )
        print(
            f"[foveal_pipeline] aggregated results ({len(res_matrix['rows'])} rows) written to {args.log_dir / 'benchmark_matrix.csv'}"
        )
    except Exception as exc:
        print(f"[foveal_pipeline] aggregation warning: {exc}")

    print(f"\n[foveal_pipeline] complete! Total jobs: {len(jobs)}, Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
