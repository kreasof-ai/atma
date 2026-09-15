"""Read-only structural checks; never launches jobs or certifies scientific results."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARMS = {"nope_memory", "temperature_softmax_memory", "polar_memory"}
READINESS = {
    "operator_semantics", "temperature_control", "matched_initialization_and_recipe",
    "immutable_inputs", "evaluation_fixtures", "runner_integrity",
    "capacity_and_storage", "instrumentation_parity",
}


def validate(plan: dict, *, require_ready: bool = False) -> list[str]:
    """Check the v1 design and evidence-file integrity, not evidence content."""
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def local_file(relative: str, root: Path = REPO) -> Path | None:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"missing or nonlocal file: {relative}")
            return None
        return path

    check(plan["schema_version"] == 1, "unsupported schema_version")
    check(plan["manifest_kind"] == "planning_only_not_runner_configuration",
          "this file must remain a planning manifest, not a runner configuration")
    check(bool(re.fullmatch(r"[0-9a-f]{40}", plan["inspected_repository_revision"])),
          "inspected repository revision must be a full Git SHA")
    for document in plan["documents"]:
        local_file(document, HERE)
    for source in plan["source_files"]:
        local_file(source)
    dates = plan["deadlines"]
    for name in ("abstract", "paper"):
        aoe = datetime.fromisoformat(dates[f"{name}_aoe"])
        wib = datetime.fromisoformat(dates[f"{name}_wib"])
        check(aoe.utcoffset() is not None and wib.utcoffset() is not None and aoe == wib,
              f"{name} deadline timezone conversion does not agree")

    experiments = plan["experiments"]
    ids = [entry["id"] for entry in experiments]
    check(len(ids) == 7 and set(ids) == {f"E{i}" for i in range(7)},
          "experiments must contain E0-E6 exactly once")
    seen: set[str] = set()
    for entry in experiments:
        check(set(entry["depends_on"]) <= seen,
              f"{entry['id']}: unknown, cyclic, or out-of-order dependency")
        check(entry["required"] == (entry["id"] in {f"E{i}" for i in range(5)}),
              f"{entry['id']}: core/conditional status differs from v1 protocol")
        check(entry["new_training"] == (entry["id"] == "E1"),
              f"{entry['id']}: unplanned training")
        document, _, anchor = entry["protocol"].partition("#")
        path = local_file(document, HERE)
        if path and anchor:
            headings = re.findall(r"^#+\s+(.+)$", path.read_text(encoding="utf-8"), re.M)
            slugs = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-") for h in headings}
            check(anchor in slugs, f"{entry['id']}: protocol heading not found: {anchor}")
        seen.add(entry["id"])

    training = plan["training"]
    arms, seeds, runs = training["arms"], training["initialization_seeds"], training["runs"]
    check(len(arms) == 3 and set(arms) == ARMS, "E1 must have exactly the three specified arms")
    check(len(seeds) == 3 and len(set(seeds)) == 3
          and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds),
          "E1 requires three distinct valid initialization seeds")
    check(len(runs) == 9 and len({r["run_id"] for r in runs}) == 9,
          "E1 must have nine unique run IDs")
    check({(r["arm"], r["init_seed"]) for r in runs} == set(itertools.product(arms, seeds)),
          "E1 arm/initialization matrix is incomplete or contains an unplanned run")
    check(training["data_order_seed_is_effective"] is False,
          "v1 does not implement a shuffled data-order seed")
    check(training["selection"] == "final_update_checkpoint", "E1 checkpoint selection changed")
    check(training["train_context_tokens"] == 2048 and training["microbatch_sequences"] == 4,
          "E1 context/microbatch does not match the fixed recipe")
    batch = training["global_batch_tokens"]
    check(batch == 524288 and training["optimizer_updates"] == 1900,
          "E1 update/token budget differs from the v1 design")
    check(batch == training["train_context_tokens"] * training["microbatch_sequences"]
          * training["gradient_accumulation_steps"], "microbatch/accumulation token arithmetic mismatch")
    tokens = batch * training["optimizer_updates"]
    check(tokens == training["training_tokens_per_run"], "per-run consumed-token total mismatch")
    check(tokens * len(runs) == training["successful_training_tokens_total"],
          "scientific training-token total mismatch")
    check(training["temperature"]["alpha_raw_initial_value"] == -1.0,
          "temperature initialization differs from the prespecified control")
    local_file(training["recipe_reference"])

    evaluation = plan["evaluation"]
    lengths, short = evaluation["lengths"], evaluation["e1_lengths"]
    check(lengths == [2048, 8192, 16384, 32768, 65536, 262144], "E2/E4 length grid changed")
    check(short == lengths[:-1], "E1 must use the five prespecified lengths through 64K")
    check(evaluation["primary_retrieval_lengths"] == short[1:], "primary endpoint lengths changed")
    check(evaluation["primary_retrieval_suite"] == "real"
          and evaluation["primary_retrieval_metric"] == "teacher_forced_exact_five_token_accuracy",
          "primary retrieval endpoint changed")
    check(evaluation["e1_primary_contrast"] == ["polar_memory", "temperature_softmax_memory"],
          "E1 primary contrast changed")
    check(evaluation["tasks"] == ["passkey", "niah"]
          and evaluation["suites"] == ["synthetic", "real"]
          and evaluation["depths"] == [0.1, 0.5, 0.9], "retrieval factorial axes changed")
    check(evaluation["real_documents"] == evaluation["synthetic_cases"] == 30,
          "retrieval must contain 30 paired case IDs per suite")
    check(evaluation["target_tokens"] == 5, "retrieval target length changed")
    per_length = len(evaluation["tasks"]) * len(evaluation["depths"]) * (
        evaluation["real_documents"] + evaluation["synthetic_cases"])
    check(per_length * len(short) == evaluation["retrieval_prompts_per_e1_model"],
          "E1 retrieval prompt count mismatch")
    check(per_length * len(lengths) == evaluation["retrieval_prompts_per_e2_e4_condition"],
          "E2/E4 retrieval prompt count mismatch")
    originals = {"primary_nope", "primary_polar"}
    check(set(evaluation["e2_checkpoints"]) == originals | {
        "foveal_nope_lm_output_kl", "foveal_polar_lm_output_kl"}
        and len(evaluation["e2_checkpoints"]) == 4, "E2 checkpoint panel changed")
    check(set(evaluation["e4_checkpoints"]) == originals | {
        "repl_seed1_nope", "repl_seed1_polar", "repl_seed2_nope", "repl_seed2_polar"}
        and len(evaluation["e4_checkpoints"]) == 6, "E4 checkpoint panel changed")
    check(evaluation["e2_cap_conditions"] == ["uncapped", "inherited_head_hl256"]
          and evaluation["e4_condition"] == "uncapped", "primary cap conditions changed")
    e2 = set(itertools.product(evaluation["e2_checkpoints"], evaluation["e2_cap_conditions"]))
    e4 = {(checkpoint, evaluation["e4_condition"]) for checkpoint in evaluation["e4_checkpoints"]}
    check(len(e2) == evaluation["e2_condition_count"] == 8, "E2 needs eight distinct conditions")
    check(len(e2 & e4) == evaluation["e2_e4_shared_conditions"] == 2, "shared-condition count mismatch")
    total = len(runs) * per_length * len(short) + len(e2 | e4) * per_length * len(lengths)
    check(total == evaluation["core_distinct_retrieval_prompts"], "deduplicated core prompt count mismatch")
    check(evaluation["bpb_documents"] == {"finepdfs": 8, "pg19": 5, "proof_pile": 5},
          "BPB corpus/document coverage changed")
    check(evaluation["bpb_target_tokens_per_document"] == 256
          and evaluation["bpb_target_start_token"] == 262144, "BPB fixed target changed")
    check(sum(evaluation["bpb_documents"].values()) * evaluation["bpb_target_tokens_per_document"]
          == evaluation["bpb_target_tokens_per_model_length"] == 4608, "BPB target-token count mismatch")
    gamma = evaluation["gamma"]
    check(gamma["target_count"] == 1 and gamma["half_life_ceiling_tokens"] == 256
          and gamma["includes_fixed_config_bias"] and gamma["zero_based_indices"],
          "gamma intervention violates the one-head final-logit protocol")
    check(plan["conditional"]["E6"]["min_timed_repetitions"] >= 10,
          "E6 requires at least ten timed repetitions")

    readiness = plan["readiness"]
    check(len(readiness) == len(READINESS) and {r["id"] for r in readiness} == READINESS,
          "readiness registry is incomplete or duplicated")
    for item in readiness:
        check(item["status"] in {"unresolved", "satisfied"}, f"{item['id']}: invalid readiness status")
        if item["status"] == "satisfied":
            check(bool(item["evidence"]), f"{item['id']}: satisfied without evidence")
        if require_ready and item["status"] != "satisfied":
            errors.append(f"unresolved readiness: {item['id']}")
        for evidence in item["evidence"]:
            path = local_file(evidence["path"])
            digest = evidence["sha256"]
            check(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"{item['id']}: invalid evidence SHA-256")
            if path:
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                check(actual == digest, f"{item['id']}: evidence file hash mismatch")
    for name in ("allocated_gpu_hours", "training_retry_token_reserve", "checkpoint_storage_bytes"):
        value = plan["resources"][name]
        if value is not None:
            check(type(value) in {int, float} and math.isfinite(value)
                  and (value >= 0 if name == "training_retry_token_reserve" else value > 0),
                  f"invalid resource value: {name}")
        elif require_ready:
            errors.append(f"unresolved resource allocation: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        errors = validate(plan, require_ready=args.require_ready)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"ERROR: malformed or unreadable plan: {exc}")
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Structurally valid: 9 training runs; 8,965,324,800 training tokens; "
          "42,120 distinct core retrieval prompts.")
    unresolved = [r["id"] for r in plan["readiness"] if r["status"] != "satisfied"]
    print(f"Unresolved readiness items: {len(unresolved)}. "
          "This check does not verify GPU behavior or scientific results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
