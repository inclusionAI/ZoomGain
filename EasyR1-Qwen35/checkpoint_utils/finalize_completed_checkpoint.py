#!/usr/bin/env python3
"""Finalize a completed training checkpoint while keeping its deployable HF model.

This utility is intentionally fail-closed: it validates the complete
``actor/huggingface`` checkpoint before deleting any resume-only state.  It is
meant to run only after the training process exits successfully.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


TRACKER_NAME = "latest_checkpointed_iteration.txt"
FINALIZED_TRACKER_NAME = "latest_checkpointed_iteration.finalized.txt"
FINALIZED_MARKER_NAME = "finalized_checkpoint.json"
ACTOR_RESUME_PATTERNS = (
    "model_world_size_*_rank_*.pt",
    "optim_world_size_*_rank_*.pt",
    "extra_state_world_size_*_rank_*.pt",
)
DATALOADER_RESUME_PATTERNS = ("data.pt", "data_*.pt")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def find_latest_checkpoint(checkpoint_root: str | Path) -> tuple[Path, Path, int]:
    root = Path(checkpoint_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root does not exist: {root}")

    tracker = root / TRACKER_NAME
    if not tracker.is_file():
        raise FileNotFoundError(f"checkpoint tracker does not exist: {tracker}")
    raw_step = tracker.read_text(encoding="utf-8").strip()
    try:
        step = int(raw_step)
    except ValueError as exc:
        raise ValueError(f"invalid checkpoint step {raw_step!r} in {tracker}") from exc
    if step < 0:
        raise ValueError(f"checkpoint step must be non-negative, got {step}")

    step_dir = (root / f"global_step_{step}").resolve()
    if step_dir.parent != root:
        raise ValueError(f"checkpoint step escapes checkpoint root: {step_dir}")
    actor_dir = (step_dir / "actor").resolve()
    if actor_dir.parent != step_dir or not actor_dir.is_dir():
        raise FileNotFoundError(f"actor checkpoint does not exist: {actor_dir}")
    return step_dir, tracker, step


def _safetensor_manifest(path: Path) -> dict:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            dtype_tensors: Counter[str] = Counter()
            dtype_elements: Counter[str] = Counter()
            for key in keys:
                tensor_slice = handle.get_slice(key)
                dtype = str(tensor_slice.get_dtype())
                elements = math.prod(tensor_slice.get_shape())
                dtype_tensors[dtype] += 1
                dtype_elements[dtype] += elements
    except Exception as exc:
        raise ValueError(f"cannot validate safetensors file: {path}") from exc
    if not keys:
        raise ValueError(f"safetensors file contains no tensors: {path}")
    return {
        "keys": keys,
        "dtype_tensors": dtype_tensors,
        "dtype_elements": dtype_elements,
    }


def validate_huggingface_checkpoint(hf_dir: str | Path) -> dict:
    """Validate config, tokenizer, safetensor files, and optional shard index."""

    model_dir = Path(hf_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Hugging Face checkpoint does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hugging Face checkpoint has no config.json: {model_dir}")
    config = _read_json(config_path)
    if not config.get("model_type") and not config.get("architectures"):
        raise ValueError(f"config has neither model_type nor architectures: {config_path}")

    tokenizer_candidates = (
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
        model_dir / "vocab.json",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_candidates):
        raise FileNotFoundError(f"Hugging Face checkpoint has no tokenizer vocabulary: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    manifests: dict[str, dict] = {}
    indexed = False

    if index_path.is_file():
        index = _read_json(index_path)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"invalid or empty weight_map in {index_path}")
        weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
        for shard_name in sorted(set(weight_map.values())):
            shard = Path(shard_name)
            if shard.is_absolute() or shard.name != shard_name:
                raise ValueError(f"unsafe shard name {shard_name!r} in {index_path}")
            shard_path = model_dir / shard_name
            if not shard_path.is_file() or shard_path.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty Hugging Face shard: {shard_path}")
            manifests[shard_name] = _safetensor_manifest(shard_path)

        indexed_keys = set(weight_map)
        actual_keys = set().union(*(manifest["keys"] for manifest in manifests.values()))
        for tensor_name, shard_name in weight_map.items():
            if tensor_name not in manifests[shard_name]["keys"]:
                raise ValueError(
                    f"index maps {tensor_name!r} to {shard_name!r}, but the tensor is absent"
                )
        if indexed_keys != actual_keys:
            raise ValueError(
                "safetensors index/header mismatch: "
                f"missing={sorted(indexed_keys - actual_keys)[:5]}, "
                f"extra={sorted(actual_keys - indexed_keys)[:5]}"
            )
        indexed = True
    elif single_path.is_file() and single_path.stat().st_size > 0:
        manifests[single_path.name] = _safetensor_manifest(single_path)
    else:
        unindexed_shards = sorted(model_dir.glob("model-*.safetensors"))
        if unindexed_shards:
            raise FileNotFoundError(
                f"found {len(unindexed_shards)} model shards but no {index_path.name}"
            )
        raise FileNotFoundError(f"Hugging Face checkpoint has no model weights: {model_dir}")

    dtype_tensors: Counter[str] = Counter()
    dtype_elements: Counter[str] = Counter()
    for manifest in manifests.values():
        dtype_tensors.update(manifest["dtype_tensors"])
        dtype_elements.update(manifest["dtype_elements"])

    shard_paths = [model_dir / name for name in sorted(manifests)]
    return {
        "hf_dir": str(model_dir),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "shard_count": len(shard_paths),
        "tensor_count": sum(len(manifest["keys"]) for manifest in manifests.values()),
        "weight_bytes": sum(path.stat().st_size for path in shard_paths),
        "dtype_tensors": dict(sorted(dtype_tensors.items())),
        "dtype_elements": dict(sorted(dtype_elements.items())),
        "indexed": indexed,
    }


def collect_resume_artifacts(step_dir: str | Path) -> list[Path]:
    step = Path(step_dir).resolve()
    actor = (step / "actor").resolve()
    if actor.parent != step:
        raise ValueError(f"actor directory escapes checkpoint step: {actor}")

    targets: set[Path] = set()
    for pattern in ACTOR_RESUME_PATTERNS:
        for candidate in actor.glob(pattern):
            target = candidate.resolve()
            if target.parent != actor or not target.is_file():
                raise ValueError(f"unsafe resume artifact: {candidate}")
            targets.add(target)
    for pattern in DATALOADER_RESUME_PATTERNS:
        for candidate in step.glob(pattern):
            target = candidate.resolve()
            if target.parent != step or not target.is_file():
                raise ValueError(f"unsafe dataloader artifact: {candidate}")
            targets.add(target)
    return sorted(targets)


def compact_actor_checkpoint(actor_dir: str | Path) -> dict:
    """Remove only resume files from an old actor, preserving its validated HF tree.

    This does not finalize the run or modify its latest-checkpoint tracker.
    Call only after a newer checkpoint has been saved successfully.
    """
    actor = Path(actor_dir).absolute()
    if actor.is_symlink() or actor.name != "actor" or not actor.parent.name.startswith("global_step_"):
        raise ValueError(f"unexpected actor checkpoint path: {actor}")
    hf_dir = actor / "huggingface"
    if hf_dir.is_symlink():
        raise ValueError(f"HF export must not be a symlink: {hf_dir}")
    validation = validate_huggingface_checkpoint(hf_dir)
    # Resolve and validate the entire allowlist before deleting anything.
    targets = collect_resume_artifacts(actor.parent)
    removed = []
    reclaimed_bytes = 0
    for target in targets:
        size = target.stat().st_size
        target.unlink()
        reclaimed_bytes += size
        removed.append(str(target.relative_to(actor.parent)))
    validate_huggingface_checkpoint(hf_dir)
    report = {
        "huggingface": validation,
        "removed": removed,
        "reclaimed_bytes": reclaimed_bytes,
        "resumable": False,
    }
    _write_json_atomic(actor.parent / "resume_state_compacted.json", report)
    return report


def _resume_group_counts(step_dir: Path) -> dict[str, int]:
    actor = step_dir / "actor"
    return {
        "model": len(list(actor.glob(ACTOR_RESUME_PATTERNS[0]))),
        "optimizer": len(list(actor.glob(ACTOR_RESUME_PATTERNS[1]))),
        "extra": len(list(actor.glob(ACTOR_RESUME_PATTERNS[2]))),
    }


def finalize_checkpoint(checkpoint_root: str | Path, *, cleanup: bool) -> dict:
    root = Path(checkpoint_root).resolve()
    marker = root / FINALIZED_MARKER_NAME
    if marker.exists():
        raise FileExistsError(f"checkpoint is already finalized: {marker}")

    step_dir, tracker, step = find_latest_checkpoint(root)
    actor = (step_dir / "actor").resolve()
    hf_dir = (actor / "huggingface").resolve()
    if hf_dir.parent != actor:
        raise ValueError(f"Hugging Face directory escapes actor checkpoint: {hf_dir}")
    validation = validate_huggingface_checkpoint(hf_dir)
    targets = collect_resume_artifacts(step_dir)
    candidate_bytes = sum(target.stat().st_size for target in targets)
    counts_before = _resume_group_counts(step_dir)
    resumable_before = all(counts_before.values()) and len(set(counts_before.values())) == 1

    cleanup_report = {
        "removed": [],
        "removed_count": 0,
        "reclaimed_bytes": 0,
    }
    finalized_tracker = None
    if cleanup:
        finalized_tracker_path = root / FINALIZED_TRACKER_NAME
        if finalized_tracker_path.exists():
            raise FileExistsError(f"finalized tracker already exists: {finalized_tracker_path}")

        removed = []
        reclaimed_bytes = 0
        for target in targets:
            size = target.stat().st_size
            target.unlink()
            removed.append(str(target.relative_to(step_dir)))
            reclaimed_bytes += size

        # Resume artifacts are independent of the HF tree. Validate again after
        # deletion so a successful marker always means the deployable model is intact.
        validation = validate_huggingface_checkpoint(hf_dir)
        tracker.replace(finalized_tracker_path)
        finalized_tracker = str(finalized_tracker_path)
        cleanup_report = {
            "removed": removed,
            "removed_count": len(removed),
            "reclaimed_bytes": reclaimed_bytes,
        }

    checked_at = datetime.now(timezone.utc).isoformat()
    result = {
        "checked_at_utc": checked_at,
        "finalized_at_utc": checked_at if cleanup else None,
        "checkpoint_root": str(root),
        "global_step": step,
        "step_dir": str(step_dir),
        "huggingface": validation,
        "cleanup_resume_state": cleanup,
        "resume_group_counts_before": counts_before,
        "resumable_before": resumable_before,
        "resumable_after": resumable_before and not cleanup,
        "resume_cleanup_candidates": {
            "count": len(targets),
            "bytes": candidate_bytes,
        },
        "resume_state": cleanup_report,
        "finalized_tracker": finalized_tracker,
    }
    if cleanup:
        _write_json_atomic(marker, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--cleanup-resume-state",
        action="store_true",
        help="After HF validation, remove only model/optimizer/extra/dataloader resume files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = finalize_checkpoint(
        args.checkpoint_root,
        cleanup=args.cleanup_resume_state,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
