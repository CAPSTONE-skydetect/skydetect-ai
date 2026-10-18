"""Package reviewed GT/A tracks and audit whether grouped C training is possible."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_label(value: str) -> str:
    label = value.strip().lower()
    if label.startswith("bird"):
        return "bird"
    if label.startswith("drone"):
        return "drone"
    return label


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups_by_label[record["label"]].add(record["source_group_id"])
        labels_by_group[record["source_group_id"]].add(record["label"])
    group_counts = {label: len(groups) for label, groups in sorted(groups_by_label.items())}
    blockers = []
    expected = {"bird", "drone"}
    if set(group_counts) != expected:
        blockers.append("both bird and drone labels are required")
    for label in sorted(expected):
        count = group_counts.get(label, 0)
        if count < 3:
            blockers.append(
                f"{label} has {count} independent source groups; at least 3 are required "
                "for non-empty grouped train/validation/test partitions"
            )
    duplicate_hashes = [
        digest for digest, count in Counter(record["gt_sha256"] for record in records).items()
        if count > 1
    ]
    return {
        "status": "ready" if not blockers else "blocked",
        "track_count": len(records),
        "label_counts": dict(Counter(record["label"] for record in records)),
        "source_group_count": len(labels_by_group),
        "source_groups_by_label": group_counts,
        "mixed_label_groups": {
            group: sorted(labels) for group, labels in sorted(labels_by_group.items())
            if len(labels) > 1
        },
        "duplicate_gt_hashes": duplicate_hashes,
        "blockers": blockers,
        "split_policy": (
            "All tracks, GT/A pairs, windows, and future augmentations from one source_group_id "
            "must remain in the same partition."
        ),
    }


def package(evaluation_root: Path, variant: str, output: Path) -> dict[str, Any]:
    summary = _json(evaluation_root / variant / "summary.json")
    labels = {
        row["sample_id"]: _class_label(row["target"]["label"])
        for row in summary["samples"]
    }
    output.mkdir(parents=True, exist_ok=True)
    tracks_dir = output / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id, label in sorted(labels.items()):
        report = _json(evaluation_root / variant / "per_video" / f"{sample_id}.json")
        metadata_path = Path(report["inputs"]["metadata_json"])
        metadata = _json(metadata_path)
        source_path = Path(metadata["source_video_path"]).expanduser().resolve()
        # Copies at different paths must remain in the same split.
        source_hash = _sha256(source_path)
        group_id = f"video:{source_hash}"
        source_dir = evaluation_root / variant / "b_inputs" / sample_id
        gt_source = source_dir / "gt_full_track_sequence.json"
        a_source = source_dir / "a_track_sequence.json"
        gt_target = tracks_dir / f"{sample_id}.gt.json"
        a_target = tracks_dir / f"{sample_id}.a.json"
        shutil.copyfile(gt_source, gt_target)
        shutil.copyfile(a_source, a_target)
        records.append({
            "sample_id": sample_id,
            "label": label,
            "domain": "real",
            "source_group_id": group_id,
            "source_video_path": str(source_path),
            "source_video_sha256": source_hash,
            "review_status": "needs_review",
            "partition": "development_unassigned",
            "gt_track": str(gt_target.relative_to(output)),
            "a_track": str(a_target.relative_to(output)),
            "gt_sha256": _sha256(gt_target),
            "a_sha256": _sha256(a_target),
            "gt_scope": "A attempted window, including GT frames A missed",
        })
    audit = audit_records(records)
    manifest = {
        "contract": "skydetect-common-c-track-dataset-1.0",
        "source_variant": variant,
        "records": records,
        "audit": audit,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Common C dataset preflight",
        "",
        f"- Status: **{audit['status']}**",
        f"- Tracks: {audit['track_count']}",
        f"- Independent source groups: {audit['source_group_count']}",
        f"- Labels: `{json.dumps(audit['label_counts'], ensure_ascii=False)}`",
        f"- Groups by label: `{json.dumps(audit['source_groups_by_label'], ensure_ascii=False)}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {reason}" for reason in audit["blockers"])
    lines.extend([
        "",
        "The six pilot objects are packaged for reproducible inference and preprocessing "
        "checks, but must not be presented as a grouped RF/MiniRocket training benchmark "
        "until the blockers above are resolved.",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--variant", default="D0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.evaluation_root, args.variant, args.output)
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
