"""Export CVAT rectangle GT as strict B TrackSequences and separate class labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from ai_server.services.feature_core import build_feature_vector
from ai_server.utils.quality import build_track_quality
from research.features import FEATURE_COLUMNS
from research.io import write_json
from research.tracking_evaluation import CvatTrack, TrackingEvaluationError, evaluate_manifest
from research.tracking_evaluation import load_cvat_track, load_video_geometry


def cvat_to_track_sequence(
    gt: CvatTrack, *, fps: float, processed_width: int, processed_height: int,
    source_video_id: str,
) -> TrackSequence:
    """Use all visible XML boxes; do not invent interpolation or frame offsets."""
    if fps <= 0:
        raise TrackingEvaluationError("GT conversion requires an explicit positive FPS")
    if abs(processed_width / processed_height - gt.source_width / gt.source_height) > .002:
        raise TrackingEvaluationError("full GT export requires an aspect-preserving resize")
    points = [TrackPoint(
        frame_index=p.frame_index, timestamp_ms=int(round(p.frame_index / fps * 1000)),
        cx=p.cx / gt.source_width, cy=p.cy / gt.source_height,
        w=p.width / gt.source_width, h=p.height / gt.source_height, conf=1.0,
    ) for p in gt.observations]
    if any(not 0 <= p.frame_index < (gt.task_frame_count or float("inf")) for p in points):
        raise TrackingEvaluationError("GT frame lies outside the CVAT task timeline")
    return TrackSequence(
        track_id=gt.track_id, source_video_id=source_video_id,
        processed_width=processed_width, processed_height=processed_height,
        stabilization=StabilizationInfo(applied=False, method="none"),
        history=points, quality=build_track_quality(points),
    )


def _class_label(xml_label: str) -> str | None:
    if xml_label == "drone":
        return "drone"
    return "bird" if re.fullmatch(r"bird(?:_\d+)?", xml_label) else None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_row(sample: dict[str, Any], track: TrackSequence, scope: str) -> tuple[dict, dict]:
    result = build_feature_vector(track)
    vector = result.feature_vector.model_dump(mode="json")
    row = {
        "sample_id": sample["sample_id"], "source_group_id": sample["source_group_id"],
        "label": sample["label"], "label_candidate_from_xml": sample["label_candidate_from_xml"],
        "label_status": sample["label_status"], "scope": scope,
        "coordinate_space": "frame" if scope == "full_gt_raw" else "camera_motion_compensated",
        "feature_status": vector["feature_status"], "feature_version": result.feature_version,
        "feature_config_id": result.feature_config_id, "coordinate_policy": result.coordinate_policy,
        "rejection_reason": ";".join(result.reasons),
        **{key: (vector["features"] or {}).get(key) for key in FEATURE_COLUMNS},
    }
    details = {"feature_vector": vector, "raw_quality": result.raw_quality,
               "feature_version": result.feature_version, "feature_config_id": result.feature_config_id,
               "coordinate_policy": result.coordinate_policy, "timebase_policy": result.timebase_policy,
               "reasons": result.reasons}
    return row, details


def export_dataset(
    input_root: Path, output_dir: Path, *, cmc_results: Path | None = None,
    labels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = labels or {}
    samples, feature_rows, hashes = [], [], {}
    manifests = sorted(input_root.rglob("sample.json"))
    if not manifests:
        raise TrackingEvaluationError("no sample.json inputs found")
    ids = set()
    for manifest in manifests:
        report = evaluate_manifest(manifest)
        legacy_id = report["sample_id"]
        settings = labels.get(legacy_id, {})
        sample_id = settings.get("export_id", legacy_id.removesuffix("_apart_fix"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise TrackingEvaluationError("export_id must be a simple file-safe identifier")
        if sample_id in ids:
            raise TrackingEvaluationError("duplicate GT export identity")
        ids.add(sample_id)
        metadata_path = Path(report["inputs"]["metadata_json"])
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        geometry = load_video_geometry(metadata_path)
        source_video = Path(meta["source_video_path"])
        if source_video not in hashes:
            hashes[source_video] = _file_hash(source_video)
        group = "video:" + hashes[source_video]
        gt = load_cvat_track(
            report["inputs"]["cvat_xml"], fps=geometry.fps,
            track_ids=tuple(report["target"]["source_track_ids"]),
        )
        if (gt.source_width, gt.source_height, gt.task_frame_count) != (
            geometry.source_width, geometry.source_height, geometry.frame_count,
        ):
            raise TrackingEvaluationError("full GT source dimensions/frame count do not match")
        candidate = _class_label(gt.label)
        needs_review = bool(settings.get("needs_label_review", candidate is None))
        label = settings.get("label", None if needs_review else candidate)
        if label is not None and label not in {"bird", "drone"}:
            raise TrackingEvaluationError("RF class labels must be bird/drone or null")
        sample = {
            "sample_id": sample_id, "legacy_evaluation_id": legacy_id,
            "source_group_id": group, "source_video": str(source_video),
            "source_video_sha256": hashes[source_video],
            "xml": report["inputs"]["cvat_xml"], "xml_sha256": _file_hash(Path(report["inputs"]["cvat_xml"])),
            "xml_track_ids": list(gt.source_track_ids), "xml_label": gt.label,
            "label_candidate_from_xml": candidate, "label": label,
            "label_status": "needs_review" if needs_review else settings.get("label_status", "from_xml"),
            "label_note": settings.get("note", ""), "split": "pilot",
            "fps": geometry.fps, "processed_width": geometry.processed_width,
            "processed_height": geometry.processed_height,
        }
        track = cvat_to_track_sequence(
            gt, fps=geometry.fps, processed_width=geometry.processed_width,
            processed_height=geometry.processed_height, source_video_id="GT_" + hashes[source_video][:16],
        )
        relative = f"full_gt_raw/{sample_id}_gt.json"
        write_json(output_dir / relative, track.model_dump(mode="json"))
        sample.update({"track_path": relative, "coordinate_space": "frame",
                       "point_count": len(track.history), "first_frame": track.history[0].frame_index,
                       "last_frame": track.history[-1].frame_index,
                       "gt_non_keyframe_count": sum(not p.keyframe for p in gt.observations),
                       "gt_occluded_count": sum(p.occluded for p in gt.observations)})
        row, details = _feature_row(sample, track, "full_gt_raw")
        feature_rows.append(row)
        write_json(output_dir / f"features/full_gt_raw/{sample_id}.json", details)
        if cmc_results is not None:
            b_dir = cmc_results / "b_inputs" / legacy_id
            comparison = json.loads((b_dir / "comparison.json").read_text(encoding="utf-8"))
            if comparison["status"] != "completed":
                raise TrackingEvaluationError(f"CMC GT not verified: {legacy_id}")
            window = TrackSequence.model_validate_json((b_dir / "gt_full_track_sequence.json").read_text(encoding="utf-8"))
            window = TrackSequence.model_validate({
                **window.model_dump(mode="json"), "track_id": gt.track_id,
                "source_video_id": track.source_video_id,
                "quality": build_track_quality(window.history).model_dump(mode="json"),
            })
            cmc_relative = f"cmc_window_gt/{sample_id}_gt.json"
            write_json(output_dir / cmc_relative, window.model_dump(mode="json"))
            sample["cmc_window"] = {
                "track_path": cmc_relative, "point_count": len(window.history),
                "first_frame": window.history[0].frame_index, "last_frame": window.history[-1].frame_index,
                "coordinate_space": "camera_motion_compensated",
                "verification": comparison["camera_verification"],
            }
            row, details = _feature_row(sample, window, "cmc_window_gt")
            feature_rows.append(row)
            write_json(output_dir / f"features/cmc_window_gt/{sample_id}.json", details)
        samples.append(sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(feature_rows[0])
    for scope in sorted({row["scope"] for row in feature_rows}):
        with (output_dir / f"{scope}_features_v4.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in feature_rows if row["scope"] == scope)
    payload = {"format": "cvat_gt_handoff_v1", "sample_count": len(samples),
               "label_policy": "separate_from_TrackSequence; unresolved labels are null",
               "timebase": "source frame_index / average FPS, rounded milliseconds; not VFR PTS",
               "confidence_policy": "conf=1 records GT inclusion, not calibrated detection confidence",
               "samples": samples}
    write_json(output_dir / "manifest.json", payload)
    lines = ["# GT file and class mapping", "",
             "TrackSequence JSON files contain geometry only. The label column is the RF class target.",
             "Do not treat filenames as classes. Unresolved labels must be reviewed before RF scoring.", "",
             "| Sample | XML label | RF label | Status | Full GT frames / points | Note |",
             "| --- | --- | --- | --- | --- | --- |"]
    for sample in samples:
        lines.append(f"| {sample['sample_id']} | {sample['xml_label']} | {sample['label'] or 'UNRESOLVED'} | {sample['label_status']} | {sample['first_frame']}-{sample['last_frame']} / {sample['point_count']} | {sample['label_note']} |")
    (output_dir / "LABEL_MAPPING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmc-results", type=Path)
    parser.add_argument("--labels", type=Path, help="Explicit class corrections and review notes keyed by evaluation ID")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    result = export_dataset(args.input_root, args.output, cmc_results=args.cmc_results, labels=labels)
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=str(Path(args.output.name) / path.relative_to(args.output)))
    print(f"Exported {result['sample_count']} GT tracks -> {args.output}; archive: {archive}")


if __name__ == "__main__":
    main()
