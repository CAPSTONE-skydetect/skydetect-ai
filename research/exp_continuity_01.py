"""Run EXP-CONTINUITY-01 with frozen MiniRocket checkpoints and saved tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/trajectory_experiments/ard100-yolo-hybrid-downstream-20261018-v4"
ARD_BASELINE = ROOT / "artifacts/trajectory_experiments/ard100-frozen-20261018-v3"
ARD_HYBRID = ROOT / "artifacts/trajectory_experiments/ard100-yolo-hybrid-gate-20261018-v2"
PILOT = ROOT / "artifacts/trajectory_experiments/pilot-20261017/evaluation/D0"
BIRD = ROOT / "artifacts/trajectory_experiments/bird-screening-20261018"
DRONE3 = ROOT / "artifacts/trajectory_experiments/holdout-drone3-20261017/evaluation/D0/b_inputs/drone_3_holdout"
LOCAL_INVENTORY = ROOT / "artifacts/trajectory_experiments/followup-20261018/local_inventory.json"
DEFAULT_RUNTIME = ROOT.parent / "skydetect-ai-minirocket-runtime"
DEFAULT_PACKAGE = Path("/Users/yuchan/Downloads/real_reference_comparison_v1.zip")
BASELINE_SHA256 = "9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a"
STACK_SHA256 = "5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def nested_values(payload: object, key: str) -> list[object]:
    found = []
    if isinstance(payload, dict):
        for item_key, value in payload.items():
            if item_key == key:
                found.append(value)
            found.extend(nested_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(nested_values(value, key))
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def canonical_start(value: float) -> str:
    return f"{float(value):.9f}"


def source_origins(track: dict, recovery_frames: set[int]) -> tuple[list[str], list[dict]]:
    history = track["history"]
    frames = [int(point["frame_index"]) for point in history]
    if len(frames) != len(set(frames)) or frames != sorted(frames):
        raise ValueError("TrackSequence frame indices must be unique and ordered")
    origins = ["yolo" if frame in recovery_frames else "d0" for frame in frames]
    if not recovery_frames.issubset(set(frames)):
        raise ValueError("Recovery provenance references a frame absent from Hybrid")
    transitions = []
    for index in range(1, len(history)):
        if origins[index - 1] == origins[index]:
            continue
        previous, current = history[index - 1], history[index]
        transitions.append({
            "from_frame": int(previous["frame_index"]),
            "to_frame": int(current["frame_index"]),
            "from_timestamp_s": float(previous["timestamp_ms"]) / 1000.0,
            "to_timestamp_s": float(current["timestamp_ms"]) / 1000.0,
            "from_origin": origins[index - 1],
            "to_origin": origins[index],
        })
    return origins, transitions


def window_sample_origins(track: dict, origins: list[str], start_s: float,
                          fps: float = 30.0, samples: int = 60) -> set[str]:
    """Find all raw observations that np.interp would use for the C sample grid."""
    history = track["history"]
    times = np.asarray([float(point["timestamp_ms"]) / 1000.0 for point in history])
    grid = float(start_s) + np.arange(samples, dtype=np.float64) / fps
    used: set[str] = set()
    for sample_time in grid:
        right = int(np.searchsorted(times, sample_time, side="left"))
        if right < len(times) and math.isclose(float(times[right]), float(sample_time), abs_tol=1e-9):
            used.add(origins[right])
            continue
        if right == 0 or right >= len(times):
            raise ValueError("Candidate window has no interpolation support")
        used.add(origins[right - 1])
        used.add(origins[right])
    return used


def audit_track_windows(track: dict, recovery_frames: set[int], config,
                        window_track_fn) -> dict:
    """Run the existing C windowing unchanged, then reject source-mixed windows."""
    origins, transitions = source_origins(track, recovery_frames)
    accepted, rejected = window_track_fn(track, config)
    accepted_by_start = {canonical_start(row["relative_start_s"]): row for row in accepted}
    candidates: dict[str, dict] = {}
    for row in accepted + rejected:
        if "relative_start_s" not in row:
            continue
        key = canonical_start(row["relative_start_s"])
        if key in candidates:
            raise ValueError(f"duplicate C window start {key}")
        candidates[key] = row

    first_s = float(track["history"][0]["timestamp_ms"]) / 1000.0
    # C's anti-alias filter is applied to an entire uninterrupted source segment.
    # If it was enabled, conservatively treat the filter as crossing any source
    # transition in this saved sequence instead of pretending its IIR support is local.
    filter_cross_boundary = bool(transitions) and any(
        bool(row.get("anti_alias_applied")) for row in accepted
    )
    audited = []
    for key, row in sorted(candidates.items(), key=lambda item: float(item[0])):
        start_s = float(row.get("window_start_s", first_s + float(key)))
        end_s = float(row.get("window_end_s", start_s + config.window_seconds))
        try:
            origins_used = sorted(window_sample_origins(
                track, origins, start_s, fps=config.fps, samples=config.samples,
            ))
        except ValueError:
            origins_used = []
        interpolation_crosses = len(origins_used) > 1
        c_accepted = key in accepted_by_start
        filter_crosses = filter_cross_boundary and c_accepted
        crosses = interpolation_crosses or filter_crosses
        if c_accepted and not origins_used:
            raise ValueError(f"accepted C window has no raw interpolation support: {key}")
        base_reason = "" if c_accepted else str(row.get("reason", "rejected"))
        continuity_reasons = []
        if interpolation_crosses:
            continuity_reasons.append("source_boundary_interpolation")
        if filter_crosses:
            continuity_reasons.append("source_boundary_filter_support")
        continuity_reason = ";".join(continuity_reasons)
        segmented_accepted = c_accepted and not crosses
        audited.append({
            "relative_start_s": float(key),
            "window_start_s": start_s,
            "window_end_s": end_s,
            "source_fps": row.get("source_fps"),
            "anti_alias_applied": row.get("anti_alias_applied"),
            "c_preprocess_accepted": c_accepted,
            "c_preprocess_reason": base_reason,
            "sources_used_for_resampling": ";".join(origins_used),
            "filter_cross_boundary_risk": filter_crosses,
            "crosses_source_boundary": crosses,
            "continuity_rejection_reason": continuity_reason,
            "segmented_accepted": segmented_accepted,
            "segmented_rejection_reason": continuity_reason or base_reason,
        })
    return {
        "origins": origins,
        "transitions": transitions,
        "accepted": accepted,
        "rejected": rejected,
        "audited": audited,
        "accepted_by_start": accepted_by_start,
        "segmented_by_start": {
            canonical_start(row["relative_start_s"]): row
            for row in accepted
            if not next(item["crosses_source_boundary"] for item in audited
                        if canonical_start(item["relative_start_s"])
                        == canonical_start(row["relative_start_s"]))
        },
    }


def frame_time_pairs(track: dict) -> dict[int, int]:
    return {int(point["frame_index"]): int(point["timestamp_ms"]) for point in track["history"]}


def compare_tracks(gt: dict, a: dict) -> dict:
    gt_clock, a_clock = frame_time_pairs(gt), frame_time_pairs(a)
    common = sorted(set(gt_clock) & set(a_clock))
    same_times = sum(gt_clock[frame] == a_clock[frame] for frame in common)
    gt_size = (gt.get("processed_width"), gt.get("processed_height"))
    a_size = (a.get("processed_width"), a.get("processed_height"))
    gt_stabilization = gt.get("stabilization") or {}
    a_stabilization = a.get("stabilization") or {}
    dimensions_known = all(value is not None and float(value) > 0 for value in (*gt_size, *a_size))
    same_stabilization_convention = (
        gt_stabilization.get("applied") == a_stabilization.get("applied")
        and gt_stabilization.get("method") == a_stabilization.get("method")
    )
    coordinate_compatible = bool(
        len(common) == len(gt_clock) == len(a_clock) and same_times == len(common)
        and dimensions_known and gt_size == a_size and same_stabilization_convention
    )
    return {
        "gt_points": len(gt_clock), "a_points": len(a_clock),
        "common_frames": len(common),
        "same_timestamp_frames": same_times,
        "frame_time_exact": len(common) == len(gt_clock) == len(a_clock) and same_times == len(common),
        "processed_resolution_equal": dimensions_known and gt_size == a_size,
        "coordinate_space_compatible": coordinate_compatible,
        "stabilization_applied_gt": gt_stabilization.get("applied"),
        "stabilization_applied_a": a_stabilization.get("applied"),
        "stabilization_method_gt": gt_stabilization.get("method"),
        "stabilization_method_a": a_stabilization.get("method"),
    }


def add_pair_record(rows: list[dict], *, source_group_id: str, sample_id: str, label: str,
                    gt_path: Path | None, a_path: Path | None, video_path: Path | None,
                    split_history: str, gt_status: str, a_status: str,
                    source_kind: str, training_hashes: set[str], development_exposed: bool,
                    extra: dict | None = None) -> dict:
    gt = read_json(gt_path) if gt_path and gt_path.exists() else None
    a = read_json(a_path) if a_path and a_path.exists() else None
    observed_source_hash = sha256_file(video_path) if video_path and video_path.exists() else ""
    extra_values = dict(extra or {})
    expected_source_hash = str(extra_values.pop("source_video_sha256", "")).lower()
    source_hash = observed_source_hash or expected_source_hash
    paired = compare_tracks(gt, a) if gt is not None and a is not None else {}
    row = {
        "record_type": "paired_or_candidate_track",
        "sample_id": sample_id,
        "source_group_id": source_group_id,
        "label": label,
        "source_kind": source_kind,
        "source_video_path": str(video_path) if video_path else "",
        "source_video_sha256": source_hash,
        "observed_source_video_sha256": observed_source_hash,
        "expected_source_video_sha256": expected_source_hash,
        "source_video_hash_matches_expected": (
            observed_source_hash == expected_source_hash if observed_source_hash and expected_source_hash else "unknown"
        ),
        "exact_hash_in_training_inventory": source_hash in training_hashes if source_hash else "unknown",
        "known_hash_overlap_only": "exact_match_search_does_not_rule_out_reencode_or_session_overlap",
        "gt_path": str(gt_path) if gt_path else "",
        "gt_track_file_available": bool(gt_path and gt_path.exists()),
        "gt_sha256": sha256_file(gt_path) if gt_path and gt_path.exists() else "",
        "gt_status": gt_status,
        "a_track_path": str(a_path) if a_path else "",
        "a_track_file_available": bool(a_path and a_path.exists()),
        "a_track_sha256": sha256_file(a_path) if a_path and a_path.exists() else "",
        "source_video_available": bool(video_path and video_path.exists()),
        "a_status": a_status,
        "development_exposed": development_exposed,
        "training_or_validation_exposure": split_history,
        "paired_frame_time_audit": json.dumps(paired, ensure_ascii=False, sort_keys=True),
        "frame_time_exact": paired.get("frame_time_exact", False),
        "processed_resolution_equal": paired.get("processed_resolution_equal", False),
        "coordinate_space_compatible": paired.get("coordinate_space_compatible", False),
    }
    row.update(extra_values)
    rows.append(row)
    return row


def package_inventory(package: Path) -> tuple[list[dict], dict, set[str]]:
    with zipfile.ZipFile(package) as archive:
        prefix = "real_reference_comparison_v1/"
        manifest = json.loads(archive.read(prefix + "dataset_manifest.json"))
        inventory_text = archive.read(prefix + "source_inventory.csv").decode("utf-8-sig")
        inventory_rows = list(csv.DictReader(inventory_text.splitlines()))
        metadata_text = archive.read(prefix + "real_metadata.csv").decode("utf-8-sig")
        metadata_rows = list(csv.DictReader(metadata_text.splitlines()))
    hashes = {row.get("video_sha256", "").strip().lower() for row in inventory_rows
              if row.get("video_sha256", "").strip()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in inventory_rows:
        grouped[row["source_group_id"]].append(row)
    out = []
    for group_id, rows in sorted(grouped.items()):
        labels = sorted({row["label"] for row in rows})
        splits = sorted({row["split"] for row in rows})
        videos = {row.get("video_sha256", "").strip().lower() for row in rows
                  if row.get("video_sha256", "").strip()}
        out.append({
            "record_type": "training_package_group",
            "source_group_id": group_id,
            "sample_id": ";".join(sorted(row["parent_track_id"] for row in rows)),
            "label": ";".join(labels),
            "source_kind": "real_A_training_windows",
            "source_video_path": "not_in_package",
            "source_video_sha256": ";".join(sorted(videos)),
            "exact_hash_in_training_inventory": "self",
            "gt_path": "",
            "gt_sha256": "",
            "gt_status": "not_verified_as_manual_GT; package contains model training windows only",
            "a_track_path": "",
            "a_track_sha256": "",
            "a_status": "real_A-derived normalized 2s windows present; full TrackSequence absent",
            "development_exposed": False,
            "training_or_validation_exposure": ";".join(splits),
            "paired_frame_time_audit": "no raw video/GT pair in package",
            "frame_time_exact": False,
            "processed_resolution_equal": False,
            "track_count": len(rows),
            "video_hash_missing_count": sum(not row.get("video_sha256", "").strip() for row in rows),
        })
    counts = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in inventory_rows if row["split"] == split]
        groups_by_label: dict[str, set[str]] = defaultdict(set)
        tracks_by_label: Counter = Counter()
        for row in split_rows:
            groups_by_label[row["label"]].add(row["source_group_id"])
            tracks_by_label[row["label"]] += 1
        counts[split] = {
            "track_count": len(split_rows),
            "group_count": len({row["source_group_id"] for row in split_rows}),
            "video_hash_known_tracks": sum(bool(row.get("video_sha256", "").strip()) for row in split_rows),
            "video_hash_missing_tracks": sum(not row.get("video_sha256", "").strip() for row in split_rows),
            "groups_by_label": {label: len(groups) for label, groups in sorted(groups_by_label.items())},
            "tracks_by_label": dict(sorted(tracks_by_label.items())),
            "raw_video_paths_available": False,
            "manual_GT_verified": False,
            "paired_GT_A_available": False,
            "npz_split_arrays_present": split != "test" or manifest.get("test_included", False),
            "test_arrays_present": bool(manifest.get("test_included", False)),
        }
    counts["metadata_rows"] = len(metadata_rows)
    counts["manifest"] = manifest
    return out, counts, hashes


def paired_manifest(package: Path) -> tuple[list[dict], dict]:
    records, package_summary, training_hashes = package_inventory(package)
    labeled = []

    pilot_labels = {
        "bird_2_apart_fix": "drone", "drone_1_bird_1": "bird",
        "drone_1_bird_2": "bird", "drone_1_drone": "drone",
        "drone_5_apart_fix": "drone", "drone_6_apart_fix": "drone",
    }
    video_inventory = read_json(LOCAL_INVENTORY).get("videos", []) if LOCAL_INVENTORY.exists() else []
    video_by_path = {str(Path(row["path"])): row for row in video_inventory}
    video_by_hash = {row.get("sha256", "").lower(): row for row in video_inventory
                     if row.get("sha256")}
    for sample_id, label in pilot_labels.items():
        item_dir = PILOT / "b_inputs" / sample_id
        per_video = read_json(PILOT / "per_video" / f"{sample_id}.json")
        video_data = per_video.get("video", {})
        video_path = Path(video_data["path"]) if video_data.get("path") else None
        metadata_candidates = [Path(path) for path in nested_values(per_video, "metadata_json")
                               if isinstance(path, str)]
        metadata_path = next((path for path in metadata_candidates if path.exists()), None)
        metadata = read_json(metadata_path) if metadata_path else {}
        if video_path is None and metadata.get("source_video_path"):
            video_path = Path(metadata["source_video_path"])
        source_hash = sha256_file(video_path) if video_path and video_path.exists() else ""
        inv = (video_by_path.get(str(video_path), {}) if video_path else {})
        if not inv and source_hash:
            inv = video_by_hash.get(source_hash.lower(), {})
        group_id = inv.get("source_group_id") or (
            f"video:{source_hash}" if source_hash else f"pilot-unknown:{sample_id}"
        )
        add_pair_record(
            labeled, source_group_id=group_id, sample_id=sample_id, label=label,
            gt_path=item_dir / "gt_matched_track_sequence.json",
            a_path=item_dir / "a_track_sequence.json", video_path=video_path,
            split_history="pilot development; repeatedly inspected",
            gt_status="CVAT-derived matched GT sequence; not an independent test",
            a_status="saved D0 TrackSequence",
            source_kind="pilot_CVAT_GT_plus_D0", training_hashes=training_hashes,
            development_exposed=True,
            extra={"gt_source_xml_path": "", "gt_source_xml_sha256": "",
                   "source_video_sha256": source_hash},
        )

    ard_ids = read_json(ARD_BASELINE / "protocol.json")["samples"]
    ard_ids = [row for row in ard_ids if row["sample_id"] in
               read_json(PARENT / "summary.json")["development_sample_ids"]]
    for sample in ard_ids:
        sample_id = sample["sample_id"]
        root = ARD_BASELINE / "samples" / sample_id
        pair_dir = root / "b_inputs"
        video_path = Path(sample["video"])
        source_hash = sample.get("video_sha256", "")
        add_pair_record(
            labeled, source_group_id=f"video:{source_hash}" if source_hash else sample.get("source_group_id", f"ARD100:{sample_id}"),
            sample_id=sample_id, label=sample.get("truth", "drone"),
            gt_path=pair_dir / "gt_matched_track_sequence.json",
            a_path=pair_dir / "a_track_sequence.json", video_path=video_path,
            split_history="ARD100 train_videos; A/C development cohort; YOLO detector exposure is possible",
            gt_status="CVAT XML-derived matched GT sequence; all labels drone",
            a_status="saved D0 TrackSequence",
            source_kind="ARD100_CVAT_GT_plus_D0", training_hashes=training_hashes,
            development_exposed=True,
            extra={"gt_source_xml_path": sample.get("gt_xml", ""),
                   "gt_source_xml_sha256": sample.get("gt_sha256", ""),
                   "source_video_sha256": source_hash},
        )

    drone3_manifest = read_json(ROOT / "artifacts/trajectory_experiments/frozen-drone3-20261018.json")
    for sample in drone3_manifest["samples"]:
        inputs = sample["inputs"]
        add_pair_record(
            labeled, source_group_id=sample["source_group_id"], sample_id="drone_3_holdout",
            label=sample["truth"],
            gt_path=ROOT / "artifacts/trajectory_experiments" / inputs["GT"]["path"],
            a_path=ROOT / "artifacts/trajectory_experiments" / inputs["D0"]["path"],
            video_path=None,
            split_history="previously inspected development/holdout comparison",
            gt_status="CVAT-derived matched GT; tail requires human review",
            a_status="saved D0 TrackSequence",
            source_kind="drone3_CVAT_GT_plus_D0", training_hashes=training_hashes,
            development_exposed=True,
            extra={"source_video_sha256": sample["source_group_id"].removeprefix("video:")},
        )

    bird_inventory = read_json(BIRD / "inventory.json")
    bird_results = {row["sample_id"]: row for row in read_csv(BIRD / "classification.csv")
                    if row["tracking_variant"] == "D0"}
    bird_fps = {row["sample_id"]: float(row.get("fps") or 0) for row in bird_inventory}
    for sample in bird_inventory:
        sample_id = sample["sample_id"]
        track_files = sorted((BIRD / "tracking/D0" / sample_id / "runs").glob("*/track_sequence.json"))
        track_path = track_files[0] if len(track_files) == 1 else None
        video_path = Path(sample["video"])
        add_pair_record(
            labeled, source_group_id=f"video:{sample['sha256']}", sample_id=sample_id,
            label="bird_user_confirmed", gt_path=None, a_path=track_path,
            video_path=video_path, split_history="bird screen; model training overlap unknown",
            gt_status="no manual GT XML/TrackSequence",
            a_status="saved D0 TrackSequence" if track_path else "no usable saved D0 TrackSequence",
            source_kind="user_confirmed_bird_A_only_candidate",
            training_hashes=training_hashes, development_exposed=True,
            extra={"source_fps": sample.get("fps"),
                   "stored_baseline_decision": bird_results.get(sample_id, {}).get("prediction", "unknown"),
                   "source_video_sha256": sample["sha256"]},
        )

    # Raw local video inventory entries remain candidates until identity and labels are reviewed.
    known_groups = {row["source_group_id"] for row in labeled}
    raw_rows = []
    for video in video_inventory:
        group_id = video.get("source_group_id") or f"video:{video.get('sha256', '')}"
        if group_id in known_groups:
            continue
        raw_rows.append({
            "record_type": "unpaired_local_video",
            "sample_id": Path(video["path"]).name,
            "source_group_id": group_id,
            "label": video.get("class_label") or "unknown",
            "source_kind": "local_video_inventory_only",
            "source_video_path": video["path"],
            "source_video_sha256": video.get("sha256", ""),
            "exact_hash_in_training_inventory": video.get("sha256", "").lower() in training_hashes
                                                 if video.get("sha256") else "unknown",
            "known_hash_overlap_only": "recording session/reencode identity unresolved",
            "gt_path": "", "gt_sha256": "", "gt_status": "no linked manual GT",
            "a_track_path": "", "a_track_sha256": "", "a_status": "no linked A TrackSequence",
            "development_exposed": bool(video.get("used_in_pilot")),
            "training_or_validation_exposure": "local inventory; needs class/source review",
            "paired_frame_time_audit": "not applicable",
            "frame_time_exact": False, "processed_resolution_equal": False,
        })
    records.extend(labeled)
    records.extend(raw_rows)

    group_classes: dict[str, set[str]] = defaultdict(set)
    source_records = [row for row in labeled if row["gt_track_file_available"]
                      and row["a_track_file_available"]]
    for row in source_records:
        group_classes[row["source_group_id"]].add(row["label"])
    package_group_count = sum(row["record_type"] == "training_package_group" for row in records)
    paired_group_rows = [row for row in source_records if row.get("coordinate_space_compatible")]
    label_group_counts: dict[str, set[str]] = defaultdict(set)
    for row in source_records:
        label_group_counts[row["label"]].add(row["source_group_id"])
    unseen_pair_groups_by_label: dict[str, set[str]] = defaultdict(set)
    for row in source_records:
        if not row["development_exposed"] and row.get("coordinate_space_compatible"):
            unseen_pair_groups_by_label[row["label"]].add(row["source_group_id"])
    unseen_counts = {label: len(unseen_pair_groups_by_label.get(label, set()))
                     for label in ("bird", "drone")}
    summary = {
        "training_package": package_summary,
        "package_source_group_rows": package_group_count,
        "local_gt_a_pair_records": len(source_records),
        "local_gt_a_source_groups": len(group_classes),
        "coordinate_compatible_gt_a_pair_records": len(paired_group_rows),
        "local_pair_labels_by_track": dict(sorted(Counter(row["label"] for row in source_records).items())),
        "local_pair_source_groups_by_label": {label: len(groups)
                                              for label, groups in sorted(label_group_counts.items())},
        "local_source_groups_with_bird": sum("bird" in labels or "bird_user_confirmed" in labels
                                              for labels in group_classes.values()),
        "local_source_groups_with_drone": sum("drone" in labels for labels in group_classes.values()),
        "local_groups_containing_both_classes": sum("drone" in labels and "bird" in labels
                                                     for labels in group_classes.values()),
        "local_pair_groups_development_exposed": sum(any(row["development_exposed"] and row["source_group_id"] == group
                                                         for row in source_records)
                                                       for group in group_classes),
        "additional_unpaired_local_videos": len(raw_rows),
        "independent_source_disjoint_paired_split_feasible": all(value >= 2 for value in unseen_counts.values()),
        "train_validation_test_paired_split_feasible": all(value >= 3 for value in unseen_counts.values()),
        "reason": "all existing paired GT/A sources were development-exposed; package has no verified manual GT or source videos; only new unseen sources count toward a leakage-free split",
        "exact_video_hash_matches_between_local_pairs_and_package": sum(
            row["exact_hash_in_training_inventory"] is True for row in source_records
        ),
        "source_disjoint_unexposed_paired_groups_by_label": unseen_counts,
        "overlap_caveat": "Missing package video hashes and re-encoded/session duplicates prevent proving complete independence",
    }
    return records, summary


def prepare_import_paths(runtime_root: Path) -> None:
    for path in (str(ROOT), str(runtime_root)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [str(runtime_root), str(ROOT)]
    research_package = importlib.import_module("research")
    local_research = str(ROOT / "research")
    if local_research not in research_package.__path__:
        research_package.__path__.insert(0, local_research)
    # Pin the exact experiment-side preprocessing module before loading runtime adapters.
    importlib.import_module("research.trajectory_sequence")


def load_models(runtime_root: Path, baseline_path: Path, stack_path: Path):
    prepare_import_paths(runtime_root)
    from ai_server.services.classifier import MiniRocketClassifier
    from ai_server.services.sequence_model import decision_scores
    from research.minirocket_stack import StackMiniRocketClassifier, decision_margins

    if sha256_file(baseline_path) != BASELINE_SHA256 or sha256_file(stack_path) != STACK_SHA256:
        raise ValueError("A frozen MiniRocket checkpoint hash does not match the protocol")
    baseline = MiniRocketClassifier(baseline_path)
    stack = StackMiniRocketClassifier(stack_path)
    return {
        "deployed_baseline": {"adapter": baseline, "model": baseline.model,
                              "score_fn": lambda windows: decision_scores(
                                  baseline.model, np.stack([row["X"] for row in windows]))},
        "frozen_stack_candidate": {"adapter": stack, "model": stack.model,
                                   "score_fn": lambda windows: decision_margins(
                                       stack.model, np.stack([row["X"] for row in windows]),
                                       [row["normalization_scale"] for row in windows])},
    }


def prediction_from_scores(scores: list[float]) -> dict:
    if not scores:
        return {"prediction": "uncertain", "decision_margin": None}
    mean = float(np.mean(scores))
    return {"prediction": "drone" if mean >= 0 else "bird", "decision_margin": mean}


def cohort_metrics(rows: list[dict], total: int) -> dict:
    decided = [row for row in rows if row["prediction"] in {"drone", "bird"}]
    correct = sum(row["prediction"] == row["truth"] for row in decided)
    wrong = len(decided) - correct
    return {
        "objects": total,
        "decided": len(decided),
        "coverage": len(decided) / total if total else None,
        "correct": correct,
        "wrong": wrong,
        "abstained": total - len(decided),
        "correct_over_all_objects": correct / total if total else None,
        "wrong_over_all_objects": wrong / total if total else None,
        "accuracy_among_decided": correct / len(decided) if decided else None,
    }


def classify_saved_sequence(models: dict, name: str, payload: dict) -> dict:
    from ai_server.schemas import TrackSequence

    track = TrackSequence.model_validate(payload)
    result = {}
    for model_name, bundle in models.items():
        prediction = bundle["adapter"].predict(track)
        result[model_name] = {
            "prediction": prediction.label,
            "decision_margin": prediction.decision_score,
            "accepted_windows": len(prediction.window_scores),
            "window_scores": prediction.window_scores,
            "window_starts_s": prediction.window_starts_s,
            "window_rejections": prediction.window_rejections,
            "abstain_reason": prediction.abstain_reason,
            "abstain_detail": prediction.abstain_detail,
        }
    return result


def score_consistency(scores: list[float]) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        return {"count": 0, "std": None, "median_absolute_deviation": None,
                "majority_sign_fraction": None}
    signs = np.sign(values)
    majority_sign = np.sign(float(np.mean(values)))
    return {
        "count": int(len(values)),
        "std": float(np.std(values)),
        "median_absolute_deviation": float(np.median(np.abs(values - np.median(values)))),
        "majority_sign_fraction": float(np.mean(signs == majority_sign)) if majority_sign else None,
    }


def window_accounting(rows: list[dict]) -> dict:
    unique = {}
    for row in rows:
        if row["model"] != "deployed_baseline":
            continue
        unique[(row["input"], row["sample_id"], canonical_start(row["relative_start_s"]))] = row
    result = {}
    for input_name in ("D0", "Hybrid"):
        selected = [row for (variant, _, _), row in unique.items() if variant == input_name]
        c_accepted = sum(bool(row["c_preprocess_accepted"]) for row in selected)
        boundary_excluded = sum(bool(row["c_preprocess_accepted"])
                                and bool(row["crosses_source_boundary"]) for row in selected)
        c_rejected = sum(not bool(row["c_preprocess_accepted"]) for row in selected)
        result[input_name] = {
            "candidate_starts": len(selected),
            "accepted_by_existing_C_preprocess": c_accepted,
            "rejected_by_existing_C_preprocess": c_rejected,
            "excluded_for_source_boundary": boundary_excluded,
            "accepted_after_segmentation": c_accepted - boundary_excluded,
            "boundary_exclusion_fraction_of_C_accepted": (
                boundary_excluded / c_accepted if c_accepted else None
            ),
            "anti_alias_filtered_window_count": sum(bool(row["anti_alias_applied"]) for row in selected),
        }
    return result


def prediction_changes(rows: list[dict]) -> list[dict]:
    changes = []
    for model_name in sorted({row["model"] for row in rows}):
        sample_ids = sorted({row["sample_id"] for row in rows if row["model"] == model_name})
        by_key = {(row["sample_id"], row["input"]): row for row in rows if row["model"] == model_name}
        for sample_id in sample_ids:
            full = by_key[sample_id, "Hybrid_full"]
            segmented = by_key[sample_id, "Hybrid_segmented"]
            changes.append({
                "sample_id": sample_id,
                "model": model_name,
                "full_prediction": full["prediction"],
                "segmented_prediction": segmented["prediction"],
                "prediction_changed": full["prediction"] != segmented["prediction"],
                "full_margin": full["decision_margin"],
                "segmented_margin": segmented["decision_margin"],
                "windows_full": full["accepted_windows"],
                "windows_segmented": segmented["accepted_windows"],
                "boundary_windows_excluded": full["boundary_excluded_windows"],
            })
    return changes


def verify_parent_reproduction(results: list[dict], parent_rows: list[dict[str, str]]) -> list[dict]:
    lookup = {(row["sample_id"], row["input"], row["model"]): row for row in parent_rows}
    checks = []
    for row in results:
        if row["input"] == "Hybrid_segmented":
            continue
        parent_input = "D0" if row["input"] == "D0" else "Hybrid"
        parent = lookup[(row["sample_id"], parent_input, row["model"])]
        parent_margin = float(parent["decision_margin"]) if parent.get("decision_margin") else None
        same_margin = (row["decision_margin"] is None and parent_margin is None) or (
            row["decision_margin"] is not None and parent_margin is not None
            and math.isclose(float(row["decision_margin"]), parent_margin, rel_tol=0, abs_tol=1e-6)
        )
        check = {
            "sample_id": row["sample_id"], "parent_input": parent_input,
            "model": row["model"], "reproduced_prediction": row["prediction"] == parent["prediction"],
            "reproduced_margin_within_1e-6": same_margin,
            "reproduced_window_count": row["accepted_windows"] == int(parent["accepted_windows"]),
            "runner_prediction": row["prediction"], "parent_prediction": parent["prediction"],
            "runner_margin": row["decision_margin"], "parent_margin": parent_margin,
            "runner_window_count": row["accepted_windows"],
            "parent_window_count": int(parent["accepted_windows"]),
        }
        if not all(check[key] for key in ("reproduced_prediction", "reproduced_margin_within_1e-6",
                                         "reproduced_window_count")):
            raise AssertionError(f"frozen parent result mismatch: {check}")
        checks.append(check)
    return checks


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    prepare_import_paths(args.runtime_root.resolve())
    from research.trajectory_sequence import SequenceConfig, window_track
    from ai_server.schemas import TrackSequence

    models = load_models(args.runtime_root.resolve(), args.baseline_model.resolve(), args.stack_model.resolve())
    parent_summary = read_json(PARENT / "summary.json")
    recoveries_by_sample: dict[str, set[int]] = defaultdict(set)
    for row in read_csv(PARENT / "recovery_audit.csv"):
        if str(row.get("sample_id", "")) and row.get("frame_index"):
            recoveries_by_sample[row["sample_id"]].add(int(row["frame_index"]))

    all_window_rows, boundary_rows, object_rows, common_rows = [], [], [], []
    sample_details = {}
    sample_ids = parent_summary["development_sample_ids"]
    config = SequenceConfig()
    for sample_id in sample_ids:
        truth = parent_summary["truth_by_sample"][sample_id]
        d0_path = PARENT / "tracks" / sample_id / "D0.json"
        hybrid_path = PARENT / "tracks" / sample_id / "Hybrid.json"
        tracks = {"D0": read_json(d0_path), "Hybrid": read_json(hybrid_path)}
        d0_frames = {int(point["frame_index"]) for point in tracks["D0"]["history"]}
        recoveries = recoveries_by_sample[sample_id]
        if d0_frames & recoveries:
            raise ValueError(f"D0 and recovery provenance overlap for {sample_id}")
        if {int(point["frame_index"]) for point in tracks["Hybrid"]["history"]} != d0_frames | recoveries:
            raise ValueError(f"Hybrid history/provenance mismatch for {sample_id}")

        audited = {
            name: audit_track_windows(track, recoveries if name == "Hybrid" else set(), config, window_track)
            for name, track in tracks.items()
        }
        sample_details[sample_id] = {}
        for transition in audited["Hybrid"]["transitions"]:
            boundary_rows.append({"sample_id": sample_id, **transition})

        per_input_windows: dict[str, dict[str, dict[str, float]]] = {}
        for input_name, audit in audited.items():
            accepted_by_start = audit["accepted_by_start"]
            segmented_by_start = audit["segmented_by_start"] if input_name == "Hybrid" else accepted_by_start
            score_sets: dict[str, dict[str, float]] = {}
            for model_name, model_bundle in models.items():
                full_windows = [accepted_by_start[key] for key in sorted(accepted_by_start)]
                segmented_windows = [segmented_by_start[key] for key in sorted(segmented_by_start)]
                full_scores = [float(value) for value in model_bundle["score_fn"](full_windows)] if full_windows else []
                segmented_scores = [float(value) for value in model_bundle["score_fn"](segmented_windows)] if segmented_windows else []
                full_by_start = dict(zip(sorted(accepted_by_start), full_scores))
                retained_scores_match = True
                for key, score in zip(sorted(segmented_by_start), segmented_scores):
                    full_score = full_by_start.get(key)
                    if full_score is None or not math.isclose(score, full_score, rel_tol=0, abs_tol=1e-9):
                        retained_scores_match = False
                        raise AssertionError(f"retained window changed after segmentation: {sample_id}/{key}/{model_name}")
                full_prediction = prediction_from_scores(full_scores)
                segmented_prediction = prediction_from_scores(segmented_scores)
                score_sets[model_name] = {
                    "full": dict(zip(sorted(accepted_by_start), full_scores)),
                    "segmented": dict(zip(sorted(segmented_by_start), segmented_scores)),
                }

                # Recheck the frozen adapter's aggregate against direct window scoring.
                adapter_prediction = model_bundle["adapter"].predict(
                    TrackSequence.model_validate(tracks[input_name])
                )
                if full_scores:
                    if adapter_prediction.label != full_prediction["prediction"] or not math.isclose(
                        float(adapter_prediction.decision_score), full_prediction["decision_margin"], abs_tol=1e-6,
                    ):
                        raise AssertionError(f"window scorer disagrees with frozen adapter: {sample_id}/{input_name}/{model_name}")
                elif adapter_prediction.label != "uncertain":
                    raise AssertionError(f"adapter decided without accepted C windows: {sample_id}/{input_name}/{model_name}")

                if input_name == "D0":
                    variant = "D0"
                    prediction = full_prediction
                    emitted_scores = score_sets[model_name]["full"]
                elif input_name == "Hybrid":
                    variant = "Hybrid_full"
                    prediction = full_prediction
                    emitted_scores = score_sets[model_name]["full"]
                else:
                    raise AssertionError(input_name)
                row = {"sample_id": sample_id, "truth": truth, "input": variant,
                       "model": model_name, **prediction,
                       "accepted_windows": len(full_scores), "rejected_by_C": len(audit["rejected"]),
                       "boundary_excluded_windows": sum(
                           item["crosses_source_boundary"] and item["c_preprocess_accepted"]
                           for item in audit["audited"]
                       ) if input_name == "Hybrid" else 0,
                       "final_accepted_windows": len(full_scores),
                       "window_scores": json.dumps(full_scores),
                       "window_starts_s": json.dumps([float(key) for key in sorted(emitted_scores)]),
                       "window_rejections": json.dumps(dict(Counter(
                           item["c_preprocess_reason"] for item in audit["audited"]
                           if not item["c_preprocess_accepted"]
                       )), sort_keys=True)}
                object_rows.append(row)
                sample_details[sample_id].setdefault(input_name, {})[model_name] = {
                    "full": full_prediction,
                    "full_window_scores": full_scores,
                    "full_window_score_consistency": score_consistency(full_scores),
                    "full_window_starts_s": [float(key) for key in sorted(accepted_by_start)],
                    "segmented": segmented_prediction if input_name == "Hybrid" else full_prediction,
                    "segmented_window_scores": segmented_scores,
                    "segmented_window_score_consistency": score_consistency(segmented_scores),
                    "segmented_window_starts_s": [float(key) for key in sorted(segmented_by_start)],
                    "retained_window_scores_match_full": retained_scores_match,
                    "track_level_c_rejections": audit["rejected"] if not audit["audited"] else [],
                }

                if input_name == "Hybrid":
                    object_rows.append({
                        "sample_id": sample_id, "truth": truth, "input": "Hybrid_segmented",
                        "model": model_name, **segmented_prediction,
                        "accepted_windows": len(segmented_scores),
                        "rejected_by_C": len(audit["rejected"]),
                        "boundary_excluded_windows": sum(
                            item["crosses_source_boundary"] and item["c_preprocess_accepted"]
                            for item in audit["audited"]
                        ),
                        "final_accepted_windows": len(segmented_scores),
                        "window_scores": json.dumps(segmented_scores),
                        "window_starts_s": json.dumps([float(key) for key in sorted(segmented_by_start)]),
                        "window_rejections": json.dumps(dict(Counter(
                            item["segmented_rejection_reason"] for item in audit["audited"]
                            if not item["segmented_accepted"]
                        )), sort_keys=True),
                    })

                for item in audit["audited"]:
                    key = canonical_start(item["relative_start_s"])
                    full_score = score_sets[model_name]["full"].get(key)
                    seg_score = score_sets[model_name]["segmented"].get(key)
                    all_window_rows.append({
                        "sample_id": sample_id, "truth": truth, "input": input_name,
                        "model": model_name, **item,
                        "window_margin_full": full_score,
                        "window_margin_segmented": seg_score,
                    })
            per_input_windows[input_name] = score_sets

        d0_keys = set(audited["D0"]["accepted_by_start"])
        full_keys = set(audited["Hybrid"]["accepted_by_start"])
        seg_keys = set(audited["Hybrid"]["segmented_by_start"])
        for comparison, right_keys in (("D0_vs_Hybrid_full", full_keys),
                                       ("D0_vs_Hybrid_segmented", seg_keys)):
            common = sorted(d0_keys & right_keys)
            for model_name in models:
                d0_map = per_input_windows["D0"][model_name]["full"]
                hy_map = per_input_windows["Hybrid"][model_name]["segmented" if "segmented" in comparison else "full"]
                left_scores = [d0_map[key] for key in common]
                right_scores = [hy_map[key] for key in common]
                common_rows.append({
                    "sample_id": sample_id, "truth": truth, "comparison": comparison,
                    "model": model_name, "common_window_count": len(common),
                    "window_starts_s": json.dumps([float(key) for key in common]),
                    "d0_mean_margin": float(np.mean(left_scores)) if left_scores else None,
                    "hybrid_mean_margin": float(np.mean(right_scores)) if right_scores else None,
                    "d0_prediction": prediction_from_scores(left_scores)["prediction"],
                    "hybrid_prediction": prediction_from_scores(right_scores)["prediction"],
                    "window_scores_d0": json.dumps(left_scores),
                    "window_scores_hybrid": json.dumps(right_scores),
                })

    classifications = []
    for input_name in ("D0", "Hybrid_full", "Hybrid_segmented"):
        for model_name in models:
            subset = [row for row in object_rows if row["input"] == input_name and row["model"] == model_name]
            classifications.append({"input": input_name, "model": model_name,
                                     **cohort_metrics(subset, len(sample_ids))})
    parent_reproduction = verify_parent_reproduction(
        object_rows, read_csv(PARENT / "classification.csv")
    )
    prediction_change_rows = prediction_changes(object_rows)
    window_counts = window_accounting(all_window_rows)

    # Regression-check the stored six-object paired GT/A pilot and bird screening inputs.
    regression_rows = []
    pilot_labels = {
        "bird_2_apart_fix": "drone", "drone_1_bird_1": "bird",
        "drone_1_bird_2": "bird", "drone_1_drone": "drone",
        "drone_5_apart_fix": "drone", "drone_6_apart_fix": "drone",
    }
    for sample_id, truth in pilot_labels.items():
        pair_dir = PILOT / "b_inputs" / sample_id
        for kind, file_name in (("GT_matched", "gt_matched_track_sequence.json"),
                                ("A_D0", "a_track_sequence.json")):
            payload = read_json(pair_dir / file_name)
            result = classify_saved_sequence(models, kind, payload)
            for model_name, prediction in result.items():
                regression_rows.append({"cohort": "pilot_6", "sample_id": sample_id,
                                        "truth": truth, "input": kind, "model": model_name,
                                        **prediction})

    bird_truths = {f"B{i:02d}": "bird" for i in range(1, 6)}
    bird_inventory = read_json(BIRD / "inventory.json")
    bird_fps = {row["sample_id"]: float(row.get("fps") or 0) for row in bird_inventory}
    for sample in bird_inventory:
        sample_id = sample["sample_id"]
        track_files = sorted((BIRD / "tracking/D0" / sample_id / "runs").glob("*/track_sequence.json"))
        if len(track_files) != 1:
            continue
        payload = read_json(track_files[0])
        result = classify_saved_sequence(models, "A_D0", payload)
        for model_name, prediction in result.items():
            regression_rows.append({"cohort": "bird_screening", "sample_id": sample_id,
                                    "truth": bird_truths[sample_id], "input": "A_D0",
                                    "source_fps": bird_fps.get(sample_id),
                                    "expected_input_status": (
                                        "unsupported_source_fps" if bird_fps.get(sample_id, 0) < 15
                                        else "evaluate_saved_track_without_relabeling"
                                    ),
                                    "model": model_name, **prediction})
            if bird_fps.get(sample_id, 0) < 15 and any(
                prediction["prediction"] != "uncertain" or prediction["accepted_windows"] != 0
                for prediction in result.values()
            ):
                raise AssertionError(f"14Hz bird input was incorrectly accepted: {sample_id}")

    pair_records, pair_summary = paired_manifest(args.package.resolve())
    hashes = {}
    paths = [
        PARENT / "summary.json", PARENT / "classification.csv", PARENT / "classification_details.json",
        PARENT / "recovery_audit.csv", PARENT / "splice_audit.csv",
        ROOT / "research/ard100_yolo_hybrid_downstream.py",
        ROOT / "research/trajectory_sequence.py", ROOT / "research/minirocket_stack.py",
        ROOT / "research/minirocket_framescale.py", ROOT / "research/minirocket_inference.py",
        ROOT / "research/exp_continuity_01.py", ROOT / "research/tests/test_exp_continuity_01.py",
        ROOT / "docs/exp_continuity_01_protocol.md",
        args.runtime_root.resolve() / "ai_server/services/classifier.py",
        args.runtime_root.resolve() / "ai_server/services/sequence_model.py",
        args.runtime_root.resolve() / "ai_server/schemas.py",
        ROOT / "docs/ard100_yolo_hybrid_gate_20261018.md",
        ROOT / "docs/minirocket_training_results_20261018.md",
        ROOT / "docs/bird_false_positive_check_20261018.md",
        ROOT / "artifacts/trajectory_experiments/minirocket-results-20261018/summary.json",
        ROOT / "artifacts/trajectory_experiments/minirocket-retrain-20261018/status.json",
        ROOT / "artifacts/trajectory_experiments/minirocket-stack-20261018/status.json",
        args.package.resolve(), args.baseline_model.resolve(), args.stack_model.resolve(),
    ]
    for path in paths:
        hashes[str(path)] = sha256_file(path)
    for sample_id in sample_ids:
        for name in ("D0", "Hybrid"):
            path = PARENT / "tracks" / sample_id / f"{name}.json"
            hashes[str(path)] = sha256_file(path)
    for path in (PARENT / "recovery_audit.csv", PILOT / "summary.csv", BIRD / "classification.csv",
                 BIRD / "inventory.json", LOCAL_INVENTORY,
                 DRONE3 / "gt_matched_track_sequence.json", DRONE3 / "a_track_sequence.json"):
        if path.exists():
            hashes[str(path)] = sha256_file(path)
    for row in pair_records:
        for key in ("gt_path", "a_track_path", "gt_source_xml_path"):
            path = Path(row[key]) if row.get(key) else None
            if path and path.exists():
                hashes[str(path)] = sha256_file(path)
        path = Path(row["source_video_path"]) if row.get("source_video_path") else None
        if path and path.exists():
            hashes[str(path)] = sha256_file(path)

    (output / "input_model_code_hashes.json").write_text(
        json.dumps({"sha256": hashes, "baseline_model_sha256": BASELINE_SHA256,
                    "stack_model_sha256": STACK_SHA256,
                    "python_version": sys.version,
                    "runtime_root": str(args.runtime_root.resolve()),
                    "training_package_manifest_sha256": read_json(ROOT / "artifacts/trajectory_experiments/minirocket-results-20261018/summary.json")["dataset_audit"]["manifest_sha256"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "paired_source_summary.json").write_text(
        json.dumps(pair_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "per_object_results.json").write_text(
        json.dumps({"classifications": object_rows, "cohort_summary": classifications,
                    "parent_reproduction_checks": parent_reproduction,
                    "prediction_changes": prediction_change_rows,
                    "window_accounting": window_counts,
                    "phantom66": sample_details.get("phantom66", {}),
                    "regression_checks": regression_rows,
                    "limitations": ["All 12 ARD100 downstream development targets are drones.",
                                    "Margins are model scores, not calibrated probabilities.",
                                    "A boundary exclusion can turn an error into abstention; this is a coverage tradeoff."]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "window_audit.csv", all_window_rows)
    write_csv(output / "source_boundaries.csv", boundary_rows)
    write_csv(output / "common_window_comparison.csv", common_rows)
    write_csv(output / "object_classification.csv", object_rows)
    write_csv(output / "regression_checks.csv", regression_rows)
    write_csv(output / "paired_source_manifest.csv", pair_records)
    write_csv(output / "parent_reproduction.csv", parent_reproduction)
    write_csv(output / "prediction_changes.csv", prediction_change_rows)

    summary = {
        "experiment_id": "EXP-CONTINUITY-01",
        "status": "completed",
        "development_samples": len(sample_ids),
        "truth_by_sample": parent_summary["truth_by_sample"],
        "source_boundaries": len(boundary_rows),
        "classifications": classifications,
        "window_accounting": window_counts,
        "prediction_changes": prediction_change_rows,
        "parent_reproduction": {
            "rows_checked": len(parent_reproduction),
            "all_predictions_margins_and_window_counts_match": True,
        },
        "paired_sources": pair_summary,
        "models": parent_summary["models"],
        "input_hashes_path": str(output / "input_model_code_hashes.json"),
        "no_A_tracking_rerun": True,
        "production_changed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/trajectory_experiments/exp-continuity-01")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--baseline-model", type=Path,
                        default=DEFAULT_RUNTIME / "models/minirocket_classifier.joblib")
    parser.add_argument("--stack-model", type=Path,
                        default=ROOT / "artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
