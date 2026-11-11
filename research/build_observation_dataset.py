"""Build C-part datasets from reviewed tracks and measured observation proxies."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import StratifiedGroupKFold

from .build_real_sequence_dataset import digest, file_hash, group_records
from . import FEATURE_VERSION
from .feature_contract import (CANONICAL_WIDTH_PX, COORDINATE_POLICY, canonical_dimensions,
                               feature_contract_id, timebase_policy)
from .features import FEATURE_COLUMNS, FeatureConfig, extract_features
from .trajectory_sequence import CHANNELS, SequenceConfig, window_track
from .verify_real_track import load_tracks


TARGETS = {"bird": (167, 55, 55), "drone": (151, 51, 51)}
SPLITS = ("train", "validation", "test")
VARIANTS = ("location_error", "short_gap", "both") * 3
SEQUENCE = SequenceConfig()
FEATURES = FeatureConfig()
RF_PROVENANCE = dict(feature_version=FEATURE_VERSION, feature_config_id=FEATURES.fingerprint,
                     feature_contract_id=feature_contract_id(FEATURES.fingerprint, target_fps=FEATURES.target_fps),
                     coordinate_policy=COORDINATE_POLICY, timebase_policy=timebase_policy(FEATURES.target_fps))
SCENARIOS = {
    "A": ["train_real", "validation_real", "test_real"],
    "B": ["train_augmented", "validation_real", "test_real"],
    "D": ["train_augmented", "validation_augmented", "test_augmented"],
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_family(path, label):
    name = re.sub(r"_track_?sequence$", "", Path(path).stem, flags=re.I)
    name = re.sub(r"^teammate_", "", name)
    name = re.sub(r"_obj\d+.*$|\(\d+\)$|_(?:아래|위에)[ _]?새$", "", name)
    if label == "bird":
        name = re.sub(r"^(?:오이도_)?새_(\d+)$", r"오이도_새_\1", name)
    return f"{label}:{name}"


def read_sources(root, label_map):
    mapped = {Path(row["renamed_relative_path"]).as_posix(): row
              for row in read_csv(label_map) if row["file_kind"] == "tracksequence"}
    records, family = [], {}
    for label in TARGETS:
        folder = root / f"final_{label}" / f"final_{label}_tracksequence"
        files = sorted(folder.glob("*.json"))
        if len(files) != sum(TARGETS[label]):
            raise ValueError(f"{label}: expected {sum(TARGETS[label])} files, got {len(files)}")
        for path in files:
            relative = path.relative_to(root).as_posix()
            entry = mapped[relative]
            if entry["class"] != label:
                raise ValueError(f"Label map disagrees with directory for {path}")
            track, = load_tracks(path)
            observations = [[p[k] for k in ("frame_index", "timestamp_ms", "cx", "cy")]
                            for p in track["history"]]
            center_hash = digest([track["processed_width"], track["processed_height"], observations])
            family[relative] = source_family(entry["original_relative_path"], label)
            records.append(dict(parent_track_id=relative, label=label, file_name=path.name,
                                source_path=str(path), file_sha256=file_hash(path),
                                original_relative_path=entry["original_relative_path"],
                                source_video_id=track.get("source_video_id", ""),
                                center_hash=center_hash, video_sha256="", track=track))
    links = group_records(records, family)
    return records, links


def calibration_clips(audit_dir, records):
    audited = {row["track_file"]: row for row in read_csv(audit_dir / "video_track_matches.csv")
               if row["raw_csv_match"].lower() == "true" and row["hash_match"].lower() == "true"}
    by_source = defaultdict(set)
    for record in records:
        by_source[record["source_video_id"]].add(record["source_group_id"])
    clips = []
    for path in sorted((audit_dir / "visual_centers").glob("*_proxy.csv")):
        rows = read_csv(path)
        audit = audited[rows[0]["track_name"]]
        groups = by_source[audit["source_video_id"]]
        if len(groups) != 1:
            raise ValueError(f"Calibration source must map to one original group: {path}")
        points = np.array([[float(r["proxy_x"]) - float(r["raw_x"]),
                            float(r["proxy_y"]) - float(r["raw_y"])]
                           for r in rows if r["proxy_x"] and r["proxy_y"]])
        points *= CANONICAL_WIDTH_PX / float(audit["video_width"])
        clips.append(dict(group=next(iter(groups)), source=str(path),
                          sha256=file_hash(path), offsets=points - np.median(points, axis=0)))
    if not clips:
        raise ValueError("No measured video-center proxies found")
    return clips


def assign_partitions(records, clips, seed):
    groups = sorted({row["source_group_id"] for row in records})
    members = defaultdict(Counter)
    for record in records:
        members[record["source_group_id"]][record["label"]] += 1
    n = len(groups)
    matrix = np.zeros((n + len(TARGETS) * len(SPLITS), n * len(SPLITS)))
    totals = [1] * n
    upper = np.ones(n * len(SPLITS))
    calibration_groups = {clip["group"] for clip in clips}
    for index, group in enumerate(groups):
        matrix[index, index * 3:index * 3 + 3] = 1
        if group in calibration_groups:
            upper[index * 3 + 1:index * 3 + 3] = 0
    for label_index, (label, targets) in enumerate(TARGETS.items()):
        counts = [members[group][label] for group in groups]
        for split_index, count in enumerate(targets):
            matrix[n + label_index * 3 + split_index, split_index::3] = counts
            totals.append(count)
    solution = milp(np.random.default_rng(seed).random(n * 3), integrality=np.ones(n * 3),
                    bounds=Bounds(0, upper), constraints=LinearConstraint(matrix, totals, totals),
                    options={"time_limit": 30})
    if solution.x is None:
        raise ValueError(f"Cannot attain requested counts without splitting source groups: {solution.message}")
    partitions = dict(zip(groups, np.argmax(solution.x.reshape(n, 3), axis=1), strict=True))
    train = []
    for record in records:
        record["split"] = SPLITS[partitions[record["source_group_id"]]]
        record["fold"] = -1
        if record["split"] == "train":
            train.append(record)
    counts = Counter((r["label"], r["split"]) for r in records)
    if any(counts[label, split] != targets[i]
           for label, targets in TARGETS.items() for i, split in enumerate(SPLITS)):
        raise ValueError("Partition solver did not attain the requested counts")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, holdout) in enumerate(splitter.split(
            train, [r["label"] for r in train], [r["source_group_id"] for r in train])):
        for i in holdout:
            train[i]["fold"] = fold
        if {train[i]["label"] for i in holdout} != set(TARGETS):
            raise ValueError(f"Fold {fold} does not contain both labels")


def observation_profile(records, clips):
    groups = {record["source_group_id"] for record in records}
    eligible = [clip for clip in clips if clip["group"] in groups]
    if not eligible:
        raise ValueError("No fit-side position observations available for calibration")
    offsets = np.concatenate([clip["offsets"] for clip in eligible])
    radius = np.linalg.norm(offsets, axis=1)
    cap = float(np.quantile(radius, .95))
    offsets = offsets[(radius > 0) & (radius <= cap)]
    gaps = Counter()
    for record in records:
        frames = [point["frame_index"] for point in record["track"]["history"]]
        gaps.update(int(step - 1) for step in np.diff(frames) if 2 <= step <= 3)
    if not gaps or not len(offsets):
        raise ValueError("Empty measured position-error or short-gap distribution")
    profile = dict(reference="reviewed image-derived center proxy, not manual ground truth",
                   canonical_width_px=CANONICAL_WIDTH_PX, radius_p95_px=cap,
                   retained_offsets=len(offsets), gap_counts=dict(gaps),
                   sources=[{k: v for k, v in clip.items() if k != "offsets"} for clip in eligible])
    return offsets, gaps, profile


def track_windows(track):
    try:
        windows, rejected = window_track(track)
        return windows, rejected, 0.002
    except ValueError as exc:
        if str(exc) != "Inconsistent frame/timestamp clock":
            return [], [{"reason": str(exc)}], 0.002
    history = track["history"]
    frame = np.array([p["frame_index"] for p in history], dtype=float)
    time = np.array([p["timestamp_ms"] for p in history], dtype=float)
    fit = np.polyval(np.polyfit(frame - frame[0], time - time[0], 1), frame - frame[0])
    if np.max(np.abs((time - time[0]) - fit)) > 3.0:
        return [], [{"reason": "Inconsistent frame/timestamp clock"}], 0.002
    try:
        windows, rejected = window_track(track, clock_step_tolerance_seconds=0.003)
        return windows, rejected, 0.003
    except ValueError as exc:
        return [], [{"reason": str(exc)}], 0.002


def augment(track, variant, rng, offsets, gaps):
    history = track["history"]
    removed = []
    if variant in {"short_gap", "both"}:
        frames = np.array([point["frame_index"] for point in history])
        times = np.array([point["timestamp_ms"] for point in history])
        period_ms = (times[-1] - times[0]) / (frames[-1] - frames[0])
        choices = {}
        for size in sorted(gaps):
            if len(frames) < size + 2:
                continue
            spans = np.lib.stride_tricks.sliding_window_view(frames, size + 2)
            candidates = np.flatnonzero(spans[:, -1] - spans[:, 0] == size + 1) + 1
            candidates = candidates[(times[candidates + size] - times[candidates - 1] - period_ms)
                                    <= (SEQUENCE.max_missing_seconds + SEQUENCE.clock_tolerance_seconds) * 1000]
            if len(candidates):
                choices[size] = candidates
        if not choices:
            raise ValueError("No interior contiguous run for short gap")
        counts = np.asarray([gaps[size] for size in choices])
        size = int(rng.choice(list(choices), p=counts / counts.sum()))
        start = int(rng.choice(choices[size]))
        removed = list(range(start, start + size))
    positions = np.array([[p["cx"], p["cy"]] for p in history], dtype=float)
    if variant in {"location_error", "both"}:
        width, height, _ = canonical_dimensions(track["processed_width"], track["processed_height"])
        perturbations = offsets[rng.integers(len(offsets), size=len(history))] / [width, height]
        shifted = positions + perturbations
        valid = ((shifted >= 0) & (shifted <= 1)).all(axis=1)
        positions[valid] = shifted[valid]
    omitted = set(removed)
    return {**track, "history": [{**p, "cx": float(x), "cy": float(y)}
                                for i, (p, (x, y)) in enumerate(zip(history, positions, strict=True))
                                if i not in omitted]}, [history[i]["frame_index"] for i in removed]


def feature_row(track):
    width, height, _ = canonical_dimensions(track["processed_width"], track["processed_height"])
    result = extract_features(track["history"], width, height, config=FEATURES)
    return dict(feature_status=result["feature_status"], reasons=";".join(result["reasons"]),
                **{column: (result["features"] or {}).get(column) for column in FEATURE_COLUMNS})


def with_rf_provenance(table):
    return table.assign(family_id=table.source_group_id, **RF_PROVENANCE)


def write_package(folder, records, seed, profile=None):
    folder.mkdir()
    recipes, features, windows = [], [], []
    for record in records:
        track = record["track"]
        base_windows, base_rejections, tolerance = record["window_result"]
        base_features = record["rf"]
        base_indices = {w["window_index"] for w in base_windows}
        exported_arrays = set()
        variants = ("original", *VARIANTS) if profile else ("original",)
        for index, variant in enumerate(variants):
            parent = record["parent_track_id"]
            sample_id = f"{parent}:{index}"
            trial_seed = int(digest([seed, parent, index])[:16], 16)
            rng = np.random.default_rng(trial_seed)
            dropped = []
            sequence, rf = base_windows, base_features
            if index:
                for attempt in range(40):
                    derived, dropped = augment(track, variant, rng, *profile[:2])
                    rf = feature_row(derived)
                    if base_indices:
                        derived_windows, _ = window_track(
                            derived, clock_step_tolerance_seconds=tolerance)
                        sequence = [w for w in derived_windows if w["window_index"] in base_indices]
                    if len(sequence) == len(base_indices) and (
                            base_features["feature_status"] != "accepted" or rf["feature_status"] == "accepted"):
                        break
                else:
                    raise ValueError(f"{sample_id}: variant failed to preserve usable model inputs")
            fields = {key: record[key] for key in ("parent_track_id", "source_group_id",
                                                   "label", "split", "fold")}
            row = dict(**fields, sample_id=sample_id, variant=variant)
            distinct = []
            for window in sequence:
                fingerprint = window["X"].tobytes()
                if not index or fingerprint not in exported_arrays:
                    distinct.append(window)
                exported_arrays.add(fingerprint)
            recipes.append(dict(**row, rng_seed_hex=f"{trial_seed:016x}" if index else "",
                                attempt=attempt + 1 if index else 0,
                                dropped_frame_indices=json.dumps(dropped),
                                clock_step_tolerance_seconds=tolerance, window_count=len(distinct),
                                repeated_window_count=len(sequence) - len(distinct),
                                original_window_rejections=json.dumps(base_rejections) if not index else ""))
            features.append(dict(**row, **rf))
            windows.extend(dict(**row, **w, window_id=f'{sample_id}:w{w["window_index"]}') for w in distinct)
    pd.DataFrame(recipes).to_csv(folder / "recipes.csv", index=False, encoding="utf-8-sig")
    with_rf_provenance(pd.DataFrame(features)).to_csv(
        folder / "rf_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in row.items() if k != "X"} for row in windows]).to_csv(
        folder / "minirocket_windows.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(folder / "minirocket.npz", X=np.stack([w["X"] for w in windows]),
                        y=[w["label"] for w in windows], group_id=[w["source_group_id"] for w in windows],
                        sample_id=[w["window_id"] for w in windows],
                        fold=[w["fold"] for w in windows])
    return dict(tracks=dict(Counter(r["label"] for r in recipes)),
                minirocket_tracks=dict(Counter(r["label"] for r in recipes if r["window_count"])),
                minirocket_windows=dict(Counter(w["label"] for w in windows)),
                rf_accepted=dict(Counter(r["label"] for r in features if r["feature_status"] == "accepted")))


def build(root, label_map, audit, output, seed):
    if output.exists():
        raise FileExistsError(output)
    records, links = read_sources(root, label_map)
    clips = calibration_clips(audit, records)
    assign_partitions(records, clips, seed)
    for record in records:
        record["window_result"] = track_windows(record["track"])
        record["rf"] = feature_row(record["track"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".observation-") as temp:
        stage = Path(temp) / output.name
        stage.mkdir()
        train = [r for r in records if r["split"] == "train"]
        profile = observation_profile(train, clips)
        profiles, packages = {"outer_train": profile[2]}, {}
        for split in SPLITS:
            selected = [r for r in records if r["split"] == split]
            for suffix, calibration in (("real", None), ("augmented", profile)):
                name = f"{split}_{suffix}"
                packages[name] = write_package(stage / name, selected, seed, calibration)
                print(f"{name}: {packages[name]}", flush=True)
        for fold in range(5):
            selected = [r for r in train if r["fold"] != fold]
            fold_profile = observation_profile(selected, clips)
            name = f"fold_{fold}_train_augmented"
            profiles[name] = fold_profile[2]
            packages[name] = write_package(stage / name, selected, seed + fold + 1, fold_profile)
            print(f"{name}: {packages[name]}", flush=True)
        pd.DataFrame([{k: v for k, v in row.items() if k not in {"track", "window_result", "rf"}}
                      for row in records]).to_csv(stage / "sources.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(links).to_csv(stage / "group_evidence.csv", index=False, encoding="utf-8-sig")
        manifest = dict(seed=seed, source_root=str(root), label_map=str(label_map),
                        scenarios=SCENARIOS, packages=packages, fold_count=5,
                        fold_policy="train_real.fold == k is real validation for all scenarios; "
                        "train_real.fold != k is A fit; fold_k_train_augmented is B/D fit",
                        group_policy="source ID, exact center hash and original-name family; "
                        "recording sessions without provenance are unverified",
                        variants_per_track=["original", *VARIANTS], calibration=profiles,
                        validation_policy="real validation for selection; augmented val/test for stress tests",
                        derived_window_policy="discard exact repeat arrays within each parent track; "
                        "original window arrays are always retained",
                        clock_policy="only retry legacy clock rejects at 3 ms if fitted-grid residual <= 3 ms; "
                        "original timestamps preserved",
                        sequence_config=asdict(SEQUENCE), sequence_contract_id=SEQUENCE.fingerprint,
                        channels=CHANNELS, rf_config=asdict(FEATURES),
                        rf_contract_id=RF_PROVENANCE["feature_contract_id"],
                        model_input_policy="use MiniRocket NPZ directly; for RF use accepted rows only")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        stage.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    build(args.input.resolve(), args.label_map.resolve(), args.audit.resolve(), args.output.resolve(), args.seed)
