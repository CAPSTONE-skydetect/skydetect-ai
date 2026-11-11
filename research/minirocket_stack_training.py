"""Opt-in portable Stack training and inference. Existing services are unchanged."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ai_server.services import sequence_model as seq
from research.minirocket_inference import decision_margins as original
from research.minirocket_stack_data import collection
from research.trajectory_sequence import SequenceConfig, window_track

VERSION = "trajectory-sequence-dual-scale-stack-1.0.0"
SCALE_VERSION = "trajectory-sequence-frame-scale-1.0.0"
GAIN = 400.0


def scale_windows(x, scales):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (len(x),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Missing or invalid per-window scales")
    out = x.copy()
    out[:, :2] = (x[:, :2].astype(np.float64) * scales[:, None, None] * GAIN).astype(np.float32)
    out[:, 2:] = np.diff(out[:, :2], axis=2, prepend=out[:, :2, :1])
    return out


def scales_for(data, metadata):
    meta = metadata.set_index("sample_id")
    if not meta.index.is_unique:
        raise ValueError("Duplicate scale metadata sample IDs")
    if not set(data["sample_id"]) <= set(meta.index):
        raise ValueError("Missing normalization_scale rows")
    rows = meta.loc[data["sample_id"]]
    if not (np.array_equal(rows.label.to_numpy(), data["y"]) and
            np.array_equal(rows.source_group_id.to_numpy(), data["group_id"])):
        raise ValueError("Scale metadata label/source mismatch")
    return rows.normalization_scale.to_numpy()


def margins(model, x, scales):
    if (model["contract_version"] != VERSION or model["classes"] != seq.LABELS or
            model["meta"].classes_.tolist() != seq.LABELS or
            model["input_contract_version"] != seq.CONTRACT_VERSION or model["input_contract_id"] != seq.CONTRACT_ID):
        raise ValueError("Stack model contract mismatch")
    frame = model["framescale"]
    if (frame["contract_version"] != SCALE_VERSION or frame["scale_gain"] != GAIN or
            frame["input_contract_version"] != seq.CONTRACT_VERSION or frame["input_contract_id"] != seq.CONTRACT_ID):
        raise ValueError("Frame-scale member contract mismatch")
    compatible = dict(frame, contract_version=seq.CONTRACT_VERSION, contract_id=seq.CONTRACT_ID)
    scores = np.column_stack([original(model["baseline"], x), original(compatible, scale_windows(x, scales))])
    return model["meta"].decision_function(model["meta_scaler"].transform(scores))


class StackMiniRocketClassifier:
    def __init__(self, path):
        self.model = joblib.load(path)  # Trusted local checkpoints only.
        margins(self.model, np.zeros((1, 4, 60), dtype=np.float32), [.0025])

    def predict(self, track):
        from ai_server.services.classifier import SequencePrediction
        try:
            windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        except (ValueError, KeyError, TypeError) as exc:
            return SequencePrediction("uncertain", None, "invalid_input", str(exc))
        rejects = dict(Counter(row["reason"] for row in rejected))
        if not windows:
            return SequencePrediction("uncertain", None, "insufficient_observation", window_rejections=rejects)
        scores = margins(self.model, np.stack([w["X"] for w in windows]), [w["normalization_scale"] for w in windows])
        mean = float(scores.mean())
        return SequencePrediction("drone" if mean >= 0 else "bird", mean, window_scores=scores.tolist(),
                                  window_starts_s=[w["relative_start_s"] for w in windows], window_rejections=rejects)


def check_splits(data):
    for key in ("train", "validation"):
        if key not in data or set(data[key]["y"]) != set(seq.LABELS):
            raise ValueError(f"Both labels need supported windows in {key}")
    seen_groups, seen_samples = set(), set()
    for name, part in data.items():
        groups, samples = set(part["group_id"]), set(part["sample_id"])
        if groups & seen_groups or samples & seen_samples or len(samples) != len(part["y"]):
            raise ValueError(f"Group/sample split overlap or duplicate in {name}")
        seen_groups |= groups
        seen_samples |= samples
        tracks = pd.DataFrame(dict(group=part["group_id"], label=part["y"], track=seq.track_ids(part["sample_id"])))
        if (tracks.groupby("group").label.nunique() > 1).any():
            raise ValueError("Mixed-label source group: inherited group metric undefined")
        if (tracks.groupby("track").group.nunique() > 1).any():
            raise ValueError("Track crosses source groups")


def cv_folds(train):
    splitter = StratifiedGroupKFold(seq.CV_SPLITS, shuffle=True, random_state=seq.RANDOM_STATE)
    folds = list(splitter.split(train["X"], train["y"], train["group_id"]))
    for fit, held in folds:
        if set(train["group_id"][fit]) & set(train["group_id"][held]):
            raise ValueError("OOF group overlap")
        if any(set(train["y"][idx]) != set(seq.LABELS) for idx in (fit, held)):
            raise ValueError("OOF fold missing a class; supply more eligible source groups")
    return folds


def load_data(kind, path, *, include_test=False):
    if kind == "collection":
        return collection(path, include_test=include_test)
    _, train, val = seq.load_package(path)
    metadata = pd.read_csv(path / "real_metadata.csv")
    return {"train": train["real_only"], "validation": val}, metadata, [], seq.file_hash(path / "dataset_manifest.json")


def fit(data, meta, digest, output):
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    check_splits(data)
    train, val = data["train"], data["validation"]
    if "sample_weight" not in train:
        train = {**train, "sample_weight": seq.balanced_group_weights(train["y"], train["group_id"])}
    folds = cv_folds(train)
    scaled = {**train, "X": scale_windows(train["X"], scales_for(train, meta))}
    val_scales = scales_for(val, meta)
    output.mkdir(parents=True, exist_ok=False)
    (output / "state.json").write_text('{"status":"started"}\n')
    oof = np.full((len(train["X"]), 2), np.nan)
    fold_ids = np.full(len(oof), -1)
    members = []
    for col, part in enumerate((train, scaled)):
        alpha, audit = seq.choose_alpha(part)
        audit.to_csv(output / f"alpha_cv_{col}.csv", index=False)
        for fold, (fitting, held) in enumerate(folds):
            x = part["X"]
            weight = seq.balanced_group_weights(train["y"][fitting], train["group_id"][fitting])
            rocket, scaler = seq.fit_representation(x[fitting])
            ridge = RidgeClassifier(alpha=alpha).fit(seq.transform(rocket, scaler, x[fitting]),
                                                     train["y"][fitting], sample_weight=weight)
            oof[held, col] = ridge.decision_function(seq.transform(rocket, scaler, x[held]))
            fold_ids[held] = fold
        member = seq.fit_arm({"real_only": part}, "real_only", alpha)
        member.update(package_manifest_sha256=digest, validation_used_for_fit=False)
        members.append(member)
    if not np.isfinite(oof).all() or (fold_ids < 0).any():
        raise ValueError("Incomplete OOF margins")
    table = pd.DataFrame(dict(sample_id=train["sample_id"], track_id=seq.track_ids(train["sample_id"]),
                              group_id=train["group_id"], label=train["y"], fold=fold_ids,
                              baseline_margin=oof[:, 0], framescale_margin=oof[:, 1]))
    tracks = table.groupby(["group_id", "track_id", "label", "fold"], as_index=False)[
        ["baseline_margin", "framescale_margin"]].mean()
    z = tracks[["baseline_margin", "framescale_margin"]].to_numpy()
    scaler = StandardScaler().fit(z)
    head = LogisticRegression(C=1.0, random_state=seq.RANDOM_STATE).fit(
        scaler.transform(z), tracks.label, sample_weight=seq.balanced_group_weights(tracks.label, tracks.group_id))
    base, frame = members
    frame.update(contract_version=SCALE_VERSION, contract_id="frame-scale-v1", scale_gain=GAIN,
                 input_contract_version=seq.CONTRACT_VERSION, input_contract_id=seq.CONTRACT_ID)
    model = {k: base[k] for k in ("classes", "channels", "alpha", "train_groups", "train_windows", "package_manifest_sha256")}
    model.update(baseline=base, framescale=frame, meta=head, meta_scaler=scaler, contract_version=VERSION,
                 contract_id="dual-scale-stack-v1", input_contract_version=seq.CONTRACT_VERSION,
                 input_contract_id=seq.CONTRACT_ID, validation_used_for_fit=False, arm="real_only_dual_scale_stack",
                 meta_tracks=len(tracks), member_alpha_policy="train-only 3-fold group CV, fixed for OOF",
                 source_sha256=seq.file_hash(Path(__file__)))
    _, val_tracks, val_groups = seq.aggregate_scores(val, margins(model, val["X"], val_scales))
    table.to_csv(output / "oof_windows.csv", index=False)
    tracks.to_csv(output / "oof_tracks.csv", index=False)
    val_tracks.to_csv(output / "validation_tracks.csv", index=False)
    checkpoint = output / "stack.joblib"
    joblib.dump(model, checkpoint)
    return dict(checkpoint_sha256=seq.file_hash(checkpoint), member_alphas=[m["alpha"] for m in members],
                train_groups=sorted(set(train["group_id"])), validation_groups=sorted(set(val["group_id"])),
                validation_track_metrics=seq.metrics(val_tracks), validation_group_metrics=seq.metrics(val_groups),
                meta_tracks=len(tracks), meta_C=1.0, seed=seq.RANDOM_STATE, n_kernels=seq.N_KERNELS,
                contract=asdict(SequenceConfig()), contract_id=seq.CONTRACT_ID, labels=seq.LABELS,
                versions={**seq.library_versions(), "pandas": pd.__version__}, python=platform.python_version(),
                os=platform.platform(), fit_source_sha256=seq.file_hash(Path(__file__)), dataset_sha256=digest,
                test_evaluated=False, oof_is_independent=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "train", "evaluate"):
        p = subs.add_parser(name)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--real-package", type=Path)
        group.add_argument("--collection", type=Path)
        if name == "train":
            p.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--split", choices=("validation", "test"), default="validation")
    p = subs.add_parser("predict")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--track", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "predict":
        from ai_server.schemas import TrackSequence
        result = asdict(StackMiniRocketClassifier(args.model).predict(TrackSequence.model_validate_json(args.track.read_bytes())))
    else:
        kind = "collection" if args.collection else "package"
        path = args.collection or args.real_package
        data, meta, audit, digest = load_data(kind, path, include_test=getattr(args, "split", None) == "test")
        check_splits(data)
        for part in data.values():
            scale_windows(part["X"], scales_for(part, meta))
        cv_folds(data["train"])
        result = dict(dataset_sha256=digest, window_counts={k: len(v["y"]) for k, v in data.items()}, audit=audit)
        if args.cmd == "train":
            result.update(fit(data, meta, digest, args.output))
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True, check=False)
            result["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
            (args.output / "status.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (args.output / "state.json").write_text('{"status":"completed"}\n')
        if args.cmd == "evaluate":
            if args.split not in data or set(data[args.split]["y"]) != set(seq.LABELS):
                raise ValueError("Evaluation split missing or lacks both classes")
            part = data[args.split]
            model = StackMiniRocketClassifier(args.model).model
            _, tracks, groups = seq.aggregate_scores(part, margins(model, part["X"], scales_for(part, meta)))
            result.update(split=args.split, model_sha256=seq.file_hash(args.model),
                          tracks=seq.metrics(tracks), groups=seq.metrics(groups))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
