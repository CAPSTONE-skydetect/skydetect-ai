"""Audit real A tracks using the exact B preprocessing; no fitting or calibration.

Run: python -m research.diagnose_real_preprocessing
Outputs are exploratory, file-weighted diagnostics, not an independent test set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, freqz, savgol_coeffs, welch

from .feature_contract import canonical_dimensions, feature_contract_descriptor
from .features import FEATURE_COLUMNS, FeatureConfig, extract_features
from .io import write_json
from .verify_real_track import load_tracks

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"raw": "#868686", "input": "#1976a3", "smooth": "#d25a24"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def track_identity(track):
    return digest({k: track.get(k) for k in
                   ("source_video_id", "processed_width", "processed_height", "history")})


def artifact_index(root):
    index, errors = defaultdict(list), []
    for path in sorted(Path(root).glob("*/track_sequence.json")):
        try:
            track = json.loads(path.read_text(encoding="utf-8-sig"))
            index[track_identity(track)].append(path.parent)
        except (ValueError, TypeError, OSError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    return index, errors


def read_raw_match(track, candidates):
    """Require exact exported-history identity, then verify the CSV coordinates."""
    errors, matches = [], []
    h = track["history"]
    frames = [p["frame_index"] for p in h]
    dims = np.array([track["processed_width"], track["processed_height"]])
    xy = np.array([[p["cx"], p["cy"]] for p in h])
    stabilized = track.get("stabilization", {}).get("applied")
    if stabilized is None:
        return None, [], ["stabilization status unknown"]
    for folder in candidates:
        try:
            table = pd.read_csv(folder / "trajectory.csv")
            if table.frame_index.duplicated().any():
                raise ValueError("duplicate CSV frame indices")
            table = table.set_index("frame_index").loc[frames]
            if not np.allclose(table.timestamp_ms, [p["timestamp_ms"] for p in h], atol=1e-6, rtol=0):
                raise ValueError("CSV timestamp mismatch")
            names = ["compensated_x", "compensated_y"] if stabilized else ["raw_x", "raw_y"]
            if not np.allclose(np.clip(table[names].to_numpy()/dims, 0, 1), xy, atol=1e-10, rtol=0):
                raise ValueError("CSV exported coordinate mismatch")
            scale = 1920. / dims[0]
            matches.append((folder, table, table[["raw_x", "raw_y"]].to_numpy()*scale))
        except (ValueError, KeyError, OSError) as error:
            errors.append(f"{folder.name}: {error}")
    if not matches:
        return None, [], errors
    if any(not np.allclose(item[2], matches[0][2], atol=1e-8, rtol=0) for item in matches[1:]):
        return None, [], errors + ["ambiguous raw coordinates across matching runs"]
    folder, table, raw = matches[0]
    compensated = table[["compensated_x", "compensated_y"]].to_numpy()*1920./dims[0]
    return dict(folder=folder, raw=raw, compensated=compensated), [str(x[0]) for x in matches], errors


def filename_family(name):
    stem = Path(name).stem
    stem = re.sub(r"_track_?sequence$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"_(?:위에|아래)[ _]?새$", "", stem)
    return stem


def assign_provisional_groups(records):
    """Conservatively union known duplicates and explicitly marked filename hints."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, rec in enumerate(records):
        keys = [("source", rec["source_video_id"]), ("history", rec["history_hash"]),
                ("filename_hint", rec["label"], filename_family(rec["file_name"]))]
        if rec.get("video_sha256"):
            keys.append(("video_bytes", rec["video_sha256"]))
        for key in keys:
            if key[-1] in (None, ""):
                continue
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, rec in enumerate(records):
        members[find(i)].append(rec)
    for group in members.values():
        group_id = "candidate-" + digest(sorted(r["sample_id"] for r in group))[:12]
        for rec in group:
            rec.update(candidate_group_id=group_id, group_size=len(group),
                       group_review="provisional_source_hash_filename_union",
                       cross_label_group=len({r["label"] for r in group}) > 1,
                       split="unassigned_development")


def longest_contiguous(history):
    frames = np.array([p["frame_index"] for p in history])
    runs = np.split(np.arange(len(history)), np.flatnonzero(np.diff(frames) != 1)+1)
    return max(runs, key=lambda run: history[run[-1]]["timestamp_ms"]-history[run[0]]["timestamp_ms"])


def spectral_diagnostic(track, cfg):
    """Use one gap-free run; exclude SG edge transients and linear position trend."""
    history = track["history"]
    run = longest_contiguous(history)
    selected = [history[int(i)] for i in run]
    traces = []
    width, height, _ = canonical_dimensions(track["processed_width"], track["processed_height"])
    result = extract_features(selected, width, height, config=cfg, diagnostics=traces)
    if result["feature_status"] != "accepted" or not traces:
        return None
    seg = traces[0]
    edge = max(1, seg["smoothing_window"]//2)
    before = seg["resampled_xy_px"][edge:-edge]
    after = seg["smoothed_xy_px"][edge:-edge]
    step = seg["step_seconds"]
    if len(before) < 16 or (len(before)-1)*step < 1.:
        return None
    # A low source rate cannot acquire new frequencies by upsampling to 30 Hz.
    source_nyquist = .5*result["quality"]["nominal_fps"]
    if source_nyquist < 10.:
        return None
    values = []
    for points in (before, after):
        f, power = welch(detrend(points, axis=0), fs=1/step, nperseg=min(128, len(points)),
                         axis=0, detrend="constant")
        values.append(power.sum(axis=1))
    band = (f >= 3.) & (f <= 10.)
    df = f[1]-f[0]
    energies = [float(p[band].sum()*df) for p in values]
    ratio = energies[1]/energies[0] if energies[0] > 1e-10 else None
    return dict(frequency=f, before=values[0], after=values[1], band_before=energies[0],
                band_after=energies[1], band_retention=ratio, seconds=(len(before)-1)*step,
                start_frame=selected[0]["frame_index"], end_frame=selected[-1]["frame_index"])


def segment_metrics(segments):
    before_length = after_length = squared = duration = 0.
    removed_peak, count = 0., 0
    for seg in segments:
        before, after = seg["resampled_xy_px"], seg["smoothed_xy_px"]
        residual = np.linalg.norm(before-after, axis=1)
        before_length += np.linalg.norm(np.diff(before, axis=0), axis=1).sum()
        after_length += np.linalg.norm(np.diff(after, axis=0), axis=1).sum()
        squared += float(np.sum(residual**2))
        count += len(before)
        removed_peak = max(removed_peak, float(residual.max()))
        duration += float(seg["time_seconds"][-1]-seg["time_seconds"][0])
    return dict(resampled_path_px=float(before_length), smoothed_path_px=float(after_length),
                path_retention=float(after_length/before_length) if before_length > 1e-8 else None,
                smoothing_displacement_rms_px=float(np.sqrt(squared/count)) if count else None,
                smoothing_displacement_max_px=removed_peak if count else None,
                retained_seconds=duration)


def plot_observations(ax, time, xy, frames, **kwargs):
    for indices in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 1)+1):
        ax.plot(xy[indices, 0], xy[indices, 1], **kwargs)
        kwargs.pop("label", None)


def plot_case(record, track, segments, raw, spectral, path):
    width, height, _ = canonical_dimensions(track["processed_width"], track["processed_height"])
    h = track["history"]
    xy = np.array([[p["cx"]*width, p["cy"]*height] for p in h])
    frames = np.array([p["frame_index"] for p in h])
    t = np.array([p["timestamp_ms"] for p in h])/1000.
    t -= t[0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    for ax in axes[0]:
        if raw is not None:
            plot_observations(ax, t, raw["raw"], frames, color=COLORS["raw"], lw=1.2, label="A overlay / raw")
        plot_observations(ax, t, xy, frames, color=COLORS["input"], lw=1., marker=".", ms=2,
                          label="B input / exported centers")
        for i, seg in enumerate(segments):
            smooth = seg["smoothed_xy_px"]
            ax.plot(*smooth.T, color=COLORS["smooth"], lw=1.3, alpha=.9,
                    label="B smoothed path" if i == 0 else None)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="canonical x (px)", ylabel="canonical y (px)")
        ax.grid(alpha=.2)
    axes[0, 0].set(xlim=(0, width), ylim=(height, 0), title="Full frame (same image aspect ratio)")
    axes[0, 1].invert_yaxis()
    # Keep a usable plotting area for near-vertical tracks without stretching geometry.
    axes[0, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 1].set_title("Same coordinates, zoomed to the observed path")
    axes[0, 1].legend(fontsize=8)
    if raw is None:
        axes[0, 0].text(.03, .03, "Raw A export unavailable; CMC effect unverified",
                        transform=axes[0, 0].transAxes, fontsize=8)
    speed_ax, residual_ax = axes[1]
    for i, seg in enumerate(segments):
        grid = seg["time_seconds"]
        mid = (grid[1:]+grid[:-1])/2
        raw_speed = np.linalg.norm(np.diff(seg["resampled_xy_px"], axis=0), axis=1)/seg["step_seconds"]
        speed_ax.plot(mid, raw_speed, color=COLORS["input"], alpha=.65,
                      label="Resampled position differences" if i == 0 else None)
        speed_ax.plot(mid, seg["speed_px_s"], color=COLORS["smooth"],
                      label="Exact B derivative speed" if i == 0 else None)
        residual = seg["resampled_xy_px"]-seg["smoothed_xy_px"]
        residual_ax.plot(grid, residual[:, 0], color="#6a5099", label="x residual" if i == 0 else None)
        residual_ax.plot(grid, residual[:, 1], color="#2c9560", label="y residual" if i == 0 else None)
    speed_ax.set(title="Apparent speed (not world speed)", xlabel="seconds", ylabel="canonical px/s")
    residual_ax.set(title="Removed by smoothing: motion + tracking error", xlabel="seconds", ylabel="canonical px")
    for ax in axes[1]:
        ax.grid(alpha=.2)
        if segments:
            ax.legend(fontsize=8)
    psd_ax, support_ax = axes[2]
    if spectral is not None:
        for key, label, color in (("before", "Before smoothing", COLORS["input"]),
                                   ("after", "After smoothing", COLORS["smooth"])):
            psd_ax.semilogy(spectral["frequency"], np.maximum(spectral[key], 1e-15), label=label, color=color)
        psd_ax.axvspan(3, 10, color="#d5b047", alpha=.12)
        psd_ax.legend(fontsize=8)
        psd_ax.set_title(f"Position PSD / gap-free frames {spectral['start_frame']}..{spectral['end_frame']}")
    else:
        psd_ax.text(.1, .5, "No sufficiently long gap-free spectral interval", transform=psd_ax.transAxes)
        psd_ax.set_title("Position PSD unavailable")
    psd_ax.set(xlabel="Hz", ylabel="canonical px^2 / Hz", xlim=(0, 15))
    for i, seg in enumerate(segments):
        support_ax.step(seg["time_seconds"][1:-1], seg["valid_turn"].astype(float), where="mid",
                        color="#3f6e83", label="Turn supported by B gate" if i == 0 else None)
    gap = np.flatnonzero(np.diff(frames) > 1)
    for i in gap:
        support_ax.axvspan(t[i], t[i+1], color="#e4a54a", alpha=.4)
    support_ax.set(title="Heading support (orange: missing frames)", xlabel="seconds",
                   ylabel="supported", ylim=(-.1, 1.15))
    if segments:
        support_ax.legend(fontsize=8)
    fig.suptitle(f"{record['sample_id']} | label={record['label']} | prediction={record.get('prediction', 'unavailable')}\n"
                 f"{len(h)} observations / {t[-1]:.2f}s | raw match={raw is not None} | no behavior annotation",
                 fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def predict_baseline(track, expected_features, classifier, rule_filter):
    from ai_server.schemas import TrackSequence
    from ai_server.services.feature_core import build_feature_vector
    result = build_feature_vector(TrackSequence.model_validate(track))
    if expected_features is not None:
        actual = result.feature_vector.features
        if actual is None or not np.allclose([getattr(actual, k) for k in FEATURE_COLUMNS],
                                           [expected_features[k] for k in FEATURE_COLUMNS], atol=1e-7, rtol=1e-7):
            raise ValueError("Research/runtime feature parity failed")
    filtered = rule_filter.apply(result.feature_vector)
    if not filtered.passed:
        return dict(prediction="uncertain", prediction_confidence=0., rule_passed=False,
                    rejection_reason=filtered.reject_reason)
    label, confidence = classifier.predict(result.feature_vector)
    return dict(prediction=label, prediction_confidence=float(confidence), rule_passed=True,
                rejection_reason="confidence_below_threshold" if label == "uncertain" else "")


def run(input_dir, artifact_dir, output_dir, model_path=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    cfg = FeatureConfig()
    index, artifact_errors = artifact_index(artifact_dir)
    classifier = rule_filter = None
    model_provenance = None
    if model_path is not None:
        import joblib
        from ai_server.services.classifier import RFClassifier
        from ai_server.services.rule_filter import RuleFilter
        classifier, rule_filter = RFClassifier(str(model_path)), RuleFilter()
        model_provenance = joblib.load(model_path).get("provenance", {})
        if classifier.feature_names != FEATURE_COLUMNS:
            raise ValueError("Model feature names/order do not match v4")
    records, feature_rows, video_hashes, errors = [], [], {}, []
    for label in ("bird", "drone"):
        for number, path in enumerate(sorted((Path(input_dir)/label).glob("*.json")), 1):
            try:
                tracks = load_tracks(path)
            except (OSError, ValueError) as error:
                errors.append(dict(path=str(path), error=str(error)))
                continue
            for track_number, track in enumerate(tracks):
                sample_id = f"{label[0].upper()}{number:03d}" + (f"_{track_number}" if len(tracks) > 1 else "")
                rec = dict(sample_id=sample_id, label=label, file_name=path.name,
                           source_path=str(path.resolve()), file_sha256=file_digest(path),
                           source_video_id=track.get("source_video_id"), track_id=track.get("track_id"),
                           history_hash=digest(track["history"]), behavior="unreviewed",
                           raw_cmc_offset_rms_px=None, cmc_export_clamp_rms_px=None)
                width, height, _ = canonical_dimensions(track["processed_width"], track["processed_height"])
                segments = []
                result = extract_features(track["history"], width, height, config=cfg, diagnostics=segments)
                rec.update(feature_status=result["feature_status"], reasons=";".join(result["reasons"]),
                           **result["quality"], processed_width=track["processed_width"],
                           processed_height=track["processed_height"],
                           stabilization_applied=track.get("stabilization", {}).get("applied"))
                if result["feature_status"] != "accepted":
                    records.append(rec)
                    continue
                history = track["history"]
                xy = np.array([[p["cx"]*width, p["cy"]*height] for p in history])
                rec.update(x_span=float(np.ptp(xy[:, 0])/width), y_span=float(np.ptp(xy[:, 1])/height),
                           **segment_metrics(segments))
                raw, matched_paths, raw_errors = read_raw_match(track, index.get(track_identity(track), []))
                rec.update(raw_match=raw is not None, matched_run_paths=matched_paths, raw_match_errors=raw_errors)
                if raw is not None:
                    cmc_offset = raw["compensated"]-raw["raw"]
                    rec["raw_cmc_offset_rms_px"] = float(np.sqrt(np.mean(np.sum(cmc_offset**2, axis=1))))
                    rec["cmc_export_clamp_rms_px"] = float(np.sqrt(np.mean(np.sum((xy-raw["compensated"])**2, axis=1)))) if rec["stabilization_applied"] else None
                    metadata_path = raw["folder"]/"metadata.json"
                    if metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                        source = Path(metadata.get("source_video_path", ""))
                        if source.is_file():
                            key = str(source.resolve())
                            if key not in video_hashes:
                                video_hashes[key] = file_digest(source)
                            rec["video_sha256"] = video_hashes[key]
                spec = spectral_diagnostic(track, cfg)
                rec.update(spectral_valid=spec is not None,
                           spectral_seconds=spec["seconds"] if spec else None,
                           band_3_10_before_px2=spec["band_before"] if spec else None,
                           band_3_10_after_px2=spec["band_after"] if spec else None,
                           band_3_10_power_retention=spec["band_retention"] if spec else None)
                for name, variant in (("default", cfg), ("no_smoothing", replace(cfg, smoothing_seconds=0.)),
                                      ("short_smoothing", replace(cfg, smoothing_seconds=.15))):
                    features = result if name == "default" else extract_features(history, width, height, config=variant)
                    feature_rows.append(dict(sample_id=sample_id, label=label, variant=name,
                                             feature_status=features["feature_status"],
                                             **(features["features"] or {}), **features["quality"]))
                if classifier is not None:
                    rec.update(predict_baseline(track, result["features"], classifier, rule_filter))
                    rec["correct"] = rec["prediction"] == label
                trace_data = {}
                for i, segment in enumerate(segments):
                    trace_data.update({f"segment_{i}_{k}": v for k, v in segment.items()})
                trace_data["b_input_xy_px"] = xy
                trace_data["frame_index"] = np.array([p["frame_index"] for p in history])
                trace_data["timestamp_ms"] = np.array([p["timestamp_ms"] for p in history])
                if raw:
                    trace_data["a_raw_xy_px"] = raw["raw"]
                    trace_data["a_compensated_unclipped_xy_px"] = raw["compensated"]
                np.savez_compressed(output / "traces" / f"{sample_id}.npz", **trace_data)
                plot_case(rec, track, segments, raw, spec, output / "plots" / f"{sample_id}.png")
                records.append(rec)
                print(f"{sample_id}: accepted; raw={raw is not None}; prediction={rec.get('prediction', 'not_run')}", flush=True)
    if not records:
        raise ValueError("No input tracks found")
    assign_provisional_groups(records)
    table = pd.DataFrame(records)
    table.to_csv(output / "track_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(output / "feature_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output / "manifest.json", records)
    if classifier is not None:
        table[table.prediction.notna() & (table.prediction != table.label)].to_csv(
            output / "errors_and_abstentions.csv", index=False, encoding="utf-8-sig")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    run_info = dict(created_at=datetime.now(timezone.utc).isoformat(), commit=commit,
                    feature_config=asdict(cfg), contract=feature_contract_descriptor(cfg.fingerprint, target_fps=cfg.target_fps),
                    code_sha256={p.name: file_digest(p) for p in (Path(__file__), ROOT/"research/features.py")},
                    model_path=str(model_path) if model_path else None,
                    model_sha256=file_digest(model_path) if model_path else None,
                    model_provenance=model_provenance,
                    model_feature_config_matches=(model_provenance.get("feature_config_id") == cfg.fingerprint) if model_provenance else None,
                    prediction_policy="Existing RFClassifier and RuleFilter defaults; no retraining; no UI overrides",
                    input_errors=errors, artifact_errors=artifact_errors,
                    spectral_policy="Longest gap-free observed run, reprocessed with B, SG edges trimmed, linear detrend; >=1s interior; source Nyquist >=10Hz",
                    grouping_policy="Provisional union of source ID, exact history, video bytes and filename-family hints; not independent-flight verification",
                    evaluation_status="Exploratory existing-model replay; no independent test or CV; file weighted")
    write_json(output / "run.json", run_info)
    write_report(table, output, cfg)
    return table


def write_report(table, output, cfg):
    valid = table[table.feature_status == "accepted"]
    summary = valid.groupby("label")[["duration_seconds", "x_span", "y_span", "path_retention",
                                       "smoothing_displacement_rms_px", "band_3_10_power_retention",
                                       "heading_valid_fraction", "raw_cmc_offset_rms_px"]].median()
    summary.to_csv(output / "class_medians.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    fields = [("duration_seconds", "Observed duration (s)"), ("x_span", "Horizontal span / width"),
              ("path_retention", "Path length retained by smoothing"),
              ("band_3_10_power_retention", "3-10 Hz position power retained"),
              ("heading_valid_fraction", "B heading support fraction"),
              ("raw_cmc_offset_rms_px", "Raw-to-CMC offset RMS (canonical px)")]
    for ax, (field, title) in zip(axes.flat, fields):
        for i, label in enumerate(("bird", "drone")):
            values = valid.loc[valid.label == label, field].dropna().to_numpy()
            ax.scatter(i+np.linspace(-.13, .13, len(values)), values, s=24, alpha=.65)
            if len(values):
                ax.plot([i-.2, i+.2], [np.median(values)]*2, color="black", lw=2)
        ax.set(xticks=[0, 1], xticklabels=["bird", "drone"], title=title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Real A -> B preprocessing audit | each dot is a file, not an independent flight", fontsize=13)
    fig.savefig(output/"overview.png", dpi=150)
    plt.close(fig)
    f, gain = freqz(savgol_coeffs(9, 2), worN=1024, fs=30.)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(f, abs(gain))
    ax.set(xlabel="Hz", ylabel="Position amplitude retained", ylim=(0, 1.1),
           title="Default 9-point quadratic SG at 30 Hz (interior only)")
    ax.grid(alpha=.3)
    fig.savefig(output/"filter_response.png", dpi=140)
    plt.close(fig)
    lines = ["# 실제 A 궤적 전처리 진단", "", "## 진단 범위", "",
             f"- 입력 {len(table)}개, 특징 계산 성공 {len(valid)}개.",
             f"- 원본 A CSV 일치 확인 {int(valid.raw_match.sum())}개. 나머지는 CMC 이전 영향을 확인할 수 없다.",
             f"- 임시 원본 그룹 {table.candidate_group_id.nunique()}개. 촬영 세션 독립성은 아직 미확인이다.",
             "- 새/드론 정답은 입력 폴더 기준이며, 비행 행동은 수동 주석하지 않았다.",
             "- 현재 모델 재실행은 개발 자료에 대한 파일 단위 기술 통계다. 독립 test, 그룹 CV, 재학습 결과가 아니다.",
             "- A 추적, 물리 시뮬레이터, 특징 수식 및 기본 전처리 설정은 변경하지 않았다.", "",
             "## 먼저 볼 결과", "",
             f"- 경로 길이 유지율 중앙값: 새 {summary.loc['bird', 'path_retention']:.1%}, 드론 {summary.loc['drone', 'path_retention']:.1%}. 전체 경로 길이가 평활화로 크게 줄어드는 상황은 중앙값에서 보이지 않는다.",
             f"- 3~10Hz 위치 진동 에너지 유지율 중앙값: 새 {summary.loc['bird', 'band_3_10_power_retention']:.1%}, 드론 {summary.loc['drone', 'band_3_10_power_retention']:.1%}. 세부 진동은 양쪽 클래스에서 약해진다. 이는 진폭 유지율과 다른 값이다.",
             "- 위 두 결과는 전체 모양이 비슷해 보여도 세부 시간 패턴은 달라질 수 있음을 보여준다. 하지만 해당 패턴이 날갯짓인지 추적 잡음인지, 오분류 원인인지는 아직 확정하지 않았다.",
             "- CMC 보정량이 큰 사례는 원본 영상과 함께 점검해야 한다. 보정량이 크다는 사실 자체는 A 오류가 아니다.", "",
             "![전체 진단](overview.png)", "", "## 클래스별 중앙값", "",
             "| 항목 | bird | drone |", "|---|---:|---:|"]
    for column in summary.columns:
        lines.append(f"| {column} | {summary.loc['bird', column]:.4f} | {summary.loc['drone', column]:.4f} |")
    lines += ["", "경로 유지율과 3~10Hz 에너지 유지율은 모두 같은 입력의 전후 비율이다. 낮아졌다는 사실만으로 새의 유효 신호를 제거했다고 단정할 수 없다. 제거된 성분에는 실제 운동, 중심 추적 오차, CMC 잔차가 섞인다.",
              "스펙트럼은 가장 긴 프레임 누락 없는 구간에서만 산출하고, 필터 가장자리와 선형 추세를 제외했다. 3~10Hz는 진단용 대역이며 날갯짓 확정 대역이 아니다. 경로 전체와 스펙트럼의 측정 구간은 다를 수 있다.",
              "raw-to-CMC offset은 보정량이며 보정 오류의 정답이 아니다. 올바른 보정인지 판단하려면 영상/수동 주석 확인이 필요하다.",
              "heading support는 B가 방향 계산에 사용한 구간의 비율이다. 값이 낮으면 방향 특징은 일부 구간만 반영한다.", "",
              "## 기존 B/C 모델 재실행", ""]
    if "prediction" in valid:
        confusion = pd.crosstab(valid.label, valid.prediction).reindex(index=["bird", "drone"],
                    columns=["bird", "drone", "uncertain"], fill_value=0)
        confusion.to_csv(output/"confusion_matrix.csv", encoding="utf-8-sig")
        lines += ["| 정답 | bird 예측 | drone 예측 | 보류 | 전체 기준 recall |", "|---|---:|---:|---:|---:|"]
        recalls = []
        for label in ("bird", "drone"):
            counts = confusion.loc[label]
            recall = counts[label]/counts.sum()
            recalls.append(recall)
            lines.append(f"| {label} | {counts['bird']} | {counts['drone']} | {counts['uncertain']} | {recall:.3f} |")
        coverage = float((valid.prediction != "uncertain").mean())
        correct = float((valid.prediction == valid.label).mean())
        lines += ["", f"전체 정답 비율(보류도 미정답): {correct:.3f}; 판정률: {coverage:.3f}; 클래스 recall 평균: {np.mean(recalls):.3f}.",
                  "표시는 기존 RF 점수이며 현실 확률로 보정되지 않았다. UI에서 품질 임계값을 변경했다면 이번 기본 설정 재실행과 결과가 다를 수 있다.",
                  "기존 모델에 기본/무평활화/짧은 평활화 특징을 무분별하게 교체해 넣어 성능을 비교하지 않았다. 다른 전처리는 학습 분포도 달라져 공정한 모델 비교에 재학습이 필요하다."]
    else:
        lines.append("모델 재실행을 생략했다.")
    lines += ["", "## 파일별 탐색", "", "[갤러리 열기](index.html)", "",
              "각 개별 그림은 왼쪽 위부터 전체 화면 경로, 확대 경로, 화면상 속도, 평활화 잔차, 위치 주파수 분포, 방향 계산에 채택된 구간을 보여준다. 오른쪽 위 확대도 두 축의 비율을 유지한다. 선은 누락 프레임에서 끊으며, 주황색 평활화 경로에는 B가 허용한 짧은 gap 보간이 포함될 수 있다.", "",
              "- `track_diagnostics.csv`: 그룹, 입력 품질, 전처리 변화, 현재 판정.",
              "- `feature_comparison.csv`: 기본 0.3초 / 0초 / 0.15초 설정의 특징 및 방향 지지율. 0.15초도 실제 창은 홀수 샘플 수로 반올림된다.",
              "- `traces/*.npz`: 실제 B 계산에서 꺼낸 좌표, 속도, 방향 gate, 시간 및 원본 매칭 좌표.",
              "- `manifest.json`: 입력 해시, 정확히 매칭된 A 실행 경로, 잠정 그룹과 검토 상태.",
              "- `errors_and_abstentions.csv`: 현행 모델의 오분류/보류 목록(모델 사용 시).", "",
              "## 해석 및 다음 단계", "",
              "1. 갤러리에서 실제 물결/활공/호버/반전을 수동 확인하고 `behavior=unreviewed`에 검토 결과를 추가한다. 파형만으로 자동 행동 정답을 만들지 않는다.",
              "2. CMC 전후와 B 평활화 전후 중 어느 단계에서 모양이 달라지는지 확인한다. 원본 없는 파일의 CMC 효과는 미확인으로 유지한다.",
              "3. source ID, 영상 바이트 해시, 파일명 계열을 묶은 잠정 그룹을 원본 촬영 정보로 확정한다. 같은 촬영의 다른 인코딩/잘라낸 영상은 해시만으로 찾지 못한다.",
              "4. 길이와 촬영 비율 편향을 통제한 그룹 평가로 현재 RF와 시간 구조를 보존한 표현을 비교한다.",
              "5. 진단 결과만으로 시뮬레이터 물리를 조정하거나 평활화를 제거하지 않는다. 실제 새 신호와 드론 추적 지터 양쪽의 영향을 확인한다.", "",
              "![필터 응답](filter_response.png)", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for rec in table.to_dict("records"):
        sid = rec["sample_id"]
        image_path = output/"plots"/f"{sid}.png"
        if not image_path.exists():
            continue
        desc = html.escape(f"{sid} | {rec['file_name']} | {rec.get('prediction', 'not_run')} | group {rec['candidate_group_id']}")
        cards.append(f'<article><h2>{desc}</h2><a href="plots/{sid}.png"><img loading="lazy" src="plots/{sid}.png" alt="{desc}"></a></article>')
    error_link = ' · <a href="errors_and_abstentions.csv">오분류·보류</a>' if "prediction" in valid else ""
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A to B preprocessing audit</title><style>body{font:16px system-ui;margin:24px;background:#fff;color:#182126}main{max-width:1500px;margin:auto}h2{font-size:16px;overflow-wrap:anywhere}article{border-top:1px solid #ccc;padding:20px 0}img{width:100%;height:auto}a{color:#076d9a}</style><main><h1>실제 A → B 전처리 진단</h1><p>회색: A 원본 / 파랑: B 입력 / 주황: B 평활화. 행동 정답은 미주석 상태입니다.</p><p><a href="report.md">진단 보고서</a> · <a href="track_diagnostics.csv">진단 수치</a>' + error_link + '</p><img src="overview.png" alt="전체 진단">' + ''.join(cards) + '</main></html>'
    (output/"index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"research/data/tracks")
    parser.add_argument("--artifacts", type=Path, default=ROOT/"artifacts/manual_tracks")
    parser.add_argument("--output", type=Path, default=ROOT/"research/output/a_preprocessing_audit")
    parser.add_argument("--model", type=Path, default=ROOT/"models/rf_classifier.pkl")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    run(args.input, args.artifacts, args.output, None if args.skip_model else args.model)


if __name__ == "__main__":
    main()
