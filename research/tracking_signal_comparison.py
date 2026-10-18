"""Export time signals from the unchanged B computation for matched A/GT inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.features import FeatureConfig, _weighted_quantile, extract_features
from research.io import write_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trace(track):
    runtime = build_feature_vector(track)
    if not runtime.accepted:
        raise ValueError(f"B rejected input: {runtime.reasons}")
    diagnostics = []
    traced = extract_features(
        [p.model_dump() for p in track.history], runtime.canonical_width,
        runtime.canonical_height, diagnostics=diagnostics,
    )
    actual = runtime.feature_vector.features.model_dump()
    if traced["feature_status"] != "accepted" or any(
        not np.isclose(actual[k], traced["features"][k], rtol=1e-12, atol=1e-9)
        for k in actual
    ):
        raise ValueError("Diagnostic arrays do not reproduce the actual B features")
    origin = track.history[0].timestamp_ms / 1000.0
    source_times = np.asarray(
        [point.timestamp_ms / 1000.0 - origin for point in track.history], dtype=float,
    )
    source_frames = np.asarray([point.frame_index for point in track.history], dtype=float)
    signals = []
    for segment_id, diagnostic in enumerate(diagnostics):
        indices = np.asarray(diagnostic["observed_indices"], dtype=int)
        grid = np.asarray(diagnostic["time_seconds"], dtype=float)
        speed = np.asarray(diagnostic["speed_px_s"], dtype=float)
        acceleration = np.abs(np.diff(speed)) / float(diagnostic["step_seconds"])
        signals.append({
            "segment_id": segment_id,
            "source_first_frame": int(source_frames[indices[0]]),
            "source_last_frame": int(source_frames[indices[-1]]),
            "smoothing_window_points": int(diagnostic["smoothing_window"]),
            "step_seconds": float(diagnostic["step_seconds"]),
            "time_seconds": (grid + origin).tolist(),
            "source_frame_position": np.interp(
                grid, source_times[indices], source_frames[indices],
            ).tolist(),
            "resampled_xy_px": np.asarray(diagnostic["resampled_xy_px"]).tolist(),
            "smoothed_xy_px": np.asarray(diagnostic["smoothed_xy_px"]).tolist(),
            "velocity_time_seconds": (
                (grid[:-1] + grid[1:]) / 2.0 + origin
            ).tolist(),
            "velocity_xy_px_s": np.asarray(diagnostic["velocity_px_s"]).tolist(),
            "speed_px_s": speed.tolist(),
            "acceleration_time_seconds": (grid[1:-1] + origin).tolist(),
            "acceleration_px_s2": acceleration.tolist(),
        })
    traced["signals"] = signals
    return traced, runtime


def compare_signals(a: TrackSequence, gt: TrackSequence):
    if (a.processed_width, a.processed_height, a.stabilization) != (
        gt.processed_width, gt.processed_height, gt.stabilization,
    ):
        raise ValueError("A/GT geometry and stabilization must match")
    clocks = lambda track: [(p.frame_index, p.timestamp_ms) for p in track.history]
    if clocks(a) != clocks(gt):
        raise ValueError("Use GT matched to the exact A frames and timestamps")
    at, runtime = _trace(a)
    gt_trace, _ = _trace(gt)
    if len(at["signals"]) != len(gt_trace["signals"]):
        raise ValueError("A/GT segmentation differs")
    observed, positions, speed, acceleration = [], [], [], []
    for p, q in zip(a.history, gt.history):
        ax, ay = p.cx * runtime.canonical_width, p.cy * runtime.canonical_height
        gx, gy = q.cx * runtime.canonical_width, q.cy * runtime.canonical_height
        observed.append(dict(time_seconds=p.timestamp_ms / 1000, frame_index=p.frame_index,
                             a_x=ax, a_y=ay, gt_x=gx, gt_y=gy,
                             error_x=ax-gx, error_y=ay-gy,
                             center_error=float(np.hypot(ax-gx, ay-gy)), confidence=p.conf))
    for s, t in zip(at["signals"], gt_trace["signals"]):
        if s["time_seconds"] != t["time_seconds"]:
            raise ValueError("A/GT signal grids differ")
        sid = s["segment_id"]
        for i, seconds in enumerate(s["time_seconds"]):
            ar, gr = s["resampled_xy_px"][i], t["resampled_xy_px"][i]
            ap, gp = s["smoothed_xy_px"][i], t["smoothed_xy_px"][i]
            positions.append(dict(segment_id=sid, time_seconds=seconds,
                                  source_frame_position=s["source_frame_position"][i],
                                  a_x=ap[0], a_y=ap[1], gt_x=gp[0], gt_y=gp[1],
                                  resampled_error_x=ar[0]-gr[0], resampled_error_y=ar[1]-gr[1],
                                  error_x=ap[0]-gp[0], error_y=ap[1]-gp[1]))
        for i, seconds in enumerate(s["velocity_time_seconds"]):
            av, gv = s["speed_px_s"][i], t["speed_px_s"][i]
            speed.append(dict(segment_id=sid, time_seconds=seconds, a=av, gt=gv,
                              signed_error=av-gv, vector_error=float(np.linalg.norm(np.subtract(
                                  s["velocity_xy_px_s"][i], t["velocity_xy_px_s"][i])))))
        edge_seconds = (s["smoothing_window_points"] // 2) * s["step_seconds"]
        for i, seconds in enumerate(s["acceleration_time_seconds"]):
            av, gv = s["acceleration_px_s2"][i], t["acceleration_px_s2"][i]
            acceleration.append(dict(
                segment_id=sid, time_seconds=seconds, a=av, gt=gv,
                signed_error=av-gv, absolute_error=abs(av-gv),
                source_frame_position=s["source_frame_position"][i+1],
                near_filter_edge=bool(min(seconds-s["time_seconds"][0],
                                          s["time_seconds"][-1]-seconds) <= edge_seconds + 1e-9),
                step_seconds=s["step_seconds"],
            ))
    error = np.array([r["signed_error"] for r in acceleration])
    weight = np.array([r["step_seconds"] for r in acceleration])
    energy = error ** 2 * weight
    count = max(1, int(np.ceil(len(error) * .1)))
    energy_sum = float(energy.sum())
    top = np.argsort(np.abs(error))[-count:]
    edge = np.array([r["near_filter_edge"] for r in acceleration], dtype=bool)
    interior = ~edge
    interior_a = _weighted_quantile(np.array([r["a"] for r in acceleration])[interior], weight[interior], .95) if interior.any() else None
    interior_gt = _weighted_quantile(np.array([r["gt"] for r in acceleration])[interior], weight[interior], .95) if interior.any() else None
    summary = dict(
        observed_points=len(observed), acceleration_samples=len(error),
        source_first_frame=a.history[0].frame_index, source_last_frame=a.history[-1].frame_index,
        first_time_seconds=a.history[0].timestamp_ms/1000,
        last_time_seconds=a.history[-1].timestamp_ms/1000,
        missing_frame_count=a.history[-1].frame_index-a.history[0].frame_index+1-len(a.history),
        acceleration_error_median=float(np.median(np.abs(error))),
        acceleration_error_p95=float(np.quantile(np.abs(error), .95)),
        acceleration_error_rmse=float(np.sqrt(np.average(error**2, weights=weight))),
        top_10_percent_error_energy_share=float(energy[top].sum()/energy_sum) if energy_sum > 1e-12 else None,
        filter_edge_error_energy_share=float(energy[edge].sum()/energy_sum) if energy_sum > 1e-12 else None,
        filter_edge_sample_fraction=float(np.mean(edge)),
        initial_center_error=observed[0]["center_error"],
        interior_a_acceleration_p95=interior_a, interior_gt_acceleration_p95=interior_gt,
        interior_acceleration_p95_change_percent=(interior_a-interior_gt)/abs(interior_gt)*100
        if interior_gt is not None and abs(interior_gt) > 1e-9 else None,
        max_acceleration_error=max(acceleration, key=lambda r: r["absolute_error"]),
        a_acceleration_p95=at["features"]["acceleration_p95"],
        gt_acceleration_p95=gt_trace["features"]["acceleration_p95"],
    )
    return dict(summary=summary, observed=observed, positions=positions, speed=speed,
                acceleration=acceleration, features={"a": at["features"], "gt": gt_trace["features"]},
                config=vars(FeatureConfig()), coordinate_width=runtime.canonical_width,
                coordinate_height=runtime.canonical_height)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(results: Path, output: Path, *, app_project: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    queries, summaries = {}, []
    names = {"bird_2_apart_fix": "bird_2 / 드론", "drone_5_apart_fix": "drone_5 / 드론",
             "drone_6_apart_fix": "drone_6 / 드론", "drone_1_drone": "drone_1 / 드론",
             "drone_1_bird_1": "drone_1 / 새 1", "drone_1_bird_2": "drone_1 / 새 2"}
    for path in sorted((results / "b_inputs").glob("*/a_track_sequence.json")):
        sid = path.parent.name
        gt_path = path.with_name("gt_matched_track_sequence.json")
        a = TrackSequence.model_validate_json(path.read_text())
        gt = TrackSequence.model_validate_json(gt_path.read_text())
        signal = compare_signals(a, gt)
        report_path = results / "per_video" / f"{sid}.json"
        report = json.loads(report_path.read_text())
        previous = json.loads(path.with_name("comparison.json").read_text())
        for which, key in [("a", "a"), ("gt", "gt_matched")]:
            expected = previous["results"][key]["features"]
            if any(not np.isclose(v, expected[k], rtol=1e-12, atol=1e-9)
                   for k, v in signal["features"][which].items()):
                raise ValueError(f"Saved B feature regression: {sid}/{which}")
        trajectory = Path(report["inputs"]["trajectory_csv"])
        debug = list(csv.DictReader(trajectory.open(encoding="utf-8")))
        by_frame = {int(r["frame_index"]): r for r in debug}
        for row in signal["observed"]:
            source = by_frame[row["frame_index"]]
            row["tracking_source"] = source["tracking_source"]
            row["lifecycle_state"] = source.get("lifecycle_state", "unknown")
        events = [dict(frame_index=int(r["frame_index"]), time_seconds=float(r["timestamp_ms"])/1000,
                       tracking_source=r["tracking_source"], lifecycle_state=r.get("lifecycle_state", "unknown"),
                       visible=r["visible"].lower() == "true") for r in debug
                  if a.history[0].frame_index <= int(r["frame_index"]) <= a.history[-1].frame_index
                  and (r["tracking_source"] != "klt" or r["visible"].lower() != "true")]
        signal["events"] = events
        peaks = [dict(r) for r in sorted(signal["acceleration"], key=lambda r: r["absolute_error"], reverse=True)[:5]]
        for row in peaks:
            frame = int(round(row["source_frame_position"]))
            row["nearest_source_frame"] = frame
            row["nearest_tracking_source"] = by_frame.get(frame, {}).get("tracking_source", "unknown")
            row["seconds_to_nearest_non_klt_event"] = min(
                (abs(row["time_seconds"]-e["time_seconds"]) for e in events), default=None,
            )
        signal["peaks"] = peaks
        summary = dict(sample_id=sid, name=names.get(sid, sid), **signal["summary"])
        signal["provenance"] = dict(
            a_track=str(path), a_sha256=_sha(path), gt_track=str(gt_path), gt_sha256=_sha(gt_path),
            trajectory=str(trajectory), trajectory_sha256=_sha(trajectory),
            research_features_sha256=_sha(Path(__file__).with_name("features.py")),
            report=str(report_path), camera_verification=previous["camera_verification"],
            scope="exact_A_observation_frames_and_timestamps; same_verified_CMC; canonical_width_1920",
            note=report.get("provenance_note", ""),
        )
        write_json(output / f"{sid}.json", signal)
        for kind in ("observed", "positions", "speed", "acceleration", "events", "peaks"):
            _write_csv(output / f"{sid}_{kind}.csv", signal[kind])
            files = [str(path), str(gt_path), str(trajectory), str(report_path)]
            queries[f"{sid}_{kind}"] = dict(rows=signal[kind], methods=[dict(
                language="python", code="python -m research.tracking_signal_comparison "
                + shlex.quote(str(results)) + " --output " + shlex.quote(str(output)))], source=dict(
                label=f"{names.get(sid, sid)} / {kind}", files=files,
                metricDefinitions=[dict(label="Coordinate/time policy", definition=signal["provenance"]["scope"]),
                                   dict(label="Acceleration", definition="abs(diff(norm(B velocity)))/dt; scalar speed-change magnitude, not vector acceleration"),
                                   dict(label="Time", definition="Absolute source timestamp seconds. Resampled frame positions are fractional; event timestamps are source observations.")],
                caveats=["GT annotation and CMC errors remain. Association with a recovery event is not proof of causality.",
                         "Short gaps are interpolated by B, long gaps split segments. SG edge samples use mode=interp polynomial fitting."] ))
        summaries.append(summary)
    if not summaries:
        raise ValueError("No verified B inputs found")
    write_json(output / "summary.json", summaries)
    queries["summary"] = dict(rows=summaries, source=dict(
        label="Six matched A/GT signal comparisons", files=[str(output / "summary.json")],
        metricDefinitions=[dict(label="Top 10% error energy", definition="Sum((A acceleration-GT acceleration)^2*dt) over the largest absolute-error 10% of samples divided by the total. Descriptive concentration, not a causal jitter test."),
                           dict(label="Edge share", definition="Same squared-error energy within half the actual SG window of each segment boundary; no edge samples removed from B features.")]))
    if app_project:
        write_json(app_project / "src/data.json", dict(
            id="skydetect-motion-signal-20261006", surface="report", title="GT ↔ A · 시간별 운동 신호 비교",
            generatedAt="2026-10-06", report={"asOf": "2026-10-06"}, buildStatus="complete", status="reviewed",
            filters=[], queries=queries,
        ))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app-project", type=Path)
    args = parser.parse_args()
    rows = export_dataset(args.results, args.output, app_project=args.app_project)
    for r in rows:
        peak = r["max_acceleration_error"]
        top_share = r["top_10_percent_error_energy_share"]
        edge_share = r["filter_edge_error_energy_share"]
        top_text = f"{top_share:.1%}" if top_share is not None else "undefined (zero error energy)"
        edge_text = f"{edge_share:.1%}" if edge_share is not None else "undefined (zero error energy)"
        print(f"{r['sample_id']}: accel error median={r['acceleration_error_median']:.2f}, "
              f"p95={r['acceleration_error_p95']:.2f}, max at {peak['time_seconds']:.3f}s, "
              f"top10% energy={top_text}, edge energy={edge_text}")


if __name__ == "__main__":
    main()
