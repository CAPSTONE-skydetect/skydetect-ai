"""Reproducible internal audit; records failures instead of declaring real-world validity."""
import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from .evaluation import evaluate_dataset
from .features import FeatureConfig, extract_features
from .io import read_jsonl, write_json
from .pipeline import BatchRunner, DRONE_SUBTYPES
from .statistical_engine import describe_dataset
from .visualizer import plot_distributions, plot_sample


def analytic_probes():
    results = []
    for fps in (15, 24, 30, 50, 60, 120):
        for missing in (False, True):
            history = [dict(frame_index=i, timestamp_ms=i*1000/fps,
                            cx=(300+12*i/fps)/1920, cy=(300+16*i/fps)/1080,
                            w=20/1920, h=10/1080, conf=.9)
                       for i in range(4*fps+1) if not missing or i % 8 not in (2, 3)]
            result = extract_features(history, 1920, 1080)
            f = result["features"]
            results.append(dict(fps=fps, dropout=missing, velocity_error=abs(f["v_mean"]-1.),
                                acceleration_error=f["a_mean"], accepted=result["feature_status"] == "accepted"))
    rng = np.random.default_rng(10)
    stationary = [dict(frame_index=i, timestamp_ms=i*1000/30,
                       cx=.5+rng.normal(0,.35)/1920, cy=.5+rng.normal(0,.35)/1080,
                       w=20/1920, h=10/1080, conf=.9) for i in range(121)]
    raw = extract_features(stationary, 1920, 1080, config=FeatureConfig(smoothing_seconds=0))["features"]
    smooth = extract_features(stationary, 1920, 1080)["features"]
    return dict(constant_motion=results, stationary_jitter=dict(raw=raw, smoothed=smooth,
                acceleration_ratio=smooth["a_mean"]/raw["a_mean"], velocity_ratio=smooth["v_mean"]/raw["v_mean"]))


def validate(output, samples_per_subtype=10, seed=20260906, stress=True):
    output = Path(output)
    table = BatchRunner(output, seed=seed).run(samples_per_subtype=samples_per_subtype)
    summary = describe_dataset(table)
    evaluation = evaluate_dataset(table)
    write_json(output / "descriptive_report.json", summary)
    write_json(output / "evaluation_report.json", evaluation)
    plot_distributions(table, output / "plots" / "distributions.png")
    selected, rejection_reasons = set(), {}
    latent_failures = 0
    pairs_checked, previous = 0, None
    minimum_altitude, max_speed = float("inf"), 0.
    for sample in read_jsonl(output / "raw_trajectories_v3.jsonl"):
        m, truth = sample["metadata"], sample["world_truth"]
        latent_failures += int(m["latent_failure"] is not None)
        for reason in sample["feature_result"]["reasons"]:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0)+1
        for point in truth:
            assert np.isfinite(point["position_m"]).all()
            minimum_altitude = min(minimum_altitude, point["position_m"][2])
            max_speed = max(max_speed, np.linalg.norm(point["velocity_m_s"]))
        if m["observation_profile"] == "ideal":
            previous = sample
        else:
            assert previous["world_truth"] == sample["world_truth"], "Paired physical states differ"
            pairs_checked += 1
        key = m["subtype"]
        if key not in selected and m["observation_profile"] == "noisy":
            plot_sample(sample, output / "plots" / f"{key}.png")
            selected.add(key)
    probes = analytic_probes()
    assert max(p["velocity_error"] for p in probes["constant_motion"]) < 1e-8
    assert max(p["acceleration_error"] for p in probes["constant_motion"]) < 1e-7
    stress_results = []
    if stress:
        for fps in (15, 30, 60):
            for n in (60, 120, 240, 420):
                for label, subtype in (("bird", "pigeon"), ("bird", "seagull"), ("bird", "falcon"),
                                       *(("drone", s) for s in DRONE_SUBTYPES)):
                    sample = BatchRunner(seed=seed+1, fps=fps).simulate("sharp_turns", label, subtype, n,
                                                                 frame_count=n, noisy=True)
                    stress_results.append(dict(fps=fps, frames=n, subtype=subtype,
                                               status=sample["feature_result"]["feature_status"],
                                               reasons=sample["feature_result"]["reasons"],
                                               latent_failure=sample["metadata"]["latent_failure"],
                                               observed_points=len(sample["track"]["history"])))
    report = dict(scope="internal_consistency_only", real_A_status="not_available_not_validated",
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   pandas=pd.__version__, sklearn=sklearn.__version__),
                  seed=seed, dataset_rows=len(table), accepted_rows=int((table.feature_status == "accepted").sum()),
                  paired_latent_equal_count=pairs_checked, latent_failures=latent_failures,
                  minimum_altitude_m=float(minimum_altitude), maximum_ground_speed_m_s=float(max_speed),
                  rejection_reasons=rejection_reasons, analytic_probes=probes, stress_matrix=stress_results,
                  acceptance_criteria=dict(analytic_velocity_abs_error=1e-8, analytic_acceleration_abs_error=1e-7,
                      paired_latent_equality="exact", training_family_leakage="zero",
                      realism="No numerical acceptance threshold until independent real-A validation"))
    write_json(output / "validation_report.json", report)
    print(f"Rows={len(table)}, accepted={report['accepted_rows']}, latent pairs checked={pairs_checked}")
    print(f"Artifacts: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"output"/"validation_sim_v4")
    parser.add_argument("--samples-per-subtype", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--no-stress", action="store_true")
    args = parser.parse_args()
    validate(args.output, args.samples_per_subtype, args.seed, not args.no_stress)


if __name__ == "__main__":
    main()
