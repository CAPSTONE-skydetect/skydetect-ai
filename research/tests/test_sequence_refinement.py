from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.camera import Camera
from research.calibrate_sequence_simulator import make_report
from research.observation import NoiseConfig, ObservationModel
from research.refine_sequence_calibration import grouped_folds
from research.sequence_comparison import compare, metric_scales, sequence_metrics
from research.sequence_simulator import AcquisitionProfile, GuidanceProfile, latent_flight, observe_flight
from research.trajectory_sequence import SequenceConfig, normalize_window


def test_second_order_jitter_has_expected_stationary_variance_and_correlation():
    camera = Camera()
    n = 4500
    truth = [dict(cx=.5, cy=.5, w=.02, h=.02, conf=1., frame_index=i, timestamp_ms=round(i*1000/30)) for i in range(n)]
    cfg = NoiseConfig(jitter_px=.5, jitter_ar1=.4, jitter_ar2=-.65, jitter_difficulty_gain=0,
                      dropout_rate=0, burst_probability=0, drift_probability=0, camera_probability=0)
    history, _ = ObservationModel(camera, np.random.default_rng(8), config=cfg).apply(truth, 30)
    residual = np.array([p["cx"]-.5 for p in history])[500:]*camera.width
    assert residual.std() == pytest.approx(.5, rel=.08)
    assert np.corrcoef(residual[:-1], residual[1:])[0, 1] == pytest.approx(.4/1.65, abs=.04)


def test_unstable_jitter_and_unsupported_second_order_fps_are_rejected():
    with pytest.raises(ValueError, match="stationary"):
        NoiseConfig(jitter_ar1=.8, jitter_ar2=.3)
    model = ObservationModel(Camera(), np.random.default_rng(1), config=NoiseConfig(jitter_ar2=-.3))
    with pytest.raises(ValueError, match="30 Hz"):
        model.apply([dict(cx=.5, cy=.5, w=.02, h=.02)], 15)


def test_default_noise_preserves_legacy_stationary_first_order_coefficient():
    assert NoiseConfig().jitter_ar1 == .55 and NoiseConfig().jitter_ar2 == 0


def test_pure_guided_hover_starts_at_rest_and_keeps_physical_force_integration():
    guide = GuidanceProfile(drone_probabilities=(0., 0., 0., 1.))
    flight = latent_flight("drone", "hover_quad", 400, duration=3, guidance=guide)
    rows = flight["world_truth"]
    assert rows[0]["velocity_m_s"] == [0., 0., 0.]
    assert flight["template"] == ["hold"]
    assert all(r["goal_m"] == rows[0]["goal_m"] for r in rows)
    assert flight["latent_failure"] is None
    assert np.isfinite([r["position_m"] for r in rows]).all()


def test_observation_aspect_is_shared_profile_setting():
    profile = AcquisitionProfile(aspect_4_3_probability=1)
    bird = observe_flight(latent_flight("bird", "pigeon", 14, duration=2.5), profile)
    drone = observe_flight(latent_flight("drone", "consumer_quad", 14, duration=2.5), profile)
    assert bird["track"]["processed_height"] == drone["track"]["processed_height"] == 1440


def test_cv_never_places_one_source_group_on_both_sides():
    table = pd.DataFrame([dict(label=c, group_id=f"{c}:{g}")
                          for c in ("bird", "drone") for g in range(6) for _ in range(g+1)])
    held_all = []
    for fit, held in grouped_folds(table):
        assert not fit & held
        assert fit | held == set(table.group_id)
        held_all.extend(held)
    assert sorted(held_all) == sorted(table.group_id.unique())


def test_report_and_acceptance_file_include_grouped_cv_blocker(tmp_path):
    import json

    x, normalization = normalize_window(
        np.column_stack((np.linspace(0, .1, 60), np.zeros(60))), SequenceConfig())
    xs = np.stack((x, x))
    meta = pd.DataFrame([dict(label=label, group_id=label,
                             **sequence_metrics(x, normalization["normalization_scale"]))
                         for label in ("bird", "drone")])
    table = compare(meta, meta, metric_scales(meta))
    metrics = pd.concat([table.assign(split=split, phase=phase)
                         for split in ("train", "validation") for phase in ("before", "after")])
    summary = dict(train_before=0., train_after=0., validation_before=0., validation_after=0.,
                   validation_coverage_before=1., validation_coverage_after=1.,
                   usable_flight_fraction_before=1., usable_flight_fraction_after=1.,
                   validation_reused=True, previous_run="fixture", test_opened=False,
                   real_groups=dict(train=2, validation=2),
                   bootstrap=dict(after_minus_before_q025_q50_q975=[0., 0., 0.]),
                   mmd_before={label: dict(mmd2=0.) for label in ("bird", "drone")},
                   mmd_after={label: dict(mmd2=0.) for label in ("bird", "drone")})
    review = dict(status="not_ready_for_bulk_generation", bulk_generation_approved=False,
                  blockers=["grouped_cv_distance_not_improved"])
    make_report(tmp_path, summary, metrics, xs, meta, (xs, meta), (xs, meta),
                AcquisitionProfile(), acceptance=review)
    assert "grouped_cv_distance_not_improved" in (tmp_path/"report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path/"acceptance_review.json").read_text(encoding="utf-8")) == review
