from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from research.calibrate_sequence_simulator import real_partition, review_acceptance, simulated_windows
from research.generators import DroneDyn, Environment
from research.sequence_comparison import compare, metric_scales, score, sequence_metrics, sequence_mmd
from research.sequence_simulator import AcquisitionProfile, CommandSchedule, latent_flight, observe_flight
from research.trajectory_sequence import SequenceConfig, normalize_window


def test_velocity_command_is_acceleration_limited_not_state_teleport():
    env = Environment(goal_pos=[100, 0, 100], wind_speed=0, gust_intensity=0)
    agent = DroneDyn(env, start_pos=[0, 0, 100], start_speed=0, rng=np.random.default_rng(1))
    env.command_velocity = np.array([-8., 0., 0.])
    before = agent.v_ground.copy()
    agent.step()
    assert np.linalg.norm(agent.v_ground-before) < 1
    assert agent.body.command_accel[0] < 0
    env.command_velocity = [float("nan"), 0, 0]
    with pytest.raises(ValueError, match="Velocity command"):
        agent.step()


def test_command_dwell_and_hard_overlap_modes_are_preserved():
    modes = {label: set() for label in ("bird", "drone")}
    for label in modes:
        for seed in range(30):
            schedule = CommandSchedule(label, 8, np.random.default_rng(seed))
            assert schedule.segments[0]["start_s"] == 0
            assert schedule.segments[-1]["end_s"] == 8
            for a, b in zip(schedule.segments, schedule.segments[1:]):
                assert a["end_s"] == b["start_s"]
            modes[label].update(s["mode"] for s in schedule.segments)
    assert {"reverse", "brake", "hold", "smooth_cruise", "climb", "descend"} <= modes["drone"]
    assert {"powered", "glide", "turn", "descending_glide", "circling_glide"} <= modes["bird"]


@pytest.mark.parametrize("label,subtype", [("bird", "pigeon"), ("drone", "consumer_quad")])
def test_reproducible_continuous_flight_and_shared_profile(label, subtype):
    a = latent_flight(label, subtype, 123, duration=3)
    b = latent_flight(label, subtype, 123, duration=3)
    assert a == b and a["latent_failure"] is None
    pos = np.array([r["position_m"] for r in a["world_truth"]])
    assert np.isfinite(pos).all() and np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) < 2
    profile = AcquisitionProfile(dropout_rate=0, burst_probability=0, drift_probability=0)
    observed = observe_flight(a, profile)
    assert observed == observe_flight(a, profile)
    assert observed["metadata"]["acquisition"] == asdict(profile)
    assert observed["track"]["stabilization"]["applied"]
    assert observed["metadata"]["observation"]["attempted_frame_count"] == 91
    x, meta, status, _ = simulated_windows([observed], SequenceConfig())
    assert x.shape[1:] == (4, 60)
    assert status.planned_points.iloc[0] == 91


def test_larger_projection_request_changes_coordinates_not_latent_motion():
    flight = latent_flight("bird", "pigeon", 88, duration=3)
    low = observe_flight(flight, AcquisitionProfile(span_multiplier=.7, dropout_rate=0, burst_probability=0))
    high = observe_flight(flight, AcquisitionProfile(span_multiplier=1.3, dropout_rate=0, burst_probability=0))
    assert low["world_truth"] == high["world_truth"]
    assert high["metadata"]["camera_distance_m"] < low["metadata"]["camera_distance_m"]
    assert high["track"]["history"] != low["track"]["history"]


def test_camera_range_does_not_cancel_actual_flight_speed():
    flight = latent_flight("bird", "pigeon", 88, duration=3)
    changed = deepcopy(flight)
    for row in changed["world_truth"]:
        row["velocity_m_s"] = (np.asarray(row["velocity_m_s"]) * 2).tolist()
    profile = AcquisitionProfile()
    original = observe_flight(flight, profile)
    doubled_speed_annotation = observe_flight(changed, profile)
    assert original["metadata"]["camera"] == doubled_speed_annotation["metadata"]["camera"]
    assert original["track"]["history"] == doubled_speed_annotation["track"]["history"]


def test_scope_excludes_fixed_wing_without_removing_legacy_support():
    with pytest.raises(ValueError, match="scope"):
        latent_flight("drone", "fixed_wing_drone", 1)


def test_calibration_refuses_test_before_reading_any_files(tmp_path):
    with pytest.raises(ValueError, match="cannot open real test"):
        real_partition(tmp_path, "test", {}, SequenceConfig())


def metric_fixture():
    xs, rows = [], []
    cfg = SequenceConfig()
    for label in ("bird", "drone"):
        for i in range(6):
            t = np.arange(60)/30
            p = np.column_stack((t*.05, .005*np.sin((i+1)*t)))
            x, meta = normalize_window(p, cfg)
            xs.append(x)
            rows.append(dict(label=label, group_id=f"{label}:{i}",
                             **sequence_metrics(x, meta["normalization_scale"])))
    return np.stack(xs), pd.DataFrame(rows)


def test_equal_distributions_have_zero_distance_and_replication_no_extra_weight():
    x, table = metric_fixture()
    scales = metric_scales(table)
    assert score(compare(table, table, scales)) == pytest.approx(0)
    repeated = pd.concat([table, table.iloc[[0]*30]], ignore_index=True)
    assert score(compare(table, repeated, scales)) == pytest.approx(0, abs=1e-12)
    result = sequence_mmd(x, table, x, table, x, table)
    assert result["bird"]["mmd2"] == pytest.approx(0, abs=1e-12)


def test_translation_is_invariant_but_physical_scale_is_measured():
    cfg = SequenceConfig()
    p = np.column_stack((np.linspace(0, .2, 60), np.zeros(60)))
    x, a = normalize_window(p, cfg)
    y, b = normalize_window(p*2+[.1, .2], cfg)
    ma, mb = sequence_metrics(x, a["normalization_scale"]), sequence_metrics(y, b["normalization_scale"])
    assert mb["screen_span"] == pytest.approx(2*ma["screen_span"])
    assert mb["straightness"] == pytest.approx(ma["straightness"])


def test_stationary_spectral_and_turn_diagnostics_remain_finite():
    x, meta = normalize_window(np.ones((60, 2)), SequenceConfig())
    result = sequence_metrics(x, meta["normalization_scale"])
    assert np.isfinite(list(result.values())).all()
    assert result["direction_support"] == 0 and result["low_motion_fraction"] == 1


def test_coverage_gain_does_not_hide_validation_regression():
    summary = dict(validation_before=.7, validation_after=.8,
                   mmd_before={"bird": {"mmd2": .08}, "drone": {"mmd2": .03}},
                   mmd_after={"bird": {"mmd2": .11}, "drone": {"mmd2": .02}},
                   bootstrap={"after_minus_before_q025_q50_q975": [-.1, .15, .4]},
                   usable_flight_fraction_before=.9, usable_flight_fraction_after=.95,
                   validation_coverage_before=.77, validation_coverage_after=.9,
                   validation_reused=True, test_opened=False)
    review = review_acceptance(summary)
    assert review["status"] == "not_ready_for_bulk_generation"
    assert "bird_sequence_mmd_worsened" in review["blockers"]
    assert not review["bulk_generation_approved"]
    summary["validation_after"] = .6
    summary["mmd_after"]["bird"]["mmd2"] = .05
    summary["bootstrap"]["after_minus_before_q025_q50_q975"] = [-.2, -.1, -.01]
    review = review_acceptance(summary)
    assert review["status"] == "candidate_for_downstream_ablation"
    assert not review["bulk_generation_approved"]
