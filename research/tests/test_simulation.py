import json

import numpy as np
import pytest

from research.camera import Camera
from research.generators import BirdDyn, DroneDyn, Environment
from research.observation import NoiseConfig, ObservationModel, drift_offset
from research.pipeline import BIRD_BEHAVIORS, DRONE_SUBTYPES, BatchRunner


def test_projection_depth_no_clipping_and_goal_invariance():
    camera = Camera(position=(0, 0, 0), look_at=(0, 1, 0))
    near = camera.project([0, 100, 0], 1., 1.)
    far = camera.project([0, 200, 0], 1., 1.)
    assert near["cx"] == near["cy"] == .5
    assert near["w"] == 2*far["w"]
    assert camera.project([0, -1, 0], 1., 1.) is None
    outside = camera.project([1000, 100, 0], 1., 1.)
    assert outside["cx"] > 1 and not camera.visible(outside)
    env = Environment(camera=camera, goal_pos=[300, 100, 0])
    agent = DroneDyn(env, start_pos=[0, 100, 0])
    old = agent.get_observation(0)
    env.x_goal[:] = [550, 200, 10]
    assert agent.get_observation(0) == old
    assert env.h_star == 10


def test_hover_holds_a_fixed_position_and_jerk_is_bounded():
    env = Environment(goal_pos=[0, 100, 100], wind_speed=0, gust_intensity=0)
    agent = DroneDyn(env, model="hover_quad", start_pos=[0, 100, 100], start_speed=5, heading=[1,0,0], rng=np.random.default_rng(2))
    agent.config.update(sigma_s=0., sigma_u=0.)
    previous = agent.accel.copy()
    for _ in range(450):
        agent.step()
        assert np.linalg.norm(agent.accel-previous) <= agent.config["jerk_max"]*env.dt+1e-8
        assert np.linalg.norm(agent.accel) <= agent.a_max+1e-8
        previous = agent.accel.copy()
    assert np.linalg.norm(agent.v_ground) < .02
    assert np.linalg.norm(agent.pos-env.x_goal) < .05


@pytest.mark.parametrize("kind", ["bird", "fixed_wing_drone"])
def test_banked_flight_constraints_and_glide(kind):
    env = Environment(goal_pos=[-300,100,180], wind_speed=0, gust_intensity=0)
    agent = BirdDyn(env, start_pos=[0,100,100]) if kind == "bird" else DroneDyn(env, model=kind, start_pos=[0,100,100])
    initial_alt = agent.pos[2]
    agent.behavior = "glide"
    for _ in range(300):
        agent.step()
        assert np.isfinite(agent.s) and agent.s > 0.
        assert abs(agent.phi) <= agent.phi_max+1e-10
        assert agent.diagnostics["thrust_n"] == 0.
        assert 0. <= agent.diagnostics["cl"] <= agent.config["cl_max"]
    assert agent.pos[2] < initial_alt


def test_drift_nonrecovering_persists_and_recovery_continuous():
    profile = dict(enabled=True, start=10, peak=30, end=60, recover=False, offset_px=[5.,-3.])
    assert np.allclose(drift_offset(100, profile), [5,-3])
    assert np.linalg.norm(drift_offset(61, profile)-drift_offset(60, profile)) == 0
    profile["recover"] = True
    assert np.allclose(drift_offset(60, profile), 0)
    assert np.linalg.norm(drift_offset(59, profile)) < .03


def test_reproducible_and_paired_latent_without_global_randomness():
    runner = BatchRunner(seed=42)
    np.random.seed(900)
    state = np.random.get_state()
    a = runner.simulate("sudden_dash", "bird", "pigeon", 0)
    np.random.normal(size=100)
    b = runner.simulate("sudden_dash", "bird", "pigeon", 0)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    ideal = runner.simulate("sudden_dash", "bird", "pigeon", 0, noisy=False)
    assert a["world_truth"] == ideal["world_truth"]
    assert a["optical_truth"] == ideal["optical_truth"]
    assert a["track"]["history"] != ideal["track"]["history"]
    np.random.set_state(state)
    expected = np.random.random()
    np.random.set_state(state)
    runner.simulate("baseline", "drone", "hover_quad", 7)
    assert np.random.random() == expected


@pytest.mark.parametrize("subtype", DRONE_SUBTYPES)
@pytest.mark.parametrize("scenario", ["baseline", "sudden_dash", "sharp_turns", "multi_mode"])
def test_all_drone_modes_finite_and_metadata_complete(subtype, scenario):
    sample = BatchRunner(seed=8).simulate(scenario, "drone", subtype, 0, frame_count=180)
    assert sample["metadata"]["frame_count"] == 180
    assert sample["metadata"]["latent_failure"] is None
    assert len(sample["world_truth"]) == 180
    assert np.isfinite(np.array([p["position_m"] for p in sample["world_truth"]])).all()
    assert sample["feature_result"]["feature_status"] in ("accepted", "rejected")
    for point in sample["track"]["history"]:
        assert 0 < point["w"] < 1 and 0 < point["h"] < 1


@pytest.mark.parametrize("species", ["pigeon", "seagull", "falcon"])
@pytest.mark.parametrize("behavior", BIRD_BEHAVIORS)
def test_species_behavior_independent(species, behavior):
    sample = BatchRunner(seed=12).simulate("baseline", "bird", species, 0, frame_count=120, behavior=behavior)
    assert sample["metadata"]["behavior_mode"] == behavior
    assert sample["metadata"]["latent_failure"] is None


def test_events_lengths_start_goals_are_diverse_and_goal_distance_enforced():
    samples = [BatchRunner(seed=4).simulate("sudden_dash", "drone", "consumer_quad", i) for i in range(12)]
    metadata = [s["metadata"] for s in samples]
    assert len({m["frame_count"] for m in metadata}) > 8
    assert len({m["events"]["dash"] for m in metadata}) > 8
    for m in metadata:
        n = m["frame_count"]
        assert 60 <= n <= 420
        assert .25*n-1 <= m["events"]["dash"] < .55*n
        assert .65*n-1 <= m["events"]["brake"] < .85*n
        assert np.linalg.norm(np.subtract(m["initial_goal_m"], m["start_position_m"])) >= 150.


def test_observation_drops_frames_and_caps_recovery_confidence():
    sample = BatchRunner(seed=7).simulate("baseline", "drone", "consumer_quad", 0, noisy=False, frame_count=300)
    truth = sample["optical_truth"]
    camera = Camera(**sample["metadata"]["camera"])
    config = NoiseConfig(dropout_rate=.2, burst_probability=1., drift_probability=1., camera_probability=1.)
    obs, meta = ObservationModel(camera, np.random.default_rng(42), True, config).apply(truth, 30)
    assert len(obs) < sum(p is not None for p in truth)
    assert any(np.diff([p["frame_index"] for p in obs]) > 1)
    assert meta["quality"]["num_points"] == len(obs)
    assert meta["quality"]["missing_ratio"] == pytest.approx(1-len(obs)/len(truth))
    for state in meta["states"]:
        if state["tracking_source"] == "appearance":
            assert state["conf"] <= .90
        if state["tracking_source"] == "motion":
            assert state["conf"] <= .86
    clean, _ = ObservationModel(camera, np.random.default_rng(4), False).apply(truth, 30)
    assert clean == [p for p in truth if p is not None]


def test_dash_is_a_speed_command_not_a_goal_teleport():
    sample = BatchRunner(seed=10).simulate("sudden_dash", "drone", "racing_quad", 0, noisy=False, frame_count=420)
    world, events = sample["world_truth"], sample["metadata"]["events"]
    assert world[events["dash"]]["speed_scale"] == events["dash_scale"]
    assert 1.2 <= events["dash_scale"] <= 1.65
    before = np.mean([np.linalg.norm(p["velocity_m_s"]) for p in world[events["dash"]-15:events["dash"]]])
    after = np.mean([np.linalg.norm(p["velocity_m_s"]) for p in world[events["dash"]+45:events["dash"]+60]])
    assert after > before*1.1
