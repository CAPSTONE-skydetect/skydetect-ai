"""Reproducible latent flight -> optical projection -> A-like observations -> v3 features."""
import argparse
from dataclasses import asdict
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import FEATURE_VERSION, SIMULATOR_VERSION
from .camera import Camera
from .behavior import BehaviorSchedule
from .features import FEATURE_COLUMNS, FeatureConfig, extract_features
from .generators import BirdDyn, DroneDyn, Environment, SPECIES_CONFIG
from .io import read_jsonl, write_json, write_jsonl
from .observation import NoiseConfig, ObservationModel
from .parameters import SAMPLING, parameter_manifest

SCENARIOS = ("baseline", "sudden_dash", "sharp_turns", "multi_mode")
DRONE_SUBTYPES = ("consumer_quad", "racing_quad", "hover_quad", "fixed_wing_drone")
BIRD_BEHAVIORS = ("glide", "flap_jitter", "thermal_circle", "foraging_zigzag", "sudden_escape")
DEPTH_MODES = ("approaching", "receding", "crossing", "passing_by")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "sim_v4"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def assign_splits(families, seed, strata=None):
    groups = sorted(set(families))
    partitions = {}
    for group in groups:
        partitions.setdefault(strata[group] if strata is not None else "all", []).append(group)
    result = {}
    for stratum, members in sorted(partitions.items()):
        if len(members) < 3:
            result.update({group: "train" for group in members})
            continue
        order = np.random.default_rng(stable_seed(seed, "split", stratum)).permutation(members)
        n_test = max(1, int(round(len(members)*.15)))
        n_val = max(1, int(round(len(members)*.15)))
        result.update({str(group): "test" if i < n_test else "validation" if i < n_test+n_val else "train"
                       for i, group in enumerate(order)})
    return result


class BatchRunner:
    def __init__(self, output_dir=DEFAULT_OUTPUT, fps=30., seed=42, feature_config=None, noise_config=None):
        if not np.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("fps must be in [1, 240]")
        self.output_dir, self.fps, self.seed = Path(output_dir), float(fps), int(seed)
        self.feature_config = feature_config or FeatureConfig()
        self.noise_config = noise_config or NoiseConfig()
        self.provenance = parameter_manifest()

    def simulate(self, scenario, agent_type, subtype, sample_index, noisy=True,
                 frame_count=None, behavior=None, depth_mode=None):
        if scenario not in SCENARIOS or agent_type not in ("bird", "drone"):
            raise ValueError("Unknown scenario or agent type")
        allowed = SPECIES_CONFIG if agent_type == "bird" else DRONE_SUBTYPES
        if subtype not in allowed:
            raise ValueError("Unknown subtype")
        family = f"{self.seed}:{scenario}:{int(sample_index)}"
        scenario_seed = stable_seed(family, "scenario")
        physical_seed = stable_seed(family, agent_type, subtype, "physics")
        observer_seed = stable_seed(family, agent_type, subtype, "observer")
        rng = np.random.default_rng(scenario_seed)
        physical_rng = np.random.default_rng(physical_seed)
        length_range = ((60, 120), (121, 240), (241, 420))[int(rng.integers(3))]
        sampled_length = int(rng.integers(length_range[0], length_range[1] + 1))
        n = sampled_length if frame_count is None else int(frame_count)
        if n < 5 or n > 10000 or (frame_count is not None and n != frame_count):
            raise ValueError("frame_count must be an integer in [5, 10000]")
        start = rng.uniform([-50., 40., 50.], [80., 180., 160.])
        chosen_depth = str(rng.choice(DEPTH_MODES))
        depth_mode = depth_mode or chosen_depth
        if depth_mode not in DEPTH_MODES:
            raise ValueError("Unknown depth mode")
        yaw = float(rng.uniform(-np.pi, np.pi))
        if depth_mode == "approaching":
            yaw = float(rng.uniform(-.8 * np.pi, -.2 * np.pi))
        elif depth_mode == "receding":
            yaw = float(rng.uniform(.2 * np.pi, .8 * np.pi))
        elif depth_mode in ("crossing", "passing_by"):
            yaw = float(rng.choice([0., np.pi]) + rng.normal(0., .12))
        direction = np.array([np.cos(yaw), np.sin(yaw), rng.uniform(-.15, .15)])
        direction /= np.linalg.norm(direction)
        distance = float(rng.uniform(150., 400.))
        goal = start + distance * direction
        goal[2] = np.clip(goal[2], 30., 180.)
        if np.linalg.norm(goal-start) < 150.:
            goal[:2] = start[:2] + (goal[:2]-start[:2]) * 151. / np.linalg.norm(goal-start)
        initial_goal = goal.copy()
        camera = Camera(horizontal_fov_deg=float(rng.uniform(50., 75.)),
                        position=(0., -70., 12.), look_at=(15., 110., 100.))
        env = Environment(fps=self.fps, goal_pos=goal, wind_speed=float(rng.uniform(*SAMPLING["wind_speed_m_s"])),
                          wind_direction=float(rng.uniform(-np.pi, np.pi)),
                          gust_intensity=float(rng.uniform(*SAMPLING["gust_std_m_s"])),
                          rng=np.random.default_rng(stable_seed(physical_seed, "wind")), camera=camera)
        events = dict(dash=int(rng.uniform(.25, .55) * n), brake=int(rng.uniform(.65, .85) * n),
                      hover_start=int(rng.uniform(.2, .45) * n), hover_end=int(rng.uniform(.6, .8) * n),
                      turns=sorted(rng.choice(np.arange(max(2, int(.15*n)), max(8, int(.85*n))),
                                             size=min(5, max(2, int(rng.integers(2, 6)))), replace=False).tolist()))
        turn_signs = rng.choice([-1., 1.], len(events["turns"]))
        events["turn_angles_rad"] = (turn_signs*np.radians(rng.uniform(*SAMPLING["turn_angle_deg"], len(turn_signs)))).tolist()
        events["dash_scale"] = float(rng.uniform(*SAMPLING["dash_scale"]))
        events["brake_scale"] = float(rng.uniform(*SAMPLING["brake_scale"]))
        sampled_behavior = str(rng.choice(BIRD_BEHAVIORS))
        behavior = behavior or (sampled_behavior if agent_type == "bird" else "cruise")
        if agent_type == "bird" and behavior not in BIRD_BEHAVIORS:
            raise ValueError("Unknown bird behavior")
        if agent_type == "drone" and behavior != "cruise":
            raise ValueError("Drone behavior is commanded by scenario")
        cls = BirdDyn if agent_type == "bird" else DroneDyn
        kwargs = {"species": subtype} if agent_type == "bird" else {"model": subtype}
        agent = cls(env, **kwargs, start_pos=start, heading=goal-start,
                    start_speed=10., rng=np.random.default_rng(stable_seed(physical_seed, "agent")))
        individual_scaling = agent.randomize_individual(physical_rng)
        schedule = BehaviorSchedule(np.random.default_rng(stable_seed(physical_seed,"behavior")), n/self.fps)
        agent.behavior = behavior
        banked = agent_type == "bird" or agent.fixed_wing
        radius = max(25., agent.s_star ** 2 / (9.81 * np.tan(np.radians(25)))) if banked else 30.
        lateral = np.array([-direction[1], direction[0], 0.])
        center = start + radius * lateral
        hold_goal = None
        latent, optical = [], []
        failure = None
        for frame in range(n):
            agent.speed_scale = 1.
            env.updraft = 0.
            mode = behavior
            if depth_mode == "passing_by":
                progress = frame / max(n-1, 1)
                env.x_goal = agent.pos + 160*direction + [0., -100*np.cos(np.pi*progress), 0.]
            if behavior == "thermal_circle":
                relative = agent.pos - center
                angle = np.arctan2(relative[1], relative[0]) + .5
                env.x_goal = center + [radius*np.cos(angle), radius*np.sin(angle), 0.]
                env.x_goal[2] = agent.pos[2]
                env.updraft = schedule.thermal_updraft(np.linalg.norm(relative[:2]), radius)
            elif behavior == "foraging_zigzag":
                env.x_goal = schedule.foraging_goal(frame/self.fps, agent.pos, direction)
            elif behavior == "sudden_escape" and frame >= events["dash"]:
                agent.speed_scale = events["dash_scale"]
            if scenario == "sudden_dash":
                agent.speed_scale = events["dash_scale"] if events["dash"] <= frame < events["brake"] else (events["brake_scale"] if frame >= events["brake"] else 1.)
            if scenario == "sharp_turns" and frame in events["turns"]:
                index = events["turns"].index(frame)
                current_yaw = np.arctan2(agent.u[1], agent.u[0]) + events["turn_angles_rad"][index]
                goal = agent.pos + [200*np.cos(current_yaw), 200*np.sin(current_yaw), 0.]
            # Scenario guidance explicitly overrides behavior guidance when active.
            if scenario == "sharp_turns":
                env.x_goal = goal.copy()
                mode = "scenario_turn"
            if scenario == "multi_mode" and events["hover_start"] <= frame < events["hover_end"]:
                if hold_goal is None:
                    hold_goal = agent.pos.copy()
                if banked:
                    angle = np.arctan2(agent.pos[1]-center[1], agent.pos[0]-center[0]) + .5
                    env.x_goal = center + [radius*np.cos(angle), radius*np.sin(angle), 0.]
                    mode = "loiter"
                else:
                    env.x_goal = hold_goal.copy()
                    mode = "position_hold"
            elif scenario == "multi_mode" and frame >= events["hover_end"]:
                env.x_goal = goal.copy()
            if agent.pos[2] <= 2. or not np.all(np.isfinite(agent.pos)):
                failure = "ground_or_nonfinite_state"
                break
            optical.append(agent.get_observation(frame))
            latent.append(dict(frame_index=frame, time_seconds=frame/self.fps, position_m=agent.pos.tolist(),
                               velocity_m_s=agent.v_ground.tolist(), acceleration_m_s2=agent.accel.tolist(),
                               airspeed_m_s=float(agent.s), bank_rad=float(agent.phi),
                               goal_m=env.x_goal.tolist(), wind_m_s=env.wind.tolist(),
                               speed_scale=agent.speed_scale, control_mode=mode,
                               dynamics=dict(agent.diagnostics)))
            if frame < n - 1:
                agent.step()
        # Keep the planned timeline: early termination is a rejection, not a shorter success.
        optical += [None] * (n-len(optical))
        observations, observation_meta = ObservationModel(
            camera, np.random.default_rng(observer_seed), noisy, self.noise_config).apply(optical, self.fps)
        if failure:
            for state in observation_meta["states"][len(latent):]:
                state["reason"] = failure
        feature_result = extract_features(observations, camera.width, camera.height, self.fps, self.feature_config)
        if failure:
            feature_result.update(feature_status="rejected", features=None)
            feature_result["reasons"].append(failure)
        sample_id = f"{family}:{agent_type}:{subtype}:{'noisy' if noisy else 'ideal'}"
        metadata = dict(sample_id=sample_id, family_id=family, label=agent_type, subtype=subtype,
                        scenario=scenario, behavior_mode=behavior, requested_depth_mode=depth_mode,
                        observation_profile="noisy" if noisy else "ideal", fps=self.fps, frame_count=n,
                        start_position_m=start.tolist(), initial_goal_m=initial_goal.tolist(),
                        commanded_goal_m=goal.tolist(), minimum_goal_distance_m=150.,
                        simulator_version=SIMULATOR_VERSION, feature_version=FEATURE_VERSION,
                        seed=self.seed, scenario_seed=scenario_seed, physics_seed=physical_seed,
                        observation_seed=observer_seed, camera=camera.to_dict(),
                        coordinate_space="post_cmc_residual_observation",
                        individual_parameters=dict(cruise_speed_m_s=agent.s_star, width_m=agent.real_width,
                                                   height_m=agent.real_height, flap_hz=agent.flap_hz,
                                                   physical_config=agent.config, scaling=individual_scaling),
                        parameter_manifest_sha256=self.provenance["sha256"],
                        behavior_schedule=schedule.metadata(), calibration_status="uncalibrated_no_real_A",
                        integration_max_step_s=agent.integration_step_s,
                        events=events, latent_failure=failure, observation=observation_meta)
        return dict(metadata=metadata, track=dict(track_id=int(sample_index), source_video_id=family,
                                                 history=observations, quality=observation_meta["quality"]),
                    world_truth=latent, optical_truth=optical, feature_result=feature_result)

    def run(self, samples_per_subtype=5, paired=True, scenarios=SCENARIOS):
        if samples_per_subtype < 1 or int(samples_per_subtype) != samples_per_subtype:
            raise ValueError("samples_per_subtype must be a positive integer")
        if not scenarios or any(s not in SCENARIOS for s in scenarios) or len(set(scenarios)) != len(scenarios):
            raise ValueError("Scenarios must be unique known names")
        families = [f"{self.seed}:{s}:{i}" for s in scenarios for i in range(samples_per_subtype)]
        strata = {f"{self.seed}:{s}:{i}": s for s in scenarios for i in range(samples_per_subtype)}
        splits = assign_splits(families, self.seed, strata)
        rows = []
        for name in ("manifest.json", "dataset_manifest.json"):
            existing_manifest = self.output_dir / name
            if existing_manifest.exists():
                import json
                if json.loads(existing_manifest.read_text(encoding="utf-8"))["simulator_version"] != SIMULATOR_VERSION:
                    raise ValueError("Refusing to overwrite artifacts from a different simulator version")
        path = self.output_dir / "raw_trajectories_v3.jsonl"

        def generate():
            for scenario in scenarios:
                for i in range(samples_per_subtype):
                    for label, subtypes in (("bird", tuple(SPECIES_CONFIG)), ("drone", DRONE_SUBTYPES)):
                        for subtype in subtypes:
                            for noisy in ((False, True) if paired else (True,)):
                                sample = self.simulate(scenario, label, subtype, i, noisy)
                                meta, feature = sample["metadata"], sample["feature_result"]
                                meta["split"] = splits[meta["family_id"]]
                                row = {k: meta[k] for k in ("sample_id", "family_id", "label", "subtype", "scenario",
                                                           "behavior_mode", "observation_profile", "split", "frame_count",
                                                           "simulator_version", "feature_version", "requested_depth_mode", "fps", "seed")}
                                row.update(attempted_missing_fraction=meta["observation"]["quality"]["missing_ratio"],
                                           camera_residual_enabled=meta["observation"]["camera_motion"]["enabled"],
                                           tracking_drift_enabled=meta["observation"]["tracking_drift"]["enabled"])
                                row.update(feature_status=feature["feature_status"],
                                           feature_config_id=feature["feature_config_id"],
                                           rejection_reason=";".join(feature["reasons"]),
                                           **{k: (feature["features"] or {}).get(k) for k in FEATURE_COLUMNS},
                                           **feature["quality"])
                                rows.append(row)
                                yield sample
        write_jsonl(path, generate())
        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / "simulation_features_v3.csv", index=False)
        manifest = dict(simulator_version=SIMULATOR_VERSION, feature_version=FEATURE_VERSION,
                        seed=self.seed, fps=self.fps, feature_config=asdict(self.feature_config),
                        feature_config_id=self.feature_config.fingerprint, noise_config=asdict(self.noise_config),
                        samples_per_subtype=samples_per_subtype, paired=paired, scenarios=list(scenarios),
                        rows=len(table), accepted=int((table.feature_status == "accepted").sum()),
                        family_splits=splits, calibration_status="uncalibrated_no_real_A",
                        split_policy="scenario-stratified families; approximately 70/15/15, rounded per scenario; <3 families all train",
                        validation_scope="physical_invariants_and_internal_consistency_only",
                        parameter_manifest_sha256=self.provenance["sha256"],
                        coordinate_space="post_cmc_residual_observation",
                        feature_columns=FEATURE_COLUMNS,
                        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
                        runtime_compatibility="NOT compatible with production v1/v2 model or RuleFilter thresholds")
        write_json(self.output_dir / "manifest.json", manifest)
        write_json(self.output_dir / "parameter_manifest.json", self.provenance)
        return table


class CoreFeatureExtractor:
    """Offline re-extraction shares exactly the function used for real tracks."""
    def __init__(self, config=None):
        self.config = config or FeatureConfig()

    def process(self, raw_jsonl):
        rows = []
        for sample in read_jsonl(raw_jsonl):
            meta = sample["metadata"]
            if meta["simulator_version"] not in ("3.0.0", SIMULATOR_VERSION):
                raise ValueError("Unsupported simulator version")
            camera = meta["camera"]
            result = extract_features(sample["track"]["history"], camera["width"], camera["height"],
                                      meta["fps"], self.config)
            if meta["latent_failure"]:
                result.update(features=None, feature_status="rejected")
                result["reasons"].append(meta["latent_failure"])
            rows.append(dict(sample_id=meta["sample_id"], **result))
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-subtype", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--no-pairs", action="store_true")
    args = parser.parse_args()
    table = BatchRunner(args.output, args.fps, args.seed).run(args.samples_per_subtype, not args.no_pairs)
    print(table.groupby(["label", "feature_status"], observed=True).size().to_string())
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
