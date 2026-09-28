"""Stage-4 force-based flight, continuous commands and calibrated optical sampling.

The physical model remains v4. This explicit sequence runner does not alter the
legacy BatchRunner distribution or warp the resulting feature vectors.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .camera import Camera
from .generators import BirdDyn, DroneDyn, Environment
from .observation import NoiseConfig, ObservationModel
from .pipeline import stable_seed


SEQUENCE_SIMULATOR_VERSION = "sequence-guidance-1.3.0"
BIRDS = ("pigeon", "seagull", "falcon")
QUADS = ("consumer_quad", "racing_quad", "hover_quad")


@dataclass(frozen=True)
class AcquisitionProfile:
    name: str = "sequence_prior"
    span_q10: float = 0.03
    span_q50: float = 0.12
    span_q90: float = 0.35
    span_multiplier: float = 1.0
    jitter_px: float = 0.35
    dropout_rate: float = 0.01
    burst_probability: float = 0.1
    drift_probability: float = 0.25
    camera_probability: float = 0.5
    fov_low_deg: float = 25.0
    fov_high_deg: float = 70.0
    jitter_ar1: float = .55
    jitter_ar2: float = 0.
    jitter_difficulty_gain: float = 1.
    jitter_motion_fraction: float = 0.
    aspect_4_3_probability: float = 0.

    def __post_init__(self):
        values = [v for k, v in asdict(self).items() if k != "name"]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite acquisition profile")
        if not 0 < self.span_q10 <= self.span_q50 <= self.span_q90 < 2:
            raise ValueError("Invalid apparent-span quantiles")
        if self.span_multiplier <= 0 or not 1 < self.fov_low_deg <= self.fov_high_deg < 175:
            raise ValueError("Invalid camera ranges")
        if not 0 <= self.aspect_4_3_probability <= 1:
            raise ValueError("Invalid aspect sampling probability")
        self.noise_config()

    def noise_config(self):
        return NoiseConfig(jitter_px=self.jitter_px, dropout_rate=self.dropout_rate,
                           burst_probability=self.burst_probability,
                           drift_probability=self.drift_probability,
                           camera_probability=self.camera_probability,
                           jitter_ar1=self.jitter_ar1, jitter_ar2=self.jitter_ar2,
                           jitter_difficulty_gain=self.jitter_difficulty_gain,
                           jitter_motion_fraction=self.jitter_motion_fraction)


@dataclass(frozen=True)
class GuidanceProfile:
    drone_probabilities: tuple = (.4, .25, .2, .15)
    bird_probabilities: tuple = (.25, .15, .2, .1, .3)
    drone_dwell_seconds: tuple = (.8, 2.4)
    bird_dwell_seconds: tuple = (2., 4.)

    def __post_init__(self):
        for values, size in ((self.drone_probabilities, 4), (self.bird_probabilities, 5)):
            if len(values) != size or not np.isfinite(values).all() or min(values) < 0 or not np.isclose(sum(values), 1):
                raise ValueError("Invalid guidance probabilities")
        for values in (self.drone_dwell_seconds, self.bird_dwell_seconds):
            if len(values) != 2 or not np.isfinite(values).all() or not 0 < values[0] <= values[1]:
                raise ValueError("Invalid command dwell range")


class CommandSchedule:
    """Dwell-time commands, not class-exclusive geometric waveform injection."""
    def __init__(self, label, duration, rng, guidance=None):
        if label not in ("bird", "drone"):
            raise ValueError("Unknown class")
        self.label = label
        guidance = guidance or GuidanceProfile()
        if label == "drone":
            templates = [("translate", "brake", "hold", "reverse", "climb", "turn"),
                         ("translate", "turn", "descend", "hold", "translate"),
                         ("smooth_cruise",), ("hold",)]
        else:
            templates = [("powered", "glide", "turn", "descending_glide", "powered"),
                         ("powered", "turn", "powered", "glide"),
                         ("glide",), ("circling_glide",), ("powered",)]
        probabilities = guidance.drone_probabilities if label == "drone" else guidance.bird_probabilities
        choice = int(rng.choice(len(templates), p=probabilities))
        self.template = templates[choice]
        self.segments = []
        elapsed, index = 0.0, 0
        while elapsed < duration:
            dwell_range = guidance.drone_dwell_seconds if label == "drone" else guidance.bird_dwell_seconds
            dwell = duration if len(self.template) == 1 else float(rng.uniform(*dwell_range))
            mode = self.template[index % len(self.template)]
            self.segments.append(dict(start_s=elapsed, end_s=min(elapsed+dwell, duration),
                                      mode=mode, speed_fraction=float(rng.uniform(.25, .85)),
                                      turn_rad=float(rng.choice([-1, 1])*rng.uniform(.5, 1.8)),
                                      vertical_m_s=float(rng.uniform(.8, 3.0))))
            elapsed += dwell
            index += 1

    def at(self, time):
        return next((s for s in self.segments if time < s["end_s"]-1e-10), self.segments[-1])


def latent_flight(label, subtype, seed, duration=8.0, fps=30, guidance=None):
    allowed = BIRDS if label == "bird" else QUADS if label == "drone" else ()
    if subtype not in allowed:
        raise ValueError("Sequence scope is birds and multicopters")
    if not np.isfinite([duration, fps]).all() or duration <= 0 or fps <= 0:
        raise ValueError("Invalid flight duration/rate")
    rng = np.random.default_rng(stable_seed(seed, label, subtype, "latent"))
    yaw = float(rng.uniform(-np.pi, np.pi))
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    env = Environment(fps=fps, goal_pos=[100, 0, 120], wind_speed=float(rng.uniform(0, 2)),
                      gust_intensity=float(rng.uniform(.05, .4)),
                      wind_direction=float(rng.uniform(-np.pi, np.pi)),
                      rng=np.random.default_rng(stable_seed(seed, "wind")))
    kwargs = dict(start_pos=[0, 0, 120], heading=direction, rng=rng)
    agent = BirdDyn(env, species=subtype, **kwargs) if label == "bird" else DroneDyn(env, model=subtype, **kwargs)
    scaling = agent.randomize_individual(rng)
    if label == "drone":
        agent.v_ground = direction * float(rng.uniform(0, 8))
        agent.s = float(np.linalg.norm(agent.v_ground-env.wind))
        agent.initialize_trim()
    schedule = CommandSchedule(label, duration, rng, guidance)
    if guidance is not None and label == "drone" and schedule.template == ("hold",):
        agent.v_ground[:] = 0.
        agent.s = float(np.linalg.norm(env.wind))
        agent.initialize_trim()
    previous = None
    hold = agent.pos.copy()
    center = agent.pos.copy()
    turn_sign = 1.0
    radius = max(18., agent.s_star**2 / (9.81*np.tan(np.radians(25))))
    rows, failure = [], None
    for frame in range(int(round(duration*fps))+1):
        time = frame / fps
        segment = schedule.at(time)
        mode = segment["mode"]
        if segment is not previous:
            if mode == "reverse":
                yaw += np.pi
            elif mode in ("turn", "circling_glide"):
                yaw += segment["turn_rad"]
            direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
            hold = agent.pos.copy()
            turn_sign = float(np.sign(segment["turn_rad"]))
            center = agent.pos + radius * np.array([-direction[1], direction[0], 0.]) * turn_sign
            previous = segment
        env.command_velocity = None
        env.updraft = 0.
        agent.speed_scale = 1.
        if label == "drone":
            if mode == "hold":
                env.x_goal = hold.copy()
            else:
                velocity = direction * agent.s_star * segment["speed_fraction"]
                if mode == "brake":
                    velocity[:] = 0.
                if mode in ("climb", "descend"):
                    velocity[2] = segment["vertical_m_s"] * (1 if mode == "climb" else -1)
                env.command_velocity = velocity
                env.x_goal = agent.pos + velocity * 5
            agent.behavior = "cruise"
        else:
            agent.behavior = "glide" if mode in ("glide", "descending_glide", "circling_glide") else "flap_jitter"
            env.x_goal = agent.pos + direction * 100
            if mode in ("turn", "circling_glide"):
                angle = np.arctan2(agent.pos[1]-center[1], agent.pos[0]-center[0]) + turn_sign*.5
                env.x_goal = center + [radius*np.cos(angle), radius*np.sin(angle), 0.]
                env.x_goal[2] = agent.pos[2]
            elif mode == "powered":
                env.x_goal[2] = max(80., agent.pos[2])
            if mode == "descending_glide":
                env.x_goal[2] = agent.pos[2]-25.
        if not np.isfinite(agent.pos).all() or agent.pos[2] <= 2:
            failure = "ground_or_nonfinite_state"
            break
        rows.append(dict(frame_index=frame, timestamp_ms=int(round(time*1000)),
                         position_m=agent.pos.tolist(), velocity_m_s=agent.v_ground.tolist(),
                         control_mode=mode, behavior=agent.behavior,
                         command_velocity=None if env.command_velocity is None else env.command_velocity.tolist(),
                         goal_m=env.x_goal.tolist(), thrust_n=float(agent.diagnostics.get("thrust_n", 0)),
                         airspeed_m_s=agent.s))
        if frame < int(round(duration*fps)):
            agent.step()
    return dict(label=label, subtype=subtype, seed=int(seed), fps=fps, duration_s=duration,
                schedule=schedule.segments, template=list(schedule.template), world_truth=rows,
                latent_failure=failure, scaling=scaling, width_m=agent.real_width, height_m=agent.real_height,
                physical_parameters=agent.config, simulator=SEQUENCE_SIMULATOR_VERSION,
                guidance=asdict(guidance or GuidanceProfile()))


def observe_flight(flight, profile):
    """Camera generation uses the same profile for both labels and physical units."""
    rng = np.random.default_rng(stable_seed(flight["seed"], "acquisition"))
    rows = flight["world_truth"]
    positions = np.array([r["position_m"] for r in rows])
    center = np.median(positions, axis=0)
    azimuth, elevation = rng.uniform(-np.pi, np.pi), np.radians(rng.uniform(5, 55))
    forward = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    u = float(rng.random())
    span = float(np.exp(np.interp(u, [0, .5, 1], np.log([profile.span_q10, profile.span_q50, profile.span_q90]))))
    span *= profile.span_multiplier
    # Apparent angular extent identifies a speed/projection ratio, not true range.
    # A common 15 m/s framing reference keeps camera sampling independent of the
    # object's actual speed. Per-flight speed compensation would erase that signal.
    reference_travel = 30.
    fov = float(rng.uniform(profile.fov_low_deg, profile.fov_high_deg))
    distance = float(np.clip(reference_travel/(2*np.tan(np.radians(fov)/2)*span), 20., 1500.))
    height = 1440 if profile.aspect_4_3_probability and rng.random() < profile.aspect_4_3_probability else 1080
    camera = Camera(height=height, position=tuple(center-distance*forward), look_at=tuple(center), horizontal_fov_deg=fov)
    optical = []
    for row in rows:
        bbox = camera.project(row["position_m"], flight["width_m"], flight["height_m"])
        optical.append(dict(frame_index=row["frame_index"], timestamp_ms=row["timestamp_ms"], conf=1., **bbox)
                       if camera.visible(bbox) else None)
    expected = int(round(flight["duration_s"]*flight["fps"]))+1
    optical += [None]*(expected-len(optical))
    history, noise = ObservationModel(camera, np.random.default_rng(stable_seed(flight["seed"], "observer")),
                                     config=profile.noise_config()).apply(optical, flight["fps"])
    return dict(track=dict(track_id=flight["seed"], source_video_id=f"sequence:{flight['seed']}",
                           processed_width=camera.width, processed_height=camera.height,
                           stabilization=dict(applied=True, method="synthetic_post_cmc_surrogate"),
                           history=history, quality=noise["quality"]),
                metadata=dict(label=flight["label"], subtype=flight["subtype"], seed=flight["seed"],
                              simulator=SEQUENCE_SIMULATOR_VERSION, acquisition=asdict(profile),
                              camera=camera.to_dict(), requested_two_second_span=span,
                              camera_distance_m=distance, schedule=flight["schedule"],
                              guidance=flight.get("guidance", asdict(GuidanceProfile())),
                              template=flight["template"], latent_failure=flight["latent_failure"],
                              observation=noise, framing="fixed camera centered on latent median; no image warping"),
                world_truth=rows, optical_truth=optical)
