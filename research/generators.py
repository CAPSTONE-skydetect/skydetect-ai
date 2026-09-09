"""Force-based flight agents; evidence and remaining priors live in parameters.py."""
from copy import deepcopy
from pathlib import Path

import numpy as np

if __package__:
    from .camera import Camera
    from .dynamics import G, RHO, MAX_STEP_S, QuadBody, drag_polar, wing_step
    from .parameters import DRONE_CONFIG, SAMPLING, SPECIES_CONFIG
else:
    from camera import Camera
    from dynamics import G, RHO, MAX_STEP_S, QuadBody, drag_polar, wing_step
    from parameters import DRONE_CONFIG, SAMPLING, SPECIES_CONFIG


def limit_vector(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Environment:
    def __init__(self, fps=30, wind_speed=1., gust_intensity=.3, goal_pos=None,
                 wind_direction=0., rng=None, camera=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if not np.all(np.isfinite([wind_speed, gust_intensity, wind_direction])) or min(wind_speed, gust_intensity) < 0:
            raise ValueError("Wind parameters must be finite and magnitudes nonnegative")
        self.fps, self.dt = float(fps), 1. / fps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.camera = camera or Camera()
        self.wind_base = wind_speed * np.array([np.cos(wind_direction), np.sin(wind_direction), 0.])
        self.gust_std, self.gust_tau = float(gust_intensity), SAMPLING["gust_tau_s"]
        self.current_gust = np.zeros(3)
        self.wind = self.wind_base.copy()
        self.x_goal = np.array(goal_pos if goal_pos is not None else [300., 50., 90.], dtype=float)
        if self.x_goal.shape != (3,) or not np.isfinite(self.x_goal).all():
            raise ValueError("Goal must be a finite 3D position")
        self.updraft = 0.

    @property
    def h_star(self):
        return self.x_goal[2]

    @h_star.setter
    def h_star(self, value):
        self.x_goal[2] = value

    def update_and_get_wind(self, apply_noise=True):
        # Exact OU transition; gust_std is stationary standard deviation (m/s).
        decay = np.exp(-self.dt / self.gust_tau)
        self.current_gust = (decay * self.current_gust
                             + self.gust_std * np.sqrt(1 - decay ** 2) * self.rng.normal(size=3))
        self.wind = self.wind_base + self.current_gust + [0., 0., self.updraft]
        return self.wind.copy()


class BaseAgent:
    def __init__(self, env, start_pos=None, start_speed=10., heading=None, rng=None):
        self.env, self.dt = env, env.dt
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pos = np.array(start_pos if start_pos is not None else [0., 100., 90.], dtype=float)
        if self.pos.shape != (3,) or not np.isfinite(self.pos).all() or not np.isfinite(start_speed) or start_speed < 0:
            raise ValueError("Start position and nonnegative speed must be finite")
        direction = env.x_goal - self.pos if heading is None else np.asarray(heading, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("Heading must be a finite 3D direction")
        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([1., 0., 0.])
        self.u = direction / np.linalg.norm(direction)
        self.s = float(start_speed)
        self.history = []
        self.phi = 0.
        self.gamma = float(np.arcsin(self.u[2]))
        self.v_ground = self.s * self.u + env.wind
        self.accel = np.zeros(3)
        self.noise_state = np.zeros(3)
        self.speed_scale = 1.
        self.behavior = "cruise"
        self.time = 0.
        self.flap_hz, self.flap_phase = 4., 0.
        self.integration_step_s = MAX_STEP_S
        self.thrust = 0.
        self.diagnostics = {"model": "initial_state"}

    def get_unit_vector_to_goal(self):
        direction = self.env.x_goal - self.pos
        return direction / max(np.linalg.norm(direction), 1e-12)

    def _disturbance(self, std):
        decay = np.exp(-self.dt / self.config["process_tau_s"])
        self.noise_state = (decay * self.noise_state
                            + std * np.sqrt(1 - decay ** 2) * self.rng.normal(size=3))
        return self.noise_state.copy()

    def update_position(self, velocity):
        self.pos += velocity * self.dt
        self.v_ground = velocity.copy()
        self.time += self.dt
        self.history.append(self.pos.copy())
        return self.pos.copy()

    def get_observation(self, frame_index, apply_noise=False):
        if apply_noise:
            raise ValueError("Use ObservationModel for noise; get_observation returns optical truth")
        width, height = self.real_width, self.real_height
        if self.is_bird and self.behavior not in ("glide", "thermal_circle"):
            phase = 2 * np.pi * self.flap_hz * self.time + self.flap_phase
            width *= .8 + .2 * np.cos(phase)
            height *= 1. + .25 * np.sin(phase)
        # Approximate silhouette foreshortening without an articulated renderer.
        view = self.env.camera.rotation[0]
        width *= .6 + .4 * np.sqrt(max(0., 1. - (self.u @ view) ** 2))
        bbox = self.env.camera.project(self.pos, width, height)
        if not self.env.camera.visible(bbox):
            return None
        return dict(frame_index=int(frame_index),
                    timestamp_ms=int(round(frame_index * 1000 / self.env.fps)),
                    **bbox, conf=1.)

    def _finish_step(self):
        self.time += self.dt
        self.history.append(self.pos.copy())
        return self.pos.copy()

    def initialize_trim(self):
        c = self.config
        if "wing_area_m2" in c:
            q_area = .5*RHO*self.s**2*c["wing_area_m2"]
            cl = np.clip(c["mass_kg"]*G/max(q_area,.01),0.,c["cl_max"])
            self.thrust = min(c["mass_kg"]*G*c["thrust_weight"],
                              q_area*drag_polar(cl,c["span_m"],c["wing_area_m2"],c["cd0"],c["oswald_e"]))
        else:
            air = self.v_ground-self.env.wind
            force = np.array([0.,0.,c["mass_kg"]*G])+c["drag_kg_m"]*np.linalg.norm(air)*air
            b3 = force/np.linalg.norm(force)
            b2 = np.cross(b3,self.body.heading)
            b2 /= np.linalg.norm(b2)
            self.body.rotation = np.column_stack((np.cross(b2,b3),b2,b3))
            self.body.rotor_speed[:] = np.sqrt(min(np.linalg.norm(force)/4,self.body.max_rotor_thrust)/c["thrust_coefficient"])

    def randomize_individual(self, rng):
        length = float(rng.uniform(*SAMPLING["length_scale"]))
        self.real_width *= length
        self.real_height *= length
        self.config["mass_kg"] *= length**3
        if "wing_area_m2" in self.config:
            self.config["wing_area_m2"] *= length**2
            self.config["span_m"] *= length
        else:
            self.config["arm_m"] *= length
            self.config["inertia_kg_m2"] = (np.asarray(self.config["inertia_kg_m2"])*length**5).tolist()
            self.config["drag_kg_m"] *= length**2
            self.body = QuadBody(self.config, np.arctan2(self.u[1], self.u[0]))
        self.s_star = float(np.clip(self.s_star*rng.uniform(*SAMPLING["cruise_scale"]),
                                    self.config.get("min_speed", 0.), self.config["max_speed"]))
        self.s = max(self.s_star*rng.uniform(*SAMPLING["initial_speed_scale"]), self.config.get("min_speed",0.))
        self.v_ground = self.s*self.u + (self.env.wind if "wing_area_m2" in self.config else 0.)
        self.s = float(np.linalg.norm(self.v_ground-self.env.wind))
        self.flap_hz = self.config.get("flap_hz",0.)*float(rng.uniform(*SAMPLING["flap_scale"]))
        self.flap_phase = float(rng.uniform(0.,2*np.pi))
        self.initialize_trim()
        return dict(length_scale=length, mass_scale=length**3, evidence="geometric_similarity_design_prior")


class BirdDyn(BaseAgent):
    def __init__(self, env, species="pigeon", apply_noise=False, **kwargs):
        super().__init__(env, **kwargs)
        self.config = deepcopy(SPECIES_CONFIG[species])
        self.is_bird = True
        self.species = species
        self.s_star, self.a_max = self.config["s_star"], self.config["a_max"]
        self.phi_max = np.radians(self.config["phi_max"])
        self.s = max(self.s, self.config["min_speed"])
        self.v_ground = self.s*self.u + env.wind
        self.real_width, self.real_height = self.config["real_w"], self.config["real_h"]
        self.flap_hz = self.config["flap_hz"]
        self.initialize_trim()

    def step(self, apply_noise=False, wind=None):
        if wind is None:
            self.env.update_and_get_wind()
        else:
            self.env.wind = np.asarray(wind, dtype=float).copy()
        wing_step(self, self._disturbance(self.config["sigma_s"]))
        return self._finish_step()


class DroneDyn(BaseAgent):
    def __init__(self, env, model="quadcopter", apply_noise=False, **kwargs):
        super().__init__(env, **kwargs)
        self.model = "consumer_quad" if model == "quadcopter" else model
        self.config = deepcopy(DRONE_CONFIG[self.model])
        self.is_bird = False
        self.s_star, self.a_max = self.config["s_star"], self.config["a_max"]
        self.real_width, self.real_height = self.config["real_w"], self.config["real_h"]
        self.fixed_wing = self.model == "fixed_wing_drone"
        self.phi_max = np.radians(self.config.get("phi_max", 0.))
        if self.fixed_wing:
            self.s = max(self.s, self.config["min_speed"])
        self.v_ground = self.s * self.u + (self.env.wind if self.fixed_wing else 0.)
        self.v_air = self.v_ground - self.env.wind
        if not self.fixed_wing:
            self.s = float(np.linalg.norm(self.v_air))
            self.body = QuadBody(self.config, np.arctan2(self.u[1],self.u[0]))
        self.flap_hz = 0.
        self.initialize_trim()

    def step(self, apply_noise=False, wind=None):
        if wind is None:
            self.env.update_and_get_wind()
        else:
            self.env.wind = np.asarray(wind, dtype=float).copy()
        if self.fixed_wing:
            wing_step(self, self._disturbance(self.config["sigma_s"]))
        else:
            disturbance = self._disturbance(self.config["sigma_s"])
            disturbance += self.config["sigma_u"]*np.array([-self.u[1],self.u[0],0.])*self.noise_state[1]
            self.body.step(self, disturbance)
        return self._finish_step()


class TrajectoryGenerator:
    """Notebook wrapper using the same observation layer as BatchRunner."""
    def __init__(self, env, output_dir="research/output"):
        self.env, self.output_dir = env, Path(output_dir)

    def generate(self, agent, num_frames=150, track_id=1, apply_noise=False, seed=0):
        """seed controls observations only; seed env/agent separately or use BatchRunner."""
        if int(num_frames) != num_frames or num_frames < 1 or agent.env is not self.env:
            raise ValueError("Need a positive integer frame count and this generator's environment")
        if __package__:
            from .observation import ObservationModel
        else:
            from observation import ObservationModel
        truth = []
        for frame in range(num_frames):
            truth.append(agent.get_observation(frame))
            agent.step()
        observer = ObservationModel(self.env.camera, np.random.default_rng(seed), apply_noise)
        observations, metadata = observer.apply(truth, self.env.fps)
        if not observations:
            raise ValueError("No visible observations; use BatchRunner to retain rejection diagnostics")
        return {"track_id": track_id, "history": observations,
                "source_video_id": f"synthetic-{seed}", "quality": metadata["quality"]}

    def save(self, data, filename=None):
        import json
        path = self.output_dir / (filename or "track_sequence.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")
        return str(path)
