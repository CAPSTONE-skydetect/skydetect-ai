"""Force-based research models in SI units, world z up; no speed/state clipping.

Wing model: translational dynamics + bank actuator, NOT articulated bird CFD.
Quad model: rigid-body 6 DOF + four first-order rotor-speed states, not blade CFD.
"""
import numpy as np
from scipy.spatial.transform import Rotation

G = 9.81
RHO = 1.225
UP = np.array([0., 0., 1.])
MAX_STEP_S = 1. / 240.


def limit(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1.e-12))


def drag_polar(cl, span, area, cd0, efficiency):
    return cd0 + cl * cl / (np.pi * (span * span / area) * efficiency)


def wing_step(agent, disturbance):
    c, dt = agent.config, agent.dt
    steps = int(np.ceil(dt / agent.integration_step_s))
    h = dt / steps
    old_v = agent.v_ground.copy()
    flapping = agent.is_bird and agent.behavior not in ("glide", "thermal_circle")
    gliding = agent.behavior in ("glide", "thermal_circle")
    for j in range(steps):
        air = agent.v_ground - agent.env.wind
        speed = max(np.linalg.norm(air), .1)
        forward = air / speed
        left = np.cross(UP, forward)
        left /= max(np.linalg.norm(left), 1.e-9)
        normal = np.cross(forward, left)
        gamma = np.arcsin(np.clip(forward[2], -1., 1.))
        goal = agent.env.x_goal - agent.pos
        yaw_error = (np.arctan2(goal[1], goal[0]) - np.arctan2(forward[1], forward[0]) + np.pi) % (2*np.pi) - np.pi
        bank = np.arctan(speed*c["k_g"]*yaw_error/G) + c["sigma_phi"]*disturbance[1]
        bank = np.clip(bank, -agent.phi_max, agent.phi_max)
        agent.phi += np.clip(4*(bank-agent.phi), -c["bank_rate_rad_s"], c["bank_rate_rad_s"])*h
        lift_dir = normal*np.cos(agent.phi) + left*np.sin(agent.phi)
        q_area = .5*RHO*speed**2*c["wing_area_m2"]
        induced = 1/(np.pi*(c["span_m"]**2/c["wing_area_m2"])*c["oswald_e"])
        if gliding:
            cl_command = min(np.sqrt(c["cd0"]/induced), c["cl_max"])
        else:
            gamma_limit = min(np.radians(c["gamma_limit_deg"]), np.arcsin(min(1., c["climb_command_m_s"]/speed)))
            desired_gamma = np.clip(np.arctan2(goal[2], max(np.linalg.norm(goal[:2]), 20.)), -gamma_limit, gamma_limit)
            cl_command = c["mass_kg"]*(G*np.cos(gamma)+speed*c["gamma_gain"]*(desired_gamma-gamma)) / max(q_area*np.cos(agent.phi), .01)
        phase = 2*np.pi*agent.flap_hz*(agent.time+(j+.5)*h) + agent.flap_phase
        lift_factor = 1+c["lift_modulation"]*np.cos(phase) if flapping else 1.
        cl_requested = max(0., cl_command)*lift_factor
        cl = np.clip(cl_requested, 0., c["cl_max"])
        lift = q_area*cl
        drag = q_area*drag_polar(cl, c["span_m"], c["wing_area_m2"], c["cd0"], c["oswald_e"])
        target_speed = np.clip(agent.s_star*agent.speed_scale, c["min_speed"], c["max_speed"])
        acceleration_command = np.clip(c["k_s"]*(target_speed-speed)+disturbance[0], -agent.a_max, agent.a_max)
        thrust_target = 0. if gliding else max(0., drag+c["mass_kg"]*(acceleration_command+G*np.sin(gamma)))
        max_thrust = c["mass_kg"]*G*c["thrust_weight"]
        thrust_target = min(thrust_target, max_thrust)
        agent.thrust += (thrust_target-agent.thrust)*(1-np.exp(-h/c["thrust_tau_s"]))
        # Glide has zero propulsive force: no hidden altitude/speed controller supplies energy.
        if gliding:
            agent.thrust = 0.
        thrust = min(max_thrust, agent.thrust*(1+c["thrust_modulation"]*np.cos(phase)) if flapping else agent.thrust)
        force = lift*lift_dir + (thrust-drag)*forward - c["mass_kg"]*G*UP
        acceleration = force/c["mass_kg"]
        agent.pos += agent.v_ground*h + .5*acceleration*h*h
        agent.v_ground += acceleration*h
        agent.diagnostics = dict(model="flapping_force" if agent.is_bird else "fixed_wing_force",
                                 lift_n=float(lift), drag_n=float(drag), thrust_n=float(thrust),
                                 cl=float(cl), cl_requested=float(cl_requested),
                                 lift_saturated=bool(cl_requested > c["cl_max"]), flapping=flapping,
                                 flap_phase_rad=float(phase % (2*np.pi)), force_world_n=force.tolist(),
                                 mechanical_energy_j=float(.5*c["mass_kg"]*np.dot(agent.v_ground, agent.v_ground)+c["mass_kg"]*G*agent.pos[2]))
    agent.accel = (agent.v_ground-old_v)/dt
    air = agent.v_ground-agent.env.wind
    agent.s = float(np.linalg.norm(air))
    agent.u = air/max(agent.s, 1.e-9)
    agent.gamma = float(np.arcsin(np.clip(agent.u[2], -1., 1.)))


class QuadBody:
    def __init__(self, config, yaw=0.):
        self.c = config
        self.rotation = Rotation.from_euler("z", yaw).as_matrix()
        self.omega = np.zeros(3)
        self.inertia = np.asarray(config["inertia_kg_m2"], dtype=float)
        self.rotor_speed = np.full(4, np.sqrt(config["mass_kg"]*G/(4*config["thrust_coefficient"])))
        self.command_accel = np.zeros(3)
        self.heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        arm = config["arm_m"] / np.sqrt(2.)
        # Rotors at (+x,+y),(-x,+y),(-x,-y),(+x,-y); thrust along body +z.
        self.mixer = np.array([[1.,1.,1.,1.], [arm,arm,-arm,-arm],
                               [-arm,arm,arm,-arm],
                               np.array([1.,-1.,1.,-1.])*config["torque_arm_m"]])
        self.inverse_mixer = np.linalg.inv(self.mixer)
        self.last = {}

    @property
    def max_rotor_thrust(self):
        return self.c["mass_kg"]*G*self.c["thrust_weight"]/4

    def advance(self, pos, velocity, wind, desired_rotor_thrust, h):
        """One open-loop rigid-body step; also used by independent force-balance tests."""
        c = self.c
        desired = np.clip(desired_rotor_thrust, 0., self.max_rotor_thrust)
        requested_speed = np.sqrt(desired/c["thrust_coefficient"])
        self.rotor_speed += (requested_speed-self.rotor_speed)*(1-np.exp(-h/c["motor_tau_s"]))
        thrusts = c["thrust_coefficient"]*self.rotor_speed**2
        wrench = self.mixer @ thrusts
        air = velocity-wind
        drag = -c["drag_kg_m"]*np.linalg.norm(air)*air
        force = self.rotation[:,2]*wrench[0] + drag - c["mass_kg"]*G*UP
        accel = force/c["mass_kg"]
        alpha = (wrench[1:]-np.cross(self.omega, self.inertia*self.omega))/self.inertia
        midpoint_omega = self.omega+.5*h*alpha
        self.rotation = self.rotation @ Rotation.from_rotvec(midpoint_omega*h).as_matrix()
        self.omega += alpha*h
        pos += velocity*h+.5*accel*h*h
        velocity += accel*h
        self.last = dict(model="quad_6dof", rotor_thrust_n=thrusts.tolist(), rotor_speed_rad_s=self.rotor_speed.tolist(),
                         rotation_body_to_world=self.rotation.tolist(), angular_velocity_rad_s=self.omega.tolist(),
                         force_world_n=force.tolist(), torque_body_nm=wrench[1:].tolist(),
                         thrust_n=float(wrench[0]), drag_n=float(np.linalg.norm(drag)),
                         allocation_saturated=bool(np.any(desired != desired_rotor_thrust)))

    def step(self, agent, disturbance):
        c = self.c
        target_velocity = limit(c["position_gain"]*(agent.env.x_goal-agent.pos), min(agent.s_star*agent.speed_scale, c["max_speed"]))
        requested = limit(c["k_a"]*(target_velocity-agent.v_ground)+disturbance, agent.a_max)
        self.command_accel += limit(requested-self.command_accel, c["jerk_max"]*agent.dt)
        steps = int(np.ceil(agent.dt/agent.integration_step_s))
        h = agent.dt/steps
        old_v = agent.v_ground.copy()
        for _ in range(steps):
            air = agent.v_ground-agent.env.wind
            drag = -c["drag_kg_m"]*np.linalg.norm(air)*air
            desired_force = c["mass_kg"]*(self.command_accel+G*UP)-drag
            desired_force[2] = max(desired_force[2], .1*c["mass_kg"]*G)
            desired_force[:2] = limit(desired_force[:2], desired_force[2]*np.tan(np.radians(c["tilt_max_deg"])))
            b3 = desired_force/np.linalg.norm(desired_force)
            b2 = np.cross(b3, self.heading)
            b2 /= np.linalg.norm(b2)
            desired_rotation = np.column_stack((np.cross(b2,b3), b2, b3))
            error_matrix = .5*(desired_rotation.T @ self.rotation - self.rotation.T @ desired_rotation)
            error = np.array([error_matrix[2,1], error_matrix[0,2], error_matrix[1,0]])
            torque = self.inertia*(-c["attitude_gain"]*error-c["angular_damping"]*self.omega) + np.cross(self.omega,self.inertia*self.omega)
            collective = max(0., np.dot(desired_force, self.rotation[:,2]))
            desired_thrust = self.inverse_mixer @ np.r_[collective,torque]
            self.advance(agent.pos, agent.v_ground, agent.env.wind, desired_thrust, h)
        agent.accel = (agent.v_ground-old_v)/agent.dt
        agent.v_air = agent.v_ground-agent.env.wind
        agent.s = float(np.linalg.norm(agent.v_air))
        if np.linalg.norm(agent.v_ground) > 1.e-9:
            agent.u = agent.v_ground/np.linalg.norm(agent.v_ground)
        agent.phi = float(np.arctan2(self.rotation[2,1],self.rotation[2,2]))
        agent.diagnostics = dict(self.last, command_acceleration_m_s2=self.command_accel.tolist())
