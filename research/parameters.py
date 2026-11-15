"""Auditable priors, NOT a fitted population model or hardware digital twin."""
from copy import deepcopy
import hashlib
import json

SOURCES = {
    "quad_equations": {
        "url": "https://arxiv.org/abs/1003.2005",
        "title": "Lee, Leok & McClamroch (2010), Control of Complex Maneuvers for a Quadrotor UAV using Geometric Methods on SE(3)",
        "scope": "Rigid-body equations only; our controller and motor model are not their validated controller."},
    "drag_polar": {
        "url": "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
        "title": "NASA Glenn: Induced Drag Coefficient",
        "scope": "CD = CD0 + CL^2/(pi*AR*e); not species-specific aerodynamic coefficients."},
    "morphology": {
        "url": "https://nora.nerc.ac.uk/id/eprint/533121/1/rsif.2022.0168.pdf",
        "title": "The role of wingbeat frequency and amplitude in flight power (2022), Table 1",
        "scope": "Pigeon sample n=9: 456 g, span .647 m, area .064 m2. Kittiwake n=3: 387 g, .965 m, .101 m2; not all gulls."},
    "pigeon_flap": {
        "url": "https://journals.biologists.com/jeb/article/218/3/480/14476/Pigeons-produce-aerodynamic-torques-through",
        "title": "Pigeons produce aerodynamic torques through changes in wing trajectory during low speed aerial turns (2015)",
        "scope": "8.3 +/- .3 Hz during low-speed turns (about 3.3 m/s); does NOT establish a cruise-frequency distribution."},
    "pigeon_force": {
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3250151/",
        "title": "Pigeons steer like helicopters and generate down- and upstroke lift during low speed turns (2011)",
        "scope": "Motivates phase-dependent forces. Our sinusoidal waveform is not a fit to the measured wingbeat."},
    "phantom": {
        "url": "https://www.dji.com/support/product/phantom-4-pro?from=landing_page&site=brandsite",
        "title": "DJI Phantom 4 Pro official specifications, Aircraft / S-mode",
        "scope": "Mass 1.388 kg, diagonal .350 m, max speed 72 km/h, max pitch 42 degrees. No inertia, motor lag or drag supplied."},
    "state_duration": {
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3751962/",
        "title": "Hidden Markov Models: The Best Models for Forager Movements? (2013)",
        "scope": "Explicit movement-state durations motivate renewal guidance; no GPS fit, inference or learned transition matrix here."},
}

WING_COMMON = dict(cd0=.035, oswald_e=.8, cl_max=1.6, thrust_weight=.7,
                   thrust_tau_s=.12, bank_rate_rad_s=.9, gamma_gain=1.2,
                   gamma_limit_deg=20., climb_command_m_s=4., process_tau_s=.5,
                   lift_modulation=.3, thrust_modulation=.25)
SPECIES_CONFIG = {
    "pigeon": dict(s_star=15., min_speed=5., max_speed=24., a_max=5., phi_max=45.,
                   k_s=1.5, k_g=1., sigma_s=.5, sigma_phi=.06, real_w=.647, real_h=.2,
                   mass_kg=.456, wing_area_m2=.064, span_m=.647, flap_hz=8.3),
    # Keep the public subtype label; the morphology anchor is specifically Rissa tridactyla.
    "seagull": dict(s_star=15., min_speed=6., max_speed=26., a_max=4., phi_max=35.,
                    k_s=.8, k_g=.8, sigma_s=.3, sigma_phi=.04, real_w=.965, real_h=.3,
                    mass_kg=.387, wing_area_m2=.101, span_m=.965, flap_hz=4.),
    "falcon": dict(s_star=25., min_speed=8., max_speed=40., a_max=8., phi_max=55.,
                   k_s=1.2, k_g=1.2, sigma_s=.4, sigma_phi=.05, real_w=1., real_h=.25,
                   mass_kg=.9, wing_area_m2=.16, span_m=1., flap_hz=5.),
}
for _config in SPECIES_CONFIG.values():
    _config.update(WING_COMMON)

QUAD_COMMON = dict(thrust_coefficient=1.e-5, torque_arm_m=.015, motor_tau_s=.045,
                   attitude_gain=80., angular_damping=18., process_tau_s=.5,
                   position_gain=.65)
DRONE_CONFIG = {
    "consumer_quad": dict(s_star=14., max_speed=20., a_max=7., jerk_max=14., k_a=1.8,
                          sigma_s=.3, sigma_u=.08, real_w=.5, real_h=.15,
                          mass_kg=1.388, arm_m=.175, inertia_kg_m2=[.02,.02,.035],
                          thrust_weight=2.5, tilt_max_deg=42., drag_kg_m=.018),
    "racing_quad": dict(s_star=24., max_speed=38., a_max=16., jerk_max=45., k_a=3.,
                        sigma_s=.6, sigma_u=.15, real_w=.32, real_h=.10,
                        mass_kg=.65, arm_m=.11, inertia_kg_m2=[.004,.004,.007],
                        thrust_weight=5., tilt_max_deg=65., drag_kg_m=.004),
    "hover_quad": dict(s_star=8., max_speed=16., a_max=5., jerk_max=10., k_a=1.8,
                       sigma_s=.2, sigma_u=.06, real_w=.6, real_h=.18,
                       mass_kg=1.388, arm_m=.175, inertia_kg_m2=[.02,.02,.035],
                       thrust_weight=2.5, tilt_max_deg=30., drag_kg_m=.018),
    "fixed_wing_drone": dict(s_star=20., min_speed=10., max_speed=32., a_max=4.,
                             phi_max=40., k_s=1., k_g=.8, sigma_s=.2, sigma_phi=.02,
                             real_w=1.2, real_h=.18, mass_kg=1.2, wing_area_m2=.24,
                             span_m=1.2, flap_hz=0.),
}
for _name, _config in DRONE_CONFIG.items():
    _config.update(WING_COMMON if _name == "fixed_wing_drone" else QUAD_COMMON)
DRONE_CONFIG["fixed_wing_drone"].update(thrust_tau_s=.25, lift_modulation=0., thrust_modulation=0.)
DRONE_CONFIG["quadcopter"] = deepcopy(DRONE_CONFIG["consumer_quad"])

SAMPLING = dict(length_scale=[.9,1.1], cruise_scale=[.8,1.2], initial_speed_scale=[.8,1.],
                flap_scale=[.85,1.15], wind_speed_m_s=[0.,3.], gust_std_m_s=[0.,.6],
                gust_tau_s=1.2, foraging_duration_s=[.7,3.5], foraging_turn_deg=[15.,65.],
                dash_scale=[1.2,1.65], brake_scale=[.5,.8], turn_angle_deg=[45.,135.],
                thermal_peak_m_s=[1.,3.], thermal_radius_scale=[1.3,2.2])
OBSERVATION_PRIORS = dict(jitter_px=.35, bbox_log_std=.035, dropout_rate=.03,
                          drift_probability=.65, burst_probability=.65, camera_probability=.75,
                          roi_scale=[.8,1.5])

UNITS = dict(s_star="m/s", min_speed="m/s (initial/command floor, not a physical bound)",
             max_speed="m/s (command ceiling, not a state clamp)", a_max="m/s2 (controller)",
             phi_max="deg (bank command)", k_s="1/s", k_g="1/s", sigma_s="m/s2",
             sigma_phi="rad/(m/s2)", real_w="m", real_h="m", mass_kg="kg", wing_area_m2="m2",
             span_m="m", flap_hz="Hz", cd0="1", oswald_e="1", cl_max="1",
             thrust_weight="N/N", thrust_tau_s="s", bank_rate_rad_s="rad/s", gamma_gain="1/s",
             gamma_limit_deg="deg", climb_command_m_s="m/s", process_tau_s="s",
             lift_modulation="1", thrust_modulation="1", jerk_max="m/s3 (command)",
             k_a="1/s", sigma_u="1", arm_m="m", inertia_kg_m2="kg*m2",
             tilt_max_deg="deg (command)", drag_kg_m="kg/m", thrust_coefficient="N/(rad/s)^2",
             torque_arm_m="m", motor_tau_s="s", attitude_gain="1/s2", angular_damping="1/s",
             position_gain="1/s")


def parameter_manifest():
    """Every active aircraft parameter has a unit, evidence level and source scope."""
    entries = {}
    for kind, configs in (("bird", SPECIES_CONFIG), ("drone", DRONE_CONFIG)):
        for subtype, config in configs.items():
            for name, value in config.items():
                evidence, source = "design_prior", None
                note = "Unfitted engineering assumption; not an empirical population interval."
                if kind == "bird" and subtype in ("pigeon", "seagull") and name in ("mass_kg", "wing_area_m2", "span_m"):
                    evidence, source = "reported_anchor", "morphology"
                    note = "A study sample anchor, not a species population distribution; seagull uses kittiwake morphology."
                elif subtype == "consumer_quad" and name in ("mass_kg", "max_speed", "tilt_max_deg", "arm_m"):
                    evidence, source = "reported_anchor" if name != "arm_m" else "derived_anchor", "phantom"
                    note = "One aircraft/S-mode anchor; arm is half the stated motor diagonal. Not a Phantom digital twin."
                elif subtype == "pigeon" and name == "flap_hz":
                    evidence, source = "extrapolated_prior", "pigeon_flap"
                    note = "Low-speed-turn frequency used as a prior outside that flight regime."
                entries[f"{kind}.{subtype}.{name}"] = dict(value=value, unit=UNITS[name], evidence=evidence,
                                                          source=source, note=note)
    result = dict(schema_version="1.0", calibrated=False, sources=deepcopy(SOURCES), parameters=entries,
                  sampling=dict(values=deepcopy(SAMPLING), evidence="design_prior",
                                note="Uniform ranges / truncated duration model, not fitted to GPS or video."),
                  observation=dict(values=deepcopy(OBSERVATION_PRIORS), evidence="stress_test_prior",
                                   note="Dropout/drift/ROI priors require measurement from A; no literature-calibrated probabilities."))
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result
