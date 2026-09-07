from copy import deepcopy

import numpy as np
import pytest

from research.behavior import BehaviorSchedule
from research.dynamics import G, QuadBody, drag_polar
from research.generators import BirdDyn, DroneDyn, Environment
from research.parameters import DRONE_CONFIG, SPECIES_CONFIG, parameter_manifest
from research.pipeline import BatchRunner
from research.build_dataset import build_dataset
from research.io import write_json


def wing(kind="bird", step=1/240, behavior="cruise"):
    env = Environment(fps=60, goal_pos=[10000,0,100], wind_speed=0, gust_intensity=0)
    kwargs = dict(start_pos=[0,0,100], heading=[1,0,0], start_speed=15,
                  rng=np.random.default_rng(7))
    agent = BirdDyn(env, **kwargs) if kind == "bird" else DroneDyn(env, model="fixed_wing_drone", **kwargs)
    agent.config.update(sigma_s=0., sigma_phi=0.)
    agent.integration_step_s = step
    agent.behavior = behavior
    return agent


def test_drag_polar_independent_reference():
    # AR=8, e=.8, CL=.6: induced coefficient = .36/(pi*8*.8).
    assert drag_polar(.6, 2., .5, .03, .8) == pytest.approx(.03+.36/(np.pi*8*.8))


def test_zero_rotor_freefall_matches_analytic_solution():
    c = deepcopy(DRONE_CONFIG["consumer_quad"])
    c["drag_kg_m"] = 0.
    body = QuadBody(c)
    body.rotor_speed[:] = 0.
    p, v = np.array([0.,0.,100.]), np.zeros(3)
    for _ in range(240):
        body.advance(p,v,np.zeros(3),np.zeros(4),1/240)
    np.testing.assert_allclose(v,[0,0,-G],atol=1.e-10)
    np.testing.assert_allclose(p,[0,0,100-G/2],atol=1.e-10)


def test_hover_force_balance_no_controller():
    c = DRONE_CONFIG["consumer_quad"]
    body = QuadBody(c)
    p, v = np.array([0.,0.,100.]), np.zeros(3)
    for _ in range(240):
        body.advance(p,v,np.zeros(3),np.full(4,c["mass_kg"]*G/4),1/240)
    np.testing.assert_allclose(p,[0,0,100.],atol=1.e-10)
    np.testing.assert_allclose(v,0.,atol=1.e-10)


def test_rotor_lag_torque_and_rotation_geometry():
    c = DRONE_CONFIG["consumer_quad"]
    body = QuadBody(c)
    body.rotor_speed[:] = 0.
    p, v = np.zeros(3), np.zeros(3)
    request = np.full(4, c["mass_kg"]*G/4)
    body.advance(p,v,np.zeros(3),request,c["motor_tau_s"])
    expected = np.sqrt(request/c["thrust_coefficient"])*(1-np.exp(-1))
    np.testing.assert_allclose(body.rotor_speed,expected,rtol=1.e-12)
    request[0] *= 1.2
    for _ in range(100):
        body.advance(p,v,np.zeros(3),request,1/240)
    assert body.omega[0] > 0 and body.omega[1] < 0 and body.omega[2] > 0
    np.testing.assert_allclose(body.rotation.T @ body.rotation,np.eye(3),atol=1.e-12)
    assert np.linalg.det(body.rotation) == pytest.approx(1.,abs=1.e-12)


@pytest.mark.parametrize("kind",["bird","fixed_wing"])
def test_unpowered_glide_dissipates_energy(kind):
    agent = wing(kind, behavior="glide")
    m = agent.config["mass_kg"]
    energy = [.5*m*np.dot(agent.v_ground,agent.v_ground)+m*G*agent.pos[2]]
    for _ in range(600):
        agent.step()
        energy.append(agent.diagnostics["mechanical_energy_j"])
        assert agent.diagnostics["thrust_n"] == 0.
        assert 0 <= agent.diagnostics["cl"] <= agent.config["cl_max"]
    assert max(np.diff(energy)) < 0.
    assert agent.pos[2] < 100.


def test_flap_changes_center_of_mass_not_only_bbox():
    flapping, averaged = wing(), wing()
    averaged.config.update(lift_modulation=0., thrust_modulation=0.)
    az, lift, lift_avg, velocity_difference = [], [], [], []
    for _ in range(600):
        flapping.step()
        averaged.step()
        az.append(flapping.accel[2])
        lift.append(flapping.diagnostics["lift_n"])
        lift_avg.append(averaged.diagnostics["lift_n"])
        velocity_difference.append(np.linalg.norm(flapping.v_ground-averaged.v_ground))
    spectrum = np.abs(np.fft.rfft(np.array(az[120:])-np.mean(az[120:])))
    frequency = np.fft.rfftfreq(len(az[120:]),1/60)
    dominant = frequency[np.argmax(spectrum[1:])+1]
    assert abs(dominant-flapping.flap_hz) <= .15
    assert np.std(lift[120:]) > 5*np.std(lift_avg[120:])
    assert np.sqrt(np.mean(np.square(velocity_difference[120:]))) > .02


@pytest.mark.parametrize("kind",["bird","fixed_wing"])
def test_timestep_convergence_wing(kind):
    endpoints = []
    for step in (1/120,1/240,1/480):
        agent = wing(kind,step)
        agent.env.x_goal[:] = [150,100,110]
        for _ in range(240):
            agent.step()
        endpoints.append(np.r_[agent.pos,agent.v_ground])
    coarse = np.linalg.norm(endpoints[0]-endpoints[2])
    fine = np.linalg.norm(endpoints[1]-endpoints[2])
    assert fine < coarse*.7
    assert np.linalg.norm(endpoints[1][:3]-endpoints[2][:3]) < .15


def test_timestep_convergence_quad():
    endpoints = []
    for step in (1/120,1/240,1/480):
        env = Environment(goal_pos=[30,10,110],wind_speed=0,gust_intensity=0)
        agent = DroneDyn(env,start_pos=[0,0,100],start_speed=0,rng=np.random.default_rng(1))
        agent.config.update(sigma_s=0.,sigma_u=0.)
        agent.integration_step_s = step
        for _ in range(120):
            agent.step()
        endpoints.append(agent.pos.copy())
    assert np.linalg.norm(endpoints[1]-endpoints[2]) < np.linalg.norm(endpoints[0]-endpoints[2])*.7
    assert np.linalg.norm(endpoints[1]-endpoints[2]) < .15


def test_low_airspeed_does_not_get_hidden_speed_or_lift_clamp():
    agent = wing()
    agent.v_ground[:] = [2.,0.,0.]
    agent.step()
    assert agent.s < agent.config["min_speed"]
    assert agent.accel[2] < -5.
    assert agent.diagnostics["lift_saturated"]


def test_wind_is_air_relative_force_not_velocity_teleport():
    agent = wing()
    before = agent.v_ground.copy()
    agent.step(wind=np.array([0.,5.,0.]))
    assert abs(agent.v_ground[1]-before[1]) < 1.
    assert not np.allclose(agent.v_ground-before,[0,5,0])


def test_parameter_provenance_complete_and_honest():
    manifest = parameter_manifest()
    expected = sum(len(c) for configs in (SPECIES_CONFIG,DRONE_CONFIG) for c in configs.values())
    assert len(manifest["parameters"]) == expected
    for param in manifest["parameters"].values():
        assert param["unit"] and param["note"]
        if param["evidence"] != "design_prior":
            assert param["source"] in manifest["sources"]
    assert not manifest["calibrated"]
    assert manifest["parameters"]["bird.falcon.max_speed"]["evidence"] == "design_prior"
    assert manifest["sha256"] == parameter_manifest()["sha256"]


def test_individual_morphology_scales_coherently():
    sample = BatchRunner(seed=6).simulate("baseline","bird","pigeon",0,frame_count=30)
    p = sample["metadata"]["individual_parameters"]
    scale = p["scaling"]["length_scale"]
    c = p["physical_config"]
    assert c["mass_kg"] == pytest.approx(SPECIES_CONFIG["pigeon"]["mass_kg"]*scale**3)
    assert c["wing_area_m2"] == pytest.approx(SPECIES_CONFIG["pigeon"]["wing_area_m2"]*scale**2)
    assert c["span_m"] == pytest.approx(SPECIES_CONFIG["pigeon"]["span_m"]*scale)


def test_renewal_durations_reproducible_nonperiodic_and_thermal_core():
    a = BehaviorSchedule(np.random.default_rng(4),60.)
    b = BehaviorSchedule(np.random.default_rng(4),60.)
    assert a.metadata() == b.metadata()
    durations = [s["end_s"]-s["start_s"] for s in a.segments]
    assert len({round(x,2) for x in durations}) > 5
    assert a.thermal_updraft(0,50) > a.thermal_updraft(50,50) > a.thermal_updraft(100,50)
    assert not a.metadata()["calibrated"]


@pytest.mark.parametrize("name",["manifest.json","dataset_manifest.json"])
def test_old_dataset_output_protected(tmp_path,name):
    write_json(tmp_path/name,{"simulator_version":"3.0.0"})
    before = (tmp_path/name).read_bytes()
    with pytest.raises(ValueError,match="Refusing to overwrite"):
        BatchRunner(tmp_path).run(1)
    with pytest.raises(ValueError,match="Refusing to overwrite"):
        build_dataset(tmp_path,train_count=2,test_count=2)
    assert (tmp_path/name).read_bytes() == before


def test_first_frame_airspeed_uses_wind_relative_velocity():
    sample = BatchRunner(seed=7).simulate("baseline","drone","consumer_quad",0,frame_count=5)
    state = sample["world_truth"][0]
    air = np.subtract(state["velocity_m_s"],state["wind_m_s"])
    assert state["airspeed_m_s"] == pytest.approx(np.linalg.norm(air))
