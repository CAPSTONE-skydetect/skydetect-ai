"""Reproducible physical checks and a scenario sweep, without classifier training."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import SIMULATOR_VERSION, FEATURE_VERSION
from .dynamics import G, QuadBody
from .generators import BirdDyn, DroneDyn, Environment
from .io import write_json, write_jsonl
from .parameters import DRONE_CONFIG, parameter_manifest
from .pipeline import BatchRunner, BIRD_BEHAVIORS, SCENARIOS, SPECIES_CONFIG, DRONE_SUBTYPES


def make_agent(kind, step=1/240, behavior="cruise", fps=60):
    env = Environment(fps=fps, goal_pos=[10000,0,100],wind_speed=0,gust_intensity=0)
    kwargs = dict(start_pos=[0,0,100],heading=[1,0,0],start_speed=15,rng=np.random.default_rng(17))
    agent = BirdDyn(env,**kwargs) if kind == "bird" else DroneDyn(env,model=kind,**kwargs)
    agent.config.update(sigma_s=0.)
    agent.integration_step_s, agent.behavior = step, behavior
    return agent


def physical_checks(output):
    checks, traces = {}, {}
    c = deepcopy(DRONE_CONFIG["consumer_quad"])
    c["drag_kg_m"] = 0.
    for name, freefall in (("hover_force_balance",False),("freefall",True)):
        body = QuadBody(c)
        p, v = np.array([0.,0.,100.]), np.zeros(3)
        target = np.zeros(4) if freefall else np.full(4,c["mass_kg"]*G/4)
        if freefall:
            body.rotor_speed[:] = 0.
        for _ in range(240):
            body.advance(p,v,np.zeros(3),target,1/240)
        expected = np.array([0.,0.,100-G/2 if freefall else 100.])
        error = float(np.linalg.norm(p-expected))
        checks[name] = dict(position_error_m=error, threshold_m=1.e-9, passed=error < 1.e-9)
    for kind in ("bird","fixed_wing_drone"):
        agent = make_agent(kind,behavior="glide")
        energies = [.5*agent.config["mass_kg"]*np.dot(agent.v_ground,agent.v_ground)+agent.config["mass_kg"]*G*agent.pos[2]]
        for _ in range(600):
            agent.step()
            energies.append(agent.diagnostics["mechanical_energy_j"])
        maximum = float(max(np.diff(energies)))
        checks[kind+"_glide"] = dict(max_energy_increment_j=maximum, final_altitude_m=float(agent.pos[2]),
                                     passed=bool(maximum < 0. and agent.pos[2] < 100.))
        traces[kind+"_energy"] = energies
    for kind in ("bird","fixed_wing_drone","consumer_quad"):
        positions = []
        for step in (1/120,1/240,1/480):
            agent = make_agent(kind,step)
            agent.env.x_goal[:] = [150,100,110]
            for _ in range(240):
                agent.step()
            positions.append(agent.pos.copy())
        coarse = float(np.linalg.norm(positions[0]-positions[2]))
        fine = float(np.linalg.norm(positions[1]-positions[2]))
        checks[kind+"_convergence"] = dict(coarse_error_m=coarse, fine_error_m=fine, threshold_m=.15,
                                           passed=fine < .15 and fine < .7*coarse)
    flapping, averaged = make_agent("bird"), make_agent("bird")
    averaged.config.update(lift_modulation=0.,thrust_modulation=0.)
    lift, az, differences = [], [], []
    for _ in range(600):
        flapping.step()
        averaged.step()
        lift.append(flapping.diagnostics["lift_n"])
        az.append(flapping.accel[2])
        differences.append(np.linalg.norm(flapping.v_ground-averaged.v_ground))
    spectrum = np.abs(np.fft.rfft(np.asarray(az[120:])-np.mean(az[120:])))
    frequencies = np.fft.rfftfreq(480,1/60)
    peak = float(frequencies[np.argmax(spectrum[1:])+1])
    rms = float(np.sqrt(np.mean(np.square(differences[120:]))))
    checks["flap_force_ablation"] = dict(command_hz=flapping.flap_hz, acceleration_peak_hz=peak,
                                        velocity_difference_rms_m_s=rms,
                                        passed=abs(peak-flapping.flap_hz) <= .15 and rms > .02)
    traces.update(lift_n=lift, acceleration_z_m_s2=az)
    # Finite physical sensitivities are not evidence that the priors are empirically correct.
    sensitivities = []
    for kind in ("bird","fixed_wing_drone"):
        for multiplier in (.5,1.,2.):
            agent = make_agent(kind,behavior="glide")
            agent.config["cd0"] *= multiplier
            for _ in range(600):
                agent.step()
            sensitivities.append(dict(kind=kind, cd0_scale=multiplier, altitude_m=float(agent.pos[2]),
                                      speed_m_s=float(agent.s)))
    write_json(output/"physical_checks.json",dict(checks=checks,sensitivities=sensitivities))
    fig, axes = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for kind in ("bird","fixed_wing_drone"):
        e = np.asarray(traces[kind+"_energy"])
        axes[0,0].plot(np.arange(len(e))/60,e/e[0],label=kind)
    axes[0,0].set(xlabel="Time (s)",ylabel="Energy / initial energy",title="Unpowered, no wind")
    axes[0,0].legend()
    axes[0,1].plot(np.arange(120)/60,lift[-120:])
    axes[0,1].set(xlabel="Time in final 2 s",ylabel="Lift (N)",title="Bird force, not bbox animation")
    axes[1,0].plot(frequencies,spectrum)
    axes[1,0].axvline(flapping.flap_hz,color="red",ls="--")
    axes[1,0].set(xlabel="Hz",ylabel="Acceleration FFT amplitude",xlim=(0,20),title="Prescribed flap frequency response")
    names = [k for k in checks if k.endswith("convergence")]
    axes[1,1].bar([k.removesuffix("_convergence") for k in names],[checks[k]["fine_error_m"] for k in names])
    axes[1,1].axhline(.15,color="red",ls="--")
    axes[1,1].set(ylabel="Position difference (m)",title="240 vs 480 Hz, after 4 seconds")
    fig.savefig(output/"physical_checks.png",dpi=150)
    plt.close(fig)
    return checks


def validate(output, seed_count=2, lengths=(60,180,420)):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    checks = physical_checks(output)
    records, previews, preview_keys = [], [], set()
    for seed in range(seed_count):
        runner = BatchRunner(seed=20260907+seed)
        for label, subtypes in (("bird",SPECIES_CONFIG),("drone",DRONE_SUBTYPES)):
            for subtype in subtypes:
                for behavior in (BIRD_BEHAVIORS if label == "bird" else ("cruise",)):
                    for scenario in SCENARIOS:
                        for n in lengths:
                            sample = runner.simulate(scenario,label,subtype,0,frame_count=n,behavior=behavior)
                            world, meta = sample["world_truth"], sample["metadata"]
                            p = np.asarray([x["position_m"] for x in world])
                            v = np.asarray([x["velocity_m_s"] for x in world])
                            dynamics = [x["dynamics"] for x in world[1:]]
                            rotations = [np.asarray(d["rotation_body_to_world"]) for d in dynamics if "rotation_body_to_world" in d]
                            orthogonal = max((np.linalg.norm(r.T@r-np.eye(3)) for r in rotations),default=0.)
                            row = dict(seed=runner.seed,label=label,subtype=subtype,behavior=behavior,scenario=scenario,
                                       frames=n,latent_failure=meta["latent_failure"],
                                       finite=bool(np.isfinite(p).all() and np.isfinite(v).all()),
                                       min_altitude_m=float(p[:,2].min()),max_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
                                       max_rotation_error=float(orthogonal),
                                       lift_saturated_fraction=float(np.mean([d.get("lift_saturated",False) for d in dynamics])),
                                       motor_saturated_fraction=float(np.mean([d.get("allocation_saturated",False) for d in dynamics])),
                                       feature_status=sample["feature_result"]["feature_status"],
                                       rejection_reasons=sample["feature_result"]["reasons"])
                            records.append(row)
                            if subtype not in preview_keys and n == max(lengths) and scenario == "baseline" and behavior in ("flap_jitter","cruise"):
                                previews.append(sample)
                                preview_keys.add(subtype)
        print(f"Completed sweep seed {seed+1}/{seed_count}: {len(records)} trajectories",flush=True)
    write_jsonl(output/"stress_records.jsonl",records)
    write_jsonl(output/"trajectory_previews.jsonl",previews)
    fig, axes = plt.subplots(len(previews),3,figsize=(14,2.6*len(previews)),squeeze=False,constrained_layout=True)
    for row,sample in enumerate(previews):
        world = sample["world_truth"]
        p = np.asarray([x["position_m"] for x in world])
        time = [x["time_seconds"] for x in world]
        speed = [x["airspeed_m_s"] for x in world]
        axes[row,0].plot(p[:,0],p[:,1])
        axes[row,0].set(xlabel="World x (m)",ylabel="World y (m)",title=sample["metadata"]["subtype"])
        axes[row,1].plot(time,p[:,2],color="teal")
        axes[row,1].set(xlabel="Time (s)",ylabel="Altitude (m)")
        axes[row,2].plot(time,speed,color="darkred")
        axes[row,2].set(xlabel="Time (s)",ylabel="Airspeed (m/s)")
    fig.savefig(output/"trajectory_previews.png",dpi=120)
    plt.close(fig)
    summary = dict(simulator_version=SIMULATOR_VERSION,feature_version=FEATURE_VERSION,
                   calibration_status="uncalibrated_no_real_A",scope="physics verification, NOT empirical validation",
                   trajectory_count=len(records),seed_count=seed_count,frame_lengths=list(lengths),
                   nonfinite_count=sum(not r["finite"] for r in records),
                   latent_failure_count=sum(r["latent_failure"] is not None for r in records),
                   feature_status_counts=dict(Counter(r["feature_status"] for r in records)),
                   rejection_counts=dict(Counter(reason for r in records for reason in r["rejection_reasons"])),
                   max_rotation_error=max(r["max_rotation_error"] for r in records),
                   physical_checks=checks,physical_checks_passed=all(x["passed"] for x in checks.values()),
                   per_subtype={sub:dict(count=sum(r["subtype"]==sub for r in records),
                                        failures=sum(r["subtype"]==sub and r["latent_failure"] is not None for r in records),
                                        max_speed_m_s=max(r["max_speed_m_s"] for r in records if r["subtype"]==sub))
                                for sub in list(SPECIES_CONFIG)+list(DRONE_SUBTYPES)},
                   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))})
    write_json(output/"parameter_manifest.json",parameter_manifest())
    write_json(output/"summary.json",summary)
    print(f"Physical checks passed: {summary['physical_checks_passed']}; latent failures: {summary['latent_failure_count']}/{len(records)}",flush=True)
    if not summary["physical_checks_passed"] or summary["nonfinite_count"] or summary["max_rotation_error"] > 1.e-9:
        raise RuntimeError("Physical verification failed; inspect the saved diagnostics")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"output"/"physics_v4")
    parser.add_argument("--seeds",type=int,default=2)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    validate(args.output,args.seeds)


if __name__ == "__main__":
    main()
