import sys, io, json, contextlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research'))
from pipeline import BatchRunner
from generators import BaseAgent, Environment, DroneDyn
from verify_real_track import RealTrackVerifier
r=BatchRunner(output_dir='artifacts',fps=30)
ver=RealTrackVerifier.__new__(RealTrackVerifier); ver.dt=1/30; ver.fps=30
subtypes=[('bird',s) for s in ['pigeon','seagull','falcon']]+[('drone',s) for s in ['consumer_quad','racing_quad','hover_quad','fixed_wing_drone']]
rows=[]
original=BaseAgent.get_observation
for noisy in [False,True]:
    for si,scenario in enumerate(['steady_cruise','sudden_dash','sharp_turns','multi_mode']):
        for ai,(typ,sub) in enumerate(subtypes):
            for idx in range(1,6):
                np.random.seed(20260904+int(noisy)*100000+si*1000+ai*100+idx)
                trace=[]
                def observed(agent,frame_index,apply_noise=False):
                    trace.append((agent.pos.copy(),agent.env.x_goal.copy()))
                    return original(agent,frame_index,apply_noise)
                with patch.object(BaseAgent,'get_observation',observed):
                    s=r._run_single_simulation(scenario,typ,sub,idx,noisy)
                obs=s['observations']; f=ver.extract_features_exact(obs)
                boundary=sum(p['cx'] in (0,1) or p['cy'] in (0,1) for p in obs)
                ghosts=[]
                for (p0,g0),(p1,g1) in zip(trace,trace[1:]):
                    before=np.clip([p0[0]/max(g0[0]*1.1,130),1-p0[2]/max(g0[2]*1.5,110)],0,1)
                    after=np.clip([p0[0]/max(g1[0]*1.1,130),1-p0[2]/max(g1[2]*1.5,110)],0,1)
                    ghosts.append(float(np.linalg.norm(after-before)))
                rows.append(dict(noisy=noisy,typ=typ,sub=sub,scenario=scenario,n=len(obs),boundary=boundary,max_projection_only_shift=max(ghosts,default=0),**f))
summary={}
for noisy in [False,True]:
    group=[x for x in rows if x['noisy']==noisy]
    summary[str(noisy)]=dict(samples=len(group),points=sum(x['n'] for x in group),boundary_points=sum(x['boundary'] for x in group),samples_with_boundary=sum(x['boundary']>0 for x in group),samples_projection_only_shift_gt_002=sum(x['max_projection_only_shift']>.02 for x in group),v_mean_gt_5=sum(x['v_mean']>5 for x in group),sigma_gt_30=sum(x['maneuverability_sigma']>30 for x in group),retained_below_60=sum(x['n']<60 for x in group),a_mean_median=float(np.median([x['a_mean'] for x in group])),max_projection_only_shift=max(x['max_projection_only_shift'] for x in group))
env=Environment(fps=30,wind_speed=0,gust_intensity=0,goal_pos=[300.,50.,50.])
agent=DroneDyn(env,start_pos=[100.,100.,80.],start_speed=0)
np.random.seed(4242)
stationary=[agent.get_observation(i,apply_noise=True) for i in range(301)]
summary['stationary_jitter_only']=ver.extract_features_exact(stationary)
print(json.dumps(summary,indent=2))
