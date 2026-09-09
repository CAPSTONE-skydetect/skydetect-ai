import sys, io, json, pickle, contextlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research'))
from generators import Environment, DroneDyn
from pipeline import BatchRunner, CoreFeatureExtractor
from verify_real_track import RealTrackVerifier
from ai_server.services.rule_filter import RuleFilter
from ai_server.schemas import FeatureVector, TrackFeatures, TrackQuality
out = {}
r = BatchRunner(output_dir='artifacts', fps=30)
a = r._run_single_simulation('steady_cruise','drone','consumer_quad',1,False)
b = r._run_single_simulation('steady_cruise','drone','consumer_quad',1,False)
out['reproducibility'] = {'same_sample_id':a['metadata']['sample_id']==b['metadata']['sample_id'],'same_start':a['metadata']['start_pos']==b['metadata']['start_pos'],'same_observations':a['observations']==b['observations']}
env = Environment(fps=30,wind_speed=0,gust_intensity=0,goal_pos=[300.,50.,50.])
agent = DroneDyn(env,start_pos=[100.,100.,80.],start_speed=0)
before = agent.get_observation(0)
env.x_goal[0] += 250
after = agent.get_observation(1)
out['goal_projection'] = {'position':agent.pos.tolist(),'cx_before':before['cx'],'cx_after':after['cx'],'apparent_widths_per_s':abs(after['cx']-before['cx'])/(after['w']/30+1e-6)}
env = Environment(fps=30,wind_speed=0,gust_intensity=0,goal_pos=[100.,100.,80.])
agent = DroneDyn(env,model='hover_quad',start_pos=[100.,100.,80.],start_speed=8)
agent.sigma_s=0
origin=agent.pos.copy()
for _ in range(60):
    env.x_goal=agent.pos.copy()
    agent.step()
out['hover']={'seconds':2,'distance_m':float(np.linalg.norm(agent.pos-origin)),'speed_m_s':float(agent.s)}
env = Environment(fps=30,wind_speed=0,gust_intensity=0,goal_pos=[300.,100.,80.])
agent=DroneDyn(env,start_pos=[100.,100.,80.],start_speed=14)
agent.sigma_s=0
env.x_goal[0] += 250
agent.step()
out['dash']={'before_m_s':14,'after_m_s':float(agent.s)}
def track(frames):
    return [dict(frame_index=i,timestamp_ms=round(i*1000/30),cx=.2+.001*i,cy=.5,w=.05,h=.02,conf=.95) for i in frames]
def features(samples):
    payload=[dict(metadata={'label':'bird','fps':30},observations=o) for o in samples]
    capture=[]
    def capture_csv(df,*args,**kwargs): capture.append(df.copy())
    ext=CoreFeatureExtractor('artifacts')
    with patch('pipeline.os.path.exists',return_value=True), patch('builtins.open',return_value=io.BytesIO(pickle.dumps(payload))), patch.object(pd.DataFrame,'to_csv',capture_csv), contextlib.redirect_stdout(io.StringIO()):
        ext.extract_features()
    return capture[0]
full=track(range(10)); gap=track([0,1,2,5,6,9])
f=features([full,gap])
out['gap']=f.to_dict(orient='records')
ver=RealTrackVerifier.__new__(RealTrackVerifier); ver.dt=1/30; ver.fps=30
out['real_verifier_gap']=ver.extract_features_exact(gap)
straight=f.iloc[0].drop('label').to_dict()
fv=FeatureVector(track_id=1,features=TrackFeatures(**straight),quality=TrackQuality(num_points=10,mean_conf=.95,missing_ratio=0,track_stability='good'))
out['straight_rule_filter']=RuleFilter().apply(fv).model_dump()
def headings_track(headings):
    increments=.001*np.column_stack([np.cos(np.radians(headings)),np.sin(np.radians(headings))])
    pts=np.vstack([[.2,.5],np.array([.2,.5])+np.cumsum(increments,axis=0)])
    return [dict(frame_index=i,timestamp_ms=round(i*1000/30),cx=float(p[0]),cy=float(p[1]),w=.05,h=.02,conf=.95) for i,p in enumerate(pts)]
fa=features([headings_track([45]*50+[-45]*50),headings_track([45,-45]*50)])
out['heading_order']={'turn_counts':[1,99],'features':fa.to_dict(orient='records'),'equal_with_tolerance':bool(np.allclose(fa.drop(columns='label').iloc[0],fa.drop(columns='label').iloc[1]))}
obs=[dict(frame_index=i,timestamp_ms=round(i*1000/30),cx=.5,cy=.5,w=.05,h=.02,conf=.9) for i in range(20)]
profile=dict(enabled=True,start=2,peak=5,end=10,recover=False,max_dx=.02,max_dy=0,duration=9)
drifted,_=r._apply_tracking_drift_noise(obs,profile)
out['no_recovery_drift']={'cx_frame_10':drifted[10]['cx'],'cx_frame_11':drifted[11]['cx']}
print(json.dumps(out,indent=2))
