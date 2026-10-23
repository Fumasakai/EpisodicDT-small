"""Independent UE waypoint routes in a fixed 24-UE HOF-search environment."""
import argparse
import numpy as np
import yaml
from src.data.scenarios import project_path,save_json
from src.data.route_csv import write_tracks,speed_warnings


def create(output,config_output,seed=20261021,variants=6):
    root,path=project_path(output),project_path(config_output)
    if root.exists() or path.exists(): raise FileExistsError('Choose new output paths')
    if variants<3: raise ValueError('Need at least three variants')
    cfg=yaml.safe_load(project_path('configs/training_dataset_v2.yaml').read_text())
    env=cfg['environment'];env['route_csv_root']=str(root);env['scenarios']=[]
    cfg['batch_dir']=str(project_path('data/processed')/root.name)
    rng=np.random.default_rng(seed);records=[]
    for index in range(variants):
        partition='train' if index<variants-2 else ('validation' if index==variants-2 else 'evaluation')
        for family in ('converging','opposing','dispersed'):
            name=f'{family}_{index:03d}';tracks={};details=[]
            for ue in range(1,19):
                reverse=family=='opposing' and ue%2==0
                x0=float(rng.uniform(110,155));y0=float(rng.uniform(5,85))
                bend=np.array([rng.uniform(170,185),rng.uniform(5,85)])
                cross=np.array([rng.uniform(220,230),rng.uniform(10,80)])
                end=np.array([rng.uniform(285,345),rng.uniform(5,85)])
                xy=np.array([[x0,y0],bend,cross,end])
                if reverse: xy[:,0]=400-xy[:,0]
                velocities=rng.uniform(32,55,3)
                dt=np.linalg.norm(np.diff(xy,axis=0),axis=1)/velocities
                crossing=float(rng.uniform(5.2,6.2) if family!='dispersed' else rng.uniform(5.2,10.))
                start=max(.5,crossing-dt[0]-dt[1]);times=start+np.r_[0,np.cumsum(dt)]
                points=[[0.,*xy[0]],*[ [float(t),*p] for t,p in zip(times,xy)], [16.,*xy[-1]]]
                if times[-1]>=16: raise ValueError('Trajectory exceeds duration')
                tracks[f'car:{ue}']=points
                details.append(dict(ue_id=ue,start_s=start,boundary_approach_s=float(times[2]),direction='west' if reverse else 'east',segment_speeds_mps=velocities.tolist()))
            for ue in range(1,5):
                initial=rng.uniform([30,20],[130,90]);angle=rng.uniform(-np.pi,np.pi)
                speed=float(rng.uniform(.5,1.6));velocity=speed*np.array([np.cos(angle),np.sin(angle)])
                mid=initial+velocity*7
                turn=angle+rng.uniform(-1,1)
                end=mid+speed*9*np.array([np.cos(turn),np.sin(turn)])
                tracks[f'walker:{ue}']=[[0.,*initial],[7.,*mid],[16.,*end]]
            for ue in (1,2):
                xy=rng.uniform([20,20],[100,100])
                tracks[f'static:{ue}']=[[0.,*xy],[16.,*xy]]
            errors=speed_warnings(tracks,env['ue_attributes'])
            if errors: raise ValueError(errors)
            # No vehicle route may be merely a spatially translated copy of another.
            shapes=[np.diff(np.asarray(tracks[f'car:{ue}'])[:,1:],axis=0) for ue in range(1,19)]
            if any(np.allclose(a,b) for i,a in enumerate(shapes) for b in shapes[i+1:]):
                raise ValueError('Duplicate translated vehicle path')
            write_tracks(root/name,tracks,env)
            env['scenarios'].append(dict(name=name,partition=partition,group=f'independent_{seed}_{index:03d}'))
            records.append(dict(name=name,partition=partition,family=family,vehicles=details))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    save_json(root/'candidates.json',dict(seed=seed,candidates=records,config=str(path)))
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='data/scenarios/new_independent_routes')
    p.add_argument('--output-config',default='configs/new_independent_routes.yaml')
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--variants',type=int,default=6)
    a=p.parse_args();cfg=create(a.output_dir,a.output_config,a.seed,a.variants)
    print('Created',len(cfg['environment']['scenarios']),'independent-route scenarios')
