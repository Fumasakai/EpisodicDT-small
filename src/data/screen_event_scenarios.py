"""Simulate candidates independently, recording failures separately from negative events."""
import argparse
import copy
import json
import subprocess
import shutil
import numpy as np
import pandas as pd
import yaml
from src.data.generate_handover_batch import run_batch
from src.data.prepare_handover import event_outcomes, LABELS
from src.data.scenarios import project_path, save_json


def screen(config, output, limit=0, runs=None):
    root=project_path(output)
    root.mkdir(parents=True,exist_ok=False)
    scenarios=config['environment']['scenarios']
    # Round-robin partitions so a small screen includes held-out conditions.
    buckets=[[s for s in scenarios if s['partition']==p] for p in ('train','validation','evaluation')]
    selected=[b[i] for i in range(max(map(len,buckets))) for b in buckets if i<len(b)]
    if limit: selected=selected[:limit]
    report=[]
    for scenario in selected:
        cfg=copy.deepcopy(config)
        cfg['environment']['scenarios']=[copy.deepcopy(scenario)]
        cfg['batch_dir']=str(root/scenario['name'])
        if runs is not None:
            cfg['execution']['runs']=runs
            cfg['environment']['scenarios'][0].pop('simulation_run', None)
        record=dict(name=scenario['name'],partition=scenario['partition'],status='invalid',event_counts={},run_results=[])
        try:
            batch=run_batch(cfg)
            for raw in sorted((batch/'raw').iterdir()):
                events=pd.read_csv(raw/'ho_events.csv')
                kpi=pd.read_csv(raw/'ue_kpi.csv')
                counts=np.zeros(len(LABELS),dtype=int)
                for imsi in kpi.imsi.unique():
                    outcomes,_=event_outcomes(events,imsi,cfg['labeling']['pingpong_window_s'])
                    counts += [sum(t>=3 for t in times) for times in outcomes]
                record['run_results'].append(dict(run=raw.name,counts=dict(zip(LABELS,map(int,counts)))))
            record['event_counts']={label:sum(r['counts'][label] for r in record['run_results']) for label in LABELS}
            # Use label masks; a crash/unknown observation is never a normal sample.
            positive=0;known=True
            for archive in (batch/'episodes').rglob('*.npz'):
                with np.load(archive,allow_pickle=False) as a:
                    positive += int(((a['labels'][:,:2]>0)&(a['label_mask'][:,:2]>0)).any(axis=1).sum())
                    known &= bool((a['label_mask'][:,:2]>0).all())
            record.update(positive_ue_windows=positive,status='positive' if positive else ('negative' if known else 'unknown'))
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            record['error']=str(error)
        report.append(record)
        save_json(root/'screening.json',{'results':report})
        print(f"SCREEN {scenario['name']}: {record['status']} {record['event_counts']}",flush=True)
    # Completed candidates, including negative controls; do not distort held-out prevalence by retaining positives only.
    accepted={r['name'] for r in report if r['status'] in ('positive','negative','unknown')}
    verified=copy.deepcopy(config)
    verified['environment']['scenarios']=[s for s in scenarios if s['name'] in accepted]
    verified['batch_dir']=str(root/'training_batch')
    if accepted: (root/'completed_scenarios.yaml').write_text(yaml.safe_dump(verified,sort_keys=False))
    if accepted:
        # Archives retain original raw-source paths and hashes; no relabelling or resimulation.
        for record in report:
            if record['name'] not in accepted: continue
            for archive in (root/record['name']/'episodes').rglob('*.npz'):
                dest=root/'episodes'/record['partition']/archive.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(archive,dest)
        first=next(r['name'] for r in report if r['name'] in accepted)
        training=yaml.safe_load((root/first/'training.yaml').read_text())
        training['simulation_config']=str(root/'completed_scenarios.yaml')
        training['data']={p+'_path':str(root/'episodes'/p) for p in ('train','validation','evaluation')}
        training['output_dir']=str(root/'mobility')
        training['diffusion_output_dir']=str(root/'diffusion')
        (root/'training.yaml').write_text(yaml.safe_dump(training,sort_keys=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/training_dataset_v2.yaml')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--limit',type=int,default=0,help='0 means all candidates')
    p.add_argument('--runs',type=int,nargs='+',default=[1])
    a=p.parse_args()
    if a.limit<0: p.error('limit must be nonnegative')
    screen(yaml.safe_load(project_path(a.config).read_text()),a.output_dir,a.limit,a.runs)


if __name__=='__main__': main()
