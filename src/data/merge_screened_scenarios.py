"""Merge completed screening runs and audit train/validation/evaluation episode counts."""
import argparse
import copy
import json
import shutil

import numpy as np
import yaml

from src.data.scenarios import project_path, save_json
from src.data.handover_dataset import assert_disjoint
from src.data.joint_scenarios import ScenarioDataset
from src.train.common import load_config


def merge(config, sources, output):
    root=project_path(output)
    if root.exists():
        raise FileExistsError(root)
    by_name={s['name']:s for s in config['environment']['scenarios']}
    results=[];seen=set();runs=set()
    for source in sources:
        source=project_path(source)
        for original in json.loads((source/'screening.json').read_text())['results']:
            record=copy.deepcopy(original)
            name=record['name']
            if name in seen:
                raise ValueError(f'Duplicate screened scenario: {name}')
            seen.add(name)
            if name not in by_name or record['partition']!=by_name[name]['partition']:
                raise ValueError(f'Unknown scenario/partition: {name}')
            record['source_batch']=str(source/name)
            if record['status']!='invalid':
                plan=json.loads((source/name/'plan.json').read_text())
                runs.update(plan['config']['execution']['runs'])
                archives=list((source/name/'episodes'/record['partition']).glob('*.npz'))
                if not archives:
                    raise ValueError(f'Missing archives for completed scenario {name}')
            results.append(record)
    accepted=[r for r in results if r['status']!='invalid']
    if not accepted:
        raise ValueError('No completed scenarios')
    root.mkdir(parents=True)
    for record in accepted:
        for archive in (project_path(record['source_batch'])/'episodes'/record['partition']).glob('*.npz'):
            dest=root/'episodes'/record['partition']/archive.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): raise FileExistsError(dest)
            shutil.copyfile(archive,dest)
    cfg=copy.deepcopy(config)
    names={r['name'] for r in accepted}
    cfg['environment']['scenarios']=[s for s in config['environment']['scenarios'] if s['name'] in names]
    cfg['execution']['runs']=sorted(runs)
    cfg['batch_dir']=str(root/'resimulation')
    (root/'simulation.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    training=yaml.safe_load((project_path(accepted[0]['source_batch'])/'training.yaml').read_text())
    training['simulation_config']=str(root/'simulation.yaml')
    training['data']={p+'_path':str(root/'episodes'/p) for p in ('train','validation','evaluation')}
    training['output_dir']=str(root/'mobility')
    training['diffusion_output_dir']=str(root/'diffusion')
    (root/'training.yaml').write_text(yaml.safe_dump(training,sort_keys=False))
    # The actual training loader checks raw hashes, fixed sites, population, radio settings,
    # contiguous warm-up history, and usable synchronized windows.
    loaded=load_config(root/'training.yaml')
    datasets={}
    for partition in ('train','validation','evaluation'):
        normalization=None if partition=='train' else datasets['train'].normalization
        datasets[partition]=ScenarioDataset(training['data'][partition+'_path'],loaded,normalization)
    assert_disjoint(*(d.episodes for d in datasets.values()))
    summary={'screened_conditions':len(results),'completed_conditions':len(accepted),
             'invalid_conditions':len(results)-len(accepted),
             'event_counts_after_3s':{e:sum(r['event_counts'].get(e,0) for r in accepted)
                                    for e in ('ho_failure','pingpong','radio_link_failure')},
             'partitions':{}}
    for partition,dataset in datasets.items():
        labels=np.asarray(dataset.arrays['labels'])
        masks=np.asarray(dataset.arrays['label_mask'],dtype=bool)
        # Collapse UE and the two supported danger-event axes, not individual UE windows.
        positive=((labels[...,:2]>0)&masks[...,:2]).any(axis=(1,2))
        known=masks[...,:2].all(axis=(1,2))
        summary['partitions'][partition]={
            'completed_conditions':sum(r['partition']==partition for r in accepted),
            'ue_windows':len(dataset.episodes),'synchronized_episodes':len(dataset),
            'dangerous_episodes':int(positive.sum()),
            'known_negative_episodes':int((~positive&known).sum()),
            'unknown_episodes':int((~positive&~known).sum()),
            'ue_count':len(dataset.slots)}
    save_json(root/'screening.json',{'results':results})
    save_json(root/'dataset_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/training_dataset_v2.yaml')
    p.add_argument('--screen-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    merge(yaml.safe_load(project_path(a.config).read_text()),a.screen_dirs,a.output_dir)


if __name__=='__main__': main()
