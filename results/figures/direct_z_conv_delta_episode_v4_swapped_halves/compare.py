"""Run from repository root after evaluation to reproduce comparison artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

out = Path(__file__).resolve().parent
configs = [yaml.safe_load(Path('configs/config.yaml').read_text()), yaml.safe_load((out/'config.yaml').read_text())]
rows = []
for label, cfg in zip(['Original', 'Swapped halves'], configs):
    metrics = json.loads((Path(cfg['evaluation']['figure_dir'])/'episode_metrics.json').read_text())
    actual = pd.read_csv(cfg['evaluation']['actual_csv'], usecols=['rsrp_actual_dbm']).to_numpy().reshape(-1, 40)
    generated = pd.read_csv(cfg['evaluation']['generated_csv'], usecols=['rsrp_generated_dbm']).to_numpy().reshape(-1, 32, 40)
    assert generated.shape[0] == actual.shape[0]
    assert metrics['seed'] == 12346 and metrics['num_samples'] == 32
    log = json.loads((Path(cfg['training']['checkpoint_dir'])/'training_metrics.json').read_text())
    row = {'experiment': label, 'evaluation_episodes': len(actual), 'epochs_run':len(log), 'best_epoch':min(log, key=lambda x:x['loss'])['epoch']}
    for key in ['reconstruction_crps_dbm', 'reconstruction_mae_dbm', 'reconstruction_coverage_90', 'within_fixed_z_std_dbm']:
        row[key] = metrics[key]
    row['interval_width_90_dbm'] = float(np.mean(metrics['reconstruction_per_step']['interval_width_90_dbm']))
    for key, values in [('source', actual), ('generated', generated)]:
        row[key+'_mean_dbm'] = float(values.mean())
        row[key+'_std_dbm'] = float(values.std())
        for order in [1,2]:
            row[f'{key}_delta{order}_std_db'] = float(np.diff(values,n=order,axis=-1).std())
    rows.append(row)
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
(out/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax,key,title in zip(axes,['reconstruction_crps_dbm','reconstruction_mae_dbm','reconstruction_coverage_90'],['Reconstruction CRPS (dBm, lower is better)','Reconstruction MAE (dBm, lower is better)','90% interval coverage']):
    vals=[r[key] for r in rows]
    bars=ax.bar([r['experiment'] for r in rows],vals,color=['tab:blue','tab:orange'])
    ax.bar_label(bars,fmt='%.3f');ax.set_title(title,fontsize=10);ax.set_ylim(0,max(vals)*1.22)
    if 'coverage' in key:
        ax.axhline(0.9,color='gray',linestyle='--');ax.set_ylim(0,1)
fig.tight_layout();fig.savefig(out/'comparison_metrics.png',dpi=150);plt.close(fig)
fig, ax=plt.subplots(figsize=(9,4))
for label,cfg in zip(['Original','Swapped halves'],configs):
    log=json.loads((Path(cfg['training']['checkpoint_dir'])/'training_metrics.json').read_text())
    ax.plot([x['epoch'] for x in log],[x['loss'] for x in log],label=label)
ax.set(xlabel='Epoch',ylabel='Validation total loss',title='Validation loss on different held-out intervals');ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(out/'comparison_training.png',dpi=150);plt.close(fig)
print(json.dumps(rows,indent=2))
