"""Plot saved mobility CSVs and the batch's recorded base-station layout."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.scenarios import digest, project_path, save_json


def draw_scenario(axes, inputs, scenario, sites):
    spatial, x_time, y_time = axes
    palette = plt.get_cmap('tab10')
    index = 0
    for kind in ('car', 'walker', 'static'):
        frame = pd.read_csv(inputs / f'{kind}.csv')
        for ue, track in frame.groupby('ue_id', sort=True):
            track = track.sort_values('timestamp')
            t = track.timestamp.to_numpy()
            xy = track[['location_x', 'location_y']].to_numpy()
            if not np.isfinite(np.column_stack([t, xy])).all() or not (np.diff(t) > 0).all():
                raise ValueError(f'Invalid coordinates/timestamps: {inputs}/{kind}.csv UE {ue}')
            color = palette(index % 10); index += 1
            label = f'{kind} UE {ue}'
            spatial.plot(*xy.T, color=color, linewidth=2, label=label)
            spatial.scatter(*xy[0], color=color, marker='o', s=55, zorder=4)
            spatial.scatter(*xy[-1], color=color, marker='x', s=65, zorder=5)
            # Direction of the first moving segment; time plots disambiguate retraced paths.
            moving = np.flatnonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8)
            if len(moving):
                j = moving[0]
                delta = xy[j + 1] - xy[j]
                spatial.annotate('', xy=xy[j] + .65 * delta, xytext=xy[j] + .35 * delta,
                                 arrowprops={'arrowstyle': '->', 'color': color, 'lw': 2})
            x_time.plot(t, xy[:, 0], '.-', color=color, label=label, markersize=5)
            y_time.plot(t, xy[:, 1], '.-', color=color, label=label, markersize=5)
    for i, (x, y, z) in enumerate(sites):
        spatial.scatter(x, y, marker='^', c='black', s=85, zorder=6,
                        label='gNB' if i == 0 else None)
        spatial.annotate(f'gNB {i} (h={z:g} m)', (x, y), xytext=(5, -16), textcoords='offset points', fontsize=8)
    spatial.set(title='XY path (o: start, x: end; arrow: initial direction)', xlabel='X [m]', ylabel='Y [m]')
    spatial.set_aspect('equal', adjustable='datalim')
    spatial.margins(.15)
    spatial.legend(loc='best', fontsize=8)
    for ax, coord in ((x_time, 'X'), (y_time, 'Y')):
        ax.set(title=f'{coord} coordinate over time', xlabel='Time [s]', ylabel=f'{coord} [m]')
        ax.margins(x=.02)
    for ax in axes:
        ax.grid(alpha=.25)
    x_time.text(.5, 1.25, f"{scenario['name']}  |  {scenario['partition']}", transform=x_time.transAxes,
                ha='center', fontsize=13, fontweight='bold')


def plot_batch(batch_dir, output_dir=None, partition=None):
    root = project_path(batch_dir)
    plan = json.loads((root / 'plan.json').read_text())
    env = plan['config']['environment']
    scenarios = [s for s in env['scenarios'] if partition is None or s['partition'] == partition]
    if not scenarios:
        raise ValueError('No scenarios for selected partition')
    sources = {'plan.json': digest(root / 'plan.json')}
    for scenario in scenarios:
        for kind in ('car', 'walker', 'static'):
            relative = f"inputs/{scenario['name']}/{kind}.csv"
            actual = digest(root / relative)
            if actual != plan['input_sha256'][relative]:
                raise ValueError(f'Input differs from recorded batch: {relative}')
            sources[relative] = actual
    output = project_path(output_dir) if output_dir else root / 'figures' / 'input_routes'
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for scenario in scenarios:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
        draw_scenario(axes, root / 'inputs' / scenario['name'], scenario,
                      env.get('sites', scenario.get('sites')))
        for suffix in ('png', 'svg'):
            path = output / f"{scenario['name']}.{suffix}"
            fig.savefig(path, dpi=170)
            files.append(path.name)
        plt.close(fig)
    # One overview per partition avoids confusing training and held-out routes.
    for split in dict.fromkeys(s['partition'] for s in scenarios):
        rows = [s for s in scenarios if s['partition'] == split]
        fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.5 * len(rows)), squeeze=False, layout='constrained')
        for scenario, row in zip(rows, axes):
            draw_scenario(row, root / 'inputs' / scenario['name'], scenario,
                          env.get('sites', scenario.get('sites')))
        path = output / f'{split}_overview.png'
        fig.savefig(path, dpi=170); plt.close(fig)
        files.append(path.name)
    save_json(output / ('provenance.json' if partition is None else f'{partition}_provenance.json'),
              {'batch_dir': str(root), 'partition': partition, 'source_sha256': sources,
               'plotter_sha256': digest(__file__), 'files': files,
               'notes': 'Input waypoints with linear interpolation, not measured radio outcomes. Units: seconds, meters.'})
    print(f'Saved mobility graphs: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--partition', choices=('train', 'validation', 'evaluation'))
    args = parser.parse_args()
    plot_batch(args.batch_dir, args.output_dir, args.partition)


if __name__ == '__main__':
    main()
