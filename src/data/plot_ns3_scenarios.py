"""Plot the actual waypoint CSVs supplied to tranData (headless rendering)."""
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def plot_scenarios(cfg, root):
    root = Path(root)
    destination = root / 'figures'
    destination.mkdir(exist_ok=True)
    for scenario in cfg['scenarios']:
        inputs = root / 'inputs' / scenario['name']
        fig = Figure(figsize=(10, 8), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        sites = pd.read_csv(inputs / 'gnb.csv')
        ax.scatter(sites.x, sites.y, marker='^', s=150, c='black', zorder=5,
                   label='Base station')
        for index, row in sites.iterrows():
            ax.annotate(f"BS {index + 1} (h={row.z:g} m)", (row.x, row.y),
                        xytext=(7, 9), textcoords='offset points')
        colors = iter(['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'])
        for kind in ('car', 'walker', 'static'):
            frame = pd.read_csv(inputs / f'{kind}.csv')
            for ue, track in frame.groupby('ue_id'):
                track = track.sort_values('timestamp')
                xy = track[['location_x', 'location_y']].to_numpy()
                color = next(colors, '#777777')
                label = f'{kind.capitalize()} {ue}'
                if kind == 'static':
                    ax.scatter(*xy[0], marker='s', s=65, color=color, label=label, zorder=4)
                    continue
                ax.plot(xy[:, 0], xy[:, 1], '.-', color=color, lw=1.7, label=label)
                ax.scatter(*xy[0], marker='o', s=80, facecolors='none', edgecolors=color, zorder=4)
                ax.scatter(*xy[-1], marker='X', s=65, color=color, zorder=4)
                for i, (start, end) in enumerate(zip(xy[:-1], xy[1:])):
                    if np.array_equal(start, end):
                        ax.annotate(f"Stop {track.iloc[i].timestamp:g}-{track.iloc[i+1].timestamp:g} s",
                                    start, xytext=(8, -15), textcoords='offset points', color=color, fontsize=8)
                    else:
                        # Two arrows distinguish outward and return travel on the same line.
                        ax.annotate('', xy=start + .65 * (end-start), xytext=start + .4 * (end-start),
                                    arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.7})
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color='gray', marker='o', markerfacecolor='none', linestyle='None'),
                    Line2D([], [], color='gray', marker='X', linestyle='None')]
        labels += ['Start (t=0)', f"End (t={cfg['duration_s']:g} s)"]
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.set(xlabel='x [m]', ylabel='y [m]',
               title=f"{scenario['name']} ({scenario['partition']})\nBase stations and UE waypoint routes")
        ax.set_aspect('equal', adjustable='box')
        ax.margins(.18)
        ax.grid(alpha=.25)
        fig.savefig(destination / f"{scenario['name']}.png", dpi=160)
        fig.clear()
    return destination
