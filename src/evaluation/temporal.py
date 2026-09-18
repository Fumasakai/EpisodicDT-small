"""Per-trajectory temporal diagnostics, with no pooling across episode boundaries."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def temporal_statistics(episodes):
    """Return lags, ACF and mean squared increments for [trajectory, step].

    ACF uses full-episode centering and a fixed sum-of-squares denominator.
    Constant trajectories have undefined ACF (NaN), but valid increments.
    Limit lags to half the episode length to retain enough contributing pairs.
    """
    x = np.asarray(episodes, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite [trajectory, step] data.")
    lags = np.arange(x.shape[1] // 2 + 1)
    centered = x - x.mean(axis=1, keepdims=True)
    energy = (centered ** 2).sum(axis=1)
    acf = np.full((len(x), len(lags)), np.nan)
    increments = np.zeros_like(acf)
    valid = energy > 0
    acf[valid, 0] = 1
    for k in lags[1:]:
        numerator = (centered[:, :-k] * centered[:, k:]).sum(axis=1)
        acf[valid, k] = numerator[valid] / energy[valid]
        increments[:, k] = ((x[:, k:] - x[:, :-k]) ** 2).mean(axis=1)
    return lags, acf, increments


def save_temporal_comparisons(generated, actual, output_dir):
    """Mean and 10–90% trajectory spread, not confidence intervals."""
    if generated.ndim != 3 or actual.shape != (generated.shape[0], generated.shape[2]):
        raise ValueError("Expected generated [N,S,L] and actual [N,L].")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = temporal_statistics(actual)
    samples = temporal_statistics(generated.reshape(-1, generated.shape[-1]))
    for index, filename, title, ylabel in (
        (1, 'episode_autocorrelation.png', 'Within-episode autocorrelation', 'Autocorrelation'),
        (2, 'episode_mean_squared_change.png', 'Mean squared change by time lag', 'Mean squared RSRP change (dB²)'),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        excluded = []
        for stats, color, label in ((source, 'tab:green', 'Source'),
                                    (samples, 'tab:blue', 'Generated')):
            lags, curves = stats[0], stats[index]
            valid = np.isfinite(curves).all(axis=1)
            excluded.append(f'{label}: {len(curves) - valid.sum()}')
            curves = curves[valid]
            if not len(curves):
                ax.plot([], [], color=color, label=f'{label}: no defined ACF')
                continue
            lower, upper = np.percentile(curves, [10, 90], axis=0)
            ax.fill_between(lags, lower, upper, color=color, alpha=0.16)
            ax.plot(lags, curves.mean(axis=0), color=color, linewidth=2,
                    label=f'{label} mean (n={len(curves):,})')
        ax.set(title=title, xlabel='Time lag (steps)', ylabel=ylabel)
        ax.set_xticks(source[0])
        ax.grid(alpha=0.25)
        ax.legend()
        if index == 1:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        note = 'Bands: 10–90% of individual trajectories (not confidence intervals).'
        if index == 1:
            note += '\nUndefined constant-trajectory ACF excluded — ' + ', '.join(excluded)
        fig.text(0.5, 0.015, note, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
