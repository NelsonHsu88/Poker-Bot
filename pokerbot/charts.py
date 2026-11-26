from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .preflop import CLASS_NAMES, equity_grid, range_percent
from .pushfold import solve_push_fold


def _grid(ax, values, title, cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.0%}"):
    grid = np.asarray(values).reshape(13, 13)
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(169):
        row, col = divmod(i, 13)
        frac = (grid[row, col] - vmin) / (vmax - vmin)
        color = "black" if frac > 0.55 else "white"
        ax.text(col, row - 0.18, CLASS_NAMES[i], ha="center", va="center", fontsize=5.5, color=color)
        ax.text(col, row + 0.25, fmt.format(grid[row, col]), ha="center", va="center", fontsize=4.5, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def preflop_equity_chart(path, opponents=4):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    eq = equity_grid(opponents)
    _grid(ax, eq, f"Preflop equity vs {opponents} random hands", vmin=eq.min(), vmax=eq.max())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def simulator_chart(path, stats):
    labels = [s.label for s in stats]
    values = [s.bb_per_100 for s in stats]
    errors = [s.ci95 for s in stats]
    colors = ["tab:green" if v > 0 else "tab:red" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, yerr=errors, color=colors, capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("big blinds won per 100 hands")
    hands = stats[0].hands
    ax.set_title(f"5-player limit hold'em bot tournament ({hands:,} hands each, 95% CI)")
    for i, s in enumerate(stats):
        ax.annotate(
            f"VPIP {100 * s.vpip / s.hands:.0f}%\nPFR {100 * s.pfr / s.hands:.0f}%",
            (i, 0),
            xytext=(0, -28 if values[i] > 0 else 14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pushfold_chart(path, solution):
    pos = solution.positions
    panels = [(f"{p} open-shove", solution.push_range(p)) for p in pos[:-1]]
    panels.append(("BB calls SB shove", solution.call_range(pos[-1], pos[-2])))
    if len(pos) >= 3:
        panels.append((f"BB calls {pos[-3]} shove", solution.call_range(pos[-1], pos[-3])))
    cols = 3
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5.2))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, freqs) in zip(axes.flat, panels):
        _grid(ax, freqs, f"{title} ({range_percent(freqs):.1f}%)", cmap="RdYlGn")
    fig.suptitle(
        f"{solution.n_players}-max push/fold equilibrium (CFR+), {solution.stack:g}bb stacks, "
        f"{solution.iterations:,} iterations",
        fontsize=13,
    )
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stack_sweep(n_players=5, stacks=(3, 4, 5, 6, 8, 10, 12, 15, 20, 25), iterations=10_000_000, seed=0):
    sweep = {}
    for s in stacks:
        sol = solve_push_fold(n_players, s, iterations=iterations, seed=seed)
        sweep[s] = {name: pct for name, pct in sol.summary()}
    return sweep


def stack_sweep_chart(path, sweep):
    stacks = sorted(sweep)
    names = list(sweep[stacks[0]])
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        ax.plot(stacks, [sweep[s][name] for s in stacks], marker="o", label=name)
    ax.set_xlabel("effective stack (big blinds)")
    ax.set_ylabel("% of hands")
    ax.set_title("How push/fold ranges shrink as stacks get deeper (5-max, no ante)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all(outdir="docs/img", sim_stats=None, iterations=20_000_000, sweep_iterations=10_000_000):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    preflop_equity_chart(out / "preflop_equity.png")
    if sim_stats is not None:
        simulator_chart(out / "simulator.png", sim_stats)
    pushfold_chart(out / "pushfold_10bb.png", solve_push_fold(5, 10, iterations=iterations))
    sweep = stack_sweep(iterations=sweep_iterations)
    stack_sweep_chart(out / "stack_sweep.png", sweep)
    return sweep
