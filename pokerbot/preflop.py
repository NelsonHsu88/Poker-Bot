import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit

from .cards import RANKS
from .equity import _equity_mc

TABLE_PATH = Path(__file__).parent / "data" / "preflop_equity.json"


@njit(cache=True)
def class_index(c1, c2):
    """Index 0..168 into a 13x13 grid: aces first, pairs on the diagonal, suited above it."""
    r1 = c1 >> 2
    r2 = c2 >> 2
    hi = max(r1, r2)
    lo = min(r1, r2)
    if hi == lo or (c1 & 3) == (c2 & 3):
        return (12 - hi) * 13 + (12 - lo)
    return (12 - lo) * 13 + (12 - hi)


def class_name(index: int) -> str:
    row, col = divmod(index, 13)
    if row == col:
        return RANKS[12 - row] * 2
    if row < col:
        return RANKS[12 - row] + RANKS[12 - col] + "s"
    return RANKS[12 - col] + RANKS[12 - row] + "o"


def class_combos(index: int) -> int:
    row, col = divmod(index, 13)
    return 6 if row == col else 4 if row < col else 12


def class_cards(index: int) -> tuple[int, int]:
    row, col = divmod(index, 13)
    if row == col:
        r = 12 - row
        return r * 4, r * 4 + 1
    if row < col:
        return (12 - row) * 4, (12 - col) * 4
    return (12 - col) * 4, (12 - row) * 4 + 1


CLASS_NAMES = [class_name(i) for i in range(169)]
COMBOS = np.array([class_combos(i) for i in range(169)], float)


def range_percent(freqs) -> float:
    """Share of the 1326 starting combos covered by per-class frequencies."""
    return float(np.dot(np.asarray(freqs), COMBOS) / 1326 * 100)


def build_table(iterations=20_000, max_opponents=4, seed=1):
    table = {}
    for n in range(1, max_opponents + 1):
        row = []
        for i in range(169):
            eq, _, _ = _equity_mc(np.array(class_cards(i), np.int64), np.empty(0, np.int64), n, iterations, seed + i)
            row.append(round(float(eq[0]), 4))
        table[str(n)] = row
    data = {"iterations": iterations, "classes": CLASS_NAMES, "equity_vs_random": table}
    TABLE_PATH.parent.mkdir(exist_ok=True)
    TABLE_PATH.write_text(json.dumps(data))
    return data


@lru_cache(maxsize=1)
def _table():
    if not TABLE_PATH.exists():
        build_table()
    data = json.loads(TABLE_PATH.read_text())["equity_vs_random"]
    return {int(k): np.array(v) for k, v in data.items()}


def equity_grid(opponents: int) -> np.ndarray:
    return _table()[opponents]


def preflop_equity(hole, opponents: int) -> float:
    return float(_table()[opponents][class_index(hole[0], hole[1])])
