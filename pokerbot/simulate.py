import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .bots import BOTS
from .engine import play_hand


@dataclass
class BotStats:
    label: str
    hands: int = 0
    net: float = 0.0
    net_sq: float = 0.0
    vpip: int = 0
    pfr: int = 0
    showdowns: int = 0
    showdown_wins: int = 0

    def merge(self, other):
        for f in ("hands", "net", "net_sq", "vpip", "pfr", "showdowns", "showdown_wins"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def bb_per_100(self):
        return 100 * self.net / self.hands

    @property
    def ci95(self):
        mean = self.net / self.hands
        var = max(self.net_sq / self.hands - mean * mean, 0.0)
        return 100 * 1.96 * math.sqrt(var / self.hands)

    def as_row(self):
        return {
            "bot": self.label,
            "hands": self.hands,
            "bb/100": round(self.bb_per_100, 2),
            "±95%": round(self.ci95, 2),
            "VPIP%": round(100 * self.vpip / self.hands, 1),
            "PFR%": round(100 * self.pfr / self.hands, 1),
            "WSD%": round(100 * self.showdown_wins / max(self.showdowns, 1), 1),
        }


def _labels(names):
    seen = {}
    labels = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        labels.append(name if names.count(name) == 1 else f"{name}#{seen[name]}")
    return labels


def _run_chunk(args):
    names, hands, seed = args
    rng = random.Random(seed)
    labels = _labels(names)
    order = list(range(len(names)))
    rng.shuffle(order)
    bots = [BOTS[names[i]]() for i in order]
    stats = {label: BotStats(label) for label in labels}
    for h in range(hands):
        result = play_hand(bots, h % len(bots), rng)
        for seat, idx in enumerate(order):
            s = stats[labels[idx]]
            net = result.net[seat]
            s.hands += 1
            s.net += net
            s.net_sq += net * net
            s.vpip += result.vpip[seat]
            s.pfr += result.pfr[seat]
            s.showdowns += result.showdown[seat]
            s.showdown_wins += result.won_showdown[seat]
    return stats


def benchmark(bot, baselines=("tag",), opponents=("tag", "rock", "maniac", "station", "math", "random"), hands=200_000, workers=1, seed=0):
    """bb/100 of `bot` and each baseline at the same tables: alone against 4 copies of each opponent, then a mixed table."""
    lineups = [(opp, [opp] * 4) for opp in opponents]
    lineups.append(("mixed table", ["tag", "rock", "maniac", "station"]))
    rows = []
    for title, others in lineups:
        row = {"opponents": title}
        for name in (bot, *baselines):
            stats = simulate([name, *others], hands, workers, seed)
            s = next(s for s in stats if s.label == name or s.label == f"{name}#1")
            row[name] = f"{s.bb_per_100:+.1f} ± {s.ci95:.1f}"
        rows.append(row)
    return rows


def simulate(names, hands=100_000, workers=1, seed=0, chunk=5_000):
    """Play bots against each other, reshuffling seats every chunk so nobody keeps a lucky seat."""
    unknown = [n for n in names if n not in BOTS]
    if unknown:
        raise ValueError(f"unknown bots {unknown}; choose from {sorted(BOTS)}")
    jobs = []
    remaining = hands
    i = 0
    while remaining > 0:
        size = min(chunk, remaining)
        jobs.append((list(names), size, seed * 1_000_003 + i))
        remaining -= size
        i += 1
    total = {label: BotStats(label) for label in _labels(list(names))}
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_run_chunk, jobs))
    else:
        results = [_run_chunk(job) for job in jobs]
    for stats in results:
        for label, s in stats.items():
            total[label].merge(s)
    return sorted(total.values(), key=lambda s: -s.bb_per_100)
