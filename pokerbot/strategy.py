"""A bot whose whole strategy is 10 numbers, so a population of them can be evolved.

`decide` is compiled with Numba and shared by the fast training engine (fastsim.py)
and the regular engine (engine.py, through EvolvedBot), so a trained genome plays
exactly the same way in both.
"""

import json
from pathlib import Path

import numpy as np
from numba import njit

from .bots import Bot

GENES = (
    ("pf_raise", 0.5, 3.5, "preflop: raise when edge is above this"),
    ("pf_reraise_step", 0.0, 1.5, "preflop: extra edge needed for each bet already in"),
    ("pf_call", 0.3, 2.5, "preflop: call one bet when edge is above this"),
    ("pf_call_raised", 0.5, 3.5, "preflop: call a raised pot when edge is above this"),
    ("position", -0.3, 0.6, "edge bonus on the button, scaled down to 0 in the small blind"),
    ("bet", 0.5, 3.5, "postflop: bet when checked to and edge is above this"),
    ("bluff", 0.0, 0.5, "postflop: chance of betting anyway when checked to"),
    ("raise", 0.8, 4.0, "postflop: raise a bet when edge is above this"),
    ("call_margin", -0.3, 0.3, "postflop: call when equity beats the pot odds by this much"),
    ("raised_margin", 0.0, 0.25, "postflop: extra equity needed per extra bet facing you"),
)
GENE_NAMES = tuple(g[0] for g in GENES)
LOW = np.array([g[1] for g in GENES])
HIGH = np.array([g[2] for g in GENES])
N_GENES = len(GENES)
FOLD, CALL, RAISE = 0, 1, 2
MODEL_PATH = Path(__file__).parent / "data" / "evolved.json"


@njit(cache=True)
def decide(g, street, equity, n_opp, to_call, pot, num_bets, can_raise, late, u):
    """Return 0 (fold), 1 (check/call) or 2 (bet/raise). late is 1 on the button and 0 in the small blind."""
    edge = equity * (n_opp + 1) + g[4] * late
    if street == 0:
        if can_raise and edge > g[0] + g[1] * (num_bets - 1):
            return RAISE
        if to_call == 0 or edge > (g[2] if num_bets <= 1 else g[3]):
            return CALL
        return FOLD
    if to_call == 0:
        if can_raise and (edge > g[5] or u < g[6]):
            return RAISE
        return CALL
    if can_raise and edge > g[7]:
        return RAISE
    if equity > to_call / (pot + to_call) + g[8] + g[9] * (num_bets - 1):
        return CALL
    return FOLD


def lateness(position, num_players):
    """0 for the small blind, who acts first after the flop, up to 1 for the button, who acts last."""
    return ((position - 1) % num_players) / (num_players - 1)


def random_genomes(n, rng):
    return LOW + rng.random((n, N_GENES)) * (HIGH - LOW)


def save_model(genome, path=MODEL_PATH, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"genes": {name: round(float(v), 5) for name, v in zip(GENE_NAMES, genome)}, **meta}
    path.write_text(json.dumps(data, indent=2))


def load_model(path=MODEL_PATH):
    data = json.loads(Path(path).read_text())
    return np.array([data["genes"][name] for name in GENE_NAMES])


class EvolvedBot(Bot):
    name = "evolved"
    description = "Champion of the genetic algorithm in evolve.py, loaded from data/evolved.json."

    def __init__(self, genome=None):
        self.genome = load_model() if genome is None else np.asarray(genome, float)

    def act(self, obs):
        action = decide(
            self.genome,
            obs.street,
            obs.equity(),
            obs.num_opponents,
            obs.to_call,
            obs.pot,
            obs.num_bets,
            obs.can_raise,
            lateness(obs.position, obs.num_players),
            obs.rng.random(),
        )
        return "fcr"[action]
