import random

import numpy as np
import pytest

from pokerbot.engine import play_hand
from pokerbot.evolve import evolve
from pokerbot.fastsim import play_tables, preflop_table
from pokerbot.strategy import EvolvedBot, load_model

TIGHT = np.array([1.8, 0.4, 1.4, 1.8, 0.1, 1.5, 0.05, 2.0, 0.0, 0.05])
STATION = np.array([99, 0, -1, -1, 0, 99, 0, 99, -1, 0])


def python_engine_bb100(hero, villain, hands, seed):
    rng = random.Random(seed)
    bots = [EvolvedBot(hero)] + [EvolvedBot(villain) for _ in range(4)]
    return 100 * sum(play_hand(bots, h % 5, rng).net[0] for h in range(hands)) / hands


def test_both_engines_agree():
    tables = np.stack([np.stack([TIGHT] + [STATION] * 4)] * 40)
    fast = 100 * play_tables(tables, 500, 11, preflop_table())[:, 0].sum() / 20_000
    slow = python_engine_bb100(TIGHT, STATION, 20_000, 12)
    assert fast > 20
    assert fast == pytest.approx(slow, abs=12)


def test_compiled_tables_are_zero_sum():
    rng = np.random.default_rng(0)
    from pokerbot.strategy import random_genomes

    net = play_tables(random_genomes(100, rng).reshape(20, 5, -1), 200, 3, preflop_table())
    assert np.abs(net.sum(axis=1)).max() < 1e-9


def test_evolution_runs_and_saves_a_model(tmp_path):
    champion, history, meta = evolve(
        population=100, generations=2, hands=40, rounds=2, out=tmp_path, finalists=5, playoff_tables=4, playoff_hands=50, log=lambda _: None
    )
    assert len(history) == 2
    assert np.allclose(load_model(tmp_path / "evolved.json"), champion, atol=1e-4)
