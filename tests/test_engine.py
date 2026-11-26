import random

from pokerbot.bots import BOTS
from pokerbot.engine import play_hand
from pokerbot.simulate import simulate


def test_hands_are_zero_sum():
    rng = random.Random(0)
    bots = [BOTS[name]() for name in ["tag", "rock", "maniac", "station", "random"]]
    for h in range(500):
        result = play_hand(bots, h % 5, rng)
        assert abs(sum(result.net)) < 1e-9


def test_everyone_folds_to_big_blind():
    class Folder:
        def act(self, obs):
            return "f"

    result = play_hand([Folder() for _ in range(5)], 0, random.Random(1))
    assert result.net == [0.0, -0.5, 0.5, 0.0, 0.0]


def test_station_beats_random_and_stats_add_up():
    stats = simulate(["station", "random", "random"], hands=3_000, seed=3)
    assert stats[0].label == "station"
    assert sum(s.net for s in stats) < 1e-6
    assert all(s.hands == 3_000 for s in stats)
