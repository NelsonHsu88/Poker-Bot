import pytest

from pokerbot.preflop import CLASS_NAMES, range_percent
from pokerbot.pushfold import solve_push_fold


@pytest.fixture(scope="module")
def heads_up():
    return solve_push_fold(2, 10, iterations=2_000_000, seed=1)


@pytest.fixture(scope="module")
def five_max():
    return solve_push_fold(5, 10, iterations=2_000_000, seed=1)


def test_heads_up_matches_known_nash_ranges(heads_up):
    push = range_percent(heads_up.push_range("SB"))
    call = range_percent(heads_up.call_range("BB", "SB"))
    assert 50 < push < 65
    assert 30 < call < 45


def test_five_max_ranges_are_ordered(five_max):
    pcts = {p: range_percent(five_max.push_range(p)) for p in ["UTG", "CO", "BTN", "SB"]}
    assert pcts["UTG"] < pcts["CO"] < pcts["BTN"] < pcts["SB"]
    aa, trash = CLASS_NAMES.index("AA"), CLASS_NAMES.index("72o")
    for p in pcts:
        assert five_max.push_range(p)[aa] > 0.95
    assert five_max.push_range("UTG")[trash] < 0.05
