import pytest

from pokerbot import equity
from pokerbot.preflop import CLASS_NAMES, class_index, equity_grid, range_percent
from pokerbot.cards import parse_cards


def test_aces_against_four_random_hands():
    [aa] = equity(["AsAh"], opponents=4, iterations=200_000, seed=1)
    assert aa["equity"] == pytest.approx(0.559, abs=0.01)


def test_coin_flip():
    ak, qq = equity(["AhKh", "QsQc"], iterations=200_000, seed=2)
    assert ak["equity"] == pytest.approx(0.46, abs=0.01)
    assert ak["equity"] + qq["equity"] == pytest.approx(1.0)


def test_made_hand_on_river():
    [nuts] = equity(["AhKh"], board="QhJhTh2c3d", opponents=3, iterations=5_000, seed=3)
    assert nuts["equity"] == 1.0


def test_split_pot():
    a, b = equity(["AhKd", "AsKc"], board="2c7d9hJsQd", iterations=1_000, seed=4)
    assert a["equity"] == b["equity"] == 0.5
    assert a["tie"] == 1.0


def test_rejects_duplicate_cards():
    with pytest.raises(ValueError):
        equity(["AsAh", "AsKd"])


def test_hand_classes():
    assert CLASS_NAMES[0] == "AA" and CLASS_NAMES[1] == "AKs" and CLASS_NAMES[13] == "AKo"
    assert CLASS_NAMES[class_index(*parse_cards("7c2d"))] == "72o"
    assert CLASS_NAMES[class_index(*parse_cards("2h7h"))] == "72s"
    assert range_percent([1.0] * 169) == pytest.approx(100.0)


def test_preflop_table_is_sane():
    heads_up = equity_grid(1)
    assert heads_up[CLASS_NAMES.index("AA")] == pytest.approx(0.852, abs=0.01)
    assert heads_up[CLASS_NAMES.index("72o")] == pytest.approx(0.346, abs=0.015)
    assert equity_grid(4).argmax() == CLASS_NAMES.index("AA")
