import random
from collections import Counter
from itertools import combinations

import numpy as np

from pokerbot import evaluate, hand_category, parse_cards


def score5(cards):
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    flush = len({c & 3 for c in cards}) == 1
    counts = Counter(ranks)
    ordered = [r for r, _ in sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)]
    shape = sorted(counts.values(), reverse=True)
    straight = None
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight = 3
    if straight is not None and flush:
        return (8, straight)
    if shape[0] == 4:
        return (7, *ordered)
    if shape == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *ranks)
    if straight is not None:
        return (4, straight)
    if shape[0] == 3:
        return (3, *ordered)
    if shape == [2, 2, 1]:
        return (2, *ordered)
    if shape[0] == 2:
        return (1, *ordered)
    return (0, *ranks)


def reference(cards):
    return max(score5(c) for c in combinations(cards, 5))


def ev(text):
    return evaluate(np.array(parse_cards(text), np.int64))


def test_matches_brute_force_reference():
    rng = random.Random(7)
    hands = [rng.sample(range(52), 7) for _ in range(3000)]
    fast = [evaluate(np.array(h, np.int64)) for h in hands]
    slow = [reference(h) for h in hands]
    for f, s in zip(fast, slow):
        assert f >> 20 == s[0]
    for i in range(len(hands) - 1):
        a, b = (fast[i] > fast[i + 1]) - (fast[i] < fast[i + 1]), (slow[i] > slow[i + 1]) - (slow[i] < slow[i + 1])
        assert a == b, (hands[i], hands[i + 1])


def test_categories_and_edge_cases():
    assert hand_category(ev("AsKsQsJsTs2c3d")) == "Straight flush"
    assert hand_category(ev("As2s3s4s5sKhKd")) == "Straight flush"
    assert ev("As2s3s4s5sKhKd") < ev("2s3s4s5s6s"[:10] + "KhKd")
    assert hand_category(ev("AhAdAcAsKdQd2c")) == "Four of a kind"
    assert hand_category(ev("KhKdKcQsQdQc2c")) == "Full house"
    assert ev("KhKdKcQsQdQc2c") > ev("KhKdKcJsJdJc2c")
    assert hand_category(ev("Ah2d3c4s5dKdQc")) == "Straight"
    assert ev("Ah2d3c4s5dKdJc") < ev("2d3c4s5d6hKdJc")
    assert hand_category(ev("AhAdKcKsQdQh2c")) == "Two pair"
    assert ev("AhAdKcKsQdQh2c") > ev("AhAdKcKsJdJh2c")
    assert ev("AhAd5c4s3d8h9c") == ev("AsAc5h4d3c8d9h")
    assert ev("7h7d7s2c2d") > ev("AhAdAsKcQd")
