import numpy as np
from numba import njit

CATEGORIES = (
    "High card",
    "Pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
)


@njit(cache=True)
def _popcount(x):
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


@njit(cache=True)
def _straight_high(mask):
    m = (mask << 1) | ((mask >> 12) & 1)
    for top in range(13, 3, -1):
        if ((m >> (top - 4)) & 0x1F) == 0x1F:
            return top - 1
    return -1


@njit(cache=True)
def _top_ranks(mask, n):
    value = 0
    taken = 0
    r = 12
    while taken < n and r >= 0:
        if (mask >> r) & 1:
            value = (value << 4) | r
            taken += 1
        r -= 1
    return value


@njit(cache=True)
def evaluate(cards):
    """Strength of the best 5-card hand in 5 to 7 cards; higher is better.

    Cards are ints 0..51 with rank = c >> 2 (0 is a deuce, 12 an ace) and suit = c & 3.
    The result is category << 20 | tiebreak ranks packed in 4-bit nibbles.
    """
    counts = np.zeros(13, np.int64)
    suit_masks = np.zeros(4, np.int64)
    rank_mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r
        rank_mask |= 1 << r

    flush_suit = -1
    for s in range(4):
        if _popcount(suit_masks[s]) >= 5:
            flush_suit = s
    if flush_suit >= 0:
        high = _straight_high(suit_masks[flush_suit])
        if high >= 0:
            return (8 << 20) | (high << 16)

    quad = -1
    trips1 = -1
    trips2 = -1
    pair1 = -1
    pair2 = -1
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4 and quad < 0:
            quad = r
        elif n == 3:
            if trips1 < 0:
                trips1 = r
            elif trips2 < 0:
                trips2 = r
        elif n == 2:
            if pair1 < 0:
                pair1 = r
            elif pair2 < 0:
                pair2 = r

    if quad >= 0:
        return (7 << 20) | (quad << 16) | (_top_ranks(rank_mask & ~(1 << quad), 1) << 12)
    if trips1 >= 0 and (trips2 >= 0 or pair1 >= 0):
        return (6 << 20) | (trips1 << 16) | (max(trips2, pair1) << 12)
    if flush_suit >= 0:
        return (5 << 20) | _top_ranks(suit_masks[flush_suit], 5)
    high = _straight_high(rank_mask)
    if high >= 0:
        return (4 << 20) | (high << 16)
    if trips1 >= 0:
        return (3 << 20) | (trips1 << 16) | (_top_ranks(rank_mask & ~(1 << trips1), 2) << 8)
    if pair2 >= 0:
        kicker = _top_ranks(rank_mask & ~(1 << pair1) & ~(1 << pair2), 1)
        return (2 << 20) | (pair1 << 16) | (pair2 << 12) | (kicker << 8)
    if pair1 >= 0:
        return (1 << 20) | (pair1 << 16) | (_top_ranks(rank_mask & ~(1 << pair1), 3) << 4)
    return _top_ranks(rank_mask, 5)


def hand_category(strength: int) -> str:
    return CATEGORIES[strength >> 20]
