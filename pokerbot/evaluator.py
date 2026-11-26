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
def _high_bit(mask):
    r = 12
    while r >= 0:
        if (mask >> r) & 1:
            return r
        r -= 1
    return -1


@njit(cache=True)
def evaluate(cards):
    """Strength of the best 5-card hand in 5 to 7 cards; higher is better.

    Cards are ints 0..51 with rank = c >> 2 (0 is a deuce, 12 an ace) and suit = c & 3.
    The result is category << 20 | tiebreak ranks packed in 4-bit nibbles. Rank counts
    are added up bitwise across the four suit masks, so nothing is allocated.
    """
    s0 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    for c in cards:
        bit = 1 << (c >> 2)
        s = c & 3
        if s == 0:
            s0 |= bit
        elif s == 1:
            s1 |= bit
        elif s == 2:
            s2 |= bit
        else:
            s3 |= bit

    flush_mask = 0
    if _popcount(s0) >= 5:
        flush_mask = s0
    elif _popcount(s1) >= 5:
        flush_mask = s1
    elif _popcount(s2) >= 5:
        flush_mask = s2
    elif _popcount(s3) >= 5:
        flush_mask = s3
    if flush_mask:
        high = _straight_high(flush_mask)
        if high >= 0:
            return (8 << 20) | (high << 16)

    a = s0 ^ s1
    b = s0 & s1
    c = s2 ^ s3
    d = s2 & s3
    ones = a ^ c
    carry = a & c
    twos = b ^ d ^ carry
    quads = (b & d) | (carry & (b ^ d))
    trips = twos & ones
    pairs = twos & ~ones
    rank_mask = s0 | s1 | s2 | s3

    if quads:
        q = _high_bit(quads)
        return (7 << 20) | (q << 16) | (_top_ranks(rank_mask & ~(1 << q), 1) << 12)
    if trips:
        t = _high_bit(trips)
        rest = (trips & ~(1 << t)) | pairs
        if rest:
            return (6 << 20) | (t << 16) | (_high_bit(rest) << 12)
    if flush_mask:
        return (5 << 20) | _top_ranks(flush_mask, 5)
    high = _straight_high(rank_mask)
    if high >= 0:
        return (4 << 20) | (high << 16)
    if trips:
        t = _high_bit(trips)
        return (3 << 20) | (t << 16) | (_top_ranks(rank_mask & ~(1 << t), 2) << 8)
    if pairs:
        p1 = _high_bit(pairs)
        rest = pairs & ~(1 << p1)
        if rest:
            p2 = _high_bit(rest)
            kicker = _top_ranks(rank_mask & ~(1 << p1) & ~(1 << p2), 1)
            return (2 << 20) | (p1 << 16) | (p2 << 12) | (kicker << 8)
        return (1 << 20) | (p1 << 16) | (_top_ranks(rank_mask & ~(1 << p1), 3) << 4)
    return _top_ranks(rank_mask, 5)


def hand_category(strength: int) -> str:
    return CATEGORIES[strength >> 20]
