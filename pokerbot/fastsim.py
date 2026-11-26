"""Compiled copy of engine.py where every seat plays a genome, for training at scale.

Rules, blinds, bet sizes and equity estimates match engine.py; tests/test_fastsim.py
checks that both engines give a genome the same results.
"""

import numpy as np
from numba import njit, prange

from .equity import _equity_core
from .evaluator import evaluate
from .preflop import class_index, equity_grid
from .strategy import FOLD, RAISE, decide

POSTFLOP_SAMPLES = 200


def preflop_table():
    return np.stack([equity_grid(k) for k in range(1, 5)])


@njit(cache=True)
def play_table(genomes, hands, seed, pre_eq, net):
    """Play `hands` hands at one table of 3 to 5 genomes, rotating the button, adding each seat's result to net."""
    np.random.seed(seed)
    n = genomes.shape[0]
    deck = np.arange(52)
    contrib = np.zeros(n)
    street_bets = np.zeros(n)
    active = np.zeros(n, np.bool_)
    eq = np.zeros(n)
    eq_opp = np.zeros(n, np.int64)
    hole = np.empty(2, np.int64)
    hand = np.empty(7, np.int64)
    strength = np.zeros(n, np.int64)
    for h in range(hands):
        button = h % n
        for i in range(2 * n + 5):
            j = np.random.randint(i, 52)
            deck[i], deck[j] = deck[j], deck[i]
        contrib[:] = 0.0
        active[:] = True
        sb = (button + 1) % n
        bb = (button + 2) % n
        contrib[sb] = 0.5
        contrib[bb] = 1.0
        n_active = n
        for street in range(4):
            board = deck[2 * n : 2 * n + (0 if street == 0 else street + 2)]
            eq_opp[:] = -1
            street_bets[:] = 0.0
            if street == 0:
                street_bets[sb] = 0.5
                street_bets[bb] = 1.0
                current = 1.0
                num_bets = 1
                seat = (button + 3) % n
            else:
                current = 0.0
                num_bets = 0
                seat = sb
            bet = 1.0 if street < 2 else 2.0
            pending = n_active
            while pending > 0 and n_active > 1:
                if active[seat]:
                    to_call = current - street_bets[seat]
                    n_opp = n_active - 1
                    if eq_opp[seat] != n_opp:
                        if street == 0:
                            eq[seat] = pre_eq[n_opp - 1, class_index(deck[2 * seat], deck[2 * seat + 1])]
                        else:
                            hole[0] = deck[2 * seat]
                            hole[1] = deck[2 * seat + 1]
                            eq[seat] = _equity_core(hole, board, n_opp, POSTFLOP_SAMPLES)[0][0]
                        eq_opp[seat] = n_opp
                    late = ((seat - button - 1 + n) % n) / (n - 1)
                    action = decide(
                        genomes[seat], street, eq[seat], n_opp, to_call, contrib.sum(),
                        num_bets, num_bets < 4, late, np.random.random(),
                    )
                    if action == RAISE and num_bets < 4:
                        current += bet
                        contrib[seat] += current - street_bets[seat]
                        street_bets[seat] = current
                        num_bets += 1
                        pending = n_active - 1
                    elif action != FOLD or to_call == 0:
                        contrib[seat] += to_call
                        street_bets[seat] = current
                        pending -= 1
                    else:
                        active[seat] = False
                        n_active -= 1
                        pending -= 1
                seat = (seat + 1) % n
            if n_active == 1:
                break

        pot = contrib.sum()
        if n_active == 1:
            for p in range(n):
                net[p] -= contrib[p]
                if active[p]:
                    net[p] += pot
            continue
        for b in range(5):
            hand[2 + b] = deck[2 * n + b]
        best = -1
        for p in range(n):
            if active[p]:
                hand[0] = deck[2 * p]
                hand[1] = deck[2 * p + 1]
                strength[p] = evaluate(hand)
                if strength[p] > best:
                    best = strength[p]
        winners = 0
        for p in range(n):
            if active[p] and strength[p] == best:
                winners += 1
        for p in range(n):
            net[p] -= contrib[p]
            if active[p] and strength[p] == best:
                net[p] += pot / winners


@njit(cache=True, parallel=True)
def play_tables(tables, hands, seed, pre_eq):
    """tables: (k, seats, genes). Returns each seat's total winnings in big blinds, shape (k, seats).

    Tables run in parallel threads; each seeds its own thread-local generator, so results
    depend only on the seed, not on how tables were spread across threads.
    """
    net = np.zeros((tables.shape[0], tables.shape[1]))
    for t in prange(tables.shape[0]):
        play_table(tables[t], hands, seed + t, pre_eq, net[t])
    return net
