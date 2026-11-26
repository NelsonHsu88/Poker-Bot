import random

import numpy as np
from numba import njit

from .cards import parse_cards
from .evaluator import evaluate


@njit(cache=True)
def _equity_mc(known, board, n_random, iterations, seed):
    np.random.seed(seed)
    n_known = known.shape[0] // 2
    n_board = board.shape[0]
    used = np.zeros(52, np.bool_)
    for c in known:
        used[c] = True
    for c in board:
        used[c] = True
    deck = np.empty(52 - known.shape[0] - n_board, np.int64)
    k = 0
    for c in range(52):
        if not used[c]:
            deck[k] = c
            k += 1

    n_players = n_known + n_random
    need = 2 * n_random + 5 - n_board
    shares = np.zeros(n_players)
    wins = np.zeros(n_players)
    ties = np.zeros(n_players)
    ranks = np.empty(n_players, np.int64)
    hand = np.empty(7, np.int64)
    for _ in range(iterations):
        for i in range(need):
            j = np.random.randint(i, deck.shape[0])
            deck[i], deck[j] = deck[j], deck[i]
        idx = 0
        for b in range(5):
            if b < n_board:
                hand[2 + b] = board[b]
            else:
                hand[2 + b] = deck[idx]
                idx += 1
        best = -1
        for p in range(n_players):
            if p < n_known:
                hand[0] = known[2 * p]
                hand[1] = known[2 * p + 1]
            else:
                hand[0] = deck[idx]
                hand[1] = deck[idx + 1]
                idx += 2
            ranks[p] = evaluate(hand)
            if ranks[p] > best:
                best = ranks[p]
        tied = 0
        for p in range(n_players):
            if ranks[p] == best:
                tied += 1
        for p in range(n_known):
            if ranks[p] == best:
                shares[p] += 1.0 / tied
                if tied == 1:
                    wins[p] += 1
                else:
                    ties[p] += 1
    return shares[:n_known] / iterations, wins[:n_known] / iterations, ties[:n_known] / iterations


def _seed(seed):
    return random.getrandbits(31) if seed is None else seed


def equity(hands, board="", opponents=0, iterations=100_000, seed=None):
    """Monte Carlo equity of each known hand against each other plus random opponents.

    hands: list of strings like ["AsAh", "KdKc"]; board: "" or up to 5 cards like "2c7d9h".
    Returns one dict per hand with equity (pot share), win and tie probabilities.
    """
    parsed = [parse_cards(h) for h in hands]
    if any(len(h) != 2 for h in parsed):
        raise ValueError("each hand needs exactly two cards")
    board_cards = parse_cards(board) if board else []
    if len(board_cards) > 5:
        raise ValueError("board has at most 5 cards")
    known = np.array([c for h in parsed for c in h], np.int64)
    if len(set(known.tolist()) | set(board_cards)) != len(known) + len(board_cards):
        raise ValueError("duplicate card across hands and board")
    if 2 * (len(parsed) + opponents) + 5 > 52:
        raise ValueError("too many players for one deck")
    eq, win, tie = _equity_mc(known, np.array(board_cards, np.int64), opponents, iterations, _seed(seed))
    return [
        {"hand": h, "equity": float(e), "win": float(w), "tie": float(t)}
        for h, e, w, t in zip(hands, eq, win, tie)
    ]


def hand_equity(hole, board, opponents, iterations=1_000, seed=None):
    """Pot share of two hole cards (ints) against random opponents, for bots that need speed."""
    eq, _, _ = _equity_mc(
        np.array(hole, np.int64), np.array(board, np.int64), opponents, iterations, _seed(seed)
    )
    return float(eq[0])
