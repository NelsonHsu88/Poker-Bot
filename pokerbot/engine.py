"""Fixed-limit Texas hold'em for 3 to 9 bots, measured in big blinds.

Blinds are 0.5/1, bets are 1 preflop and on the flop and 2 on the turn and river,
and each street is capped at 4 bets. Stacks are unlimited, so nobody is ever all-in.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from .equity import hand_equity
from .evaluator import evaluate
from .preflop import preflop_equity

SMALL_BLIND = 0.5
BIG_BLIND = 1.0
BET_SIZES = (1.0, 1.0, 2.0, 2.0)
BET_CAP = 4
STREETS = ("preflop", "flop", "turn", "river")
BOARD_SIZES = (0, 3, 4, 5)
POSTFLOP_SAMPLES = 200


@dataclass
class Observation:
    position: int
    num_players: int
    hole: tuple
    board: tuple
    street: int
    pot: float
    to_call: float
    bet_size: float
    can_raise: bool
    num_opponents: int
    num_bets: int
    rng: random.Random
    _cache: dict = field(repr=False)

    def equity(self) -> float:
        """Pot share against the remaining opponents holding random hands."""
        key = (self.hole, self.street, self.num_opponents)
        if key not in self._cache:
            if self.street == 0:
                value = preflop_equity(self.hole, self.num_opponents)
            else:
                value = hand_equity(
                    self.hole, self.board, self.num_opponents, POSTFLOP_SAMPLES, self.rng.getrandbits(31)
                )
            self._cache[key] = value
        return self._cache[key]

    @property
    def edge(self) -> float:
        """Equity relative to a fair share: 1.0 is average, 2.0 is twice the fair share."""
        return self.equity() * (self.num_opponents + 1)

    @property
    def pot_odds(self) -> float:
        return self.to_call / (self.pot + self.to_call) if self.to_call > 0 else 0.0


@dataclass
class HandResult:
    net: list
    vpip: list
    pfr: list
    showdown: list
    won_showdown: list


def play_hand(bots, button: int, rng: random.Random) -> HandResult:
    n = len(bots)
    if n < 3:
        raise ValueError("the engine needs at least 3 players")
    deck = list(range(52))
    rng.shuffle(deck)
    holes = [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]
    runout = deck[2 * n : 2 * n + 5]

    contrib = [0.0] * n
    active = [True] * n
    vpip = [False] * n
    pfr = [False] * n
    sb, bb = (button + 1) % n, (button + 2) % n
    contrib[sb] = SMALL_BLIND
    contrib[bb] = BIG_BLIND
    n_active = n
    cache = {}

    for street in range(4):
        board = tuple(runout[: BOARD_SIZES[street]])
        street_bets = [0.0] * n
        if street == 0:
            street_bets[sb], street_bets[bb] = SMALL_BLIND, BIG_BLIND
            current, num_bets, seat = BIG_BLIND, 1, (button + 3) % n
        else:
            current, num_bets, seat = 0.0, 0, (button + 1) % n
        bet_size = BET_SIZES[street]
        pending = n_active
        while pending > 0 and n_active > 1:
            if active[seat]:
                to_call = current - street_bets[seat]
                obs = Observation(
                    position=(seat - button) % n,
                    num_players=n,
                    hole=holes[seat],
                    board=board,
                    street=street,
                    pot=sum(contrib),
                    to_call=to_call,
                    bet_size=bet_size,
                    can_raise=num_bets < BET_CAP,
                    num_opponents=n_active - 1,
                    num_bets=num_bets,
                    rng=rng,
                    _cache=cache,
                )
                action = bots[seat].act(obs)
                if action == "f" and to_call > 0:
                    active[seat] = False
                    n_active -= 1
                    pending -= 1
                elif action == "r" and num_bets < BET_CAP:
                    current += bet_size
                    contrib[seat] += current - street_bets[seat]
                    street_bets[seat] = current
                    num_bets += 1
                    pending = n_active - 1
                    if street == 0:
                        vpip[seat] = pfr[seat] = True
                else:
                    contrib[seat] += to_call
                    street_bets[seat] = current
                    pending -= 1
                    if street == 0 and to_call > 0:
                        vpip[seat] = True
            seat = (seat + 1) % n
        if n_active == 1:
            break

    pot = sum(contrib)
    live = [i for i in range(n) if active[i]]
    won = [0.0] * n
    showdown = [False] * n
    won_showdown = [False] * n
    if len(live) == 1:
        won[live[0]] = pot
    else:
        strengths = {i: evaluate(np.array(holes[i] + tuple(runout), np.int64)) for i in live}
        best = max(strengths.values())
        winners = [i for i in live if strengths[i] == best]
        for i in live:
            showdown[i] = True
        for i in winners:
            won[i] = pot / len(winners)
            won_showdown[i] = True
    return HandResult([won[i] - contrib[i] for i in range(n)], vpip, pfr, showdown, won_showdown)
