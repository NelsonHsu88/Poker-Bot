"""Push-or-fold preflop solver using chance-sampled CFR+.

Every player either shoves their whole stack or folds; once someone has shoved,
"in" means calling. With equal stacks there are no side pots, so a deal plus a
sampled board fixes the payoff of every one of the 2**n in/out combinations.
Each iteration samples one deal, walks the full action tree bottom-up, and
updates regrets for every player at once. Payoffs are chip EV in big blinds.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .evaluator import evaluate
from .preflop import CLASS_NAMES, class_index, range_percent

POSITION_NAMES = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["UTG", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
}


@njit(cache=True)
def _cfr(n, stack, ante, iterations, start, regret, strat_sum, seed):
    np.random.seed(seed)
    deck = np.arange(52)
    posted = np.full(n, ante)
    posted[n - 2] += 0.5
    posted[n - 1] += 1.0
    stack_in = stack + ante
    cls = np.empty(n, np.int64)
    rank = np.empty(n, np.int64)
    hand = np.empty(7, np.int64)
    n_nodes = (1 << (n + 1)) - 1
    value = np.zeros((n_nodes, n))
    reach = np.zeros(n_nodes)
    sigma = np.zeros((n_nodes, 2))
    for it in range(iterations):
        for i in range(2 * n + 5):
            j = np.random.randint(i, 52)
            deck[i], deck[j] = deck[j], deck[i]
        for b in range(5):
            hand[2 + b] = deck[2 * n + b]
        for p in range(n):
            hand[0] = deck[2 * p]
            hand[1] = deck[2 * p + 1]
            rank[p] = evaluate(hand)
            cls[p] = class_index(deck[2 * p], deck[2 * p + 1])

        leaf = (1 << n) - 1
        for m in range(1 << n):
            pot = 0.0
            best = -1
            for p in range(n):
                if (m >> p) & 1:
                    pot += stack_in
                    if rank[p] > best:
                        best = rank[p]
                else:
                    pot += posted[p]
            winners = 0
            for p in range(n):
                if (m >> p) & 1 and rank[p] == best:
                    winners += 1
            for p in range(n):
                if (m >> p) & 1:
                    share = pot / winners if rank[p] == best else 0.0
                    value[leaf + m, p] = share - stack_in
                else:
                    value[leaf + m, p] = -posted[p]

        for d in range(n):
            for m in range(1 << d):
                node = (1 << d) - 1 + m
                if d == n - 1 and m == 0:
                    sigma[node, 0] = 0.0
                    sigma[node, 1] = 1.0
                    continue
                r0 = max(regret[node, cls[d], 0], 0.0)
                r1 = max(regret[node, cls[d], 1], 0.0)
                if r0 + r1 > 0:
                    sigma[node, 0] = r0 / (r0 + r1)
                    sigma[node, 1] = r1 / (r0 + r1)
                else:
                    sigma[node, 0] = 0.5
                    sigma[node, 1] = 0.5

        reach[0] = 1.0
        for d in range(n):
            for m in range(1 << d):
                node = (1 << d) - 1 + m
                base = (1 << (d + 1)) - 1
                reach[base + m] = reach[node] * sigma[node, 0]
                reach[base + (m | (1 << d))] = reach[node] * sigma[node, 1]

        weight = start + it + 1.0
        for d in range(n - 1, -1, -1):
            base = (1 << (d + 1)) - 1
            for m in range(1 << d):
                node = (1 << d) - 1 + m
                fold_child = base + m
                in_child = base + (m | (1 << d))
                for p in range(n):
                    value[node, p] = sigma[node, 0] * value[fold_child, p] + sigma[node, 1] * value[in_child, p]
                if d == n - 1 and m == 0:
                    continue
                c = cls[d]
                v = value[node, d]
                regret[node, c, 0] = max(regret[node, c, 0] + reach[node] * (value[fold_child, d] - v), 0.0)
                regret[node, c, 1] = max(regret[node, c, 1] + reach[node] * (value[in_child, d] - v), 0.0)
                strat_sum[node, c, 0] += weight * sigma[node, 0]
                strat_sum[node, c, 1] += weight * sigma[node, 1]


@dataclass
class PushFoldSolution:
    n_players: int
    stack: float
    ante: float
    iterations: int
    strat_sum: np.ndarray

    @property
    def positions(self):
        return POSITION_NAMES[self.n_players]

    def _node(self, position: int, earlier_in: int) -> int:
        return (1 << position) - 1 + earlier_in

    def in_frequency(self, position, earlier_in=()) -> np.ndarray:
        """Probability per hand class (169) that a player goes in, given which earlier seats are already in."""
        pos = self.positions.index(position) if isinstance(position, str) else position
        mask = 0
        for seat in earlier_in:
            mask |= 1 << (self.positions.index(seat) if isinstance(seat, str) else seat)
        s = self.strat_sum[self._node(pos, mask)]
        total = s.sum(axis=1)
        return np.where(total > 0, s[:, 1] / np.maximum(total, 1e-12), 0.5)

    def push_range(self, position) -> np.ndarray:
        """Open-shove frequency when everyone before this seat folded."""
        return self.in_frequency(position, ())

    def call_range(self, position, shover) -> np.ndarray:
        """Calling frequency facing one shove from an earlier seat, with everyone else folded."""
        return self.in_frequency(position, (shover,))

    def summary(self):
        rows = []
        for pos in self.positions[:-1]:
            rows.append((f"{pos} open-shove", range_percent(self.push_range(pos))))
        rows.append((f"BB call vs SB", range_percent(self.call_range(self.positions[-1], self.positions[-2]))))
        return rows

    def hands(self, freqs, threshold=0.5):
        return [CLASS_NAMES[i] for i in np.argsort(-freqs) if freqs[i] >= threshold]


def solve_push_fold(n_players=5, stack=10.0, ante=0.0, iterations=3_000_000, seed=0, batch=500_000):
    if n_players not in POSITION_NAMES:
        raise ValueError(f"n_players must be one of {sorted(POSITION_NAMES)}")
    n_nodes = (1 << n_players) - 1
    regret = np.zeros((n_nodes, 169, 2))
    strat_sum = np.zeros((n_nodes, 169, 2))
    done = 0
    while done < iterations:
        step = min(batch, iterations - done)
        _cfr(n_players, float(stack), float(ante), step, done, regret, strat_sum, seed + done)
        done += step
    return PushFoldSolution(n_players, stack, ante, iterations, strat_sum)
