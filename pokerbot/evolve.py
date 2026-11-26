"""Train a bot by evolving a large population of genome bots in a league.

Each generation every bot plays three kinds of tables:

1. population only: 5 random members of the population;
2. one outsider: 4 members plus one league opponent;
3. outnumbered: 1 member against 4 copies of one league opponent.

The league is a set of fixed playing styles (ARCHETYPES), every past generation's best
bot (a hall of fame), and fresh random bots. Without it the population drifts into
one style and its champion only beats that style.

The top `elite` share survives unchanged; the rest of the next generation are children
of tournament-selected parents, made by uniform crossover plus Gaussian mutation. At the
end the finalists play every opponent pool and the champion is the one whose worst
matchup is best (maximin), so it has to beat every style, not just the average.
"""

import json
import time
from pathlib import Path

import numpy as np

from .fastsim import play_tables, preflop_table
from .strategy import GENE_NAMES, HIGH, LOW, N_GENES, random_genomes, save_model

SEATS = 5
NEVER = 99.0
ARCHETYPES = {
    "station": [NEVER, 0, -1, -1, 0, NEVER, 0, NEVER, -1, 0],
    "maniac": [0.9, 0, 0.7, 0.7, 0, 1.1, 0.3, 1.1, -0.2, 0],
    "rock": [2.2, 0, 1.5, 1.5, 0, 2.4, 0, 2.4, 0.08, 0],
    "tag": [1.3, 0.3, 1.2, 1.6, 0, 1.6, 0, 1.6, 0, 0],
    "math": [1.4, 0, 0.8, 1.0, 0, 1.4, 0, 1.4, 0, 0],
}


def _seed(*parts):
    return int(np.random.default_rng(list(parts)).integers(0, 2**31 - 1))


def _against(candidates, opponents, tables_each, hands, seed, pre_eq, rng, copies=False):
    """bb/100 of each candidate in seat 0; the other seats are drawn from opponents (all the same one if copies)."""
    k = len(candidates)
    n = k * tables_each
    tables = np.empty((n, SEATS, N_GENES))
    tables[:, 0] = np.repeat(candidates, tables_each, axis=0)
    if copies:
        tables[:, 1:] = opponents[rng.integers(0, len(opponents), n)][:, None]
    else:
        tables[:, 1:] = opponents[rng.integers(0, len(opponents), (n, SEATS - 1))]
    net = play_tables(tables, hands, seed, pre_eq)[:, 0].reshape(k, tables_each)
    return 100 * net.sum(axis=1) / (tables_each * hands)


def _league(hall_of_fame, n, rng):
    """n league opponents: half archetypes, 30% past champions, 20% fresh random bots."""
    archetypes = np.array(list(ARCHETYPES.values()), float)
    pick = rng.random(n)
    out = archetypes[rng.integers(0, len(archetypes), n)]
    if len(hall_of_fame):
        hof = pick < 0.3
        out[hof] = hall_of_fame[rng.integers(0, len(hall_of_fame), hof.sum())]
    fresh = pick > 0.8
    out[fresh] = random_genomes(fresh.sum(), rng)
    return out


def _fitness(pop, hall_of_fame, hands, seed, gen, pre_eq, rng):
    """bb/100 of every member over the three kinds of tables."""
    total = np.zeros(len(pop))
    played = 0

    perm = rng.permutation(len(pop))
    net = play_tables(pop[perm].reshape(-1, SEATS, N_GENES), hands, _seed(seed, gen, 0), pre_eq)
    total[perm] += net.reshape(-1)
    played += hands

    perm = rng.permutation(len(pop))
    members = pop[perm].reshape(-1, SEATS - 1, N_GENES)
    tables = np.concatenate([members, _league(hall_of_fame, len(members), rng)[:, None]], axis=1)
    net = play_tables(tables, hands, _seed(seed, gen, 1), pre_eq)
    total[perm] += net[:, : SEATS - 1].reshape(-1)
    played += hands

    short = max(hands // 2, 1)
    for r in (2, 3):
        tables = np.repeat(_league(hall_of_fame, len(pop), rng)[:, None], SEATS, axis=1)
        tables[:, 0] = pop
        total += play_tables(tables, short, _seed(seed, gen, r), pre_eq)[:, 0]
        played += short
    return 100 * total / played


def _next_generation(pop, fitness, elite, mutation, rng):
    n = len(pop)
    order = np.argsort(-fitness)
    n_elite = max(1, int(elite * n))
    k = n - n_elite
    picks = rng.integers(0, n, (2, k, 3))
    parents = np.take_along_axis(picks, np.argmax(fitness[picks], axis=2)[..., None], axis=2)[..., 0]
    children = np.where(rng.random((k, N_GENES)) < 0.5, pop[parents[0]], pop[parents[1]])
    mutate = rng.random((k, N_GENES)) < 0.3
    children = children + mutate * rng.normal(0.0, mutation * (HIGH - LOW), (k, N_GENES))
    return np.concatenate([pop[order[:n_elite]], np.clip(children, LOW, HIGH)])


def matchups(candidates, pools, tables_each, hands, seed, pre_eq):
    """bb/100 of each candidate against each named pool; a pool of one genome means a table of 4 copies."""
    rng = np.random.default_rng(seed)
    return {
        name: _against(candidates, pool, tables_each, hands, _seed(seed, i), pre_eq, rng, copies=len(pool) == 1)
        for i, (name, pool) in enumerate(pools.items())
    }


def evolve(
    population=10_000,
    generations=50,
    hands=250,
    elite=0.05,
    mutation=0.08,
    seed=0,
    out="runs/evolve",
    finalists=50,
    playoff_tables=100,
    playoff_hands=400,
    log=print,
):
    """hands: hands per kind of table per generation; each bot plays 3 x hands in total."""
    if population % (SEATS * (SEATS - 1)):
        raise ValueError(f"population must be a multiple of {SEATS * (SEATS - 1)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pre_eq = preflop_table()
    fresh = random_genomes(5_000, np.random.default_rng(seed + 1))
    yardsticks = {"random bots": fresh, **{name: np.array([g], float) for name, g in ARCHETYPES.items()}}
    checkpoint = out / "checkpoint.npz"
    history_path = out / "history.json"
    if checkpoint.exists() and history_path.exists():
        saved = np.load(checkpoint)
        pop, hall_of_fame = saved["population"], saved["hall_of_fame"]
        history = json.loads(history_path.read_text())
        log(f"resuming at generation {len(history)}")
    else:
        pop = random_genomes(population, np.random.default_rng(seed))
        hall_of_fame = np.empty((0, N_GENES))
        history = []

    for gen in range(len(history), generations):
        start = time.perf_counter()
        rng = np.random.default_rng([seed, gen])
        fitness = _fitness(pop, hall_of_fame, hands, seed, gen, pre_eq, rng)
        best = pop[np.argmax(fitness)]
        checks = {k: float(v[0]) for k, v in matchups(best[None], yardsticks, 40, 250, _seed(seed, gen, 99), pre_eq).items()}
        hall_of_fame = np.concatenate([hall_of_fame, best[None]])
        history.append(
            {
                "generation": gen,
                "population": len(pop),
                "best_fitness": float(fitness.max()),
                "top10pct_fitness": float(np.sort(fitness)[-len(pop) // 10 :].mean()),
                "best_matchups": checks,
                "best_worst_matchup": min(checks.values()),
                "best": dict(zip(GENE_NAMES, map(float, best))),
                "mean": dict(zip(GENE_NAMES, map(float, pop.mean(axis=0)))),
                "std": dict(zip(GENE_NAMES, map(float, pop.std(axis=0)))),
            }
        )
        if gen < generations - 1:
            pop = _next_generation(pop, fitness, elite, mutation, rng)
        history[-1]["seconds"] = round(time.perf_counter() - start, 1)
        np.savez(checkpoint, population=pop, fitness=fitness, hall_of_fame=hall_of_fame)
        history_path.write_text(json.dumps(history))
        worst = min(checks, key=checks.get)
        log(
            f"gen {gen:3d}  top10% {history[-1]['top10pct_fitness']:6.1f}  best bot: vs random {checks['random bots']:6.1f}, "
            f"worst matchup {worst} {checks[worst]:6.1f} bb/100  ({history[-1]['seconds']}s)"
        )

    fitness = np.load(checkpoint)["fitness"]
    candidates = np.concatenate([pop[np.argsort(-fitness)[:finalists]], hall_of_fame[-10:]])
    pools = {"final population": pop, "hall of fame": hall_of_fame, **yardsticks}
    scores = matchups(candidates, pools, playoff_tables, playoff_hands, seed + 2, pre_eq)
    worst = np.min(np.stack(list(scores.values())), axis=0)
    winner = int(np.argmax(worst))
    champion = candidates[winner]
    playoff = {name: round(float(v[winner]), 1) for name, v in scores.items()}
    log(f"champion playoff bb/100: {playoff}")
    meta = {
        "population": len(pop),
        "generations": generations,
        "hands_per_bot_per_generation": 3 * hands,
        "seed": seed,
        "playoff_bb100": playoff,
        "playoff_hands_per_matchup": playoff_tables * playoff_hands,
    }
    save_model(champion, out / "evolved.json", **meta)
    return champion, history, meta
