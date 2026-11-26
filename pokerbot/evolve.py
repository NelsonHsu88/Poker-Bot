"""Train a bot by evolving a large population of genome bots against each other.

Each generation every bot plays `hands` hands at randomly drawn 5-player tables
(reshuffled `rounds` times so nobody is stuck with one set of opponents). The top
`elite` share survives unchanged; the rest of the next generation are children of
tournament-selected parents, made by uniform crossover plus Gaussian mutation.

Fitness inside the population is relative (the population always sums to zero), so
progress is tracked against a frozen yardstick: the generation's best bot playing
tables of fresh random bots that never evolve.
"""

import json
import time
from pathlib import Path

import numpy as np

from .fastsim import play_tables, preflop_table
from .strategy import GENE_NAMES, HIGH, LOW, N_GENES, random_genomes, save_model

SEATS = 5


def _against(candidates, opponents, tables_each, hands, seed, pre_eq, rng):
    """bb/100 of each candidate in seat 0 against opponents drawn at random."""
    k = len(candidates)
    tables = np.empty((k * tables_each, SEATS, N_GENES))
    tables[:, 0] = np.repeat(candidates, tables_each, axis=0)
    tables[:, 1:] = opponents[rng.integers(0, len(opponents), (k * tables_each, SEATS - 1))]
    net = play_tables(tables, hands, seed, pre_eq)[:, 0].reshape(k, tables_each)
    return 100 * net.sum(axis=1) / (tables_each * hands)


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


def evolve(
    population=10_000,
    generations=60,
    hands=1_000,
    rounds=4,
    elite=0.05,
    mutation=0.08,
    seed=0,
    out="runs/evolve",
    finalists=100,
    playoff_tables=200,
    playoff_hands=500,
    log=print,
):
    if population % SEATS:
        raise ValueError(f"population must be a multiple of {SEATS}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pre_eq = preflop_table()
    rng = np.random.default_rng(seed)
    yardstick = random_genomes(5_000, np.random.default_rng(seed + 1))
    checkpoint = out / "checkpoint.npz"
    history_path = out / "history.json"
    if checkpoint.exists() and history_path.exists():
        pop = np.load(checkpoint)["population"]
        history = json.loads(history_path.read_text())
        log(f"resuming at generation {len(history)}")
    else:
        pop = random_genomes(population, rng)
        history = []

    per_round = hands // rounds
    for gen in range(len(history), generations):
        start = time.perf_counter()
        gen_rng = np.random.default_rng([seed, gen])
        fitness = np.zeros(len(pop))
        for r in range(rounds):
            perm = gen_rng.permutation(len(pop))
            net = play_tables(pop[perm].reshape(-1, SEATS, N_GENES), per_round, (seed * 1000 + gen) * 100_000 + r * 10_000, pre_eq)
            fitness[perm] += net.reshape(-1)
        fitness *= 100 / (per_round * rounds)
        best = pop[np.argmax(fitness)]
        vs_random = float(_against(best[None], yardstick, 200, 250, 7_000_000 + gen * 1000, pre_eq, gen_rng)[0])
        history.append(
            {
                "generation": gen,
                "best_fitness": float(fitness.max()),
                "top10pct_fitness": float(np.sort(fitness)[-len(pop) // 10 :].mean()),
                "best_vs_random_bots": vs_random,
                "best": dict(zip(GENE_NAMES, map(float, best))),
                "mean": dict(zip(GENE_NAMES, map(float, pop.mean(axis=0)))),
                "std": dict(zip(GENE_NAMES, map(float, pop.std(axis=0)))),
                "seconds": 0.0,
            }
        )
        if gen < generations - 1:
            pop = _next_generation(pop, fitness, elite, mutation, gen_rng)
        history[-1]["seconds"] = round(time.perf_counter() - start, 1)
        np.savez(checkpoint, population=pop, fitness=fitness)
        history_path.write_text(json.dumps(history))
        log(
            f"gen {gen:3d}  best {fitness.max():7.1f}  top10% {history[-1]['top10pct_fitness']:7.1f}  "
            f"best vs random bots {vs_random:7.1f} bb/100  ({history[-1]['seconds']}s)"
        )

    fitness = np.load(checkpoint)["fitness"]
    candidates = pop[np.argsort(-fitness)[:finalists]]
    rng = np.random.default_rng(seed + 2)
    vs_pop = _against(candidates, pop, playoff_tables, playoff_hands, 9_000_000, pre_eq, rng)
    vs_rand = _against(candidates, yardstick, playoff_tables, playoff_hands, 9_500_000, pre_eq, rng)
    score = (vs_pop + vs_rand) / 2
    winner = int(np.argmax(score))
    champion = candidates[winner]
    playoff = {
        "vs_final_population_bb100": float(vs_pop[winner]),
        "vs_random_bots_bb100": float(vs_rand[winner]),
        "hands_each": playoff_tables * playoff_hands,
    }
    log(f"champion: {playoff}")
    meta = {
        "population": len(pop),
        "generations": generations,
        "hands_per_bot_per_generation": per_round * rounds,
        "seed": seed,
        "playoff": playoff,
    }
    save_model(champion, out / "evolved.json", **meta)
    return champion, history, meta
