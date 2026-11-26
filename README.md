# poker-bot

A poker AI playground in Python: a fast hand evaluator, a Monte Carlo equity calculator, a 5-player bot simulator, and a counterfactual regret minimization (CFR) solver for push-or-fold poker, plus the charts that come out of them. The hot loops are compiled with [Numba](https://numba.pydata.org/), so 200,000 equity runouts take a quarter of a second, the simulator plays about 13,000 hands a second per core, and the solver runs 600,000 CFR iterations a second.

```bash
pip install -e ".[dev]"
pytest -q
```

## 1. Monte Carlo equity calculator

Deal random runouts thousands of times and count how often each hand wins. Known hands, a partial board, and any number of random opponents can be mixed.

```
$ pokerbot equity AsAh -o 4
AsAh + 4 random, board [-], 200,000 runouts in 0.26s
hand  equity     win    tie
AsAh  55.76%  55.54%  0.56%

$ pokerbot equity AhKh QsQc --board Qh7h2c
hand  equity     win    tie
AhKh  25.40%  25.40%  0.00%
QsQc  74.60%  74.60%  0.00%
```

`pokerbot table -o 4` prints the 169-hand preflop table, which ships precomputed in [pokerbot/data/preflop_equity.json](pokerbot/data/preflop_equity.json) (100,000 runouts per hand against 1 to 4 random opponents).

![Preflop equity vs 4 random hands](docs/img/preflop_equity.png)

From Python:

```python
from pokerbot import equity
equity(["AsAh", "KdKc"], board="2c7d9h", opponents=2, iterations=100_000)
```

## 2. Five-player bot simulator

[pokerbot/engine.py](pokerbot/engine.py) deals fixed-limit hold'em (blinds 0.5/1, bets 1/1/2/2, 4-bet cap) between bots defined in [pokerbot/bots.py](pokerbot/bots.py). Each bot sees its cards, the board, the pot, and the price to call, and can ask for its equity against the opponents still in the hand. Seats are reshuffled every 5,000 hands and the button rotates every hand, so position luck washes out.

| bot | style |
| --- | --- |
| `tag` | tight-aggressive: raises good hands, calls when equity beats the pot odds |
| `rock` | tight-passive: premium hands only, mostly calls |
| `maniac` | loose-aggressive: plays most hands, raises often, bluffs 30% of the time |
| `station` | calls everything |
| `random` | uniform among legal actions |
| `math` | pure pot-odds caller that raises with a clear edge |

```
$ pokerbot simulate -n 1000000 -w 4
    bot    hands  bb/100  ±95%  VPIP%  PFR%  WSD%
 maniac  1000000  155.03  2.12   83.6  59.0  48.5
    tag  1000000    81.2  1.55   15.8   9.9  53.3
   rock  1000000   16.78   0.9    5.5   1.1  57.6
station  1000000  -88.31  1.98   95.3   0.0  40.8
 random  1000000  -164.7  1.24   65.6  38.2  42.4
```

![Bot tournament results](docs/img/simulator.png)

Against a table with a calling station and a random bot, relentless aggression pays best: the maniac's raises get called by worse hands and its bluffs are only called by the station. Swap the lineup (`--bots tag rock maniac math math`) and the ordering changes, which is the fun part. To add a bot, subclass `Bot`, implement `act(obs)` returning `"f"`, `"c"`, or `"r"`, and register it in `BOTS`.

## 3. Push/fold equilibrium with CFR

[pokerbot/pushfold.py](pokerbot/pushfold.py) solves the short-stack game where every player either shoves all-in or folds, and anyone facing a shove calls or folds. Information sets are (seat, which earlier seats are already in, hand class), so a 5-max game has 31 decision nodes times 169 hands.

The solver is chance-sampled CFR+: each iteration deals five hands and a board, which fixes the payoff of all 32 in/out combinations, then walks the tree bottom-up and updates every player's regrets at once. Regrets are floored at zero and the average strategy is weighted linearly by iteration. Heads-up at 10bb it lands on the known Nash ranges (SB shoves 57%, BB calls 39%), and two seeds at 60M iterations agree to within 0.2 points.

```
$ pokerbot solve -p 5 -s 10
5-max push/fold at 10bb, ante 0: 20,000,000 CFR+ iterations in 34s
UTG open-shoves 20.5% ...
```

![5-max push/fold ranges at 10bb](docs/img/pushfold_10bb.png)

The grids show how often each hand goes in. Hands that sit near 0% or 100% are clear decisions; the mixed ones in between are close to break-even, and the few-percent floor on trash hands is left over from the early, near-random iterations in the average.

## 4. Analysis: how ranges move with stack size

`pokerbot charts` reruns the solver at every stack depth from 3bb to 25bb and redraws every chart in [docs/img](docs/img).

![Push/fold ranges vs stack size](docs/img/stack_sweep.png)

| stack | UTG | CO | BTN | SB | BB calls SB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3bb | 33.2% | 37.4% | 43.3% | 75.6% | 85.1% |
| 5bb | 30.4% | 33.8% | 40.3% | 68.3% | 58.7% |
| 10bb | 21.3% | 26.5% | 32.5% | 54.5% | 36.9% |
| 15bb | 16.0% | 20.7% | 28.0% | 43.6% | 28.8% |
| 25bb | 11.2% | 14.2% | 19.7% | 34.7% | 20.1% |

## 5. A trained bot: 10,000 bots evolved in a league

[pokerbot/strategy.py](pokerbot/strategy.py) describes a whole limit hold'em strategy with 10 numbers: preflop raise and call thresholds, how much each extra bet raises them, a position bonus, postflop bet, raise and call thresholds, and a bluff frequency. [pokerbot/fastsim.py](pokerbot/fastsim.py) is a compiled copy of the game engine in which every seat plays one of these strategies, running about 75,000 hands a second on 16 cores. [pokerbot/evolve.py](pokerbot/evolve.py) trains them with a genetic algorithm:

1. Start 10,000 bots with random settings, which covers everything from calling stations to maniacs.
2. Each generation, each bot plays 750 hands at three kinds of tables: other population members only, four members plus one league opponent, and alone against four copies of one league opponent. The league holds fixed playing styles (station, maniac, rock, TAG, pot-odds), every earlier generation's best bot, and fresh random bots.
3. The top 5% survive. The rest are replaced by children of tournament-selected parents, made by crossover plus mutation.
4. After 50 generations, the 50 best bots and the 10 latest generation champions play 40,000 hands against each opponent pool. The winner is the bot whose worst matchup is best.

The league is there because a first run without it evolved a single aggressive style. Its champion beat five of the six hand-written bots but lost 68 bb/100 to a table of maniacs, because nothing left in the population punished it for folding too much under pressure.

```bash
pokerbot evolve --population 10000 --generations 50 --out runs/league --install   # about 2 hours on 16 cores
pokerbot benchmark evolved --baselines tag maniac -n 300000
```

![Training curves](docs/img/evolution.png)

The left chart shows the generation's best bot measured on only 10,000 hands, so the lines are noisy. The final pick is made by the 40,000-hand playoff. The right chart shows the population settling on a strategy of raising or folding preflop, betting when checked to, and calling postflop only with a clear margin over the pot odds.

**Held-out test.** The trained model ships as [pokerbot/data/evolved.json](pokerbot/data/evolved.json) and plays as the `evolved` bot. It is tested in the regular Python engine against the hand-written bots from section 2, which are separate code from the training league. Each cell is 300,000 hands with a 95% confidence interval. Every cell sits at the same table with the same opponents, so the columns compare directly:

| opponents | evolved | tag | maniac |
| --- | ---: | ---: | ---: |
| 4 × tag | **+26.8 ± 1.5** | +0.3 ± 0.9 | −78.0 ± 2.7 |
| 4 × rock | **+53.0 ± 0.6** | +17.1 ± 0.4 | +48.8 ± 1.0 |
| 4 × maniac | **+99.6 ± 5.7** | +41.3 ± 3.5 | −0.0 ± 10.7 |
| 4 × station | +161.6 ± 3.5 | +112.4 ± 2.3 | +166.7 ± 3.8 |
| 4 × math | **+50.5 ± 2.8** | +7.5 ± 2.0 | −164.0 ± 4.7 |
| 4 × random | +231.8 ± 3.0 | +48.9 ± 1.7 | +643.0 ± 4.4 |
| tag, rock, maniac, station | **+170.1 ± 4.0** | +75.3 ± 2.7 | +95.0 ± 6.1 |

The trained bot wins against every opponent, and every interval is far from zero. It beats the hand-written TAG in every lineup and the maniac in five of seven. The maniac still does better against pure calling stations and random bots, because it bets more into opponents who never fold correctly.

The trained genes:

| gene | value | meaning |
| --- | ---: | --- |
| pf_raise | 0.87 | opens with a raise on any hand with at least 87% of a fair share of equity |
| pf_call | 1.96 | almost never just calls an unraised pot: raise or fold |
| pf_reraise_step | 0.52 | each extra bet raises the bar for re-raising |
| pf_call_raised | 0.85 | defends raised pots wide, since the pot odds are good |
| bet | 1.00 | bets when checked to with an above-average hand |
| bluff | 0.20 | bluffs 20% of the time when checked to |
| raise | 1.82 | raises a bet only with nearly twice its fair share of equity |
| call_margin | 0.12 | calls a bet only when equity beats the pot odds by 12 points |
| raised_margin | 0.09 | needs 9 more points of equity for each extra bet |
| position | −0.09 | barely uses position |

## Caveats

- The "edge" is measured against bots in this simulator. Real players adapt, read betting patterns, and don't play from fixed thresholds, so these win rates say nothing about a real table. Running a bot on real-money poker sites also breaks their terms of service.

- Bots estimate equity against random hands, not against what their opponents' actions say they hold. That is the obvious next upgrade.
- CFR is only guaranteed to find a Nash equilibrium in two-player zero-sum games. In 5-player push/fold it converges to a stable, sensible strategy, but that is an empirical result, not a theorem.
- Payoffs are chip EV. Tournament pay jumps (ICM) are not modeled, and the solver allows no limping or min-raising.

## Layout

```
pokerbot/
  cards.py        card parsing ("As" -> 48)
  evaluator.py    7-card hand evaluator (Numba)
  equity.py       Monte Carlo equity
  preflop.py      169 hand classes and the precomputed preflop table
  engine.py       fixed-limit hold'em dealer
  bots.py         rule-based bots
  simulate.py     multi-process tournament runner and stats
  pushfold.py     CFR+ push/fold solver
  strategy.py     10-number genome strategy and the EvolvedBot
  fastsim.py      compiled engine for training genomes
  evolve.py       league genetic algorithm
  charts.py       matplotlib charts
  cli.py          the `pokerbot` command
tests/            evaluator vs brute force, equity sanity, engine zero-sum, solver ranges
```
