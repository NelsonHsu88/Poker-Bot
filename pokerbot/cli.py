import argparse
import time

from .bots import BOTS
from .cards import RANKS
from .equity import equity
from .preflop import CLASS_NAMES, equity_grid, range_percent


def _print_grid(values, fmt):
    print("     " + "".join(f"{r:>5}" for r in reversed(RANKS)))
    for row in range(13):
        cells = "".join(f"{fmt(values[row * 13 + col]):>5}" for col in range(13))
        print(f"{RANKS[12 - row]:>5}" + cells)


def _table(rows):
    cols = list(rows[0])
    widths = [max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(str(c).rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))


def cmd_equity(args):
    start = time.perf_counter()
    results = equity(args.hands, args.board, args.opponents, args.iterations, args.seed)
    took = time.perf_counter() - start
    opp = f" + {args.opponents} random" if args.opponents else ""
    print(f"{' vs '.join(args.hands)}{opp}, board [{args.board or '-'}], {args.iterations:,} runouts in {took:.2f}s")
    _table([{k: (f"{v:.2%}" if isinstance(v, float) else v) for k, v in r.items()} for r in results])


def cmd_table(args):
    print(f"Preflop equity vs {args.opponents} random hand(s); rows are the first card, suited above the diagonal")
    _print_grid(equity_grid(args.opponents), lambda v: f"{100 * v:.0f}")


def cmd_simulate(args):
    from .simulate import simulate

    start = time.perf_counter()
    stats = simulate(args.bots, args.hands, args.workers, args.seed)
    took = time.perf_counter() - start
    print(f"{args.hands:,} hands in {took:.1f}s ({args.hands / took:,.0f} hands/s)")
    _table([s.as_row() for s in stats])
    if args.chart:
        from .charts import simulator_chart

        simulator_chart(args.chart, stats)
        print(f"chart written to {args.chart}")


def cmd_solve(args):
    from .pushfold import solve_push_fold

    start = time.perf_counter()
    sol = solve_push_fold(args.players, args.stack, args.ante, args.iterations, args.seed)
    print(f"{args.players}-max push/fold at {args.stack:g}bb, ante {args.ante:g}: {args.iterations:,} CFR+ iterations in {time.perf_counter() - start:.1f}s")
    for pos in sol.positions[:-1]:
        freqs = sol.push_range(pos)
        print(f"\n{pos} open-shoves {range_percent(freqs):.1f}% (percent of the time each hand shoves):")
        _print_grid(freqs, lambda v: "." if v < 0.05 else f"{100 * v:.0f}")
    bb, sb = sol.positions[-1], sol.positions[-2]
    freqs = sol.call_range(bb, sb)
    print(f"\n{bb} calls a {sb} shove {range_percent(freqs):.1f}%:")
    _print_grid(freqs, lambda v: "." if v < 0.05 else f"{100 * v:.0f}")
    if args.chart:
        from .charts import pushfold_chart

        pushfold_chart(args.chart, sol)
        print(f"chart written to {args.chart}")


def cmd_charts(args):
    from .charts import make_all, simulator_chart
    from .simulate import simulate

    stats = simulate(args.bots, args.hands, args.workers, args.seed)
    sweep = make_all(args.out, stats, args.iterations, args.sweep_iterations)
    print(f"charts written to {args.out}")
    _table([s.as_row() for s in stats])
    rows = [{"stack": s, **{k: f"{v:.1f}" for k, v in pcts.items()}} for s, pcts in sweep.items()]
    _table(rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pokerbot", description="Poker AI playground")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("equity", help="Monte Carlo equity of hands against each other and random opponents")
    e.add_argument("hands", nargs="+", help="hole cards like AsAh KdKc")
    e.add_argument("--board", default="", help="0 to 5 board cards like 2c7d9h")
    e.add_argument("-o", "--opponents", type=int, default=0, help="extra opponents with random hands")
    e.add_argument("-n", "--iterations", type=int, default=200_000)
    e.add_argument("--seed", type=int)
    e.set_defaults(func=cmd_equity)

    t = sub.add_parser("table", help="print the 13x13 preflop equity table")
    t.add_argument("-o", "--opponents", type=int, default=4, choices=[1, 2, 3, 4])
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("simulate", help="5-player limit hold'em bot tournament")
    s.add_argument("--bots", nargs="+", default=["tag", "rock", "maniac", "station", "random"], choices=sorted(BOTS))
    s.add_argument("-n", "--hands", type=int, default=100_000)
    s.add_argument("-w", "--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--chart", help="write a bar chart PNG here")
    s.set_defaults(func=cmd_simulate)

    v = sub.add_parser("solve", help="push/fold equilibrium with CFR+")
    v.add_argument("-p", "--players", type=int, default=5, choices=[2, 3, 4, 5, 6])
    v.add_argument("-s", "--stack", type=float, default=10.0, help="effective stack in big blinds")
    v.add_argument("--ante", type=float, default=0.0, help="ante per player in big blinds")
    v.add_argument("-n", "--iterations", type=int, default=20_000_000)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--chart", help="write a range-grid PNG here")
    v.set_defaults(func=cmd_solve)

    c = sub.add_parser("charts", help="regenerate every chart in docs/img")
    c.add_argument("--out", default="docs/img")
    c.add_argument("--bots", nargs="+", default=["tag", "rock", "maniac", "station", "random"], choices=sorted(BOTS))
    c.add_argument("--hands", type=int, default=200_000)
    c.add_argument("-w", "--workers", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--iterations", type=int, default=20_000_000)
    c.add_argument("--sweep-iterations", type=int, default=10_000_000)
    c.set_defaults(func=cmd_charts)

    args = p.parse_args(argv)
    args.func(args)
