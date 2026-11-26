"""Simple rule-based bots. Each one turns an Observation into "f" (fold), "c" (check/call) or "r" (bet/raise)."""


class Bot:
    name = "bot"
    description = ""

    def act(self, obs) -> str:
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"
    description = "Picks uniformly among the legal actions."

    def act(self, obs):
        choices = ["c"]
        if obs.can_raise:
            choices.append("r")
        if obs.to_call > 0:
            choices.append("f")
        return obs.rng.choice(choices)


class CallingStation(Bot):
    name = "station"
    description = "Calls everything, never raises, never folds."

    def act(self, obs):
        return "c"


class Rock(Bot):
    name = "rock"
    description = "Tight-passive: plays premium hands and mostly calls."

    def act(self, obs):
        edge = obs.edge
        if obs.street == 0:
            if edge > 2.2:
                return "r"
            return "c" if edge > 1.5 else "f"
        if edge > 2.4:
            return "r"
        return "c" if obs.equity() > obs.pot_odds + 0.08 else "f"


class TightAggressive(Bot):
    name = "tag"
    description = "Tight-aggressive: raises good hands, calls with pot odds, folds the rest."

    def act(self, obs):
        edge = obs.edge
        if obs.street == 0:
            if edge > 1.6 or (edge > 1.3 and obs.num_bets == 1):
                return "r"
            if edge > 1.2 and obs.to_call <= obs.bet_size:
                return "c"
            return "f"
        if edge > 1.6:
            return "r"
        return "c" if obs.equity() > obs.pot_odds else "f"


class Maniac(Bot):
    name = "maniac"
    description = "Loose-aggressive: plays most hands, raises often, bluffs a third of the time."

    def act(self, obs):
        edge = obs.edge
        if obs.street == 0:
            if edge > 0.9:
                return "r"
            return "c" if edge > 0.7 else "f"
        if edge > 1.1 or obs.rng.random() < 0.3:
            return "r"
        return "c" if obs.equity() > 0.7 * obs.pot_odds else "f"


class PotOddsBot(Bot):
    name = "math"
    description = "Calls whenever equity beats the pot odds and raises with a clear edge."

    def act(self, obs):
        if obs.edge > 1.4:
            return "r"
        return "c" if obs.equity() > obs.pot_odds else "f"


BOTS = {cls.name: cls for cls in (TightAggressive, Rock, Maniac, CallingStation, RandomBot, PotOddsBot)}
