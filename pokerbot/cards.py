RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(text: str) -> int:
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"bad card {text!r}")
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def parse_cards(text: str) -> list[int]:
    text = "".join(text.split()).replace(",", "")
    if len(text) % 2:
        raise ValueError(f"cannot parse cards from {text!r}")
    cards = [card(text[i : i + 2]) for i in range(0, len(text), 2)]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate card in {text!r}")
    return cards


def card_str(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]
