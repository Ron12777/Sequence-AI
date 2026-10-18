"""Search inputs derived from public discards and the observer's own cards."""


def unseen_cards(game, observer):
    """Two decks minus public discards and own hand; never inspect hidden identities.

    This variant exposes skipped dead cards as discards. Opponent hand size is
    public, but its contents and the actual remaining deck order are not inputs.
    """
    if observer not in (1, 2):
        raise ValueError("observer must be player 1 or 2")
    counts = [2] * 52
    for card in [*game.hands[observer], *game.discard_piles[1], *game.discard_piles[2]]:
        counts[card.to_int()] -= 1
    if min(counts) < 0:
        raise ValueError("public card history exceeds the two-deck card counts")
    return [card for card, count in enumerate(counts) for _ in range(count)]


def search_class(mode):
    import sequence_native

    if mode == "ensemble":
        return sequence_native.EnsembleMCTS
    if mode == "single":
        return sequence_native.CMCTS
    raise ValueError("search mode must be ensemble or single")
