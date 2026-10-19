/** Stable display slots, separate from the engine/model's ordered hand arrays. */
class HandLayout {
    constructor() {
        this.game = null;
        this.slots = {};
    }

    sync(game) {
        if (this.game === game) return;
        this.game = game;
        this.slots = Object.fromEntries([1, 2].map(player =>
            [player, game.hands[player].map((_, index) => index)]));
    }

    cards(game, player) {
        this.sync(game);
        return game.hands[player].map((card, index) =>
            ({card, index, slot: this.slots[player][index]}))
            .sort((a, b) => a.slot - b.slot);
    }

    makeMove(game, move, preferredIndex = null) {
        this.sync(game);
        const player = game.currentPlayer;
        const hand = game.hands[player];
        const length = hand.length;
        const removedIndex = hand.indexOf(move.card);
        const selectedIndex = Number.isInteger(preferredIndex) && hand[preferredIndex] === move.card
            ? preferredIndex : removedIndex;
        if (!game.makeMove(move)) return false;
        const slots = this.slots[player];
        // Identical copies are interchangeable to the engine, but the UI must
        // replace the particular copy the person clicked.
        [slots[removedIndex], slots[selectedIndex]] = [slots[selectedIndex], slots[removedIndex]];
        const [vacatedSlot] = slots.splice(removedIndex, 1);
        const exchanges = game.lastCardExchanges || [];
        const removedDeadCards = exchanges.filter(e => e.player === player && !e.replacement).length;
        if (game.hands[player].length + removedDeadCards === length) slots.push(vacatedSlot);
        for (const exchange of exchanges) {
            if (!exchange.replacement) this.slots[exchange.player].splice(exchange.index, 1);
        }
        // With no replacement, leave that grid column empty; other cards stay put.
        return true;
    }
}

if (typeof module !== 'undefined' && module.exports) module.exports = {HandLayout};
