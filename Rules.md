# Implemented rules

This repository implements a two-player Sequence variant, not the complete multiplayer tabletop rules.

- Each player starts with seven cards from two combined decks and needs two sequences to win.
- A sequence is five connected chips in a straight horizontal, vertical, or diagonal line.
- Corners are free spaces shared by both players.
- Play a card to place a chip on its matching empty square, then draw a replacement.
- Clubs/diamonds Jacks place a chip on any empty square. Hearts/spades Jacks remove an opposing chip that is not part of a completed sequence.
- Completed sequences are protected. Two sequences can overlap in at most one chip; a nine-chip line can therefore count as two.
- Ordinary cards with no empty matching square are automatically discarded when drawing. Jacks are retained. This rule applies equally to both players.
- Discards are not recycled into the deck. If no replacement remains, the hand shrinks.
- In the browser, if the player whose turn it is has no legal move, the game ends as a draw. An already completed win takes precedence.
- Training additionally treats 200 moves as a draw.

Manual dead-card exchange, passes, missed-draw penalties, and multiplayer teams are not implemented. Editing is a sandbox and can create positions or hands that would not arise in a normal shuffled game.
