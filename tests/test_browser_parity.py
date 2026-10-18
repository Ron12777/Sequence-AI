"""Replay a seeded Python game through the actual browser engine (requires Node)."""

import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import pytest
from src.game import SequenceGame


def test_browser_replays_reference_game(tmp_path):
    node = os.environ.get("SEQUENCE_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    random.seed(109)
    game = SequenceGame()
    game.reset()
    fixture = {
        "hands": {p: [str(c) for c in h] for p, h in game.hands.items()},
        "deck": [str(c) for c in game.deck.cards],
        "steps": [],
    }
    for _ in range(200):
        legal = game.get_legal_moves()
        if not legal:
            break
        move = random.choice(legal)
        assert game.make_move(move)
        fixture["steps"].append(
            {
                "move": {
                    "card": str(move.card),
                    "row": move.row,
                    "col": move.col,
                    "is_removal": move.is_removal,
                },
                "board": game.board.tolist(),
                "winner": game.winner,
                "over": game.game_over,
                "hands": {p: [str(c) for c in h] for p, h in game.hands.items()},
                "deck": [str(c) for c in game.deck.cards],
                "discards": {
                    p: [str(c) for c in cards]
                    for p, cards in game.discard_piles.items()
                },
            }
        )
    path = tmp_path / "game.json"
    path.write_text(json.dumps(fixture))
    script = """
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context=vm.createContext({});
vm.runInContext(fs.readFileSync('web/static/game_engine.js','utf8')+';globalThis.Game=SequenceGame;',context);
const fixture=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const game=new context.Game();game.hands=fixture.hands;game.deck=fixture.deck;
for(const expected of fixture.steps){
  assert.equal(game.makeMove(expected.move),true);
  assert.equal(JSON.stringify(game.board),JSON.stringify(expected.board));
  assert.equal(JSON.stringify(game.hands),JSON.stringify(expected.hands));
  assert.equal(JSON.stringify(game.deck),JSON.stringify(expected.deck));
  assert.equal(JSON.stringify(game.discardPiles),JSON.stringify(expected.discards));
  assert.equal(game.winner,expected.winner);assert.equal(game.gameOver,expected.over);
}
"""
    subprocess.run(
        [node, "-e", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
