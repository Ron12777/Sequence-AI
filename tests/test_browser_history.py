"""Browser public-history inputs must exactly match training observations."""

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, observe


def test_browser_history_matches_training_and_excludes_hidden_cards(tmp_path):
    node = os.environ.get("SEQUENCE_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    fixtures, event_kinds = [], set()
    for seed in range(8):
        random.seed(7100 + seed)
        game, events = FastSequenceGame(), []
        fixture = {
            "hands": {p: list(map(str, h)) for p, h in game.hands.items()},
            "deck": list(map(str, game.deck.cards)),
            "steps": [],
        }
        while not game.game_over:
            moves = game.get_legal_moves()
            if not moves:
                break
            board, context, history = observe(game, events)
            move = random.choice(moves)
            fixture["steps"].append(
                {
                    "board": board.flatten().tolist(),
                    "context": context.tolist(),
                    "history": history.flatten().tolist(),
                    "move": {
                        "card": str(move.card),
                        "row": move.row,
                        "col": move.col,
                        "is_removal": move.is_removal,
                    },
                }
            )
            apply_public_move(game, events, move)
        event_kinds.update(e[3] for e in events)
        fixture["events"] = events
        fixtures.append(fixture)
    assert event_kinds == {0, 1, 2}  # Placements, removals, and skipped dead draws.
    path = tmp_path / "history.json"
    path.write_text(json.dumps(fixtures))
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const ctx=vm.createContext({});
vm.runInContext(fs.readFileSync('web/static/game_engine.js','utf8')+'\n'+
                fs.readFileSync('web/static/mcts.js','utf8')+
                ';globalThis.api={SequenceGame,cloneGame};',ctx);
for(const fixture of JSON.parse(fs.readFileSync(process.argv[1],'utf8'))){
  const game=new ctx.api.SequenceGame();game.hands=fixture.hands;game.deck=fixture.deck;
  for(const expected of fixture.steps){
    const inputs=game.getHistoryInputs();
    for(const name of ['board','context','history'])
      assert.deepEqual(Array.from(inputs[name],Number),expected[name]);
    const opponent=3-game.currentPlayer, hand=game.hands[opponent], deck=game.deck;
    game.hands[opponent]=hand.map(()=>'JC');game.deck=[...deck].reverse();
    const hiddenChanged=game.getHistoryInputs();
    for(const name of ['board','context','history'])
      assert.deepEqual(Array.from(hiddenChanged[name],Number),expected[name]);
    game.hands[opponent]=hand;game.deck=deck;
    assert.equal(game.makeMove(expected.move),true);
  }
  assert.equal(JSON.stringify(game.publicEvents),JSON.stringify(fixture.events));
  const clone=ctx.api.cloneGame(game);
  clone.publicEvents[0][0]=-1;
  assert.notEqual(game.publicEvents[0][0],-1);
  game.publicHistoryComplete=false;
  assert.throws(()=>game.getHistoryInputs(),/complete move history/);
  game.reset();
  assert.equal(game.publicEvents.length,0);
  assert.equal(game.publicHistoryComplete,true);
}
"""
    subprocess.run(
        [node, "-e", script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
