"""The browser must execute the same exact-card policy on the same public inputs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.browser_exact_fixtures import generate
from src.exact_policy import ExactPolicy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


def test_complete_games_browser_inputs_moves_and_heatmap_match_python(tmp_path):
    node = os.environ.get("SEQUENCE_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(918)
        model = ExactPolicy.from_history_policy(HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8))
        with torch.no_grad():
            model.spatial_projection.weight.normal_(std=0.1)
            model.card_bias.normal_(std=0.1)
        fixtures = generate(model)
    finally:
        torch.set_num_threads(previous)
    assert {event[3] for fixture in fixtures for event in fixture["events"]} == {0, 1, 2}
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(fixtures))
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const ctx=vm.createContext({});
vm.runInContext(fs.readFileSync('web/static/game_engine.js','utf8')+'\n'+
                fs.readFileSync('web/static/direct_player.js','utf8')+
                ';globalThis.api={SequenceGame,DirectPlayer,cardToIndex};',ctx);
const runtime={Tensor:class {constructor(type,data,dims){Object.assign(this,{type,data,dims});}}};
const config={policyKind:'exact_sequence_policy',actionEncoding:'card_id * 100 + row * 10 + col'};
(async()=>{
 for(const fixture of JSON.parse(fs.readFileSync(process.argv[1],'utf8'))){
  const game=new ctx.api.SequenceGame();game.hands=fixture.hands;game.deck=fixture.deck;
  for(const expected of fixture.steps){
   const before=JSON.stringify(game);
   const session={inputNames:['board','context','history','actions'],run:async feeds=>{
    for(const name of ['board','context','history','actions'])
     assert.deepEqual(Array.from(feeds[name].data,Number),expected[name]);
    assert.equal(feeds.actions.type,'int64');
    assert.equal(feeds.board.type,'float32');
    assert.equal(feeds.actions.dims[1],expected.actions.length);
    return {logits:{data:Float32Array.from(expected.logits)},value:{data:[expected.value]}};
   }};
   const result=await new ctx.api.DirectPlayer(session,runtime,config).chooseMove(game);
   assert.equal(ctx.api.cardToIndex(result.move.card)*100+result.move.row*10+result.move.col,expected.chosen);
   assert.equal(result.value,expected.value);
   for(let i=0;i<100;i++)assert.ok(Math.abs(result.policy[i]-expected.policy[i])<2e-7);
   assert.equal(JSON.stringify(game),before);
   const opponent=3-game.currentPlayer, oldOpponent=game.hands[opponent], deck=game.deck;
   const own=game.hands[game.currentPlayer];
   game.hands[opponent]=oldOpponent.map(()=>'JC');game.deck=[...deck].reverse();
   game.hands[game.currentPlayer]=[...own].reverse();
   const changed=game.getExactInputs();
   for(const name of ['board','context','history','actions'])
    assert.deepEqual(Array.from(changed[name],Number),expected[name]);
   game.hands[opponent]=oldOpponent;game.hands[game.currentPlayer]=own;game.deck=deck;
   assert.equal(game.makeMove(expected.move),true);
  }
  assert.equal(JSON.stringify(game.publicEvents),JSON.stringify(fixture.events));
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    subprocess.run([node, "-e", script, str(path)], cwd=Path(__file__).resolve().parents[1], check=True)
