const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({console, setTimeout});
vm.runInContext(fs.readFileSync('web/static/game_engine.js', 'utf8') + '\n' +
                fs.readFileSync('web/static/mcts.js', 'utf8') + '\n' +
                'globalThis.engine = {SequenceGame, MCTSNode, MCTS, cloneGame};', context);
const {SequenceGame, MCTSNode, MCTS, cloneGame} = context.engine;

test('illegal browser moves cannot mutate hand or board', () => {
  const game = new SequenceGame(); game.hands[1] = ['AC'];
  const before = JSON.stringify(game);
  assert.equal(game.makeMove({card:'AC',row:4,col:4,is_removal:false}), false);
  assert.equal(JSON.stringify(game),before);
  assert.equal(game.makeMove({card:'AC',row:-1,col:0,is_removal:false}), false);
});

test('nine-chip line can count as two sequences sharing one chip', () => {
  const game=new SequenceGame();
  for(let c=0;c<9;c++)game.board[3][c]=1;
  game.completedSequences[1]=[['3,0','3,1','3,2','3,3','3,4']];
  assert.deepEqual(Array.from(game.findNewSequences(1,3,8)[0]),['3,4','3,5','3,6','3,7','3,8']);
});

test('policy aggregates shared squares and includes removals', () => {
  const root=new MCTSNode(new SequenceGame());
  for(const [visits,removal] of [[2,false],[3,false],[5,true]]) {
    const child=new MCTSNode(root.game,root,{row:1,col:1,is_removal:removal});
    child.visits=visits;root.children.push(child);
  }
  assert.equal(root.getPolicy()[11],1);
});

test('unvisited search actions use neural priors', () => {
  const root=new MCTSNode(new SequenceGame());root.visits=1;
  const a=new MCTSNode(root.game,root),b=new MCTSNode(root.game,root);
  a.prior=.01;b.prior=.99;root.children=[a,b];
  assert.equal(root.selectChild(),b);
});

test('search returns a legal move and normalized policy', async () => {
  const game=new SequenceGame();game.reset();
  const result=await new MCTS(null,4).search(game);
  assert.ok(game.getLegalMoves().some(m=>JSON.stringify(m)===JSON.stringify(result.move)));
  assert.ok(Math.abs(Array.from(result.policy).reduce((a,b)=>a+b,0)-1)<1e-6);
});

test('duplicate cards do not duplicate actions', () => {
  const game = new SequenceGame();game.hands[1]=['AC'];
  const count=game.getLegalMoves().length;
  game.hands[1].push('AC');
  assert.equal(game.getLegalMoves().length,count);
});

test('one-simulation search selects the highest prior', () => {
  const root=new MCTSNode(new SequenceGame());
  const a=new MCTSNode(root.game,root,{row:1,col:1});
  const b=new MCTSNode(root.game,root,{row:2,col:2});
  a.prior=.1;b.prior=.9;root.children=[a,b];
  assert.equal(root.getBestMove(),b.move);
});

test('multi-deal search preserves the total simulation budget and public state', async () => {
  const game = new SequenceGame();
  const before = JSON.stringify(game);
  const search = new MCTS(null, 128, 4);
  const seen = new Set();
  let calls = 0;
  const simulate = search.simulate.bind(search);
  search.simulate = async root => {
    seen.add(JSON.stringify(root.game.hands[2])); calls++;
    return simulate(root);
  };
  const result = await search.search(game);
  assert.equal(calls,128);
  assert.equal(result.deals,4);
  assert.equal(seen.size,4);
  assert.equal(JSON.stringify(game),before);
  assert.ok(game.getLegalMoves().some(m=>JSON.stringify(m)===JSON.stringify(result.move)));
  assert.ok(Math.abs(Array.from(result.policy).reduce((a,b)=>a+b,0)-1)<1e-6);
});

test('browser records both played cards and skipped dead cards', () => {
  const game = new SequenceGame();
  game.hands[1] = ['JC'];
  game.deck = ['AD','AC'];
  game.board[0][1]=1; game.board[5][2]=2; // Both AC targets occupied.
  assert.equal(game.makeMove({card:'JC',row:4,col:4,is_removal:false}),true);
  assert.deepEqual(Array.from(game.discardPiles[1]),['JC','AC']);
  assert.deepEqual(Array.from(game.hands[1]),['AD']);
});


vm.runInContext(fs.readFileSync('web/static/direct_player.js', 'utf8') +
                ';globalThis.DirectPlayer=DirectPlayer;',context);
const {DirectPlayer}=context;
const runtime={Tensor:class {constructor(type,data,dims){Object.assign(this,{type,data,dims});}}};

test('direct policy masks illegal squares and preserves ordinary cards on score ties', async () => {
  const game=new SequenceGame();game.hands[1]=['JC','AC'];
  const logits=new Float32Array(100).fill(-10000);logits[0]=10000;logits[1]=3;
  let calls=0;
  const player=new DirectPlayer({inputNames:['board','context','history'],run:async inputs=>{
    calls++;assert.equal(inputs.history.type,'int64');assert.equal(inputs.context.data.length,157);
    return {policy:{data:logits},value:{data:[.25]}};
  }},runtime);
  const before=JSON.stringify(game),result=await player.chooseMove(game);
  assert.equal(result.move.card,'AC');assert.equal(result.move.col,1);
  assert.equal(result.policy[0],0);assert.equal(result.value,.25);
  assert.equal(calls,1);assert.equal(JSON.stringify(game),before);
  game.publicHistoryComplete=false;
  await assert.rejects(()=>player.chooseMove(game),/complete move history/);
  assert.equal(calls,1);
});

test('direct policy rejects invalid inference instead of guessing a move', async () => {
  const game=new SequenceGame();game.hands[1]=['AC'];
  const player=new DirectPlayer({inputNames:['board','context','history'],run:async()=>({
    policy:{data:new Float32Array(100).fill(NaN)},value:{data:[0]}
  })},runtime);
  await assert.rejects(()=>player.chooseMove(game),/invalid predictions/);
});

test('direct board policy works without public history and never invokes search', async () => {
  const game=new SequenceGame();game.hands[1]=['AC'];game.publicHistoryComplete=false;
  game.getHistoryInputs=()=>{throw Error('Board policy must not request history');};
  const player=new DirectPlayer({inputNames:['input'],run:async feeds=>{
    assert.equal(Object.keys(feeds).join(','),'input');
    assert.equal(feeds.input.data.length,1100);
    const logits=new Float32Array(100);logits[1]=2;
    return {policy:{data:logits},value:{data:[0]}};
  }},runtime);
  assert.equal((await player.chooseMove(game)).move.col,1);
  assert.equal(player.usesHistory,false);
});

const exactConfig={policyKind:'exact_sequence_policy',actionEncoding:'card_id * 100 + row * 10 + col'};
const exactInputNames=['board','context','history','actions'];

test('exact policy spends its chosen Jack even when a normal card reaches the same square', async () => {
  const game=new SequenceGame();game.hands[1]=['JC','AC','AC'];game.deck=[];
  const player=new DirectPlayer({inputNames:exactInputNames,run:async feeds=>{
    const ids=Array.from(feeds.actions.data,Number);
    assert.equal(new Set(ids).size,ids.length);
    assert.equal(ids.filter(id=>id===1).length,1); // AC at square 1.
    const logits=new Float32Array(ids.length).fill(-4);
    logits[ids.indexOf(1001)]=9; // JC at square 1.
    return {logits:{data:logits},value:{data:[.4]}};
  }},runtime,exactConfig);
  const result=await player.chooseMove(game);
  assert.equal(result.move.card,'JC');assert.equal(result.move.col,1);
  assert.equal(result.move.is_removal,false);
  assert.ok(Math.abs(Array.from(result.policy).reduce((a,b)=>a+b,0)-1)<1e-6);
  assert.equal(game.makeMove(result.move),true);
  assert.deepEqual(Array.from(game.hands[1]),['AC','AC']);
});

test('exact policy preserves one-eyed Jack removal identity and protected-square legality', async () => {
  const game=new SequenceGame();game.hands[1]=['JH','JS'];game.deck=[];
  game.board[4][4]=2;game.board[3][3]=2;
  game.completedSequences[2]=[['3,3','3,4','3,5','3,6','3,7']];
  const player=new DirectPlayer({inputNames:exactInputNames,run:async feeds=>{
    const ids=Array.from(feeds.actions.data,Number);
    assert.ok(!ids.some(id=>id%100===33));
    const logits=new Float32Array(ids.length);
    logits[ids.indexOf(4944)]=5; // JS rank10 + suit3*13 -> card49.
    return {logits:{data:logits},value:{data:[-.2]}};
  }},runtime,exactConfig);
  const result=await player.chooseMove(game);
  assert.equal(result.move.card,'JS');assert.equal(result.move.is_removal,true);
  assert.equal(game.makeMove(result.move),true);assert.equal(game.board[4][4],0);
  assert.deepEqual(Array.from(game.hands[1]),['JH']);
});

test('exact policy requires explicit compatible manifest and validates predictions', async () => {
  const session={inputNames:exactInputNames,run:async()=>({logits:{data:[NaN]},value:{data:[0]}})};
  assert.throws(()=>new DirectPlayer(session,runtime),/Unsupported direct policy inputs/);
  assert.throws(()=>new DirectPlayer(session,runtime,{...exactConfig,actionEncoding:'square'}),/action encoding/);
  assert.throws(()=>new DirectPlayer({inputNames:['input']},runtime,exactConfig),/requires board/);
  const game=new SequenceGame();game.hands[1]=['AC'];
  await assert.rejects(()=>new DirectPlayer(session,runtime,exactConfig).chooseMove(game),/invalid predictions/);
  game.publicHistoryComplete=false;
  await assert.rejects(()=>new DirectPlayer(session,runtime,exactConfig).chooseMove(game),/complete move history/);
});

test('unbounded regularized values are displayed as scores rather than win percentages', () => {
  const source=fs.readFileSync('web/static/game.js','utf8');
  const start=source.indexOf('function updateEvalBar(value)');
  const end=source.indexOf('\n}\n',start)+2;
  Object.assign(context,{game:{currentPlayer:1},evalBarContainer:{},hasEvalValue:false,
    showEvalBar:{checked:false},evalBarFill:{style:{}},evalText:{},
    window:{matchMedia:()=>({matches:false})},
    aiPlayer:new DirectPlayer({inputNames:exactInputNames},runtime,{...exactConfig,valueMode:'unbounded'})});
  vm.runInContext(source.slice(start,end),context);
  context.updateEvalBar(2.5);
  assert.equal(context.evalText.textContent,'Red +2.50 value');
  assert.ok(parseFloat(context.evalBarFill.style.width)<100);
  context.game.currentPlayer=2;context.updateEvalBar(2.5);
  assert.equal(context.evalText.textContent,'Blue +2.50 value');
  context.aiPlayer.valueMode='bounded';context.updateEvalBar(.4);
  assert.equal(context.evalText.textContent,'Blue 70%');
});

test('explicit board-hand CNN analyzes edited positions and hint retains exact card', async () => {
  const game=new SequenceGame();game.hands[1]=['AC','JC'];game.publicHistoryComplete=false;
  game.board[4][4]=2;
  game.getHistoryInputs=()=>{throw Error('Unavailable history must not be requested');};
  const before=JSON.stringify(game);
  const config={...exactConfig,variant:'cnn',historyArchitecture:'board',observationMode:'board_hand'};
  let calls=0;
  const player=new DirectPlayer({inputNames:exactInputNames,run:async feeds=>{
    calls++;
    assert.equal(feeds.board.data[100+44],1); // Actual edited opponent chip.
    assert.equal(feeds.context.data[0],.5);assert.equal(feeds.context.data[10],.5);
    assert.ok(Array.from(feeds.context.data).slice(52).every(x=>x===0));
    assert.ok(Array.from(feeds.history.data).every(x=>x===0n));
    const ids=Array.from(feeds.actions.data,Number),logits=new Float32Array(ids.length);
    logits[ids.indexOf(1001)]=7;
    return {logits:{data:logits},value:{data:[.25]}};
  }},runtime,config);
  const result=await player.chooseMove(game);
  assert.equal(result.move.card,'JC');assert.equal(result.move.col,1);
  assert.equal(JSON.stringify(game),before);
  // Exercise the actual hint function through the same direct player.
  const source=fs.readFileSync('web/static/game.js','utf8');
  const start=source.indexOf('async function getHint()'),end=source.indexOf('\n}\n',start)+2;
  let highlighted=false;
  Object.assign(context,{game,aiPlayer:player,selectedCard:null,aiProgressContainer:null,
    document:{getElementById:()=>null,querySelector:()=>({classList:{add:()=>{highlighted=true;}}})},
    chooseAiMove:()=>player.chooseMove(game),renderP1Hand:()=>{},renderBoard:()=>{},showAiError:e=>{throw e;}});
  vm.runInContext(source.slice(start,end),context);
  await context.getHint();
  assert.equal(context.selectedCard,'JC');assert.ok(highlighted);assert.equal(calls,2);
  assert.equal(JSON.stringify(game),before);
});

test('edited positions fail closed without explicit board-only export capability', async () => {
  const game=new SequenceGame();game.publicHistoryComplete=false;
  const session={inputNames:exactInputNames,run:async()=>{throw Error('Must reject before inference');}};
  for(const config of [exactConfig,{...exactConfig,variant:'cnn'},
    {...exactConfig,variant:'transformer',historyArchitecture:'board',observationMode:'public_history'},
    {...exactConfig,variant:'cnn',historyArchitecture:'gru',observationMode:'public_history'}]) {
    await assert.rejects(()=>new DirectPlayer(session,runtime,config).chooseMove(game),/complete move history/);
  }
  for(const [variant,historyArchitecture] of [['transformer','board'],['cnn','gru'],['cnn','summary'],['cnn',undefined]]) {
    assert.throws(()=>new DirectPlayer(session,runtime,{...exactConfig,variant,historyArchitecture,observationMode:'board_hand'}),/board-only CNN/);
  }
  assert.throws(()=>new DirectPlayer(session,runtime,{...exactConfig,observationMode:'unknown'}),/Unsupported observation/);
});
