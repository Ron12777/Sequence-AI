/** Actual ORT Web 1.18 WASM + production DirectPlayer, full-game Python fixtures.
 * node SCRIPT RUNTIME_DIR MODEL FIXTURES CHECKPOINT_SHA OUTPUT
 * Local VM supplies browser globals. This does not test a rendered browser UI.
 */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),crypto=require('node:crypto'),os=require('node:os');
(async()=>{
 const [runtimeDir,modelPath,fixturePath,checkpointHash,outputPath]=process.argv.slice(2);
 assert.ok(outputPath,'runtime, model, fixtures, checkpoint SHA and output required');
 const context=vm.createContext({console,setTimeout,clearTimeout,TextDecoder,TextEncoder,URL,Response,performance,
  fetch:async url=>new Response(fs.readFileSync(path.join(runtimeDir,path.basename(new URL(url).pathname))),{headers:{'Content-Type':'application/wasm'}})});
 vm.runInContext(`globalThis.self=globalThis;globalThis.window=globalThis;
  globalThis.location={href:'http://localhost/'};globalThis.document={currentScript:{src:'http://localhost/ort.min.js'}};`,context);
 vm.runInContext(fs.readFileSync(path.join(runtimeDir,'ort.min.js'),'utf8'),context);
 for(const file of ['web/static/game_engine.js','web/static/direct_player.js'])vm.runInContext(fs.readFileSync(file,'utf8'),context);
 context.modelBytes=Array.from(fs.readFileSync(modelPath));
 context.fixtures=JSON.parse(fs.readFileSync(fixturePath,'utf8'));
 const nativeReport=JSON.parse(fs.readFileSync(modelPath.replace(/\.onnx$/,'.validation.json'),'utf8'));
 assert.equal(nativeReport.checkpoint_sha256,checkpointHash);
 context.capabilities=nativeReport.browserManifestFields;
 assert.equal(context.capabilities.policyKind,'exact_sequence_policy');
 context.assert=assert;
 const result=await vm.runInContext(`(async()=>{
  ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='http://localhost/';
  const session=await ort.InferenceSession.create(Uint8Array.from(modelBytes),{executionProviders:['wasm']});
  let expected,raw,states=0,logitError=0,valueError=0,policyError=0,ambiguities=0,normalSpend=0,jackSpend=0,totalJackMoves=0,maxEvents=0;
  const timing=[],kinds=new Set();
  const checkedSession={inputNames:session.inputNames,run:async feeds=>{
   for(const name of ['board','context','history','actions'])assert.equal(JSON.stringify(Array.from(feeds[name].data,Number)),JSON.stringify(expected[name]),name+' Python/browser input mismatch');
   raw=await session.run(feeds);return raw;
  }};
  const player=new DirectPlayer(checkedSession,ort,capabilities);
  const otherQuery=new DirectPlayer(session,ort,capabilities);
  for(const fixture of fixtures){
   const game=new SequenceGame();game.hands=JSON.parse(JSON.stringify(fixture.hands));game.deck=[...fixture.deck];
   for(expected of fixture.steps){
    const before=JSON.stringify(game),tick=performance.now();
    const selected=await player.chooseMove(game);
    timing.push(performance.now()-tick);
    assert.equal(JSON.stringify(game),before,'Inference mutated state');
    const exact=cardToIndex(selected.move.card)*100+selected.move.row*10+selected.move.col;
    assert.equal(exact,expected.chosen,'Exact selected card/square mismatch at state '+states);
    for(let i=0;i<expected.logits.length;i++){
     const error=Math.abs(raw.logits.data[i]-expected.logits[i]);logitError=Math.max(logitError,error);
     assert.ok(Number.isFinite(raw.logits.data[i]) && error<=2e-4+2e-4*Math.abs(expected.logits[i]),'logit mismatch');
    }
    const ve=Math.abs(selected.value-expected.value);valueError=Math.max(valueError,ve);assert.ok(ve<=2e-5,'value mismatch');
    for(let i=0;i<100;i++){const error=Math.abs(selected.policy[i]-expected.policy[i]);policyError=Math.max(policyError,error);assert.ok(error<=2e-5,'heatmap mismatch');}
    const alternatives=expected.actions.filter(a=>a%100===exact%100);
    const chosenJack=(Math.floor(exact/100)%13)===10;
    if(chosenJack)totalJackMoves++;
    if(alternatives.some(a=>Math.floor(a/100)%13===10)&&alternatives.some(a=>Math.floor(a/100)%13!==10)){
     ambiguities++;if(chosenJack)jackSpend++;else normalSpend++;
    }
    // Execute the selected move on an independent copy, verifying exact spend
    // via the first newly public event and discard, even when a Jack can share its square.
    const copy=new SequenceGame();Object.assign(copy,JSON.parse(before));
    const actor=copy.currentPlayer,eventCount=copy.publicEvents.length,discardCount=copy.discardPiles[actor].length;
    assert.ok(copy.makeMove(selected.move));
    assert.equal(copy.publicEvents[eventCount][0],Math.floor(exact/100));
    assert.equal(copy.discardPiles[actor][discardCount],selected.move.card);
    // Query inputs cannot depend on opponent identities or deck order, and own
    // hand channel ordering remains canonical under arbitrary permutation.
    const opponent=3-game.currentPlayer,oldOpponent=game.hands[opponent],deck=game.deck,own=game.hands[game.currentPlayer];
    game.hands[opponent]=oldOpponent.map(()=>'JC');game.deck=[...deck].reverse();game.hands[game.currentPlayer]=[...own].reverse();
    const changed=game.getExactInputs();
    for(const name of ['board','context','history','actions'])assert.equal(JSON.stringify(Array.from(changed[name],Number)),JSON.stringify(expected[name]));
    const changedDecision=await player.chooseMove(game);
    assert.equal(JSON.stringify(changedDecision),JSON.stringify(selected),'Hidden replacement changed decision');
    game.hands[opponent]=new Proxy(oldOpponent,{get(target,key){
     if(key==='length')return target.length;
     throw Error('Private opponent identity read: '+String(key));
    }});
    Object.defineProperty(game,'deck',{configurable:true,get(){throw Error('Draw pile read during inference');}});
    const guarded=await player.chooseMove(game);
    assert.equal(JSON.stringify(guarded),JSON.stringify(selected),'Guarded decision changed');
    Object.defineProperty(game,'deck',{configurable:true,enumerable:true,writable:true,value:deck});
    game.hands[opponent]=oldOpponent;game.hands[game.currentPlayer]=own;game.deck=deck;
    // An intervening hint/evaluation for the other side must not leak across calls.
    const hintPosition=new SequenceGame();Object.assign(hintPosition,JSON.parse(before));
    hintPosition.currentPlayer=opponent;
    await otherQuery.chooseMove(hintPosition);
    const repeated=await player.chooseMove(game);
    assert.equal(JSON.stringify(repeated),JSON.stringify(selected),'Other-player query changed decision');
    assert.equal(JSON.stringify(game),before,'Audit queries mutated game');
    assert.ok(game.makeMove(expected.move));
    maxEvents=Math.max(maxEvents,game.publicEvents.length);states++;
   }
   assert.equal(JSON.stringify(game.publicEvents),JSON.stringify(fixture.events));
   assert.ok(game.gameOver||game.getLegalMoves().length===0,'Fixture was not a completed game');
   for(const event of game.publicEvents)kinds.add(event[3]);
  }
  assert.ok(ambiguities>0 && normalSpend>0,'No normal-versus-Jack spending decision tested');
  assert.ok(totalJackMoves>0,'No learned Jack move tested');
  assert.ok(kinds.has(0)&&kinds.has(1)&&kinds.has(2),'Missing placement/removal/dead-skip coverage');
  await session.release();
  return {states,games:fixtures.length,max_public_events:maxEvents,exact_move_matches:states,
   normal_versus_jack_same_square:{states:ambiguities,selected_normal:normalSpend,selected_jack:jackSpend},
   total_selected_jack_moves:totalJackMoves,event_kinds:Array.from(kinds).sort(),
   max_absolute_error:{logits:logitError,value:valueError,display_policy:policyError},
   canonical_inputs_match_python:true,hidden_identity_invariance:true,hidden_access_traps_passed:true,interleaved_other_player_queries_invariant:true,inference_did_not_mutate_game:true,
   selected_card_spend_verified:true,timing};
 })()`,context);
 const timing=Array.from(result.timing).slice(2).sort((a,b)=>a-b);delete result.timing;
 const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
 const report={passed:true,timestamp_utc:new Date().toISOString(),model:modelPath,onnx_sha256:hash(modelPath),checkpoint_sha256:checkpointHash,
  fixtures_sha256:hash(fixturePath),runtime:'onnxruntime-web@1.18.0 WASM single thread in isolated Node VM',
  execution:'Actual browser WASM runtime and production DirectPlayer; not a rendered browser UI test',
  runtime_sha256:hash(path.join(runtimeDir,'ort.min.js')),wasm_sha256:hash(path.join(runtimeDir,'ort-wasm-simd.wasm')),
  engine_sha256:hash('web/static/game_engine.js'),player_sha256:hash('web/static/direct_player.js'),
  validator_sha256:hash(__filename),capabilities:context.capabilities,cpu:os.cpus()[0]?.model,node:process.version,...result,
  milliseconds:{samples:timing.length,median:timing[Math.floor(timing.length/2)],p95:timing[Math.ceil(timing.length*.95)-1],scope:'Observed full-game decisions with input parity assertions, excludes first two calls, UI/network/loading'},deployment_changed:false};
 fs.writeFileSync(outputPath,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
