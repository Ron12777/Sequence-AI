/** Run a legacy board ONNX through real ORT Web WASM and the actual DirectPlayer.
 * node scripts/check_direct_web_runtime.cjs RUNTIME_DIR MODEL FIXTURES OUTPUT
 * A Node VM supplies browser globals; no browser UI or deployment is performed.
 */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),crypto=require('node:crypto'),os=require('node:os');

(async()=>{
    const [runtimeDir,modelPath,fixturePath,outputPath]=process.argv.slice(2);
    assert.ok(outputPath,'Runtime directory, model, fixtures, and output are required');
    const context=vm.createContext({console,setTimeout,clearTimeout,TextDecoder,TextEncoder,URL,Response,performance,
        fetch:async url=>new Response(fs.readFileSync(path.join(runtimeDir,path.basename(new URL(url).pathname))),
            {headers:{'Content-Type':'application/wasm'}})});
    vm.runInContext(`globalThis.self=globalThis;globalThis.window=globalThis;
        globalThis.location={href:'http://localhost/'};
        globalThis.document={currentScript:{src:'http://localhost/ort.min.js'}};`,context);
    vm.runInContext(fs.readFileSync(path.join(runtimeDir,'ort.min.js'),'utf8'),context);
    for(const script of ['web/static/game_engine.js','web/static/direct_player.js'])
        vm.runInContext(fs.readFileSync(script,'utf8'),context);
    context.modelBytes=Array.from(fs.readFileSync(modelPath));
    const fixture=JSON.parse(fs.readFileSync(fixturePath,'utf8'));
    context.rows=fixture.rows;
    const result=await vm.runInContext(`(async()=>{
        ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='http://localhost/';
        const session=await ort.InferenceSession.create(Uint8Array.from(modelBytes),{executionProviders:['wasm']});
        const player=new DirectPlayer(session,ort);
        const checked=[];
        for(const row of rows){
            const game=new SequenceGame();Object.assign(game,row.state);game.deck=[];
            const input=Array.from(game.getStateTensor(game.currentPlayer));
            const raw=await session.run({input:new ort.Tensor('float32',Float32Array.from(input),[1,11,10,10])});
            const before=JSON.stringify(game);
            const start=performance.now(),chosen=await player.chooseMove(game),milliseconds=performance.now()-start;
            checked.push({input,logits:Array.from(raw.policy.data),value:chosen.value,
                policy:Array.from(chosen.policy),move:chosen.move,milliseconds,
                unchanged:before===JSON.stringify(game)});
        }
        // Exclude two warm-up calls; benchmark the same public position repeatedly.
        const game=new SequenceGame();Object.assign(game,rows[0].state);game.deck=[];
        const timing=[];
        for(let index=0;index<22;index++){
            const start=performance.now();await player.chooseMove(game);
            if(index>=2)timing.push(performance.now()-start);
        }
        await session.release();return {checked,timing};
    })()`,context);
    let logitError=0,valueError=0,policyError=0;
    for(let index=0;index<fixture.rows.length;index++){
        const expected=fixture.rows[index],actual=result.checked[index];
        assert.deepEqual(Array.from(actual.input),expected.input,'Browser input must retain original hand order');
        assert.equal(JSON.stringify(actual.move),JSON.stringify(expected.move),'Browser must choose the identical exact legacy move');
        assert.ok(actual.unchanged,'Inference must not mutate the game');
        for(let j=0;j<100;j++){
            const error=Math.abs(actual.logits[j]-expected.logits[j]);
            assert.ok(error<=1e-5+1e-5*Math.abs(expected.logits[j]));logitError=Math.max(logitError,error);
            const pError=Math.abs(actual.policy[j]-expected.policy[j]);
            assert.ok(pError<=2e-5);policyError=Math.max(policyError,pError);
        }
        valueError=Math.max(valueError,Math.abs(actual.value-expected.value));
        assert.ok(Math.abs(actual.value-expected.value)<=2e-5);
    }
    const times=Array.from(result.timing).sort((a,b)=>a-b);
    const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    const report={passed:true,timestamp_utc:new Date().toISOString(),model:modelPath,onnx_sha256:hash(modelPath),
        checkpoint_sha256:fixture.checkpoint_sha256,fixtures_sha256:hash(fixturePath),
        runtime:'onnxruntime-web@1.18.0 WASM, single thread, isolated browser-like Node VM',
        runtime_sha256:hash(path.join(runtimeDir,'ort.min.js')),wasm_sha256:hash(path.join(runtimeDir,'ort-wasm-simd.wasm')),
        cpu:os.cpus()[0]?.model,node:process.version,states:fixture.rows.length,
        max_public_events:Math.max(...fixture.rows.map(row=>row.public_events)),original_hand_order:true,
        identical_selected_moves:true,inference_did_not_mutate_game:true,
        max_absolute_error:{logits:logitError,value:valueError,display_policy:policyError},
        single_move_milliseconds:{samples:times.length,median:times[Math.floor(times.length/2)],p95:times[Math.ceil(times.length*.95)-1],
            scope:'Warm DirectPlayer observation, WASM inference, legal-move selection and heatmap; one repeated state; excludes rendered UI/network/model loading'},
        deployment_changed:false};
    fs.writeFileSync(outputPath,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
