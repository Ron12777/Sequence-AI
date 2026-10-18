const {chromium}=require('/home/rohan-mistry/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),crypto=require('crypto'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'../..');
const help='Usage: node validate-exact-rendered-browser.cjs --url http://127.0.0.1:PORT/ --model-path artifacts/SELECTED.onnx --output-prefix experiments/2026-10-17/SELECTED-rendered [--manifest-path web/static/model.json]';
const args={};
for(let i=2;i<process.argv.length;i++){
 const key=process.argv[i];
 if(key==='--help'){console.log(help);process.exit(0);}
 if(!['--url','--model-path','--output-prefix','--manifest-path'].includes(key)||!process.argv[i+1]||process.argv[i+1].startsWith('--'))throw Error(help);
 if(args[key])throw Error('Duplicate argument '+key);
 args[key]=process.argv[++i];
}
if(!args['--url']||!args['--model-path'])throw Error(help);
const targetUrl=new URL(args['--url']);
assert.ok(['http:','https:'].includes(targetUrl.protocol),'URL must use HTTP(S)');
const modelPath=path.resolve(root,args['--model-path']);
const manifestPath=path.resolve(root,args['--manifest-path']||'web/static/model.json');
const prefix=path.resolve(root,args['--output-prefix']||'experiments/2026-10-17/exact-cnn-rendered');
fs.mkdirSync(path.dirname(prefix),{recursive:true});
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const hash=p=>sha(fs.readFileSync(path.resolve(root,p)));
const expectedManifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
const expectedModelSha=hash(modelPath);
const report={started_utc:new Date().toISOString(),scope:'Actual headless Chrome rendered UI against served deployment; loaded manifest and ONNX bytes checked against local selected files',url:targetUrl.href,expected_model_path:modelPath,expected_model_sha256:expectedModelSha,expected_manifest_path:manifestPath,checks:{},console_errors:[],page_errors:[],request_failures:[],responses:[],loaded_assets:[]};
const assetReads=[];
let browser;
(async()=>{
 browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 report.browser_version=browser.version();
 const context=await browser.newContext({viewport:{width:1440,height:1100}});
 const page=await context.newPage();page.setDefaultTimeout(12000);
 page.on('console',msg=>{if(msg.type()==='error')report.console_errors.push({text:msg.text(),location:msg.location()});});
 page.on('pageerror',err=>report.page_errors.push(String(err)));
 page.on('requestfailed',request=>report.request_failures.push({url:request.url(),failure:request.failure()}));
 page.on('response',response=>{
  if(response.status()>=400)report.responses.push({url:response.url(),status:response.status()});
  const pathname=new URL(response.url()).pathname;
  if(pathname.endsWith('/model.json')||pathname.endsWith('.onnx'))assetReads.push(response.body().then(bytes=>report.loaded_assets.push({url:response.url(),status:response.status(),sha256:sha(bytes)})).catch(error=>{throw Error('Could not read loaded model asset: '+error);}));
 });
 await page.goto(report.url,{waitUntil:'networkidle',timeout:45000});
 await page.waitForFunction(()=>typeof aiPlayer!=='undefined'&&aiPlayer!==null&&typeof game!=='undefined'&&game!==null,{timeout:45000});
 report.manifest=await page.evaluate(()=>fetch('model.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Manifest HTTP '+r.status);return r.json();}));
 await Promise.all(assetReads);
 assert.deepEqual(report.manifest,expectedManifest,'Served manifest differs from local deployed manifest');
 const manifestUrl=new URL('model.json',report.url).href;
 const modelUrl=new URL(report.manifest.model,report.url).href;
 const loadedModels=report.loaded_assets.filter(asset=>asset.url===modelUrl);
 assert.ok(loadedModels.length>0,'No actual ONNX model load captured');
 for(const asset of loadedModels){assert.equal(asset.status,200);assert.equal(asset.sha256,expectedModelSha,'Loaded ONNX differs from selected model');}
 const loadedManifests=report.loaded_assets.filter(asset=>asset.url===manifestUrl);
 assert.ok(loadedManifests.length>0,'No actual manifest load captured');
 for(const asset of loadedManifests){assert.equal(asset.status,200);assert.equal(asset.sha256,hash(manifestPath),'Loaded manifest bytes differ from deployed file');}
 report.checks.served_manifest_matches_local=true;
 report.checks.loaded_onnx_sha256_matches_selected=true;
 report.checks.loaded=await page.evaluate(()=>({direct:aiPlayer instanceof DirectPlayer,exact:aiPlayer.exactActions,boardHandOnly:aiPlayer.boardHandOnly,inputs:onnxSession.inputNames,outputs:onnxSession.outputNames,depthControlsHidden:[...document.querySelectorAll('.depth-control')].every(e=>getComputedStyle(e).display==='none')}));
 assert.ok(Object.values(report.checks.loaded).filter(x=>typeof x==='boolean').every(Boolean));
 // Fail loudly if any UI route unexpectedly falls back to search.
 await page.evaluate(()=>{window.__searchCalls=0;MCTS.prototype.search=async()=>{window.__searchCalls++;throw Error('Unexpected MCTS search');};window.__directCalls=0;const original=DirectPlayer.prototype.chooseMove;DirectPlayer.prototype.chooseMove=async function(...args){window.__directCalls++;return original.apply(this,args);};});
 await page.locator('#hintBtn').click();
 await page.waitForFunction(()=>selectedCard!==null&&document.querySelector('.cell.best-move'));
 const hint=await page.evaluate(()=>({card:selectedCard,row:Number(document.querySelector('.cell.best-move').dataset.row),col:Number(document.querySelector('.cell.best-move').dataset.col)}));
 await page.locator(`.cell[data-row="${hint.row}"][data-col="${hint.col}"]`).click();
 await page.waitForFunction(()=>game.publicEvents.filter(e=>e[3]!==2).length>=2&&game.currentPlayer===1&&!isThinking);
 report.checks.human_hint_move_ai_reply=await page.evaluate(()=>({events:game.publicEvents.length,currentPlayer:game.currentPlayer,error:document.querySelector('#aiError').textContent,historyItems:document.querySelector('#moveHistory').children.length}));
 assert.equal(report.checks.human_hint_move_ai_reply.error,'');
 await page.locator('label.toggle-switch').click();
 assert.ok(await page.locator('#showEvalBar').isChecked());
 await page.locator('#hintBtn').click();
 await page.screenshot({path:prefix+'-human-game.png',fullPage:true});
 await page.locator('#watchAiBtn').click();
 await page.waitForFunction(()=>watchMode&&game.publicEvents.filter(e=>e[3]!==2).length>=4);
 await page.locator('#newGameBtn').click();
 await page.waitForFunction(()=>isPaused&&!isThinking);
 const pausedEvents=await page.evaluate(()=>game.publicEvents.length);
 await page.waitForTimeout(800);
 assert.equal(await page.evaluate(()=>game.publicEvents.length),pausedEvents);
 await page.locator('#pauseControls button').filter({hasText:'Resume Game'}).click();
 await page.waitForFunction(n=>game.publicEvents.length>n,pausedEvents);
 await page.locator('#newGameBtn').click();await page.waitForFunction(()=>isPaused&&!isThinking);
 await page.locator('#pauseControls button').filter({hasText:'Reset Board'}).click();
 await page.waitForFunction(()=>!watchMode&&!isPaused&&game.publicEvents.length===0);
 report.checks.watch_pause_resume_reset=true;
 await page.locator('#analysisBtn').click();
 await page.locator('.cell[data-row="4"][data-col="4"]').click();
 assert.equal(await page.evaluate(()=>game.publicHistoryComplete),false);
 // Exercise edited own hand via the actual picker, then edited-position best move.
 await page.locator('#handP1 > *').last().click();
 await page.locator('#cardPickerGrid > *').first().click();
 await page.locator('#bestMoveBtn').click();
 await page.waitForFunction(()=>!isThinking&&document.querySelectorAll('.highlight-orb').length>0);
 const analysis=await page.evaluate(()=>({publicHistoryComplete:game.publicHistoryComplete,error:document.querySelector('#aiError').textContent,editedChip:game.board[4][4],hand:game.hands[1],directCalls:window.__directCalls}));
 assert.equal(analysis.error,'');assert.equal(analysis.editedChip,1);
 await page.locator('#toggleTurnBtn').click();await page.locator('#bestMoveBtn').click();await page.waitForFunction(()=>!isThinking);
 assert.equal(await page.evaluate(()=>document.querySelector('#aiError').textContent),'');
 await page.locator('#toggleTurnBtn').click();await page.locator('#analysisBtn').click();
 await page.locator('#hintBtn').click();await page.waitForFunction(()=>selectedCard!==null&&document.querySelector('.cell.best-move'));
 report.checks.edited_board_and_hand_analysis_and_hint=analysis;
 report.checks.search_calls=await page.evaluate(()=>window.__searchCalls);assert.equal(report.checks.search_calls,0);
 await page.screenshot({path:prefix+'-edited-analysis.png',fullPage:true});
 // Reload within the same browser context so cached static/model assets are exercised.
 await page.reload({waitUntil:'networkidle',timeout:45000});
 await page.waitForFunction(()=>typeof aiPlayer!=='undefined'&&aiPlayer!==null&&typeof game!=='undefined'&&game!==null,{timeout:45000});
 await Promise.all(assetReads);
 assert.ok(report.loaded_assets.filter(asset=>asset.url===modelUrl).length>=2,'Reload did not fetch/load the selected model');
 for(const asset of report.loaded_assets.filter(asset=>asset.url===modelUrl))assert.equal(asset.sha256,expectedModelSha);
 assert.deepEqual(await page.evaluate(()=>fetch('model.json',{cache:'no-store'}).then(r=>r.json())),expectedManifest);
 report.checks.same_session_reload=await page.evaluate(()=>aiPlayer instanceof DirectPlayer&&aiPlayer.exactActions&&aiPlayer.boardHandOnly);
 assert.ok(report.checks.same_session_reload);
 report.checks.ai_error=await page.locator('#aiError').textContent();assert.equal(report.checks.ai_error,'');
 assert.equal(report.page_errors.length,0);
 // Preserve missing-favicon diagnostics, but do not confuse them with AI/runtime failures.
 const faviconError=entry=>{try{return new URL(entry.url||entry.location?.url).pathname==='/favicon.ico';}catch{return false;}};
 report.actionable_console_errors=report.console_errors.filter(entry=>!faviconError(entry));
 assert.equal(report.actionable_console_errors.length,0,'Browser console errors');
 assert.equal(report.request_failures.filter(entry=>!faviconError(entry)).length,0,'Failed application requests');
 assert.equal(report.responses.filter(entry=>!faviconError(entry)).length,0,'Application HTTP errors');
 report.passed=true;
})().catch(e=>{report.passed=false;report.failure=String(e.stack||e);process.exitCode=1;}).finally(async()=>{
 report.finished_utc=new Date().toISOString();
 report.source_sha256=Object.fromEntries(['web/static/game.js','web/static/game_engine.js','web/static/direct_player.js','web/static/index.html','web/static/style.css',modelPath,manifestPath].map(p=>[p,fs.existsSync(path.resolve(root,p))?hash(p):null]));
 report.script_sha256=crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
 fs.writeFileSync(prefix+'-browser-validation.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));if(browser)await browser.close();
});
