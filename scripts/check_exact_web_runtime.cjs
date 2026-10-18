/** Check the actual pinned ONNX Web WASM kernels in a browser-like JS context.
 * Usage: node scripts/check_exact_web_runtime.cjs /tmp/fixture-directory cnn transformer
 * Directory contains ort.min.js, ort-wasm-simd.wasm, NAME.onnx, NAME.inputs.json.
 * Runtime files are intentionally external; this script does not deploy assets.
 */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

async function check(directory, name) {
    const context = vm.createContext({
        console, setTimeout, clearTimeout, TextDecoder, TextEncoder, URL, Response, performance,
        fetch: async url => new Response(fs.readFileSync(path.join(directory, path.basename(new URL(url).pathname))),
            {headers: {'Content-Type': 'application/wasm'}}),
    });
    vm.runInContext(`globalThis.self=globalThis;globalThis.window=globalThis;
        globalThis.location={href:'http://localhost/'};
        globalThis.document={currentScript:{src:'http://localhost/ort.min.js'}};`, context);
    vm.runInContext(fs.readFileSync(path.join(directory, 'ort.min.js'), 'utf8'), context);
    context.modelBytes = Array.from(fs.readFileSync(path.join(directory, `${name}.onnx`)));
    context.fixture = JSON.parse(fs.readFileSync(path.join(directory, `${name}.inputs.json`), 'utf8'));
    const result = await vm.runInContext(`(async()=>{
        ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='http://localhost/';
        const session=await ort.InferenceSession.create(Uint8Array.from(modelBytes),{executionProviders:['wasm']});
        const feeds={};
        for(const [name,input] of Object.entries(fixture.inputs)){
            const data=input.dtype==='int64'?BigInt64Array.from(input.data,BigInt):Float32Array.from(input.data);
            feeds[name]=new ort.Tensor(input.dtype,data,input.shape);
        }
        const outputs=await session.run(feeds);
        const result={};
        for(const name of ['logits','value'])result[name]=Array.from(outputs[name].data);
        await session.release();
        return result;
    })()`, context);
    const expected = context.fixture.expected;
    const errors = {};
    for (const key of ['logits', 'value']) {
        const flat = expected[key].data.flat(Infinity);
        assert.equal(result[key].length, flat.length);
        errors[key] = Math.max(...flat.map((value, index) => {
            const error = Math.abs(value - result[key][index]);
            assert.ok(Number.isFinite(result[key][index]) && error <= 2e-4 + 2e-4 * Math.abs(value),
                `${name} ${key}[${index}]: ${result[key][index]} != ${value}`);
            return error;
        }));
    }
    return {model: name, provider: 'onnxruntime-web WASM', maxAbsoluteError: errors};
}

(async () => {
    const directory = process.argv[2];
    assert.ok(directory, 'A runtime and fixture directory is required');
    const results = [];
    for (const name of process.argv.slice(3)) results.push(await check(directory, name));
    assert.ok(results.length, 'At least one fixture name is required');
    console.log(JSON.stringify({passed: true, results}, null, 2));
})().catch(error => {console.error(error); process.exitCode = 1;});
