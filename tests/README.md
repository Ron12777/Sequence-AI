# Tests

Install the CPU version of PyTorch (or your appropriate CUDA build), then:

```sh
python -m pip install -r requirements-test.txt .
OMP_NUM_THREADS=1 python -m pytest -q -rs
npm ci
npm test
npx playwright install --with-deps chromium webkit
npm run test:ui
```

`tests/ui/` opens the shipped page in Chromium and WebKit at desktop and phone sizes. It
loads the actual ONNX model and ORT Web WASM, clicks the page's real handlers,
and checks play, evaluation before a move, Watch speed/pause/resume, no-move draws, editing, the
52-card picker, keyboard focus, corner markers, layout bounds and load failure.
The pinned npm runtime substitutes for CDN transport only; model outputs and
application scripts are not mocked. Test screenshots and failure traces appear
in the GitHub Actions `browser-test-report` artifact.

`browser.test.cjs` contains fast engine/inference unit tests and isolated UI
state/race tests. Its small fake DOM does not test rendering or event wiring;
the browser suite covers those. Native export parity requires ONNX Runtime:
missing export dependencies fail test collection instead of silently skipping.
CUDA-only tests can still skip when no CUDA device is available.

The main-branch deployment job depends on both the Python/unit checks and the
browser suite. It deploys the tested commit only if that commit is still main's
tip. The previous independent deployment workflow has been removed.

These are functional/layout checks, not proof of visual quality, model strength,
or calibrated win probabilities. Real-game training evaluations are separate.

Hidden-information regression tests trap access to opposing card identities and the draw pile for both seats. `experiments/2026-10-18/audit-browser-fairness.cjs` additionally exercises the actual WASM model and intervening other-player queries; its JSON report identifies the exact audited engine/model hashes.
