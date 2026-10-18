const {test: base, expect} = require('@playwright/test');
const fs = require('node:fs/promises');
const path = require('node:path');
const {createHash} = require('node:crypto');

// Load the real HTML, scripts, CSS, model and WASM runtime. Only CDN transport
// is replaced with the identical pinned npm files, so CI needs no live CDN.
const test = base.extend({
  page: async ({page}, use) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const pkg = JSON.parse(await fs.readFile('node_modules/onnxruntime-web/package.json', 'utf8'));
    await page.route('https://cdn.jsdelivr.net/npm/onnxruntime-web@*/dist/*', async route => {
      const url = new URL(route.request().url());
      expect(url.pathname).toContain(`onnxruntime-web@${pkg.version}/dist/`);
      const file = path.basename(url.pathname);
      await route.fulfill({
        path: path.resolve('node_modules/onnxruntime-web/dist', file),
        contentType: file.endsWith('.wasm') ? 'application/wasm' : 'application/javascript',
      });
    });
    await use(page);
    expect(errors, 'uncaught errors from the full page').toEqual([]);
  },
});

async function openGame(page) {
  const manifest = JSON.parse(await fs.readFile('web/static/model.json', 'utf8'));
  const modelResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/' + manifest.model);
  await page.goto('/');
  const response = await modelResponse;
  expect(response.ok()).toBe(true);
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  expect(hash(await response.body())).toBe(hash(await fs.readFile(path.join('web/static', manifest.model))));
  await expect(page.locator('body')).toHaveClass(/direct-policy/);
  await expect(page.getByRole('button', {name: 'Hint', exact: true})).toBeEnabled();
  await expect(page.locator('#aiError')).toBeHidden();
  await expect(page.locator('#board .cell')).toHaveCount(100);
}

async function noHorizontalOverflow(page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}

test('evaluation before any move, Hint, human move and real AI reply', async ({page}) => {
  await openGame(page);
  await page.getByRole('checkbox', {name: 'Show evaluation'}).check();
  await expect(page.locator('#evalBarContainer')).toBeVisible();
  await expect(page.locator('#evalText')).toHaveText(/^(Red|Blue) \d+%$/);
  await expect(page.locator('.history-item')).toHaveCount(0);
  await page.getByRole('button', {name: 'Hint', exact: true}).click();
  const suggested = page.locator('.cell.best-move');
  await expect(suggested).toHaveCount(1);
  const beforeHand = await page.locator('#handP1 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  const playedSlot = Number(await page.locator('#handP1 .card.selected').getAttribute('data-slot'));
  await suggested.click();
  await expect(page.locator('.history-item')).toHaveCount(2);
  const afterHand = await page.locator('#handP1 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  expect(afterHand).toHaveLength(7);
  for (let slot = 0; slot < 7; slot++) if (slot !== playedSlot) expect(afterHand[slot]).toBe(beforeHand[slot]);
  await expect(page.locator('#turnIndicator')).toContainText('Your turn');
  await expect(page.getByRole('button', {name: 'Hint', exact: true})).toBeEnabled();
  await expect(page.locator('#evalText')).toHaveText(/^(Red|Blue) \d+%$/);
  await expect(page.locator('#aiError')).toBeHidden();
  await expect(page.locator('#aiProgressContainer')).toHaveCount(0);
  await noHorizontalOverflow(page);
});

test('Watch button starts play, pause freezes it, resume and editing preserve modes', async ({page}) => {
  await openGame(page);
  await page.getByRole('button', {name: 'Watch AI', exact: true}).click();
  await expect.poll(() => page.locator('.history-item').count()).toBeGreaterThan(0);
  await expect(page.locator('#hintBtn')).toBeHidden();
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Resume', exact: true})).toBeVisible();
  const pausedCount = await page.locator('.history-item').count();
  // Cross several real 500ms inter-move sleeps: no old loop may wake and move.
  await page.waitForTimeout(1200);
  await expect(page.locator('.history-item')).toHaveCount(pausedCount);
  await page.getByRole('button', {name: 'Resume', exact: true}).click();
  await expect.poll(() => page.locator('.history-item').count()).toBeGreaterThan(pausedCount);
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  await expect(page.locator('#hintBtn')).toBeHidden();
  await page.getByRole('button', {name: 'Done editing', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Resume', exact: true})).toBeVisible();
  await expect(page.locator('#hintBtn')).toBeHidden();
  await page.getByRole('button', {name: 'New game', exact: true}).click();
  await expect(page.locator('.history-item')).toHaveCount(0);
  await expect(page.locator('#hintBtn')).toBeVisible();
  await expect(page.locator('#watchAiBtn')).toBeVisible();
});

test('four unlabelled rows show all 52 cards, selection and keyboard focus work', async ({page}, testInfo) => {
  await openGame(page);
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  const otherHand = await page.locator('#handP2 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  const dialog = page.getByRole('dialog', {name: 'Choose card', exact: true});
  for (const [suit, name] of [['C', 'clubs'], ['D', 'diamonds'], ['H', 'hearts'], ['S', 'spades']]) {
    await page.locator('#handP1 .card').first().click();
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('.picker-card')).toHaveCount(52);
    await expect(dialog.getByRole('group')).toHaveCount(4);
    await expect(dialog.locator('.current-card')).toHaveCount(1);
    const bounds = await dialog.locator('.picker-card').evaluateAll(cards => cards.map(card => {
      const r = card.getBoundingClientRect();
      return {top: r.top, visible: r.top >= 0 && r.bottom <= window.innerHeight && r.left >= 0 && r.right <= window.innerWidth, width: r.width, height: r.height};
    }));
    expect(bounds.every(r => r.visible && r.width >= 24 && r.height >= 24)).toBe(true);
    expect(new Set(bounds.map(r => Math.round(r.top))).size).toBe(4);
    expect(await dialog.innerText()).not.toMatch(/clubs|diamonds|hearts|spades/i);
    await expect(dialog.locator('.picker-rank')).toHaveCount(52);
    await expect(dialog.locator('.picker-symbol')).toHaveCount(52);
    if (suit === 'C') await testInfo.attach('all-cards', {body: await page.screenshot(), contentType: 'image/png'});
    await dialog.getByRole('button', {name: new RegExp(`^J of ${name}(, current card)?$`)}).click();
    await expect(dialog).toBeHidden();
    await expect(page.locator('#handP1 .card').first()).toHaveAttribute('data-card', 'J' + suit);
  }
  expect(await page.locator('#handP2 .card').evaluateAll(cards => cards.map(c => c.dataset.card))).toEqual(otherHand);
  await page.locator('#handP1 .card').first().press('Enter');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', {name: 'Close card picker'})).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(dialog.getByRole('button', {name: /^K of spades/})).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(page.locator('#handP1 .card').first()).toBeFocused();
  await noHorizontalOverflow(page);
});

test('editor makes and breaks a sequence without marking or changing free corners', async ({page}) => {
  await openGame(page);
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  for (let row = 1; row <= 4; row++) await page.locator(`.cell[data-row="${row}"][data-col="0"]`).click();
  await expect(page.locator('#playerSequences')).toHaveText('1');
  await expect(page.locator('.sequence-mark')).toHaveCount(4);
  const corner = page.locator('.cell[data-row="0"][data-col="0"]');
  await corner.click();
  await expect(corner).toHaveClass(/free-space/);
  await expect(corner).toHaveClass(/free-chip/);
  await expect(corner.locator('.sequence-mark')).toHaveCount(0);
  await page.locator('.cell[data-row="2"][data-col="0"]').click();
  await expect(page.locator('#playerSequences')).toHaveText('0');
  await expect(page.locator('.sequence-mark')).toHaveCount(0);
  const cell = await corner.boundingBox();
  expect(cell.height / cell.width).toBeCloseTo(1.25, 1);
});

test('failed model load is visible and disables AI controls', async ({page}) => {
  await page.route('**/model.json', route => route.fulfill({status: 503, body: 'unavailable'}));
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('AI unavailable');
  await expect(page.locator('#hintBtn')).toBeDisabled();
  await expect(page.locator('#watchAiBtn')).toBeDisabled();
  await expect(page.getByRole('button', {name: 'Edit board', exact: true})).toBeEnabled();
});

test('playing the second duplicate replaces only that chosen hand slot', async ({page}) => {
  await openGame(page);
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  for (const slot of [0, 6]) {
    await page.locator(`#handP1 .card[data-slot="${slot}"]`).click();
    await page.getByRole('dialog', {name: 'Choose card'}).getByRole('button', {name: /^J of clubs/}).click();
  }
  await page.getByRole('button', {name: 'Done editing', exact: true}).click();
  const before = await page.locator('#handP1 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  await page.locator('#handP1 .card[data-slot="6"]').click();
  await expect(page.locator('#handP1 .card.selected')).toHaveCount(1);
  await expect(page.locator('#handP1 .card.selected')).toHaveAttribute('data-slot', '6');
  await page.locator('.cell.valid-move').first().click();
  await expect(page.locator('.history-item')).toHaveCount(2);
  const after = await page.locator('#handP1 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  expect(after).toHaveLength(7);
  expect(after.slice(0, 6)).toEqual(before.slice(0, 6));
  await expect(page.locator('#hintBtn')).toBeEnabled();
});
