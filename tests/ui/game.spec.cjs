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

test('leaving an unplayable edited hand ends as a draw and Play again resets it', async ({page}) => {
  await openGame(page);
  await page.getByRole('checkbox', {name: 'Show evaluation'}).check();
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  // Removal Jacks cannot move on an empty board. Use the editor to reach
  // this edge case without waiting for a particular shuffled game.
  for (let index = 0; index < 7; index++) {
    await page.locator('#handP1 .card').nth(index).click();
    await page.getByRole('dialog', {name: 'Choose card', exact: true})
      .getByRole('button', {name: /^J of spades/}).click();
  }
  await page.getByRole('button', {name: 'Done editing', exact: true}).click();
  await expect(page.getByRole('dialog', {name: 'Draw', exact: true})).toBeVisible();
  await expect(page.locator('#evalText')).toHaveText('Draw');
  await expect(page.locator('#aiError')).toBeHidden();
  await expect(page.locator('#hintBtn')).toBeHidden();
  await page.getByRole('button', {name: 'Play again', exact: true}).click();
  await expect(page.locator('#gameOverOverlay')).not.toBeVisible();
  await expect(page.locator('#hintBtn')).toBeEnabled();
  await expect(page.locator('#handP1 .card')).toHaveCount(7);
});

test('evaluation before any move, Hint, human move and real AI reply', async ({page}) => {
  await openGame(page);
  await page.getByRole('checkbox', {name: 'Show evaluation'}).check();
  await expect(page.locator('#evalBarContainer')).toBeVisible();
  await expect(page.locator('#evalText')).toHaveText(/^(Red|Blue) \d+%$/);
  await expect(page.locator('.history-item')).toHaveCount(0);
  await page.getByRole('button', {name: 'Hint', exact: true}).click();
  const suggested = page.locator('.cell.best-move');
  await expect(suggested).toHaveCount(1);
  await expect(page.locator('.cell.valid-move, .cell.removal-target')).toHaveCount(0);
  await expect(page.locator('#handP1 .hint-card')).toHaveCount(1);
  // Hover changes highlights; the single gold recommendation must survive.
  await page.locator('#handP1 .hint-card').hover();
  await expect(suggested).toHaveCount(1);
  await expect(page.locator('.cell.valid-move, .cell.removal-target')).toHaveCount(0);
  await page.locator('#hintBtn').hover();
  await expect(suggested).toHaveCount(1);
  await expect(page.locator('.cell.valid-move, .cell.removal-target')).toHaveCount(0);
  await page.locator('#handP1 .card:not(.hint-card)').first().hover();
  await page.locator('#hintBtn').hover();
  await expect(suggested).toHaveCount(1);
  await expect(page.locator('.cell.valid-move, .cell.removal-target')).toHaveCount(0);
  const beforeHand = await page.locator('#handP1 .card').evaluateAll(cards => cards.map(c => c.dataset.card));
  const playedSlot = Number(await page.locator('#handP1 .card.selected').getAttribute('data-slot'));
  await suggested.click();
  await expect(page.locator('.history-item')).toHaveCount(2);
  await expect(page.locator('.cell.best-move, #handP1 .hint-card')).toHaveCount(0);
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

test('moving from a hand card to the board keeps the same click targets', async ({page}) => {
  await openGame(page);
  const target = await page.locator('#board .cell').nth(1).elementHandle();
  await page.locator('#handP1 .card').first().hover();
  expect(await target.evaluate(cell => cell.isConnected)).toBe(true);
  await page.locator('#hintBtn').hover();
  expect(await target.evaluate(cell => cell.isConnected)).toBe(true);
});

test('selected 2S keeps its own targets even after hovering AS', async ({page}, testInfo) => {
  await openGame(page);
  await page.getByRole('button', {name: 'Edit board', exact: true}).click();
  for (const [index, rank] of [[0, 'A'], [1, '2']]) {
    await page.locator('#handP1 .card').nth(index).click();
    await page.getByRole('dialog', {name: 'Choose card', exact: true})
      .getByRole('button', {name: new RegExp(`^${rank} of spades`)}).click();
  }
  await page.getByRole('button', {name: 'Done editing', exact: true}).click();
  for (const slot of [0, 1]) {
    const card = page.locator(`#handP1 .card[data-slot="${slot}"]`);
    if (testInfo.project.use.hasTouch) await card.tap();
    else await card.click();
  }
  await expect(page.locator('#handP1 .card.selected')).toHaveAttribute('data-card', '2S');
  // Reproduce a lingering preview from a different card while 2S is selected.
  await page.locator('#handP1 .card[data-slot="0"]').hover();
  await expect(page.locator('.cell.valid-move')).toHaveCount(2);
  for (const cell of await page.locator('.cell.valid-move').all()) {
    await expect(cell).toHaveAttribute('aria-label', /^2S, /);
  }
  const target = page.locator('.cell.valid-move').first();
  if (testInfo.project.use.hasTouch) await target.tap();
  else await target.click();
  await expect(page.locator('.history-item')).toHaveCount(2);
  await expect(page.locator('.history-player-1')).toHaveAttribute('aria-label', /You, 2S, placed chip/);
  await expect(page.locator('#aiError')).toBeHidden();
});

test('one card tap then a center or edge tap places a move and gets an AI reply', async ({page}, testInfo) => {
  await openGame(page);
  for (const point of ['center', 'left edge', 'bottom edge']) {
    if (point !== 'center') await page.getByRole('button', {name: 'New game', exact: true}).click();
    // Every initial seven-card hand contains a placing card (only four removal Jacks exist).
    const card = page.locator('#handP1 .card:not([data-card="JS"]):not([data-card="JH"])').first();
    const slot = await card.getAttribute('data-slot');
    if (testInfo.project.use.hasTouch) await card.tap();
    else await card.click();
    await expect(page.locator(`#handP1 .card[data-slot="${slot}"]`)).toHaveClass(/selected/);
    const target = page.locator('.cell.valid-move').first();
    await expect(target).toBeVisible();
    const row = Number(await target.getAttribute('data-row'));
    const col = Number(await target.getAttribute('data-col'));
    await target.scrollIntoViewIfNeeded();
    const box = await target.boundingBox();
    const x = point === 'left edge' ? box.x + 1 : box.x + box.width / 2;
    const y = point === 'bottom edge' ? box.y + box.height - 1 : box.y + box.height / 2;
    if (testInfo.project.use.hasTouch) await page.touchscreen.tap(x, y);
    else await page.mouse.click(x, y);
    await expect(page.locator('.history-item')).toHaveCount(2);
    await expect(page.locator('.history-player-1')).toHaveAttribute('aria-label', new RegExp(`placed chip at column ${col + 1}, row ${row + 1}$`));
    await expect(page.locator('#turnIndicator')).toContainText('Your turn');
    await expect(page.locator('#aiError')).toBeHidden();
  }
});

for (const speed of ['500', '0']) {
  test(`evaluation can overlap Watch start and resume at speed ${speed}`, async ({page}) => {
    await openGame(page);
    await page.getByLabel('Watch speed').selectOption(speed);
    for (let attempt = 0; attempt < 3; attempt++) {
      await page.getByRole('checkbox', {name: 'Show evaluation'}).uncheck();
      const before = await page.locator('.history-item').count();
      // Start evaluation and play in the same event turn. Pause through the
      // real button as soon as two moves appear: Max can finish an entire
      // game before a later automation round trip reaches Pause.
      const after = await page.evaluate(resuming => new Promise(resolve => {
        const history = document.getElementById('moveHistory');
        const count = history.querySelectorAll('.history-item').length;
        const observer = new MutationObserver(() => {
          const moves = history.querySelectorAll('.history-item').length;
          if (moves >= count + 2) {
            observer.disconnect();
            document.getElementById('newGameBtn').click();
            resolve(moves);
          }
        });
        observer.observe(history, {childList: true});
        document.getElementById('showEvalBar').click();
        (resuming ? document.querySelector('#pauseControls button')
          : document.getElementById('watchAiBtn')).click();
      }), attempt > 0);
      expect(after).toBeGreaterThan(before);
      await expect(page.getByRole('button', {name: 'Resume', exact: true})).toBeVisible();
      await expect(page.locator('#aiError')).toBeHidden();
    }
    // Request a fresh paused evaluation and verify the model session still works.
    await page.getByRole('checkbox', {name: 'Show evaluation'}).uncheck();
    await page.getByRole('checkbox', {name: 'Show evaluation'}).check();
    await expect(page.locator('#evalText')).toHaveText(/^(Red|Blue) \d+%$/);
    await expect(page.locator('#aiError')).toBeHidden();
  });
}

test('Watch button starts play, pause freezes it, resume and editing preserve modes', async ({page}) => {
  await openGame(page);
  await expect(page.getByLabel('Watch speed')).toBeVisible();
  await page.getByRole('button', {name: 'Watch AI', exact: true}).click();
  await expect.poll(() => page.locator('.history-item').count()).toBeGreaterThan(0);
  await expect(page.locator('#hintBtn')).toBeHidden();
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Resume', exact: true})).toBeVisible();
  await expect(page.getByLabel('Watch speed')).toBeVisible();
  await page.getByLabel('Watch speed').selectOption('0');
  await expect(page.getByLabel('Watch speed')).toHaveValue('0');
  await page.getByLabel('Watch speed').selectOption('1000');
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

test('a Max-speed game can finish, slow down, and restart', async ({page}) => {
  test.setTimeout(120000);
  await openGame(page);
  await page.getByLabel('Watch speed').selectOption('0');
  await page.getByRole('button', {name: 'Watch AI', exact: true}).click();
  const result = page.locator('#gameOverOverlay');
  await expect(result).toBeVisible({timeout: 90000});
  await result.getByRole('button', {name: 'Close', exact: true}).click();
  await expect(page.getByLabel('Watch speed')).toBeVisible();
  await page.getByLabel('Watch speed').selectOption('1000');
  await page.getByRole('button', {name: 'Watch AI again', exact: true}).click();
  await expect(result).not.toBeVisible();
  await expect.poll(() => page.locator('.history-item').count()).toBeGreaterThan(0);
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
  await expect(page.getByLabel('Watch speed')).toHaveValue('1000');
  await page.getByRole('button', {name: 'New game', exact: true}).click();
  await page.getByLabel('Watch speed').selectOption('500');
  await expect(page.getByRole('button', {name: 'Watch AI', exact: true})).toBeVisible();
  await expect(page.locator('.history-item')).toHaveCount(0);
  await noHorizontalOverflow(page);
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
