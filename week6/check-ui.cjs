/** UI-only browser regression. No artifacts are written; saved files are hashed. */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const editable = new Set(['index.html', 'script.js', 'style.css', 'readme.md', 'check-ui.cjs']);
function protectedFiles(directory = __dirname) {
    return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
        // Office lock files may be unreadable while the user's document is open.
        if (entry.name.startsWith('~$')) return [];
        const file = path.join(directory, entry.name);
        return entry.isDirectory() ? protectedFiles(file) : editable.has(path.relative(__dirname, file)) ? [] : [path.relative(root, file)];
    });
}
const inputs = protectedFiles();
const hashes = () => execFileSync('git', ['hash-object', ...inputs], { cwd: root, encoding: 'utf8' });
const originalHashes = hashes();
const graphSpy = `;(() => {
 const wrap = (name, original) => async (...args) => {
  const opts = args[name === 'pageRank' ? 1 : 2] || {};
  const graph = opts.graph || getGraph();
  const n = opts.transition ? opts.transition.Q.length : graph?.nodeCount;
  if (n > 32) {
   if (name !== 'personalizedPageRank' || args[0] !== graph.nodeOfUser(1) || args[1] !== 0.50) {
    window.uiAudit.forbiddenCalculations++; throw new Error('Unexpected real-data calculation in UI verification');
   }
   window.uiAudit.pprExampleCalls++;
  }
  return original(...args);
 };
 pageRank = wrap('pageRank', pageRank); personalizedPageRank = wrap('personalizedPageRank', personalizedPageRank);
})();`;
const append = {
    'graph.js': graphSpy,
    'batch-ppr.js': `;HW6_BATCH_PPR.create = () => { window.uiAudit.forbiddenCalculations++; throw new Error('Full batch evaluation forbidden'); };`,
    'mf.js': `;for(const name of ['ensure','fit','retrain']) HW6_MF_ENGINE[name] = () => { window.uiAudit.forbiddenCalculations++; throw new Error('MF training forbidden'); };`,
    'alpha-selection.js': `;HW6_ALPHA_SELECTION.run = () => { window.uiAudit.forbiddenCalculations++; throw new Error('Alpha sweep forbidden'); };`,
    'test-comparison.js': `;HW6_TEST_COMPARISON.run = () => { window.uiAudit.forbiddenCalculations++; throw new Error('Population evaluation forbidden'); };`,
    'script.js': `;(() => {
     const original = runSelfChecks;
     runSelfChecks = async (...args) => {
      window.uiAudit.testCalls++;
      if(window.uiAudit.delayTests) await new Promise(resolve => setTimeout(resolve, 500));
      return original(...args);
     };
    })();`,
};
const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { res.writeHead(404); res.end(); return; }
        if (append[path.basename(file)]) body = body.toString('utf8') + append[path.basename(file)];
        res.writeHead(200, { 'Content-Type': { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
            '.json': 'application/json', '.css': 'text/css' }[path.extname(file)] || 'application/octet-stream' }); res.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    const errors = [];
    try {
        browser = await chromium.launch({ headless: true });
        const newPage = async () => {
            const page = await browser.newPage(); page.setDefaultTimeout(120000);
            page.on('pageerror', error => errors.push(error.message));
            await page.addInitScript(() => {
                window.uiAudit = { pprExampleCalls: 0, forbiddenCalculations: 0, testCalls: 0, delayTests: false, fetches: [] };
                const original = window.fetch;
                window.fetch = (...args) => { window.uiAudit.fetches.push(String(args[0])); return original(...args); };
            });
            return page;
        };
        const url = `http://127.0.0.1:${server.address().port}`;
        const page = await newPage();
        let releaseData; let requested;
        const waitForData = new Promise(resolve => { releaseData = resolve; });
        const dataRequested = new Promise(resolve => { requested = resolve; });
        await page.route('**/u.data', async route => { requested(); await waitForData; await route.continue(); });
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await dataRequested;
        assert.equal(await page.locator('#run-tests').isDisabled(), true);
        assert.equal(await page.locator('#recommend-btn').isDisabled(), true);
        assert.equal(await page.locator('#application-gate button:enabled').count(), 0);
        assert.match(await page.locator('#app-readiness').textContent(), /Loading/);
        await page.locator('#run-tests').scrollIntoViewIfNeeded();
        const box = await page.locator('#run-tests').boundingBox();
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
        await page.locator('#run-tests').evaluate(button => button.click());
        assert.equal(await page.evaluate(() => window.uiAudit.testCalls), 0);
        assert.equal(await page.evaluate(() => HW6_STATE.lastVerification), null);
        releaseData();
        await page.waitForFunction(() => HW6_STATE.loaded && HW6_STATE.uiStage === 'ready');
        assert.equal(await page.locator('#run-tests').isEnabled(), true);
        assert.match(await page.locator('#app-readiness').textContent(), /^Ready\./);
        assert.equal(await page.locator('#alpha').inputValue(), '0.85');
        assert.equal(await page.locator('#saved-alpha-label').textContent(), '0.50');
        assert.match(await page.locator('#alpha-explanation').textContent(), /does not change/);
        await page.evaluate(() => { window.uiAudit.delayTests = true; });
        await page.locator('#run-tests').click();
        await page.waitForFunction(() => document.getElementById('tests-summary').textContent.includes('Running tests'));
        assert.equal(await page.locator('#run-tests').isDisabled(), true);
        assert.equal(await page.locator('#run-tests').getAttribute('aria-busy'), 'true');
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && document.getElementById('tests-summary').textContent === '12 passed, 0 failed, 0 pending');
        assert.equal(await page.locator('#run-tests').isEnabled(), true);
        assert.equal(await page.locator('#run-tests').textContent(), 'Run tests');
        await page.locator('#tests-details-link').click();
        assert.equal(await page.evaluate(() => location.hash), '#self-checks');
        assert.equal(await page.locator('#self-checks').evaluate(element => {
            const rect = element.getBoundingClientRect(); return rect.top < innerHeight && rect.bottom > 0;
        }), true);
        assert.match(await page.locator('#self-checks').textContent(), /pass 12, fail 0, pending 0/);
        assert.equal(await page.locator('#ppr-settings-results table').count(), 3);
        assert.deepEqual(await page.evaluate(() => HW6_PPR_PROFILES.result.examples.map(x => [x.userId, x.alpha])), [[1, .85], [2, .85], [1, .50]]);

        await page.selectOption('#method', 'ppr');
        await page.selectOption('#user-select', '1');
        await page.locator('#alpha').evaluate(input => { input.value = '0.50'; input.dispatchEvent(new Event('input', { bubbles: true })); });
        await page.locator('#recommend-btn').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && document.querySelectorAll('#recommendations tbody tr').length === 10);
        assert.equal(await page.locator('#recommendations tbody tr').first().locator('td').nth(2).textContent(), 'Fargo (1996)');
        assert.match(await page.locator('#recommendations').textContent(), /Mean training popularity: 228\.40/);
        const audit = await page.evaluate(() => window.uiAudit);
        assert.equal(audit.pprExampleCalls, 1); assert.equal(audit.forbiddenCalculations, 0); assert.equal(audit.testCalls, 1);
        await page.close();

        const failed = await newPage();
        await failed.route('**/u.data', route => route.fulfill({ status: 503, body: 'Injected data loading failure' }));
        await failed.goto(url, { waitUntil: 'domcontentloaded' });
        await failed.waitForFunction(() => HW6_STATE.uiStage === 'error');
        assert.equal(await failed.locator('#run-tests').isDisabled(), true);
        assert.match(await failed.locator('#app-readiness').textContent(), /could not finish loading.*503/);
        assert.equal(await failed.locator('#app-readiness').getAttribute('role'), 'alert');
        assert.match(await failed.locator('#tests-summary').textContent(), /unavailable/);
        assert.equal(await failed.evaluate(() => HW6_STATE.loaded), false);
        await failed.close();

        const noTf = await newPage();
        await noTf.route(/cdn\.jsdelivr\.net\/npm\/@tensorflow\/tfjs/, route => route.abort());
        await noTf.goto(url, { waitUntil: 'domcontentloaded' });
        await noTf.waitForFunction(() => HW6_STATE.uiStage === 'error');
        assert.match(await noTf.locator('#app-readiness').textContent(), /TensorFlow\.js could not start/);
        assert.equal(await noTf.locator('#run-tests').isDisabled(), true);
        await noTf.close();

        const filePage = await newPage();
        await filePage.route(/cdn\.jsdelivr\.net\/npm\/@tensorflow\/tfjs/, route => route.abort());
        await filePage.goto(pathToFileURL(path.join(__dirname, 'index.html')).href, { waitUntil: 'domcontentloaded' });
        await filePage.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.uiStage === 'error');
        const message = await filePage.locator('#app-readiness').textContent();
        assert.match(message, /file:\/\//); assert.match(message, /GitHub Pages/); assert.match(message, /python -m http\.server/);
        assert.equal(await filePage.locator('#run-tests').isDisabled(), true);
        assert.deepEqual(await filePage.evaluate(() => window.uiAudit.fetches), []);
        await filePage.close();
        assert.deepEqual(errors, []);
        console.log('UI_CHECKS_PASS', JSON.stringify({ slowLoadingAndIgnoredEarlyClick: true,
            explicitReadyAndLoadErrors: true, blockedTensorFlowError: true, fileProtocolHelpWithoutFetch: true,
            visibleBusyTestsAndDetailsLink: true, tests: { passed: 12, failed: 0, pending: 0 },
            ppr: { userId: 1, alpha: 0.50, first: 'Fargo (1996)', meanPopularity: '228.40' },
            controlledSettingsPreserved: true, forbiddenCalculations: 0, pageErrors: errors }));
    } finally {
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(hashes(), originalHashes, 'Algorithms, datasets, weights, saved results, reports or session files changed');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
