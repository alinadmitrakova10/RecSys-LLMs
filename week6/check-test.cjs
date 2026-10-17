/**
 * node week6/check-test.cjs --run-test     One TEST run, only when no file exists.
 * node week6/check-test.cjs --verify-saved Fast saved-result/UI verification.
 * Default is verification. Never trains MF or repeats validation selection.
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const mode = process.argv[2] || '--verify-saved';
assert.ok(['--run-test', '--verify-saved'].includes(mode) && process.argv.length <= 3, 'Choose one documented test mode');
const protectedNames = ['u.data', 'u.item', 'graph.js', 'batch-ppr.js', 'mf.js', 'mf-checkpoint.json', 'mf-run.json',
    ...fs.readdirSync(__dirname).filter(name => /^alpha-.*\.(json|csv)$/.test(name)),
    'recommendations-alpha085.json', 'recommendations-alpha085.csv', 'recommendations-summary-alpha085.csv'];
if (mode === '--verify-saved') {
    for (const name of ['test-results.json', 'results.csv', 'recommendations.csv', 'recommendations-summary.csv']) {
        assert.ok(fs.existsSync(path.join(__dirname, name)), 'Missing saved artifact: ' + name);
        protectedNames.push(name);
    }
}
const hashes = () => execFileSync('git', ['hash-object', ...protectedNames.map(name => 'week6/' + name)], { cwd: root, encoding: 'utf8' });
const beforeHashes = hashes();
const outputPath = path.join(__dirname, 'test-results.json');
if (mode === '--run-test') assert.equal(fs.existsSync(outputPath), false, 'Saved TEST results already exist; use --verify-saved instead of recalculating');
else assert.ok(fs.existsSync(outputPath), 'No saved test result to verify');

const graphSpy = `;(() => {
 const wrap = (name, original) => async function(...args) {
  const opts = args[name === 'pageRank' ? 1 : 2] || {};
  const n = opts.transition ? opts.transition.Q.length : (opts.graph || getGraph())?.nodeCount;
  if (n > 32) {
   if (window.finalTestAudit.forbidPropagation) throw new Error('Unexpected MovieLens propagation during saved-result verification');
   window.finalTestAudit[name]++;
   if (name === 'pageRank') window.finalTestAudit.globalAlphas.push(args[0]);
  }
  return original(...args);
 };
 pageRank = wrap('pageRank', pageRank); personalizedPageRank = wrap('personalizedPageRank', personalizedPageRank);
})();`;
const batchSpy = `;(() => {
 const original = HW6_BATCH_PPR.create;
 HW6_BATCH_PPR.create = async (...args) => {
  if (window.finalTestAudit.forbidPropagation) throw new Error('Unexpected batch engine during saved-result verification');
  window.finalTestAudit.batchCreates++;
  const engine = await original(...args); const run = engine.runUsers;
  engine.runUsers = async (ids, alpha, opts) => {
   window.finalTestAudit.batchCalls++; window.finalTestAudit.batchAlphas.push(alpha);
   return run(ids, alpha, opts);
  }; return engine;
 };
})();`;
const mfSpy = `;for(const name of ['ensure','fit','retrain']) HW6_MF_ENGINE[name] = () => {
 window.finalTestAudit.trainingAttempts++; throw new Error('MF training is forbidden');
};`;
const alphaSpy = `;(() => {
 const original = HW6_ALPHA_SELECTION.run;
 HW6_ALPHA_SELECTION.run = (...args) => {
  if (args[0]?.force || HW6_ALPHA_SELECTION.result?.status !== 'complete') {
   window.finalTestAudit.alphaSelectionAttempts++; throw new Error('Validation selection is forbidden in this test step');
  } return original(...args);
 };
})();`;
const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { res.writeHead(404); res.end(); return; }
        const additions = { 'graph.js': graphSpy, 'batch-ppr.js': batchSpy, 'mf.js': mfSpy, 'alpha-selection.js': alphaSpy };
        if (additions[path.basename(file)]) body = body.toString('utf8') + additions[path.basename(file)];
        res.writeHead(200, { 'Content-Type': { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
            '.json': 'application/json', '.css': 'text/css' }[path.extname(file)] || 'application/octet-stream' }); res.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser; let progress;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage(); page.setDefaultTimeout(180000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => {
            window.finalTestAudit = { forbidPropagation: new URLSearchParams(location.search).has('verify'),
                pageRank: 0, personalizedPageRank: 0, batchCreates: 0, batchCalls: 0, globalAlphas: [], batchAlphas: [],
                trainingAttempts: 0, alphaSelectionAttempts: 0 };
        });
        const baseUrl = `http://127.0.0.1:${server.address().port}`;
        await page.goto(baseUrl + (mode === '--verify-saved' ? '/?verify=1' : '/'), { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);
        let result;
        if (mode === '--run-test') {
            // Deliberately disagree with selected alpha to catch accidental slider use.
            await page.evaluate(() => hw6SetAlpha(0.95));
            let last = '';
            progress = setInterval(async () => {
                try { const status = await page.locator('#status').textContent(); if (status !== last) { console.log('TEST_PROGRESS', status); last = status; } }
                catch (_) { /* Last poll can race page closure. */ }
            }, 15000);
            await page.locator('#recall-btn').click();
            await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_TEST_COMPARISON.result);
            result = await page.evaluate(() => HW6_TEST_COMPARISON.result);
            clearInterval(progress); progress = null;
            fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
            assert.equal(result.status, 'complete', result.error);
            const outputs = await page.evaluate(value => ({ results: HW6_TEST_COMPARISON.csvResults(value),
                recommendations: HW6_TEST_COMPARISON.csvRecommendations(value), summary: HW6_TEST_COMPARISON.csvSummary(value) }), result);
            fs.writeFileSync(path.join(__dirname, 'results.csv'), outputs.results);
            fs.writeFileSync(path.join(__dirname, 'recommendations.csv'), outputs.recommendations);
            fs.writeFileSync(path.join(__dirname, 'recommendations-summary.csv'), outputs.summary);
            const audit = await page.evaluate(() => window.finalTestAudit);
            assert.equal(audit.pageRank, 1); assert.equal(audit.personalizedPageRank, 0);
            assert.equal(audit.batchCreates, 1); assert.equal(audit.batchCalls, 54);
            assert.deepEqual(audit.globalAlphas, [0.5]); assert.deepEqual([...new Set(audit.batchAlphas)], [0.5]);
            assert.equal(audit.trainingAttempts, 0); assert.equal(audit.alphaSelectionAttempts, 0);
        } else {
            result = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
            assert.equal(await page.evaluate(() => HW6_TEST_COMPARISON.result?.status), 'complete');
        }
        const verification = await page.evaluate(value => HW6_TEST_COMPARISON.verify(value), result);
        const expectedCsv = await page.evaluate(value => ({ results: HW6_TEST_COMPARISON.csvResults(value),
            recommendations: HW6_TEST_COMPARISON.csvRecommendations(value), summary: HW6_TEST_COMPARISON.csvSummary(value) }), result);
        assert.equal(fs.readFileSync(path.join(__dirname, 'results.csv'), 'utf8'), expectedCsv.results);
        assert.equal(fs.readFileSync(path.join(__dirname, 'recommendations.csv'), 'utf8'), expectedCsv.recommendations);
        assert.equal(fs.readFileSync(path.join(__dirname, 'recommendations-summary.csv'), 'utf8'), expectedCsv.summary);
        assert.equal(verification.propagationRerun, false);
        assert.equal(await page.locator('#comparison-results tbody tr').count(), 4);
        assert.equal(await page.locator('#test-recommendations li').count(), 80);
        assert.match(await page.locator('#comparison-results').textContent(), /not independently tuned/);

        // Check schema/settings, duplicate/filter/order guards and macro integrity.
        const tamperChecks = await page.evaluate(async value => {
            const cases = [];
            async function rejects(name, mutate) {
                const bad = structuredClone(value); mutate(bad);
                let rejected = false;
                try { await HW6_TEST_COMPARISON.load(bad); } catch (_) { rejected = true; }
                if (!rejected || HW6_TEST_COMPARISON.result.status === 'complete') throw new Error('Tampering passed: ' + name);
                await HW6_TEST_COMPARISON.load(value);
                cases.push(name);
            }
            await rejects('duplicate IDs', bad => { bad.rows[0].perUser[0].recommendedIds[1] = bad.rows[0].perUser[0].recommendedIds[0]; });
            await rejects('previously rated movie', bad => {
                bad.rows[0].perUser[0].recommendedIds[0] = HW6_SPLIT.train.find(row => row.userId === bad.cohort[0]).movieId;
            });
            await rejects('ranking order', bad => {
                const user = bad.rows[0].perUser[0];
                for (const key of ['recommendedIds', 'recommendedScores', 'trainingPopularities']) [user[key][0], user[key][1]] = [user[key][1], user[key][0]];
            });
            await rejects('different cohort', bad => { bad.rows[1].perUser.pop(); });
            await rejects('wrong alpha', bad => { bad.rows[2].alpha = 0.85; });
            await rejects('wrong MF checkpoint', bad => { bad.mf.checkpoint++; });
            await rejects('wrong macro Recall', bad => { bad.rows[0].recall += 0.01; });
            await rejects('nonconverged user', bad => { bad.rows[2].perUser[0].convergence.converged = false; });
            return cases;
        }, result);

        // A real page refresh must restore without even creating a batch engine.
        await page.evaluate(url => history.replaceState({}, '', url + '/?verify=1'), baseUrl);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => HW6_STATE.loaded);
        assert.equal(await page.evaluate(() => HW6_TEST_COMPARISON.result?.loadedFrom), 'test-results.json');
        await page.evaluate(() => hw6SetAlpha(0.95));
        await page.locator('#recall-btn').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy);
        assert.deepEqual(await page.evaluate(() => HW6_TEST_COMPARISON.result.rows.filter(row => row.alpha !== null).map(row => row.alpha)), [0.5, 0.5]);
        const reloadAudit = await page.evaluate(() => window.finalTestAudit);
        assert.equal(reloadAudit.pageRank, 0); assert.equal(reloadAudit.personalizedPageRank, 0);
        assert.equal(reloadAudit.batchCreates, 0); assert.equal(reloadAudit.trainingAttempts, 0); assert.equal(reloadAudit.alphaSelectionAttempts, 0);
        assert.equal(await page.locator('#comparison-results tbody tr').count(), 4);
        assert.equal(await page.locator('#test-recommendations li').count(), 80);
        assert.deepEqual(errors, []);
        fs.writeFileSync(path.join(__dirname, 'test-verification.json'), JSON.stringify({ generatedAt: new Date().toISOString(),
            status: 'passed', verification, tamperChecks, reloadAudit, noPropagationAfterRefresh: true,
            csvMatchesJson: true, protectedInputsUnchanged: hashes() === beforeHashes, pageErrors: errors }, null, 2) + '\n');
        console.log('TEST_COMPARISON_PASS', JSON.stringify({ metrics: result.rows.map(row => ({ method: row.method,
            recall: row.recall, meanPopularity: row.meanPopularity, runtimeSeconds: row.runtimeMs / 1000, convergence: row.convergence })),
            cohortSize: result.cohortSize, alpha: result.alpha, totalSeconds: result.runtimeMs / 1000,
            savedVerification: verification.status, tamperChecks: tamperChecks.length,
            noPropagationAfterRefresh: true, pageErrors: errors }, null, 2));
    } finally {
        if (progress) clearInterval(progress);
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(hashes(), beforeHashes, 'Data, saved MF, alpha selection or fixed-alpha examples changed');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
