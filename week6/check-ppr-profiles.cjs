/**
 * --build builds the controlled examples from cached lists and train records.
 * Default/--verify-saved checks artifacts and page without real propagation,
 * metric evaluation, alpha selection or training. Only new profile files write.
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const mode = process.argv[2] || '--verify-saved';
assert.ok(['--build', '--verify-saved'].includes(mode) && process.argv.length <= 3, 'Choose --build or --verify-saved');
const protectedNames = ['u.data', 'u.item', 'graph.js', 'batch-ppr.js', 'mf.js', 'mf-checkpoint.json', 'mf-run.json',
    'test-results.json', 'results.csv', 'recommendations.csv', 'recommendations-summary.csv', 'test-verification.json',
    'recommendations-alpha085.json', 'recommendations-alpha085.csv', 'recommendations-summary-alpha085.csv',
    ...fs.readdirSync(__dirname).filter(file => /^alpha-.*\.(json|csv)$/.test(file))];
const artifacts = { lists: 'ppr-settings-examples.csv', comparisons: 'ppr-settings-comparisons.csv',
    histories: 'ppr-train-histories.csv', profiles: 'ppr-train-summary.csv', genres: 'ppr-train-genres.csv' };
if (mode === '--verify-saved') {
    for (const file of ['ppr-settings-examples.json', 'ppr-settings-observations.md', ...Object.values(artifacts)]) {
        assert.ok(fs.existsSync(path.join(__dirname, file)), 'Missing profile artifact: ' + file);
        protectedNames.push(file);
    }
}
const hashes = () => execFileSync('git', ['hash-object', ...protectedNames.map(file => 'week6/' + file)], { cwd: root, encoding: 'utf8' });
const beforeHashes = hashes();

const graphSpy = `;(() => {
 const wrap = (name, original) => async (...args) => {
  const opts = args[name === 'pageRank' ? 1 : 2] || {};
  const n = opts.transition ? opts.transition.Q.length : (opts.graph || getGraph())?.nodeCount;
  if (n > 32) { window.profileAudit.propagationAttempts++; throw new Error('MovieLens propagation is forbidden for cached examples'); }
  return original(...args);
 };
 pageRank = wrap('pageRank', pageRank); personalizedPageRank = wrap('personalizedPageRank', personalizedPageRank);
})();`;
const forbidden = {
    'graph.js': graphSpy,
    'batch-ppr.js': `;HW6_BATCH_PPR.create = () => { window.profileAudit.propagationAttempts++; throw new Error('Batch propagation is forbidden'); };`,
    'mf.js': `;for(const key of ['ensure','fit','retrain']) HW6_MF_ENGINE[key] = () => { window.profileAudit.trainingAttempts++; throw new Error('Training is forbidden'); };`,
    'alpha-selection.js': `;HW6_ALPHA_SELECTION.run = () => { window.profileAudit.selectionAttempts++; throw new Error('Alpha selection is forbidden'); };`,
    'test-comparison.js': `;HW6_TEST_COMPARISON.run = () => { window.profileAudit.testEvaluationAttempts++; throw new Error('Test evaluation is forbidden'); };`,
};
const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { res.writeHead(404); res.end(); return; }
        if (forbidden[path.basename(file)]) body = body.toString('utf8') + forbidden[path.basename(file)];
        res.writeHead(200, { 'Content-Type': { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
            '.json': 'application/json', '.csv': 'text/csv; charset=utf-8', '.md': 'text/markdown; charset=utf-8', '.css': 'text/css' }[path.extname(file)] || 'application/octet-stream' });
        res.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage(); page.setDefaultTimeout(180000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => { window.profileAudit = {
            propagationAttempts: 0, trainingAttempts: 0, selectionAttempts: 0, testEvaluationAttempts: 0 }; });
        await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);
        const value = mode === '--build' ? await page.evaluate(() => HW6_PPR_PROFILES.build()) :
            JSON.parse(fs.readFileSync(path.join(__dirname, 'ppr-settings-examples.json'), 'utf8'));
        const verification = await page.evaluate(value => HW6_PPR_PROFILES.verify(value), value);
        const output = await page.evaluate(value => ({ csv: HW6_PPR_PROFILES.csv(value), markdown: HW6_PPR_PROFILES.markdown(value) }), value);
        if (mode === '--build') {
            fs.writeFileSync(path.join(__dirname, 'ppr-settings-examples.json'), JSON.stringify(value, null, 2) + '\n');
            for (const [key, file] of Object.entries(artifacts)) fs.writeFileSync(path.join(__dirname, file), output.csv[key]);
            fs.writeFileSync(path.join(__dirname, 'ppr-settings-observations.md'), output.markdown);
        }
        for (const [key, file] of Object.entries(artifacts)) assert.equal(fs.readFileSync(path.join(__dirname, file), 'utf8'), output.csv[key]);
        assert.equal(fs.readFileSync(path.join(__dirname, 'ppr-settings-observations.md'), 'utf8'), output.markdown);
        assert.deepEqual(value.examples.map(row => [row.userId, row.alpha]), [[1, 0.85], [2, 0.85], [1, 0.50]]);
        assert.deepEqual(value.profiles.map(row => [row.userId, row.ratingCount, row.positiveRatingCount]), [[1, 217, 131], [2, 49, 32]]);
        assert.deepEqual(value.comparisons.map(row => row.commonCount), [0, 7]);
        assert.deepEqual(value.examples.map(row => row.meanPopularity), [245.5, 254.9, 228.4]);
        assert.deepEqual(value.examples.map(row => row.zeroScoreCount), [0, 0, 0]);
        assert.equal(verification.trainHistoryRows, 266);

        const tamperChecks = await page.evaluate(async value => {
            const cases = [];
            async function rejects(name, mutate) {
                const bad = structuredClone(value); mutate(bad);
                let rejected = false;
                try { await HW6_PPR_PROFILES.load(bad); } catch (_) { rejected = true; }
                if (!rejected) throw new Error('Tampering passed: ' + name);
                cases.push(name);
            }
            await rejects('fourth setting', bad => bad.examples.push(structuredClone(bad.examples[0])));
            await rejects('wrong user/alpha', bad => { bad.examples[1].alpha = 0.5; });
            await rejects('duplicate/unfiltered IDs', bad => { bad.examples[0].recommendations[1].movieId = bad.examples[0].recommendations[0].movieId; });
            await rejects('wrong intersection', bad => { bad.comparisons[1].commonCount++; });
            await rejects('wrong mean popularity', bad => { bad.examples[0].meanPopularity++; });
            await rejects('wrong zero count', bad => { bad.examples[0].zeroScoreCount++; });
            await rejects('held-out profile label', bad => { bad.profiles[0].sourceSplit = 'test'; });
            await rejects('wrong positive genre count', bad => { bad.profiles[0].genreDistribution[0].count++; });
            await rejects('non-train profile sample', bad => { bad.profiles[0].examples[0] = { ...bad.profiles[0].examples[0], movieId: 100, rating: 5 }; });
            await HW6_PPR_PROFILES.load(value);
            return cases;
        }, value);

        await page.evaluate(() => hw6SetAlpha(0.95));
        await page.locator('#two-user-demo-btn').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy);
        assert.equal(await page.locator('#ppr-settings-results table tbody tr').count(), 30);
        assert.equal(await page.locator('#ppr-settings-results table').count(), 3);
        assert.match(await page.locator('#ppr-settings-results').textContent(), /training popularity|train popularity/);
        assert.match(await page.locator('#ppr-settings-results').textContent(), /not prove better quality/);
        assert.equal(await page.evaluate(() => HW6_ALPHA_SELECTION.comparisonAlpha()), 0.50);
        await page.locator('#run-tests').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.lastVerification);
        const oldTests = await page.evaluate(() => HW6_STATE.lastVerification);
        assert.equal(oldTests.failed, 0, JSON.stringify(oldTests.checks));
        assert.equal(oldTests.passed, 12); assert.equal(oldTests.pending, 0);
        const beforeReloadAudit = await page.evaluate(() => window.profileAudit);
        assert.deepEqual(beforeReloadAudit, { propagationAttempts: 0, trainingAttempts: 0, selectionAttempts: 0, testEvaluationAttempts: 0 });

        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => HW6_STATE.loaded);
        assert.equal(await page.locator('#ppr-settings-results table tbody tr').count(), 30);
        assert.equal(await page.locator('#comparison-results tbody tr').count(), 4);
        assert.equal(await page.locator('#test-recommendations li').count(), 80);
        const finalVerification = await page.evaluate(() => HW6_TEST_COMPARISON.verify(HW6_TEST_COMPARISON.result));
        const audit = await page.evaluate(() => window.profileAudit);
        assert.deepEqual(audit, { propagationAttempts: 0, trainingAttempts: 0, selectionAttempts: 0, testEvaluationAttempts: 0 });
        assert.deepEqual(errors, []);
        fs.writeFileSync(path.join(__dirname, 'ppr-settings-verification.json'), JSON.stringify({ generatedAt: new Date().toISOString(),
            status: 'passed', verification, tamperChecks, csvAndMarkdownMatchJson: true, beforeReloadAudit, afterReloadAudit: audit,
            previousFinalResultsVerifiedWithoutRerun: finalVerification.status === 'passed',
            previousSelfChecks: { passed: oldTests.passed, failed: oldTests.failed, pending: oldTests.pending },
            protectedInputsUnchanged: hashes() === beforeHashes, pageErrors: errors }, null, 2) + '\n');
        console.log('PPR_SETTINGS_PROFILES_PASS', JSON.stringify({ settings: value.examples.map(row => ({ userId: row.userId, alpha: row.alpha,
            meanPopularity: row.meanPopularity, zeros: row.zeroScoreCount })), comparisons: value.comparisons.map(row => ({ change: row.changedId,
            common: row.commonCount, appeared: row.appearedIds, disappeared: row.disappearedIds })),
            profiles: value.profiles.map(row => ({ userId: row.userId, train: row.ratingCount, positive: row.positiveRatingCount })),
            trainHistoryRows: verification.trainHistoryRows, tamperChecks: tamperChecks.length,
            oldSelfChecks: { passed: oldTests.passed, failed: oldTests.failed, pending: oldTests.pending }, audit }, null, 2));
    } finally {
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(hashes(), beforeHashes, 'Protected data, models or previous numerical results changed');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
