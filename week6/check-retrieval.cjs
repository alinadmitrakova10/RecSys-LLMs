/**
 * node week6/check-retrieval.cjs --fixtures-only
 * node week6/check-retrieval.cjs
 * Browser known-answer tests, then eight fixed-alpha lists (not population
 * Recall). Generated lists/verification are written only to week6.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const protectedPaths = ['graph.js', 'mf.js', 'mf-checkpoint.json', 'mf-run.json', 'u.data', 'u.item'].map(name => 'week6/' + name);
const hash = () => execFileSync('git', ['hash-object', ...protectedPaths], { cwd: root, encoding: 'utf8' });
const beforeHash = hash();
const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { res.writeHead(404); res.end(); return; }
        const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.css': 'text/css' };
        res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' }); res.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage();
        page.setDefaultTimeout(180000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);
        await page.evaluate(() => {
            window.retrievalAudit = { fixtureRecallCalls: 0, fullRecallAttempts: 0, trainingAttempts: 0 };
            const originalRecall = recallAtK;
            recallAtK = async (...args) => {
                if (HW6_STATE.trainUserIds.length > 32) {
                    window.retrievalAudit.fullRecallAttempts++;
                    throw new Error('Population Recall is forbidden in this staged verification');
                }
                window.retrievalAudit.fixtureRecallCalls++;
                return originalRecall(...args);
            };
            for (const key of ['ensure', 'retrain', 'fit']) {
                HW6_MF_ENGINE[key] = () => {
                    window.retrievalAudit.trainingAttempts++;
                    throw new Error('MF training/automatic training fallback is forbidden');
                };
            }
        });
        const fixtures = await page.evaluate(() => HW6_RETRIEVAL_CHECKS.run());
        assert.equal(fixtures.passed, 5);
        assert.equal(fixtures.fullMovieLensRecallRun, false);
        assert.equal(fixtures.tensorsBefore, fixtures.tensorsAfter);
        console.log('RETRIEVAL_FIXTURES_PASS', JSON.stringify(fixtures, null, 2));
        if (process.argv.includes('--fixtures-only')) return;

        await page.locator('#build-graph').click();
        assert.equal(await page.locator('#recall-btn').isDisabled(), false);
        assert.equal(await page.locator('#alpha-sweep-btn').isDisabled(), false);
        // This regression targets the earlier eight-list preview, not the new
        // controlled-three-settings UI. The focused controller has its own test.
        await page.evaluate(() => hw6RunTwoUserDemo());
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.demo && HW6_STATE.demo.lists && HW6_STATE.demo.lists.length === 8);
        const preview = await page.evaluate(() => HW6_STATE.demo);
        assert.equal(preview.alpha, 0.85); assert.equal(preview.candidateCount, 1611);
        assert.equal(preview.mf.trainingRuns, 0);
        assert.deepEqual(preview.users, [1, 2]);
        assert.equal(preview.lists.length, 8);
        assert.equal(preview.exclusions.val.cohortSize, 846);
        assert.equal(preview.exclusions.test.cohortSize, 861);
        assert.equal(preview.exclusions.test.excludedTotal, 82);
        assert.equal(preview.exclusions.test.excludedNoPositiveEdge, 1);
        assert.equal(preview.exclusions.test.excludedMfUnavailable, 0);
        assert.equal(preview.exclusions.test.excludedNoEligible, 82);
        assert.equal(preview.exclusions.test.overlapUsers.length, 1);
        assert.equal(preview.exclusions.test.excludedUnseen, 13);

        // Independent filtering/sorting from raw histories and method scores.
        const oracle = await page.evaluate(async () => {
            const output = [];
            const candidateIds = Array.from(new Set(HW6_SPLIT.train.map(row => row.movieId))).sort((a, b) => a - b);
            const popularityUsers = new Map();
            for (const row of HW6_SPLIT.train) if (row.rating >= 4) {
                if (!popularityUsers.has(row.movieId)) popularityUsers.set(row.movieId, new Set());
                popularityUsers.get(row.movieId).add(row.userId);
            }
            const userFactors = await HW6_MF_MODEL.userFactors.array();
            const itemFactors = await HW6_MF_MODEL.itemFactors.array();
            for (const list of HW6_STATE.demo.lists) {
                const excluded = new Set([...HW6_SPLIT.train, ...HW6_SPLIT.val]
                    .filter(row => row.userId === list.userId).map(row => row.movieId));
                let scores;
                if (list.method === 'popularity') scores = Object.fromEntries(candidateIds.map(id => [id, popularityUsers.get(id)?.size || 0]));
                else if (list.method === 'mf') scores = Object.fromEntries(candidateIds.map(id => [id,
                    HW6_MF_MODEL.globalMean + userFactors[list.userId].reduce((sum, value, f) => sum + value * itemFactors[id][f], 0)]));
                else {
                    const g = getGraph();
                    const propagation = list.method === 'ppr' ? await hw6RunPPR(g, list.userId, 0.85) : await hw6RunPageRank(g, 0.85);
                    scores = Object.fromEntries(g.movieIds.map((id, i) => [id, propagation.scores[g.nUsers + i]]));
                }
                const expected = candidateIds.filter(id => !excluded.has(id))
                    .sort((a, b) => scores[b] - scores[a] || a - b).slice(0, 10);
                const actual = list.recommendations.map(rec => rec.movieId);
                if (JSON.stringify(expected) !== JSON.stringify(actual)) throw new Error('Independent Top-10 mismatch: ' + list.method + '/' + list.userId);
                for (const rec of list.recommendations) {
                    if (excluded.has(rec.movieId) || !candidateIds.includes(rec.movieId)) throw new Error('Invalid candidate');
                    if (Math.abs(rec.score - scores[rec.movieId]) > 1e-5) throw new Error('Raw score mismatch');
                    if (rec.popularity !== (popularityUsers.get(rec.movieId)?.size || 0)) throw new Error('Training popularity mismatch');
                    const movie = movies.find(movie => movie.id === rec.movieId);
                    if (movie.title !== rec.title || JSON.stringify(movie.genres) !== JSON.stringify(rec.genres)) throw new Error('Metadata mismatch');
                }
                const mean = list.recommendations.reduce((sum, rec) => sum + rec.popularity, 0) / 10;
                if (mean !== list.meanPopularity) throw new Error('Mean popularity mismatch');
                output.push({ method: list.method, userId: list.userId, length: actual.length, meanPopularity: mean });
            }
            for (const overlap of HW6_STATE.demo.overlap) {
                const pair = HW6_STATE.demo.lists.filter(list => list.method === overlap.method);
                const common = pair[0].recommendations.filter(rec => pair[1].recommendations.some(other => other.movieId === rec.movieId));
                if (common.length !== overlap.count) throw new Error('Overlap mismatch');
            }
            return output;
        });
        assert.equal(oracle.length, 8);
        await page.selectOption('#user-select', '1');
        await page.selectOption('#method', 'popularity');
        await page.locator('#recommend-btn').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && document.querySelectorAll('#recommendations tbody tr').length === 10);
        assert.match(await page.locator('#recommendations').textContent(), /Mean training popularity/);
        assert.match(await page.locator('#recommendations').textContent(), /Reasons overlap/);
        await page.selectOption('#method', 'mf');
        await page.selectOption('#user-select', '2');
        await page.locator('#recommend-btn').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && document.querySelectorAll('#recommendations tbody tr').length === 10);
        // The final TEST action is allowed now, but this regression must not
        // create a population experiment. Reuse a saved result if one exists.
        const savedTest = await page.evaluate(() => HW6_TEST_COMPARISON.result?.status === 'complete');
        if (savedTest) {
            await page.evaluate(async () => { await hw6HandleRecall(); });
            assert.equal(await page.evaluate(() => HW6_STATE.comparisonRows.length), 4);
        } else assert.equal(await page.evaluate(() => HW6_STATE.comparisonRows), null);

        await page.locator('#run-tests').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.lastVerification);
        const selfChecks = await page.evaluate(() => HW6_STATE.lastVerification);
        assert.equal(selfChecks.failed, 0, JSON.stringify(selfChecks.checks));
        assert.equal(selfChecks.passed, 12); assert.equal(selfChecks.pending, 0);
        // An empty implementation must fail a known-answer fixture, not pass vacuously.
        const emptyTrap = await page.evaluate(async () => {
            const original = recommendTopK;
            try { recommendTopK = async () => []; return await runSelfChecks(); }
            finally { recommendTopK = original; }
        });
        assert.ok(emptyTrap.failed > 0); assert.equal(emptyTrap.pending, 0);
        await page.evaluate(() => { HW6_STATE.lastVerification = null; });
        await page.locator('#run-tests').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.lastVerification?.failed === 0);

        const exported = await page.evaluate(() => ({
            csv: serializeRecommendations({ exampleOnly: true }),
            summaryCsv: hw6CsvDocument(['method', 'user_id', 'alpha', 'mean_training_popularity', 'zero_score_count', 'common_movies_between_users', 'score_meaning'],
                HW6_STATE.demo.lists.map(list => [list.method, list.userId, list.alpha, list.meanPopularity, list.zeroScoreCount,
                    HW6_STATE.demo.overlap.find(row => row.method === list.method).count, list.scoreMeaning])),
            verification: HW6_STATE.lastVerification, audit: window.retrievalAudit,
            trainingRuns: HW6_MF_ENGINE.trainingRuns, recommendationRows: HW6_STATE.recommendationRows.length,
        }));
        assert.equal(exported.recommendationRows, 80);
        assert.equal(exported.audit.fullRecallAttempts, 0);
        assert.equal(exported.audit.trainingAttempts, 0);
        assert.equal(exported.trainingRuns, 0);
        assert.deepEqual(errors, []);
        const verification = { generatedAt: new Date().toISOString(), fixtures, selfChecks: exported.verification,
            emptyListRejected: true, oracle, audit: exported.audit, trainingRuns: 0,
            fullMovieLensRecallRun: false, alphaSweepRun: false, pageErrors: errors,
            browser: browser.version(), protectedFilesUnchanged: hash() === beforeHash };
        if (!process.argv.includes('--verify-only')) {
            fs.writeFileSync(path.join(__dirname, 'recommendations-alpha085.json'), JSON.stringify(preview, null, 2) + '\n');
            fs.writeFileSync(path.join(__dirname, 'recommendations-alpha085.csv'), exported.csv);
            fs.writeFileSync(path.join(__dirname, 'recommendations-summary-alpha085.csv'), exported.summaryCsv);
            fs.writeFileSync(path.join(__dirname, 'retrieval-verification.json'), JSON.stringify(verification, null, 2) + '\n');
        }
        console.log('RETRIEVAL_BROWSER_PASS', JSON.stringify({ lists: oracle, overlap: preview.overlap,
            cohortSizes: { val: preview.exclusions.val.cohortSize, test: preview.exclusions.test.cohortSize },
            selfChecks: { passed: selfChecks.passed, failed: selfChecks.failed, pending: selfChecks.pending },
            trainingRuns: 0, fullMovieLensRecallRun: false, alphaSweepRun: false, pageErrors: errors }, null, 2));
    } finally {
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(hash(), beforeHash, 'PageRank, MF method/checkpoint or data changed');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
