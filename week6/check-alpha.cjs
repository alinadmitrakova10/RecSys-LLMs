/**
 * Validation-only tools. --benchmark-only measures existing scalar PPR first.
 * --checks-only verifies batch equivalence; --run-validation runs the grid.
 * Generated artifacts stay in week6. Test retrieval and MF training are forbidden.
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const modes = ['--benchmark-only', '--diagnose-roundoff', '--checks-only', '--run-validation', '--retry-capped', '--retry-cpu', '--verify-saved'];
if (process.argv.slice(2).length > 1 || process.argv.slice(2).some(mode => !modes.includes(mode))) throw new Error('Choose one documented alpha-tool mode');
const verifySaved = process.argv.includes('--verify-saved') || process.argv.slice(2).length === 0;
const root = path.resolve(__dirname, '..');
const protectedPaths = ['u.data', 'u.item', 'mf.js', 'mf-checkpoint.json', 'mf-run.json'].map(file => 'week6/' + file);
const hashes = () => execFileSync('git', ['hash-object', ...protectedPaths], { cwd: root, encoding: 'utf8' });
const beforeHashes = hashes();
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
    let progress;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage();
        page.setDefaultTimeout(180000);
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);
        await page.evaluate(() => {
            window.alphaAudit = { testRecallAttempts: 0, trainingAttempts: 0 };
            const originalRecall = recallAtK;
            recallAtK = async (method, K, split, ...rest) => {
                if (split === 'test' && HW6_STATE.trainUserIds.length > 32) {
                    window.alphaAudit.testRecallAttempts++;
                    throw new Error('Full test Recall is forbidden during validation selection');
                }
                return originalRecall(method, K, split, ...rest);
            };
            for (const key of ['ensure', 'retrain', 'fit']) HW6_MF_ENGINE[key] = () => {
                window.alphaAudit.trainingAttempts++;
                throw new Error('MF training is forbidden');
            };
        });
        await page.locator('#build-graph').click();
        if (process.argv.includes('--diagnose-roundoff')) {
            const diagnostic = await page.evaluate(async () => {
                const graph = buildGraph([{ userId: 42, movieId: 10, rating: 5 },
                    { userId: 42, movieId: 20, rating: 4 }, { userId: 700, movieId: 30, rating: 1 }]);
                const blocks = rowNormalize(graph);
                const v = tf.tensor2d([1, 0, 0, 0, 0], [5, 1]);
                const mask = tf.tensor2d(blocks.danglingMask.map(x => x ? 1 : 0), [5, 1]);
                const output = [];
                try {
                    for (const variant of ['original', 'interpolation', 'complement']) {
                        let p = v.clone(); let residual; let iteration;
                        try {
                            for (iteration = 1; iteration <= 500; iteration++) {
                                const step = tf.tidy(() => {
                                    const linked = tf.concat([tf.matMul(blocks.movieToUser, p.slice([2, 0], [3, 1]), true),
                                        tf.matMul(blocks.userToMovie, p.slice([0, 0], [2, 1]), true)], 0);
                                    const d = p.mul(mask).sum();
                                    let next;
                                    if (variant === 'original') next = linked.mul(0.95).add(v.mul(d.mul(0.95).add(0.05)));
                                    else if (variant === 'complement') next = linked.mul(0.95).add(v.mul(d.mul(0.95).add(tf.scalar(1).sub(tf.scalar(0.95)))));
                                    else { const incoming = linked.add(v.mul(d)); next = incoming.add(v.sub(incoming).mul(0.05)); }
                                    return { next, residual: next.sub(p).abs().sum() };
                                });
                                residual = (await step.residual.data())[0]; step.residual.dispose();
                                p.dispose(); p = step.next;
                                if (residual < 1e-6) break;
                            }
                            output.push({ variant, iterations: Math.min(iteration, 500), residual, p: Array.from(await p.data()) });
                        } finally { p.dispose(); }
                    }
                } finally { blocks.dispose(); v.dispose(); mask.dispose(); }
                return output;
            });
            console.log('ROUNDOFF_DIAGNOSTIC', JSON.stringify(diagnostic, null, 2));
            return;
        }
        if (process.argv.includes('--benchmark-only')) {
            const benchmark = await page.evaluate(async () => {
                const cohort = await hw6SharedCohort('val');
                if (cohort.length !== 846) throw new Error('Expected 846 validation users');
                const users = [cohort[0], cohort[1], cohort[cohort.length - 1]];
                const graph = getGraph();
                const runs = [];
                const before = tf.memory().numTensors;
                for (const userId of users) {
                    const result = await personalizedPageRank(graph.nodeOfUser(userId), 0.85, { graph });
                    if (!result.converged) throw new Error('Scalar benchmark failed to converge');
                    runs.push({ userId, alpha: 0.85, runtimeMs: result.runtimeMs, iterations: result.iterations, residual: result.residual });
                }
                const meanMs = runs.reduce((sum, run) => sum + run.runtimeMs, 0) / runs.length;
                return { generatedAt: new Date().toISOString(), backend: tf.getBackend(), tensorflowVersion: tf.version.tfjs,
                    cohortSize: cohort.length, runs, meanScalarUserMs: meanMs,
                    illustrativeGridEstimateMs: meanMs * cohort.length * 4,
                    estimateNote: 'Extrapolation at alpha=0.85, not a measured grid time; alpha=0.95 can need more updates.',
                    tensorsBefore: before, tensorsAfter: tf.memory().numTensors, trainingRuns: HW6_MF_ENGINE.trainingRuns };
            });
            assert.equal(benchmark.tensorsBefore, benchmark.tensorsAfter);
            fs.writeFileSync(path.join(__dirname, 'alpha-benchmark.json'), JSON.stringify(benchmark, null, 2) + '\n');
            console.log('SCALAR_BENCHMARK', JSON.stringify(benchmark, null, 2));
            return;
        }
        if (!process.argv.includes('--retry-capped') && !process.argv.includes('--retry-cpu') && !verifySaved) {
            const verification = await page.evaluate(() => HW6_ALPHA_CHECKS.run());
            fs.writeFileSync(path.join(__dirname, 'alpha-batch-verification.json'), JSON.stringify(verification, null, 2) + '\n');
            console.log('BATCH_CHECKS_PASS', JSON.stringify(verification));
        }
        if (process.argv.includes('--checks-only')) return;
        let lastProgress = '';
        await page.exposeFunction('persistAlphaProgress', value => {
            fs.writeFileSync(path.join(__dirname, 'alpha-validation-progress.json'), JSON.stringify(value, null, 2) + '\n');
        });
        progress = setInterval(async () => {
            try {
                const value = await page.locator('#status').textContent();
                if (value !== lastProgress) { console.log('VALIDATION_PROGRESS', value); lastProgress = value; }
            } catch (_) { /* The final progress poll may race page closure. */ }
        }, 15000);
        const result = verifySaved ? await page.evaluate(async () => {
            const saved = await (await fetch('alpha-validation.json', { cache: 'no-store' })).json();
            await HW6_ALPHA_SELECTION.load(saved, 'verification without rerunning PPR');
            return saved;
        }) : process.argv.includes('--retry-cpu') ? await page.evaluate(async () => {
            const saved = await (await fetch('alpha-validation.json', { cache: 'no-store' })).json();
            return HW6_ALPHA_SELECTION.recoverCpu(saved);
        }) : process.argv.includes('--retry-capped') ? await page.evaluate(async () => {
            const saved = await (await fetch('alpha-validation.json', { cache: 'no-store' })).json();
            return HW6_ALPHA_SELECTION.recover(saved);
        }) : await page.evaluate(() => hw6RunAlphaSweep({ force: true, onProgress: value => window.persistAlphaProgress(value) }));
        clearInterval(progress); progress = null;
        const csv = await page.evaluate(result => HW6_ALPHA_SELECTION.csv(result), result);
        if (!verifySaved) {
            fs.writeFileSync(path.join(__dirname, 'alpha-validation.json'), JSON.stringify(result, null, 2) + '\n');
            fs.writeFileSync(path.join(__dirname, 'alpha-validation.csv'), csv.summary);
            fs.writeFileSync(path.join(__dirname, 'alpha-validation-users.csv'), csv.users);
            if (result.cpuVerification) fs.writeFileSync(path.join(__dirname, 'alpha-cpu-verification.json'), JSON.stringify(result.cpuVerification, null, 2) + '\n');
        }
        console.log('VALIDATION_STATUS', JSON.stringify({ status: result.status, error: result.error, selectedAlpha: result.selectedAlpha,
            rows: result.rows.map(row => ({ alpha: row.alpha, recall: row.recall, convergence: row.convergence })) }, null, 2));
        assert.equal(result.status, 'complete', result.error);
        assert.equal(result.cohortSize, 846);
        for (const row of result.rows) {
            const macro = row.perUser.reduce((sum, user) => sum + user.hits / user.relevant, 0) / 846;
            assert.equal(macro, row.recall);
        }
        await page.evaluate(result => hw6RenderAlphaSweep(result), result);
        const savedAlpha = result.selectedAlpha;
        const invalidResultRejected = await page.evaluate(async value => {
            const incomplete = structuredClone(value); incomplete.rows.pop();
            let rejected = false;
            try { await HW6_ALPHA_SELECTION.load(incomplete); } catch (_) { rejected = true; }
            if (!rejected || HW6_ALPHA_SELECTION.selectedAlpha !== null) throw new Error('Incomplete saved grid announced a winner');
            await HW6_ALPHA_SELECTION.load(value);
            return true;
        }, result);
        await page.evaluate(alpha => hw6SetAlpha(alpha === 0.50 ? 0.95 : 0.50), savedAlpha);
        assert.equal(await page.evaluate(() => HW6_ALPHA_SELECTION.comparisonAlpha()), savedAlpha);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => HW6_STATE.loaded);
        const restored = await page.evaluate(() => ({ alpha: HW6_ALPHA_SELECTION.selectedAlpha,
            source: HW6_ALPHA_SELECTION.result?.loadedFrom, trainingRuns: HW6_MF_ENGINE.trainingRuns }));
        assert.equal(restored.alpha, savedAlpha);
        assert.equal(restored.trainingRuns, 0);
        assert.match(await page.locator('#alpha-sweep-results').textContent(), /selected|Selected/);
        const audit = await page.evaluate(() => window.alphaAudit || { testRecallAttempts: 0, trainingAttempts: 0 });
        assert.equal(audit.testRecallAttempts, 0); assert.equal(audit.trainingAttempts, 0);
        fs.writeFileSync(path.join(__dirname, 'alpha-result-verification.json'), JSON.stringify({
            generatedAt: new Date().toISOString(), status: 'passed', cohortSize: result.cohortSize,
            macroRecomputedFromAllUsers: true, incompleteSavedGridRejected: invalidResultRejected,
            sliderDoesNotChangeWinner: true, restoredAlpha: restored.alpha, audit,
            finalBackendCounts: result.rows.map(row => ({ alpha: row.alpha, users: row.perUser.length,
                cpuUsers: row.perUser.filter(user => user.backend === 'cpu').length,
                failedAttemptsRetained: row.perUser.reduce((n, user) => n + (user.attempts || []).filter(attempt => !attempt.converged).length, 0) })),
            pageErrors: errors,
        }, null, 2) + '\n');
        assert.deepEqual(errors, []);
        console.log('VALIDATION_GRID_PASS', JSON.stringify({ rows: result.rows.map(({ alpha, recall, runtimeMs, convergence }) =>
            ({ alpha, recall, runtimeMs, convergence })), selectedAlpha: savedAlpha, cohortSize: result.cohortSize,
            runtimeMs: result.runtimeMs, restored, invalidResultRejected, pageErrors: errors }, null, 2));
    } finally {
        if (progress) clearInterval(progress);
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(hashes(), beforeHashes, 'Data/saved MF must remain unchanged');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
