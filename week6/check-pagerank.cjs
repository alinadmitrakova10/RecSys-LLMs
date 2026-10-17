/**
 * Run: node week6/check-pagerank.cjs
 * Uses the existing Playwright installation (or PLAYWRIGHT_MODULE) and a local
 * HTTP server. Requires access to the TF.js CDN used by index.html. No training,
 * retrieval experiments, screenshots, downloads or result files are produced.
 */
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');

const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/favicon.ico') { response.writeHead(204); response.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { response.writeHead(403); response.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { response.writeHead(404); response.end(); return; }
        const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8' };
        response.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
        response.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage();
        page.setDefaultTimeout(120000);
        const pageErrors = [];
        const failedRequests = [];
        page.on('pageerror', error => pageErrors.push(error.message));
        page.on('requestfailed', request => failedRequests.push(request.url() + ': ' + request.failure().errorText));
        const url = `http://127.0.0.1:${server.address().port}`;
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);

        // Exercise real solver functions first on small graphs, not the teaching demo.
        const small = await page.evaluate(async () => {
            const ensure = (condition, message) => { if (!condition) throw new Error(message); };
            const close = (a, b, tolerance = 1e-6) => ensure(Math.abs(a - b) < tolerance, `${a} != ${b}`);
            const vector = (actual, expected) => {
                ensure(actual.length === expected.length, 'vector length mismatch');
                expected.forEach((value, i) => close(actual[i], value));
            };
            const probability = result => {
                close(Array.from(result.p).reduce((sum, value) => sum + value, 0), 1);
                ensure(Array.from(result.p).every(value => value >= 0), 'negative probability');
                for (const step of result.trace) {
                    close(step.mass, 1);
                    ensure(step.minimum >= 0, 'negative intermediate probability');
                }
                ensure(result.converged === (result.residual < 1e-6), 'invalid convergence verdict');
            };
            const reject = async (fn, pattern) => {
                let message = '';
                try { await fn(); } catch (error) { message = error.message; }
                ensure(pattern.test(message), 'missing expected error: ' + pattern + '; got ' + message);
            };
            const brief = result => ({ p: Array.from(result.p), iterations: result.iterations,
                residual: result.residual, converged: result.converged,
                mass: Array.from(result.p).reduce((sum, value) => sum + value, 0),
                maxMassError: Math.max(...result.trace.map(step => Math.abs(step.mass - 1))) });

            await tf.ready();
            const originalDemo = graphFixturePropagate;
            graphFixturePropagate = () => { throw new Error('The teaching demo must not be used'); };
            let uniform, seeded;
            try {
                uniform = await pageRank(0.85, { transition: REFERENCE_GRAPH.transition, trace: true });
                seeded = await personalizedPageRank(0, 0.85, { transition: REFERENCE_GRAPH.transition, trace: true });
            } finally { graphFixturePropagate = originalDemo; }
            vector(uniform.p, [0.3725268513, 0.1958239118, 0.3941492369, 0.0375]);
            vector(seeded.p, [0.4522328999, 0.1921989825, 0.3555681176, 0]);
            probability(uniform); probability(seeded);
            ensure(uniform.converged && seeded.converged, 'reference did not converge');

            // Directed one-step result distinguishes Q^T p from Q p, alpha from restart.
            const directed = { Q: [[0, 1], [0, 1]], danglingMask: [0, 0] };
            const oneStep = await personalizedPageRank(0, 0.7, { transition: directed, maxIter: 1, trace: true });
            vector(oneStep.p, [0.3, 0.7]);
            ensure(!oneStep.converged && oneStep.iterations === 1, 'cap is not convergence');

            const star = hw6MakeSyntheticGraph([[1, 1, 0], [1, 0, 0]]);
            const beforeBlocks = tf.memory().numTensors;
            const blocks = rowNormalize(star);
            try {
                ensure(!('Q' in blocks), 'normalization must not allocate square Q');
                vector((await blocks.userToMovie.array())[0], [0.5, 0.5, 0]);
                vector((await blocks.userToMovie.array())[1], [1, 0, 0]);
                vector((await blocks.movieToUser.array())[0], [0.5, 0.5]);
                vector((await blocks.movieToUser.array())[2], [0, 0]);
                ensure(JSON.stringify(blocks.danglingMask) === '[false,false,false,false,true]', 'wrong mask');
            } finally { blocks.dispose(); }
            ensure(tf.memory().numTensors === beforeBlocks, 'normalization leaked tensors');

            // An independent dense, double-precision equation check for a small graph.
            const normalized = await hw6NormalizedFromGraph(star);
            const global = await pageRank(0.85, { graph: star, trace: true });
            probability(global);
            const next = hw6Step(0.85, normalized, hw6UniformRestart(star.nodeCount), global.p);
            ensure(next.reduce((sum, value, i) => sum + Math.abs(value - global.p[i]), 0) < 1e-6, 'not stationary');

            const isolated = hw6MakeSyntheticGraph([[1, 1, 0], [0, 0, 0]]);
            const danglingGlobal = await pageRank(0.85, { graph: isolated, trace: true });
            const danglingPPR = await personalizedPageRank(1, 0.85, { graph: isolated, trace: true });
            probability(danglingGlobal); probability(danglingPPR);
            vector(danglingPPR.p, [0, 1, 0, 0, 0]);
            const allZero = await pageRank(0.85, { graph: hw6MakeSyntheticGraph([[0]]), trace: true });
            vector(allZero.p, [0.5, 0.5]); probability(allZero);

            // Raw IDs 42/700 are deliberately unrelated to their node indices 0/1.
            const sparse = buildGraph([{ userId: 42, movieId: 80, rating: 4 }, { userId: 700, movieId: 90, rating: 5 }]);
            const correctUser = await hw6RunPPR(sparse, 700, 0);
            vector(correctUser.scores, [0, 1, 0, 0]);
            await reject(() => hw6RunPPR(sparse, 1, 0.85), /cannot map user/);
            await reject(() => hw6AdapterPPR(sparse, 0, 0.85, 700), /does not match/);
            await reject(() => personalizedPageRank(700, 0.85, { graph: sparse }), /restart node index/);
            await reject(() => personalizedPageRank(2, 0.85, { graph: sparse }), /restart node index/);
            await reject(() => pageRank(NaN, { graph: sparse }), /alpha/);
            await reject(() => pageRank(0.85, { graph: sparse, tol: 0 }), /tol/);
            await reject(() => pageRank(0.85, { graph: sparse, maxIter: 0 }), /maxIter/);
            await reject(() => pageRank(0.85, { transition: { Q: [[0]], danglingMask: [0] } }), /does not sum/);
            await reject(() => pageRank(0.85, { graph: buildGraph([]) }), /nonempty graph/);

            // Bipartite periodic walk with alpha=1 must reach 500 without convergence.
            const periodic = await personalizedPageRank(0, 1, { graph: hw6MakeSyntheticGraph([[1]]), trace: true });
            ensure(periodic.iterations === 500 && !periodic.converged && periodic.residual === 2, 'periodic cap verdict');
            probability(periodic);
            const restartOnly = await personalizedPageRank(0, 0, { graph: sparse, maxIter: 1 });
            ensure(restartOnly.iterations === 1 && restartOnly.converged && restartOnly.residual === 0, 'last update must use measured residual');
            ensure(hw6ConvergenceVerdict({ iterations: 1, residual: null, converged: null }).value === null, 'iterations alone are not convergence');

            // Fail after tensor allocation, not just during input validation.
            const beforeFailure = tf.memory().numTensors;
            const originalData = tf.Tensor.prototype.data;
            try {
                tf.Tensor.prototype.data = async function () { throw new Error('injected asynchronous tensor read failure'); };
                await reject(() => pageRank(0.85, { graph: isolated }), /injected asynchronous tensor read failure/);
            } finally { tf.Tensor.prototype.data = originalData; }
            ensure(tf.memory().numTensors === beforeFailure, 'failed asynchronous read leaked tensors');

            const memoryBefore = tf.memory();
            for (let i = 0; i < 5; i++) {
                await pageRank(0.85, { transition: REFERENCE_GRAPH.transition });
                await personalizedPageRank(0, 0.85, { graph: isolated });
            }
            const memoryAfter = tf.memory();
            ensure(memoryBefore.numTensors === memoryAfter.numTensors && memoryBefore.numDataBuffers === memoryAfter.numDataBuffers,
                'repeated propagation leaked tensors/data buffers');
            return { backend: tf.getBackend(), uniform: brief(uniform), restartAtA: brief(seeded),
                directedOneStep: brief(oneStep), danglingGlobal: brief(danglingGlobal), danglingSeed: brief(danglingPPR),
                periodic: { iterations: periodic.iterations, residual: periodic.residual, converged: periodic.converged },
                memory: { before: memoryBefore.numTensors, after: memoryAfter.numTensors, buffersBefore: memoryBefore.numDataBuffers, buffersAfter: memoryAfter.numDataBuffers } };
        });
        console.log('SMALL_GRAPHS_PASS', JSON.stringify(small, null, 2));

        // Use the real buttons on MovieLens; no recommendations/MF/sweeps are run.
        await page.locator('#build-graph').click();
        await page.selectOption('#method', 'pagerank');
        await page.locator('#run-pagerank').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.convergence.pagerank);
        await page.selectOption('#seed-user-select', '1');
        await page.selectOption('#method', 'ppr');
        await page.locator('#run-ppr').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.convergence.ppr);
        const movieLens = await page.evaluate(async () => {
            const g = getGraph();
            const brief = result => ({ iterations: result.iterations, residual: result.residual,
                converged: result.converged, backend: result.backend, runtimeMs: result.runtimeMs,
                mass: Array.from(result.scores).reduce((sum, x) => sum + x, 0),
                minimum: Math.min(...result.scores) });
            const result = { nodes: g.nodeCount, userId: 1, seedNode: g.nodeOfUser(1),
                pagerank: brief(HW6_STATE.pageRank), ppr: brief(HW6_STATE.ppr),
                seedProbability: HW6_STATE.ppr.scores[g.nodeOfUser(1)] };
            const blocks = rowNormalize(g);
            try { result.blockShapes = [blocks.userToMovie.shape, blocks.movieToUser.shape]; }
            finally { blocks.dispose(); }
            const before = tf.memory();
            for (let i = 0; i < 2; i++) {
                await pageRank(0.85, { graph: g });
                await personalizedPageRank(g.nodeOfUser(1), 0.85, { graph: g });
            }
            const after = tf.memory();
            result.memory = { before: before.numTensors, after: after.numTensors,
                buffersBefore: before.numDataBuffers, buffersAfter: after.numDataBuffers };
            return result;
        });
        assert.equal(movieLens.nodes, 2554);
        assert.equal(movieLens.seedNode, 0);
        for (const method of ['pagerank', 'ppr']) {
            assert.equal(movieLens[method].converged, true);
            assert.ok(movieLens[method].residual < 1e-6);
            assert.ok(Math.abs(movieLens[method].mass - 1) < 1e-5);
            assert.ok(movieLens[method].minimum >= 0);
        }
        assert.deepEqual(movieLens.blockShapes, [[943, 1611], [1611, 943]]);
        assert.equal(movieLens.memory.before, movieLens.memory.after);
        assert.equal(movieLens.memory.buffersBefore, movieLens.memory.buffersAfter);
        assert.match(await page.locator('#convergence-results').textContent(), /runtime/);
        assert.ok(await page.locator('#graph-view g.graph-node').count() > 0);
        console.log('MOVIELENS_BUTTONS_PASS', JSON.stringify(movieLens, null, 2));

        await page.locator('#run-tests').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.lastVerification);
        const checks = await page.evaluate(() => HW6_STATE.lastVerification);
        assert.equal(checks.failed, 0, JSON.stringify(checks.checks));
        assert.equal(checks.passed, 12);
        assert.equal(checks.pending, 0);
        console.log('IN_PAGE_CHECKS_PASS', JSON.stringify(checks));

        // Deliberately break the interface: the harness and buttons must expose errors.
        await page.evaluate(() => {
            window.originalPageRankForCheck = pageRank;
            pageRank = async () => { throw new Error('HW6_INTERFACE: injected propagation failure'); };
            hw6InvalidateCaches();
        });
        await page.locator('#run-pagerank').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && document.getElementById('status').textContent.includes('injected propagation failure'));
        assert.match(await page.locator('#convergence-results').textContent(), /injected propagation failure/);
        const failedChecks = await page.evaluate(() => runSelfChecks());
        assert.ok(failedChecks.failed > 0);
        assert.equal(failedChecks.pending, 0);
        await page.evaluate(() => { pageRank = window.originalPageRankForCheck; delete window.originalPageRankForCheck; });
        // A malformed output must also fail instead of being accepted or retried.
        const malformed = await page.evaluate(async () => {
            const original = pageRank;
            pageRank = async () => ({ p: [1], iterations: 1, residual: 0, converged: true });
            try { await hw6AdapterPageRank(getGraph(), 0.85); return false; }
            catch (error) { return error.message.includes('invalid probabilities'); }
            finally { pageRank = original; }
        });
        assert.equal(malformed, true);
        await page.locator('#run-tests').click();
        await page.waitForFunction(() => !HW6_STATE.propagationBusy && HW6_STATE.lastVerification.failed === 0);
        assert.equal(await page.evaluate(() => HW6_MF_ENGINE.trainingRuns), 0);
        assert.deepEqual(pageErrors, []);
        assert.deepEqual(failedRequests, []);
        console.log('BROWSER_PASS', JSON.stringify({ url, browser: browser.version(), pageErrors, failedRequests,
            injectedErrors: 'reported as failures, not PENDING', mfTrained: false }));
    } finally {
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
