/**
 * node week6/check-mf.cjs --fixtures-only : small browser/Git parity checks.
 * node week6/check-mf.cjs --train-once    : one fixed full run if no artifact exists.
 * node week6/check-mf.cjs                 : verify/reload the existing artifact.
 * Generated mf-checkpoint.json/mf-run.json are written only inside week6.
 * Playwright is reused as a dependency, never as a source of the MF method.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../week5/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const source = execFileSync('git', ['show', '40a20cc:week5/script.js'], { cwd: root, encoding: 'utf8' });
const unchangedPaths = ['week6/graph.js', 'week6/u.data', 'week6/u.item', 'week6/check-pagerank.cjs'];
const beforeHashes = execFileSync('git', ['hash-object', ...unchangedPaths], { cwd: root, encoding: 'utf8' });
const checkpointPath = path.join(__dirname, 'mf-checkpoint.json');
const runPath = path.join(__dirname, 'mf-run.json');
const artifactExisted = fs.existsSync(checkpointPath);
const fixturesOnly = process.argv.includes('--fixtures-only');
const allowTraining = process.argv.includes('--train-once');
if (!fixturesOnly && !artifactExisted && !allowTraining) throw new Error('No checkpoint; use --train-once to authorize the first full run');

const server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    // MF-focused fixtures must not preload a later validation experiment/model.
    if (pathname === '/alpha-validation.json') { res.writeHead(404); res.end(); return; }
    if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(__dirname, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(__dirname + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (error, body) => {
        if (error) { res.writeHead(404); res.end(); return; }
        const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.css': 'text/css' };
        res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
        res.end(body);
    });
});

async function main() {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const url = `http://127.0.0.1:${server.address().port}`;
    let browser;
    let progress;
    const errors = [];
    const open = async () => {
        const page = await browser.newPage();
        page.setDefaultTimeout(120000);
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => typeof HW6_STATE !== 'undefined' && HW6_STATE.loaded);
        return page;
    };
    try {
        browser = await chromium.launch({ headless: true });
        let fixtures = { status: 'not rerun in checkpoint-load-only mode' };
        if (!process.argv.includes('--load-only')) {
        const fixturePage = await open();
        fixtures = await fixturePage.evaluate(async oldSource => {
            const ensure = (ok, message) => { if (!ok) throw new Error(message); };
            const close = (a, b, tol = 1e-6) => ensure(Math.abs(a - b) <= tol, `${a} != ${b}`);
            const arrayClose = (a, b) => { ensure(a.length === b.length, 'length mismatch'); a.forEach((x, i) => close(x, b[i])); };
            const reject = async (fn, pattern) => {
                let message = '';
                try { await fn(); } catch (error) { message = error.message; }
                ensure(pattern.test(message), `expected ${pattern}, received ${message}`);
            };
            const dispose = model => tf.dispose([model.userFactors, model.itemFactors]);
            const trackedOptimizers = [];
            const legacyTf = { ...tf, train: { ...tf.train, adam(...args) {
                const optimizer = tf.train.adam(...args); trackedOptimizers.push(optimizer); return optimizer;
            } } };
            const legacyMath = Object.create(Math);
            legacyMath.random = hw6Mulberry32(MF_SEED + 2);
            // The exact Git object runs in a local scope, with inert UI/bootstrap.
            const legacy = new Function('tf', 'Math', 'numUsers', 'movies',
                'const window = {}; const document = {getElementById:()=>null};\n' + oldSource +
                '\nreturn {predictRating,computeRMSEFor,trainMF,selectBestCheckpoint,setModel:x=>model=x};'
            )(legacyTf, legacyMath, 2, [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }]);

            const analytic = tf.tidy(() => ({
                userFactors: tf.variable(tf.tensor2d([[0, 0], [1, 2], [3, 4]])),
                itemFactors: tf.variable(tf.tensor2d([[0, 0], [3, 4], [1, -1]])), globalMean: 3,
            }));
            legacy.setModel(analytic);
            close(HW6_MF_ENGINE.scores(1, [1, 2], analytic)[1], 14);
            close(HW6_MF_ENGINE.predictRating(1, 1, analytic), 5);
            close(HW6_MF_ENGINE.predictRating(999, 1, analytic), legacy.predictRating(999, 1));
            const evalRows = [{ userId: 1, movieId: 1, rating: 4 }, { userId: 1, movieId: 2, rating: 1 }];
            close(HW6_MF_ENGINE.rmse(evalRows, analytic), 1);
            close(HW6_MF_ENGINE.rmse(evalRows, analytic), legacy.computeRMSEFor(evalRows));
            const loss = tf.tidy(() => HW6_MF_ENGINE.batchLoss(analytic, tf.tensor1d([1, 1], 'int32'),
                tf.tensor1d([1, 2], 'int32'), tf.tensor1d([5, 1]), 0.05).dataSync()[0]);
            close(loss, 41.4625, 1e-5);
            dispose(analytic);
            const history = [{ epoch: 3, valRMSE: 0.4 }, { epoch: 2, valRMSE: 0.3 }, { epoch: 1, valRMSE: 0.3 }];
            ensure(HW6_MF_ENGINE.selectBest(history).epoch === 1, 'earliest checkpoint tie');
            ensure(HW6_MF_ENGINE.selectBest(history).epoch === legacy.selectBestCheckpoint(history).epoch, 'HW5 checkpoint mismatch');
            ensure(HW6_MF_ENGINE.selectBest([]) === null, 'empty checkpoint history');

            const train = [
                { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
                { userId: 1, movieId: 3, rating: 5, timestamp: 2 },
                { userId: 2, movieId: 2, rating: 1, timestamp: 3 },
                { userId: 2, movieId: 4, rating: 1, timestamp: 4 },
            ];
            const val = train.map(row => ({ ...row, rating: 6 - row.rating, timestamp: row.timestamp + 4 }));
            const seededA = HW6_MF_ENGINE.initialize(train, 3, 5);
            const seededB = HW6_MF_ENGINE.initialize(train, 3, 5);
            arrayClose(Array.from(await seededA.userFactors.data()), Array.from(await seededB.userFactors.data()));
            arrayClose(Array.from(await seededA.itemFactors.data()), Array.from(await seededB.itemFactors.data()));
            dispose(seededA); dispose(seededB);
            const make = () => tf.tidy(() => {
                const u = new Float32Array(3 * 16); u[16] = 0.5; u[32] = 0.5;
                const i = new Float32Array(5 * 16); i[16] = 0.5; i[32] = -0.5; i[48] = 0.5; i[64] = -0.5;
                return { userFactors: tf.variable(tf.tensor2d(u, [3, 16])),
                    itemFactors: tf.variable(tf.tensor2d(i, [5, 16])), globalMean: 3 };
            });
            const oldModel = make(); const newModel = make();
            legacy.setModel(oldModel);
            const config = { epochs: 3, lr: 0.01, reg: 0.05, batchSize: 2 };
            const oldHistory = legacy.trainMF(train, val, config);
            const snapshots = [];
            const fitted = await HW6_MF_ENGINE.fit(newModel, train, val, config, MF_SEED + 2, () => {
                snapshots.push(Array.from(newModel.userFactors.dataSync()));
            });
            oldHistory.forEach((row, i) => { close(row.valRMSE, fitted.history[i].valRMSE); close(row.trainRMSE, fitted.history[i].trainRMSE); });
            arrayClose(Array.from(await oldModel.userFactors.data()), Array.from(await newModel.userFactors.data()));
            arrayClose(Array.from(await oldModel.itemFactors.data()), Array.from(await newModel.itemFactors.data()));
            ensure(fitted.best.epoch < 3, 'fixture must distinguish best and final epoch');
            arrayClose(Array.from(await newModel.userFactors.data()), snapshots[fitted.best.epoch - 1]);
            ensure(snapshots[2].some((value, i) => Math.abs(value - snapshots[fitted.best.epoch - 1][i]) > 1e-6), 'fixture did not move after best epoch');
            dispose(oldModel); dispose(newModel); trackedOptimizers.forEach(optimizer => optimizer.dispose());

            // A real 20-epoch tiny fit supplies the persistence fixture's history.
            // It is not a MovieLens experiment and is never written to the artifact.
            movies = [1, 2, 3, 4].map(id => ({ id, title: 'Fixture ' + id, genres: [] }));
            ratings = [...train, ...val]; numUsers = 2; HW6_MAX_MOVIE_ID = 4;
            Object.assign(HW6_SPLIT, { train, val, test: val });
            hw6BuildSplitIndices(HW6_SPLIT);
            HW6_MF_MODEL = make();
            const tiny = await HW6_MF_ENGINE.fit(HW6_MF_MODEL, train, val);
            const data = await HW6_MF_ENGINE.describeData();
            Object.assign(MF_INFO, { checkpoint: tiny.best.epoch, validationRMSE: tiny.best.valRMSE,
                testRMSE: HW6_MF_ENGINE.rmse(val, HW6_MF_MODEL), history: tiny.history,
                trainedAt: new Date().toISOString(), datasetSignature: data.signature, runId: 'fixture-only',
                backend: tf.getBackend(), runKind: HW6_MF_ENGINE.spec().runKind });
            const scoreBefore = mfScores(1);
            const payload = await HW6_MF_ENGINE.save('fixture-only');
            hw6DisposeMf();
            const saved = await HW6_MF_ENGINE.storage('get', 'fixture-only');
            await HW6_MF_ENGINE.load(saved, 'fixture IndexedDB');
            arrayClose(Object.values(mfScores(1)), Object.values(scoreBefore));
            ensure(MF_INFO.loadVerification.maxScoreError === 0, 'same-backend roundtrip changed scores');
            const memoryBefore = tf.memory().numTensors;
            const corrupt = structuredClone(payload); corrupt.weights.userValues[0] += 1;
            await reject(() => HW6_MF_ENGINE.load(corrupt), /checksum/);
            const wrongConfig = structuredClone(payload); wrongConfig.spec.lr = 0.02;
            await reject(() => HW6_MF_ENGINE.load(wrongConfig), /settings/);
            const originalTest = HW6_SPLIT.test;
            HW6_SPLIT.test = [{ ...val[0], timestamp: 999 }];
            await reject(() => HW6_MF_ENGINE.load(payload), /data\/split\/candidates/);
            HW6_SPLIT.test = originalTest;
            ensure(tf.memory().numTensors === memoryBefore, 'rejected load leaked tensors');
            const existing = HW6_MF_MODEL;
            const repeated = await Promise.all([hw6EnsureMf(), hw6EnsureMf(), hw6EnsureMf()]);
            ensure(repeated.every(model => model === existing), 'repeated calls replaced the loaded model');
            ensure(HW6_MF_ENGINE.trainingRuns === 0, 'fixture restore triggered full training');
            await HW6_MF_ENGINE.storage('delete', 'fixture-only');
            hw6DisposeMf();
            ensure(tf.memory().numTensors === 0, 'fixture leaked tensors');
            return { analyticRawScore: 14, analyticClippedPrediction: 5, analyticRMSE: 1,
                batchLoss: loss, parityWithGit40a20cc: true, tinyBestEpoch: fitted.best.epoch,
                checkpointRestored: true, initializationSeeded: true, indexedDBRoundtripMaxScoreError: 0,
                corruptionAndIncompatibleDataRejected: true, fullTrainingRuns: HW6_MF_ENGINE.trainingRuns };
        }, source);
        console.log('MF_FIXTURES_PASS', JSON.stringify(fixtures, null, 2));
        await fixturePage.close();
        if (fixturesOnly) return;
        }

        const page = await open();
        let lastEpoch = -1;
        progress = setInterval(async () => {
            try {
                const status = await page.evaluate(() => ({ epoch: MF_INFO.history.length, status: document.getElementById('status').textContent }));
                if (status.epoch !== lastEpoch) { console.log('MF_PROGRESS', status.status); lastEpoch = status.epoch; }
            } catch (_) { /* Page closure may race the last progress poll. */ }
        }, 15000);
        const trained = await page.evaluate(async () => {
            const a = hw6EnsureMf(); const b = hw6EnsureMf();
            const [first, second] = await Promise.all([a, b]);
            if (first !== second) throw new Error('Concurrent MF callers did not share the model');
            return { info: JSON.parse(JSON.stringify(MF_INFO)), trainingRuns: HW6_MF_ENGINE.trainingRuns };
        });
        clearInterval(progress); progress = null;
        assert.equal(trained.trainingRuns, artifactExisted ? 0 : 1);
        assert.equal(trained.info.history.length, 20);
        const payload = await page.evaluate(() => HW6_MF_ENGINE.pack());
        // Retain the selected factors before any subsequent verification/reload.
        if (!artifactExisted) fs.writeFileSync(checkpointPath, JSON.stringify(payload) + '\n', { flag: 'wx' });

        const coverage = await page.evaluate(async () => {
            const model = HW6_MF_MODEL;
            const finite = tf.tidy(() => {
                const users = tf.gather(model.userFactors, tf.tensor1d(HW6_STATE.trainUserIds, 'int32'));
                const items = tf.gather(model.itemFactors, tf.tensor1d(HW6_STATE.candidateIds, 'int32'));
                return tf.matMul(users, items, false, true).add(model.globalMean).isFinite().all();
            });
            try { return { users: HW6_STATE.trainUserIds.length, candidates: HW6_STATE.candidateIds.length, finite: Boolean((await finite.data())[0]) }; }
            finally { finite.dispose(); }
        });
        assert.deepEqual(coverage, { users: 943, candidates: 1611, finite: true });
        const scoresBefore = await page.evaluate(() => [1, 2, 943].map(userId => Object.values(mfScores(userId))));
        await page.evaluate(async payload => { await HW6_MF_ENGINE.load(payload, 'JSON roundtrip'); }, payload);
        const scoresAfter = await page.evaluate(() => [1, 2, 943].map(userId => Object.values(mfScores(userId))));
        assert.deepEqual(scoresAfter, scoresBefore);
        const loadVerification = await page.evaluate(() => MF_INFO.loadVerification);
        assert.equal(loadVerification.maxScoreError, 0);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => HW6_STATE.loaded);
        await page.selectOption('#method', 'mf');
        const reload = await page.evaluate(async () => {
            await Promise.all([hw6EnsureMf(), hw6EnsureMf()]);
            return { loadedFrom: MF_INFO.loadedFrom, trainingRuns: HW6_MF_ENGINE.trainingRuns,
                runId: MF_INFO.runId, verification: MF_INFO.loadVerification, tensors: tf.memory().numTensors };
        });
        assert.equal(reload.loadedFrom, 'IndexedDB'); assert.equal(reload.trainingRuns, 0);
        assert.equal(reload.runId, trained.info.runId); assert.equal(reload.tensors, 2);
        assert.deepEqual(await page.evaluate(() => [1, 2, 943].map(userId => Object.values(mfScores(userId)))), scoresBefore);
        await page.close();

        // A fresh context has no IndexedDB cache: restore the portable artifact.
        const fresh = await open();
        const portable = await fresh.evaluate(async () => {
            await hw6EnsureMf();
            return { loadedFrom: MF_INFO.loadedFrom, trainingRuns: HW6_MF_ENGINE.trainingRuns,
                runId: MF_INFO.runId, verification: MF_INFO.loadVerification };
        });
        assert.equal(portable.loadedFrom, 'mf-checkpoint.json'); assert.equal(portable.trainingRuns, 0);
        assert.equal(portable.runId, trained.info.runId);
        const beforeTensorCount = await fresh.evaluate(() => tf.memory().numTensors);
        await fresh.evaluate(async () => { for (let i = 0; i < 3; i++) { await hw6EnsureMf(); mfScores(1); } });
        assert.equal(await fresh.evaluate(() => tf.memory().numTensors), beforeTensorCount);
        const report = { ...trained, spec: payload.spec, data: payload.data, checksum: payload.checksum,
            coverage, fixtures, loadVerification, reload, portable, browser: browser.version(),
            noTestBasedSelection: true, oldRecallNotReused: true };
        if (!artifactExisted) fs.writeFileSync(runPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
        assert.deepEqual(errors, []);
        console.log('MF_RUN_PASS', JSON.stringify(report, null, 2));
        await fresh.close();
    } finally {
        if (progress) clearInterval(progress);
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
        assert.equal(execFileSync('git', ['hash-object', ...unchangedPaths], { cwd: root, encoding: 'utf8' }), beforeHashes,
            'PageRank/data files must remain unchanged');
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
