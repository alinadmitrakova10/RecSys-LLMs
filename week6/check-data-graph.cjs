/**
 * Data/graph regression checks. Run: node week6/check-data-graph.cjs
 * Uses Node built-ins only; performs no training, propagation or file writes.
 * Reads the final HW5 data layer directly from its Git branch for split parity.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');

const root = path.resolve(__dirname, '..');
const bytes = name => fs.readFileSync(path.join(__dirname, name));
const hash = value => createHash('sha256').update(value).digest('hex');
const initialHashes = Object.fromEntries(['u.item', 'u.data'].map(name => [name, hash(bytes(name))]));
const plain = value => JSON.parse(JSON.stringify(value));
const summary = { innerHTML: '' };
const responses = new Map(['u.item', 'u.data'].map(name => [name, bytes(name)]));
const context = {
    console: { log() {}, error() {} }, TextDecoder,
    document: {
        readyState: 'loading', addEventListener() {},
        getElementById(id) { return id === 'dataset-summary-body' ? summary : null; },
    },
    async fetch(name) {
        const buffer = responses.get(name);
        return {
            ok: true,
            async arrayBuffer() { return Uint8Array.from(buffer).buffer; },
            async text() { return buffer.toString('utf8'); },
        };
    },
};
context.window = context;
vm.createContext(context);
for (const name of ['data.js', 'graph.js', 'script.js']) {
    vm.runInContext(bytes(name).toString('utf8'), context, { filename: name });
}
const run = code => vm.runInContext(code, context);
const itemRow = (id, title, flags = {}) =>
    [id, title, '', '', '', ...Array.from({ length: 19 }, (_, i) => flags[i] || 0)].join('|');

async function main() {
    // Metadata: unknown does not shift Action/Western, and missing dates are valid.
    context.itemFixture = [itemRow(42, 'Café', { 0: 1 }), itemRow(7, 'Other', { 1: 1, 18: 1 })].join('\n');
    run('parseItemData(itemFixture); parseItemData(itemFixture)');
    assert.deepEqual(plain(run('movies')), [
        { id: 42, title: 'Café', genres: ['unknown'] },
        { id: 7, title: 'Other', genres: ['Action', 'Western'] },
    ]);
    context.badItems = context.itemFixture + '\n' + itemRow(42, 'Duplicate');
    assert.throws(() => run('parseItemData(badItems)'), /u.item:.*line 3: duplicate movie ID 42/);
    assert.equal(run('movies.length'), 2);
    context.badItems = itemRow(50, 'Bad flags', { 2: 2 });
    assert.throws(() => run('parseItemData(badItems)'), /genre flags/);
    assert.throws(() => run("parseItemData('1|incomplete')"), /exactly 24 fields/);

    context.ratingFixture = '10\t42\t4\t100\n30\t7\t1\t101\n';
    run('parseRatingData(ratingFixture); parseRatingData(ratingFixture)');
    assert.equal(run('ratings.length'), 2);
    for (const [text, error] of [
        [context.ratingFixture + '10\t42\t5\t102', /line 3: duplicate pair 10:42/],
        ['10\t999\t4\t100', /unknown movie ID/],
        ['10x\t42\t4\t100', /positive integer/],
        ['10\t42\t6\t100', /rating must be/],
        ['10\t42\t0\t100', /positive integer/],
        ['10\t42\t4\tNaN', /positive integer/],
        ['10\t42\t4', /exactly 4 fields/],
        ['10\t42\t4\t100\n\n', /blank record/],
    ]) {
        context.badRatings = text;
        assert.throws(() => run('parseRatingData(badRatings)'), error);
        assert.equal(run('ratings.length'), 2, 'failed parses must not partially replace ratings');
    }

    // User/movie IDs have separate namespaces; low ratings still create nodes.
    context.trainFixture = [
        { userId: 42, movieId: 42, rating: 4 },
        { userId: 10, movieId: 7, rating: 5 },
        { userId: 42, movieId: 7, rating: 2 },
        { userId: 99, movieId: 200, rating: 1 },
    ];
    const graph = run('buildGraph(trainFixture)');
    assert.deepEqual(plain(graph.userIds), [10, 42, 99]);
    assert.deepEqual(plain(graph.movieIds), [7, 42, 200]);
    assert.equal(graph.nodeCount, 6);
    assert.equal(graph.nodeOfUser(42), 1);
    assert.equal(graph.movieOfNode(4), 42);
    assert.equal(graph.nodeOfUser(0), -1);
    assert.equal(graph.isUserNode(-1), false);
    assert.equal(graph.movieOfNode(0), -1);
    assert.deepEqual(plain(graph.R), [[1, 0, 0], [0, 1, 0], [0, 0, 0]]);
    assert.deepEqual(Array.from(graph.degree), [1, 1, 0, 1, 1, 0]);
    assert.deepEqual(plain(graph.zeroDegreeUsers), [99]);
    assert.deepEqual(plain(graph.zeroDegreeMovies), [200]);
    assert.equal(graph.positiveEdges, 2);
    assert.throws(() => run('buildGraph([...trainFixture, trainFixture[0]])'), /duplicate pair/);
    assert.throws(() => run('buildGraph([{userId:0,movieId:7,rating:4}])'), /invalid training/);
    assert.equal(run('buildGraph(null).nodeCount'), 0);
    assert.equal(run('buildGraph([{userId:9,movieId:8,rating:3}]).zeroDegreeMovies[0]'), 8);

    // Rebuilding split indices must remove every previous split's stale entries.
    run('hw6BuildSplitIndices({train:trainFixture,val:[{userId:42,movieId:999,rating:5}],test:[{userId:99,movieId:888,rating:4}]})');
    run('setGraph(buildGraph(trainFixture))');
    run('hw6BuildSplitIndices({train:[{userId:3,movieId:4,rating:4}],val:[],test:[]})');
    assert.deepEqual(plain(run('HW6_STATE.candidateIds')), [4]);
    assert.equal(run('HW6_STATE.pairsBySplit.train.has(42)'), false);
    assert.equal(run('HW6_STATE.gradesBySplit.val.size + HW6_STATE.gradesBySplit.test.size'), 0);
    assert.equal(run('HW6_STATE.popularity.has(7)'), false);
    assert.equal(run('getGraph()'), null);
    run('hw6BuildSplitIndices({train:[],val:[],test:[]})');
    assert.equal(run('HW6_STATE.candidateIds.length + HW6_STATE.popularity.size'), 0);

    // With sparse IDs, two observations in a 2 x 2 universe mean 50% sparsity.
    run('hw6BuildSplitIndices({train:ratings,val:[],test:[]}); hw6RenderDatasetSummary()');
    assert.match(summary.innerHTML, /50\.00%/);

    // Timestamp ties are resolved by movie ID, with floor counts and a remainder.
    context.tiedRatings = Array.from({ length: 11 }, (_, i) => ({
        userId: 10, movieId: 200 - i * 3, rating: 4, timestamp: 100,
    }));
    const tiedSplit = run('ratings = tiedRatings; splitByTimestamp()');
    assert.deepEqual(['train', 'val', 'test'].map(name => tiedSplit[name].length), [8, 1, 2]);
    assert.deepEqual(plain(tiedSplit.train.map(row => row.movieId)), [170, 173, 176, 179, 182, 185, 188, 191]);
    assert.equal(tiedSplit.val[0].movieId, 194);
    assert.deepEqual(plain(tiedSplit.test.map(row => row.movieId)), [197, 200]);

    // Both files are validated before a new load is committed.
    const beforeFailure = JSON.stringify(plain(run('({movies, ratings})')));
    responses.set('u.data', Buffer.from('broken'));
    await assert.rejects(context.loadData(), /u.data/);
    assert.equal(JSON.stringify(plain(run('({movies, ratings})'))), beforeFailure);
    responses.set('u.data', bytes('u.data'));

    await context.loadData();
    await context.loadData();
    assert.equal(run('movies.length'), 1682);
    assert.equal(run('ratings.length'), 100000);
    assert.equal(run("movies.find(m=>m.id===543).title"), 'Misérables, Les (1995)');
    assert.equal(run("movies.some(m=>m.title.includes('\uFFFD'))"), false);
    assert.deepEqual(plain(run('movies.filter(m=>m.genres.includes("unknown")).map(m=>m.id)')), [267, 1373]);
    const expectedTitles = bytes('u.item').toString('latin1').trimEnd().split(/\r?\n/).map(line => line.split('|')[1]);
    assert.deepEqual(plain(run('movies.map(movie => movie.title)')), expectedTitles);

    // Compare all split records (including ordering) with final HW5 from Git.
    const oldContext = vm.createContext({ dataText: bytes('u.data').toString('utf8') });
    const oldSource = execFileSync('git', ['show', 'alina-05-matrix-factorization:week5/data.js'], { cwd: root, encoding: 'utf8' });
    vm.runInContext(oldSource, oldContext);
    for (const name of ['u.item', 'u.data']) {
        const oldBytes = execFileSync('git', ['show', `alina-05-matrix-factorization:week5/${name}`], { cwd: root, maxBuffer: 4 * 1024 * 1024 });
        assert.equal(bytes(name).toString('latin1').replace(/\r\n/g, '\n'), oldBytes.toString('latin1').replace(/\r\n/g, '\n'));
    }
    const oldSplit = vm.runInContext('parseRatingData(dataText); splitByTimestamp()', oldContext);
    const split = run('splitByTimestamp()');
    const splitHashes = {};
    for (const name of ['train', 'val', 'test']) {
        splitHashes[name] = hash(JSON.stringify(split[name]));
        assert.equal(splitHashes[name], hash(JSON.stringify(oldSplit[name])));
    }
    assert.deepEqual(['train', 'val', 'test'].map(name => split[name].length), [79619, 9596, 10785]);
    context.actualSplit = split;
    run('Object.assign(HW6_SPLIT, actualSplit); hw6BuildSplitIndices(HW6_SPLIT)');
    const actualGraph = run('buildGraph(HW6_SPLIT.train)');
    assert.equal(actualGraph.nodeCount, 2554);
    assert.equal(actualGraph.nUsers, 943);
    assert.equal(actualGraph.nMovies, 1611);
    assert.equal(actualGraph.positiveEdges, 45602);
    assert.equal(actualGraph.degree.reduce((sum, n) => sum + n, 0), 91204);
    assert.deepEqual(plain(actualGraph.zeroDegreeUsers), [685]);
    assert.equal(actualGraph.zeroDegreeMovies.length, 253);
    assert.equal(run('HW6_STATE.positivePairs'), actualGraph.positiveEdges);
    assert.deepEqual(plain(run('HW6_STATE.candidateIds')), plain(actualGraph.movieIds));
    run('hw6RenderDatasetSummary()');
    assert.match(summary.innerHTML, /93\.70%/);

    // Held-out-only nodes and changed held-out values cannot enter the graph.
    const signature = g => JSON.stringify({ users: g.userIds, movies: g.movieIds, R: g.R, degree: Array.from(g.degree) });
    const originalSignature = signature(actualGraph);
    run('HW6_SPLIT.val = [{userId:99999,movieId:99999,rating:5}]; HW6_SPLIT.test = [{userId:88888,movieId:88888,rating:5}]; ratingMatrix = null');
    assert.equal(signature(run('buildGraph(HW6_SPLIT.train)')), originalSignature);

    // Confirm the dataset bytes have not changed.
    for (const name of ['u.item', 'u.data']) assert.equal(hash(bytes(name)), initialHashes[name]);
    console.log('PASS: metadata, strict validation, atomic reloads, sparse IDs, graph isolates, duplicate rejection, index replacement, sparsity, HW5 split parity and held-out isolation.');
    console.log(JSON.stringify({ users: 943, movies: 1682, ratings: 100000, trainMovies: actualGraph.nMovies,
        nodes: actualGraph.nodeCount, positivePairs: actualGraph.positiveEdges,
        zeroDegreeUsers: plain(actualGraph.zeroDegreeUsers), zeroDegreeMovies: actualGraph.zeroDegreeMovies.length,
        splitHashes }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
