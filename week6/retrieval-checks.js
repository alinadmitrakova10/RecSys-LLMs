/** Known-answer retrieval fixtures used by the in-page harness and browser CLI. */
const HW6_RETRIEVAL_CHECKS = (function () {
  const ensure = (ok, message) => { if (!ok) throw new Error(message); };
  const equal = (a, b) => ensure(JSON.stringify(a) === JSON.stringify(b), `${JSON.stringify(a)} != ${JSON.stringify(b)}`);
  const close = (a, b) => ensure(Math.abs(a - b) < 1e-6, `${a} != ${b}`);
  const row = (userId, movieId, rating) => ({ userId, movieId, rating, timestamp: movieId });

  /** Arbitrary raw scores via hand-set one-hot user factors; never trained. */
  function modelFor(fixture) {
    const users = HW6_STATE.trainUserIds;
    const f = users.length;
    const u = new Float32Array((numUsers + 1) * f);
    const i = new Float32Array((HW6_MAX_MOVIE_ID + 1) * f);
    users.forEach((userId, index) => {
      u[userId * f + index] = fixture.unavailableUsers && fixture.unavailableUsers.includes(userId) ? NaN : 1;
      for (const movieId of fixture.movieIds) i[movieId * f + index] =
        fixture.scores && fixture.scores[userId] && fixture.scores[userId][movieId] !== undefined ? fixture.scores[userId][movieId] : 0;
    });
    return tf.tidy(() => ({ userFactors: tf.variable(tf.tensor2d(u, [numUsers + 1, f])),
      itemFactors: tf.variable(tf.tensor2d(i, [HW6_MAX_MOVIE_ID + 1, f])), globalMean: 0 }));
  }

  /** Isolate fixtures, including tensors, caches and original live-model metadata. */
  async function withFixture(fixture, fn) {
    if (HW6_MF_TRAINING) throw new Error("Cannot replace data while the MF checkpoint is loading");
    await tf.ready();
    const saved = { movies, ratings, numUsers, numMovies, ratingMatrix, max: HW6_MAX_MOVIE_ID,
      split: { ...HW6_SPLIT }, state: { ...HW6_STATE }, info: { ...MF_INFO }, model: HW6_MF_MODEL,
      graph: GRAPH, retrieval: HW6_RETRIEVAL.snapshot(), scorer: hw6ScoreVector, mfScorer: mfScores,
      alphaSelection: typeof HW6_ALPHA_SELECTION === "undefined" ? null : HW6_ALPHA_SELECTION.snapshot(),
      testComparison: typeof HW6_TEST_COMPARISON === "undefined" ? null : HW6_TEST_COMPARISON.snapshot(),
      pprProfiles: typeof HW6_PPR_PROFILES === "undefined" ? null : HW6_PPR_PROFILES.snapshot() };
    try {
      HW6_MF_MODEL = null; // Keep the saved live model owned by the caller.
      movies = fixture.movieIds.map(id => ({ id, title: `Fixture movie ${id}`, genres: id % 2 ? ["Action"] : ["unknown"] }));
      Object.assign(HW6_SPLIT, { train: fixture.train, val: fixture.val || [], test: fixture.test || [] });
      ratings = [...HW6_SPLIT.train, ...HW6_SPLIT.val, ...HW6_SPLIT.test];
      numUsers = Math.max(...fixture.train.map(record => record.userId));
      numMovies = movies.length;
      HW6_MAX_MOVIE_ID = Math.max(...fixture.movieIds);
      hw6BuildSplitIndices(HW6_SPLIT);
      HW6_STATE.recommendationRows = [];
      HW6_STATE.recommendationKeys = new Set();
      GRAPH = buildGraph(fixture.train);
      HW6_MF_MODEL = modelFor(fixture);
      Object.assign(MF_INFO, { checkpoint: 1, trainedAt: "fixture", loadedFrom: "hand-set fixture" });
      return await fn();
    } finally {
      hw6ScoreVector = saved.scorer;
      mfScores = saved.mfScorer;
      hw6DisposeMf();
      HW6_MF_MODEL = saved.model;
      movies = saved.movies; ratings = saved.ratings; numUsers = saved.numUsers;
      numMovies = saved.numMovies; ratingMatrix = saved.ratingMatrix; HW6_MAX_MOVIE_ID = saved.max;
      Object.assign(HW6_SPLIT, saved.split);
      Object.keys(HW6_STATE).forEach(key => { delete HW6_STATE[key]; });
      Object.assign(HW6_STATE, saved.state);
      Object.keys(MF_INFO).forEach(key => { delete MF_INFO[key]; });
      Object.assign(MF_INFO, saved.info);
      GRAPH = saved.graph;
      HW6_RETRIEVAL.restore(saved.retrieval);
      if (saved.alphaSelection) HW6_ALPHA_SELECTION.restoreSnapshot(saved.alphaSelection);
      if (typeof HW6_TEST_COMPARISON !== "undefined") HW6_TEST_COMPARISON.restoreSnapshot(saved.testComparison);
      if (typeof HW6_PPR_PROFILES !== "undefined") HW6_PPR_PROFILES.restoreSnapshot(saved.pprProfiles);
    }
  }

  async function rejects(fn, pattern) {
    let message = "";
    try { await fn(); } catch (error) { message = error.message; }
    ensure(pattern.test(message), `Expected ${pattern}, got ${message}`);
  }

  async function run() {
    const beforeTensors = tf.memory().numTensors;
    const checks = [];
    const graph = { userIds: [10, 42], movieIds: [90, 7], nodeCount: 4 };
    equal(Array.from(movieScores([0.6, 0.2, 0, 0.2], graph)), [0, 0.2]);
    equal(Array.from(movieScores([0.6, 0.2, 0, 0.2], { ...graph, movieNodes: [3, 2] })), [0.2, 0]);
    await rejects(() => movieScores([0.6, 0.2, 0, 0.2], { ...graph, movieNodes: [0, 3] }), /movie block/);
    await rejects(() => movieScores([0.6, 0.2, null, 0.2], graph), /invalid probability/);
    await rejects(() => movieScores([0.6], graph), /dimensions/);
    checks.push("movie-only extraction, explicit mapping, invalid inputs and zero preservation");

    await withFixture({ movieIds: [50, 30, 10, 20, 40],
      train: [row(1, 10, 1), row(1, 20, 5), row(2, 30, 4), row(2, 40, 4), row(2, 50, 4)],
      val: [row(1, 30, 1)], test: [row(1, 40, 5)],
      scores: { 1: { 10: 100, 20: 100, 30: 1, 40: 1, 50: 1 } },
    }, async () => {
      equal((await recommendTopK("mf", 1, 10, "val")).map(rec => rec.movieId), [30, 40, 50]);
      equal((await recommendTopK("mf", 1, 10, "test")).map(rec => rec.movieId), [40, 50]);
      equal((await recommendTopK("popularity", 1, 10, "test")).map(rec => rec.movieId), [40, 50]);
      equal((await recommendTopK("mf", 1, 1, "test")).map(rec => rec.movieId), [40]);
      equal(await recommendTopK("mf", 1, 0, "test"), []);
      equal(await recommendTopK("unknown", 1, 10, "test"), []);
      const zero = await recommendTopK("ppr", 1, 10, "test", { alpha: 0.85 });
      equal(zero.map(rec => rec.movieId), [40, 50]);
      ensure(zero.every(rec => rec.score === 0), "Unreachable movie scores must remain zero");
      const original = mfScores;
      for (const invalid of [undefined, null, NaN, Infinity]) {
        mfScores = userId => ({ ...original(userId), 40: invalid });
        await rejects(() => recommendTopK("mf", 1, 10, "test"), /missing or invalid.*not a zero score/);
      }
      mfScores = original;
      const scoreVector = hw6ScoreVector;
      hw6ScoreVector = async (...args) => {
        const scored = await scoreVector(...args);
        scored.scores[HW6_STATE.candidateIds.indexOf(40)] = NaN;
        return scored;
      };
      await rejects(() => recommendTopK("ppr", 1, 10, "test", { alpha: 0.85 }), /missing or invalid.*not a zero score/);
      hw6ScoreVector = scoreVector;
      const one = await recommendTopK("mf", 1, 1, "test");
      hw6RecordRecommendationRows(1, "mf", null, zero);
      hw6RecordRecommendationRows(1, "mf", null, one);
      ensure(HW6_STATE.recommendationRows.length === 1, "Stale exported ranks were retained");
    });
    checks.push("validation/test low-rating filters, score ties, short lists, zero PPR vs invalid scores, export replacement");

    await withFixture({ movieIds: [1, 2, 3, 4],
      train: [row(1, 1, 5), row(2, 2, 4), row(2, 3, 4), row(2, 4, 4)],
      val: [row(1, 2, 5)], test: [row(1, 3, 5)],
      scores: { 1: { 1: 100, 2: 9, 3: 8, 4: 7 } },
    }, async () => {
      for (const method of HW6_RETRIEVAL.methods) {
        const validation = await recallAtK(method, 1, "val", { alpha: 0.85 });
        const test = await recallAtK(method, 1, "test", { alpha: 0.85 });
        equal(validation.cohort, [1]); equal(test.cohort, [1]);
        close(validation.recall, 1); close(test.recall, 1);
        equal((await recommendTopK(method, 1, 1, "val", { alpha: 0.85 })).map(rec => rec.movieId), [2]);
        equal((await recommendTopK(method, 1, 1, "test", { alpha: 0.85 })).map(rec => rec.movieId), [3]);
      }
    });
    checks.push("nonempty validation/test cohorts for four methods; fresh split-specific lists, not reused test rankings");

    for (const targetRank of [10, 11]) {
      const values = Object.fromEntries(Array.from({ length: 14 }, (_, i) => [i + 1, 14 - i]));
      if (targetRank === 11) [values[10], values[11]] = [values[11], values[10]];
      await withFixture({ movieIds: Array.from({ length: 14 }, (_, i) => i + 1),
        train: [row(1, 14, 5), ...Array.from({ length: 13 }, (_, i) => row(2, i + 1, 4))],
        test: [row(1, 10, 5)], scores: { 1: values },
      }, async () => {
        const recs = await recommendTopK("mf", 1, 10, "test");
        ensure(recs.length === 10, "An empty/short list cannot pass the rank fixture");
        const report = await recallAtK("mf", 10, "test");
        close(report.recall, targetRank === 10 ? 1 : 0);
        equal(report.cohort, [1]);
        close((await recallAtK("mf", 11, "test")).recall, 1);
      });
    }
    checks.push("known relevant movie at rank 10 hits, rank 11 misses, K=11 hits");

    const train = [row(1, 12, 5), row(2, 12, 5), row(3, 12, 1), row(4, 12, 5), row(5, 12, 1),
      ...Array.from({ length: 11 }, (_, i) => row(6, i + 1, 4))];
    const test = [row(1, 1, 5), row(1, 12, 5), row(1, 14, 5),
      row(2, 2, 5), row(2, 10, 5), row(2, 11, 5), row(2, 13, 5),
      row(3, 1, 5), row(4, 1, 5), row(5, 13, 5), row(6, 12, 2), row(99, 1, 5)];
    const values = Object.fromEntries(Array.from({ length: 14 }, (_, i) => [i + 1, 100 - i]));
    await withFixture({ movieIds: Array.from({ length: 14 }, (_, i) => i + 1), train, test,
      scores: Object.fromEntries([1, 2, 3, 4, 5, 6].map(id => [id, values])), unavailableUsers: [4],
    }, async () => {
      for (const method of HW6_RETRIEVAL.methods) {
        const report = await recallAtK(method, 10, "test", { alpha: 0.85 });
        equal(report.cohort, [1, 2]);
        close(report.recall, (1 + 2 / 3) / 2);
        ensure(Math.abs(report.recall - 3 / 4) > 0.01, "Recall was pooled instead of macro-averaged");
        equal(report.perUser.map(user => [user.userId, user.relevant, user.hit]), [[1, 1, 1], [2, 3, 2]]);
        const x = report.exclusions;
        equal([x.populationUsers, x.excludedTotal, x.excludedNoTrain, x.excludedNoPositiveEdge,
          x.excludedMfUnavailable, x.excludedNoEligible], [7, 5, 1, 3, 2, 2]);
        equal(x.overlapUsers, [5, 99]);
        equal([x.excludedUnseen, x.unseenDistinctMovies, x.previouslyRatedRelevant], [3, 2, 1]);
        const empty = await recallAtK(method, 10, "val");
        equal(empty.cohort, []); close(empty.recall, 0); equal(empty.perUser, []);
      }
      const kZero = await recallAtK("mf", 0, "test");
      close(kZero.recall, 0); equal(kZero.cohort, [1, 2]);
    });
    checks.push("macro 5/6 (not pooled 3/4), common users for four methods, MF coverage, overlapping reasons, unseen records vs IDs, empty cohort");
    ensure(tf.memory().numTensors === beforeTensors, "Retrieval fixtures leaked tensors");
    return { passed: checks.length, checks, fullMovieLensRecallRun: false, tensorsBefore: beforeTensors,
      tensorsAfter: tf.memory().numTensors };
  }

  return { run, withFixture };
})();
