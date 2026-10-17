/** Equivalence and resource checks, executed before the validation grid. */
const HW6_ALPHA_CHECKS = (function () {
  let lastReport = null;
  let lastGraph = null;
  const ensure = (ok, message) => { if (!ok) throw new Error(message); };
  async function reject(fn, pattern) {
    let message = "";
    try { await fn(); } catch (error) { message = error.message; }
    ensure(pattern.test(message), `Expected ${pattern}, got ${message}`);
  }
  function compare(scalar, batched) {
    ensure(scalar.p.length === batched.p.length, "Batch vector length mismatch");
    const maxError = scalar.p.reduce((max, value, i) => Math.max(max, Math.abs(value - batched.p[i])), 0);
    ensure(maxError < 1e-6, `Batch/scalar score discrepancy ${maxError}`);
    ensure(scalar.converged && batched.converged && scalar.residual < 1e-6 && batched.residual < 1e-6,
      "Nonconverged runs cannot pass equivalence checks");
    ensure(Math.abs(batched.p.reduce((sum, value) => sum + value, 0) - 1) < 1e-5, "Batch lost probability mass");
    for (const row of batched.trace) ensure(Math.abs(row.mass - 1) < 1e-5 && row.minimum >= 0, "Invalid intermediate batch probabilities");
    return maxError;
  }
  async function run() {
    const started = performance.now();
    await hw6EnsureMf();
    const tied = ALPHA_GRID.map(alpha => ({ alpha, status: "complete", recall: 0.25, convergence: { failedUsers: 0 } }));
    ensure(HW6_ALPHA_SELECTION.choose(tied) === 0.50, "Exact ties must select the smaller alpha");
    ensure(HW6_ALPHA_SELECTION.choose(tied.slice(0, 3)) === null, "Incomplete grids must not select a winner");
    const failed = tied.map(row => ({ ...row })); failed[3] = { ...failed[3], status: "failed" };
    ensure(HW6_ALPHA_SELECTION.choose(failed) === null, "Failed grids must not select a winner");
    const before = tf.memory().numTensors;
    const fixtureChecks = [];
    let engine = await HW6_BATCH_PPR.create(null, { transition: REFERENCE_GRAPH.transition });
    try {
      for (const alpha of ALPHA_GRID) {
        const batch = await engine.runNodes([0, 1, 3], alpha, { trace: true });
        for (const result of batch.results) {
          const scalar = await personalizedPageRank(result.seedNode, alpha, { transition: REFERENCE_GRAPH.transition });
          fixtureChecks.push({ fixture: "directed README", alpha, seedNode: result.seedNode, maxError: compare(scalar, result) });
        }
      }
    } finally { engine.dispose(); }
    await HW6_RETRIEVAL_CHECKS.withFixture({ movieIds: [10, 20, 30],
      train: [{ userId: 42, movieId: 10, rating: 5 }, { userId: 42, movieId: 20, rating: 4 },
        { userId: 700, movieId: 30, rating: 1 }],
    }, async () => {
      const graph = getGraph();
      const baseline = tf.memory().numTensors;
      const fixtureEngine = await HW6_BATCH_PPR.create(graph);
      try {
        for (const alpha of ALPHA_GRID) {
          const batch = await fixtureEngine.runUsers([42, 700], alpha, { trace: true });
          for (const result of batch.results) {
            const scalar = await personalizedPageRank(graph.nodeOfUser(result.userId), alpha, { graph });
            if (!scalar.converged || !result.converged) {
              ensure(alpha === 0.95 && result.userId === 42 && !scalar.converged && !result.converged &&
                scalar.iterations === 500 && result.iterations === 500, "Unexpected fixture nonconvergence");
              await reject(() => HW6_RETRIEVAL.rankPropagation(result.userId, result, graph), /did not converge/);
              fixtureChecks.push({ fixture: "float32 threshold plateau (negative test)", alpha, userId: result.userId,
                scalarResidual: scalar.residual, batchResidual: result.residual, iterations: 500,
                acceptedAsConverged: false, rankingRejected: true });
              continue;
            }
            const a = HW6_RETRIEVAL.rankPropagation(result.userId, scalar, graph).map(row => row.movieId);
            const b = HW6_RETRIEVAL.rankPropagation(result.userId, result, graph).map(row => row.movieId);
            ensure(JSON.stringify(a) === JSON.stringify(b), "Small-graph Top-10 mismatch");
            fixtureChecks.push({ fixture: "sparse IDs and isolates", alpha, userId: result.userId,
              maxError: compare(scalar, result), scalarIterations: scalar.iterations, batchIterations: result.iterations });
          }
          ensure(batch.results[1].iterations === 1 && batch.results[1].p[graph.nodeOfUser(700)] === 1,
            "Isolated seed was not frozen independently at iteration one");
        }
        const capped = await fixtureEngine.runUsers([42, 700], 1);
        ensure(!capped.results[0].converged && capped.results[0].iterations === 500,
          "A periodic capped row was incorrectly called converged");
        ensure(capped.results[1].converged && capped.results[1].iterations === 1, "Rows must have independent convergence");
        await reject(() => fixtureEngine.runUsers([1], 0.85), /restart node/);
        const held = tf.memory().numTensors;
        for (let i = 0; i < 3; i++) await fixtureEngine.runUsers([42, 700], 0.85);
        ensure(tf.memory().numTensors === held, "Repeated batch calls leaked tensors");
        const originalData = tf.Tensor.prototype.data;
        try {
          tf.Tensor.prototype.data = async function () { throw new Error("injected batch read failure"); };
          await reject(() => fixtureEngine.runUsers([42, 700], 0.85), /injected batch read failure/);
        } finally { tf.Tensor.prototype.data = originalData; }
        ensure(tf.memory().numTensors === held, "Failed batch call leaked tensors");
      } finally { fixtureEngine.dispose(); }
      ensure(tf.memory().numTensors === baseline, "Disposed batch engine retained transitions");
      await reject(() => fixtureEngine.runUsers([42], 0.85), /disposed/);
    });
    const cohort = await hw6SharedCohort("val");
    ensure(cohort.length === 846, "Expected the common validation cohort of 846 users");
    const graph = getGraph();
    const sampledUsers = [cohort[0], cohort[1], cohort[cohort.length - 1]];
    const batchUsers = [...sampledUsers, ...cohort.filter(id => !sampledUsers.includes(id)).slice(0, 13)];
    const liveChecks = [];
    const performanceRows = [];
    engine = await HW6_BATCH_PPR.create(graph, { maxBatchSize: 16 });
    const held = tf.memory().numTensors;
    try {
      for (const alpha of ALPHA_GRID) {
        const scalarResults = [];
        for (const userId of sampledUsers) scalarResults.push(await personalizedPageRank(graph.nodeOfUser(userId), alpha, { graph }));
        const batch = await engine.runUsers(batchUsers, alpha);
        ensure(batch.results.every(result => result.converged), "Live batch includes nonconverged users");
        for (let i = 0; i < sampledUsers.length; i++) {
          const userId = sampledUsers[i];
          const scalar = scalarResults[i]; const batched = batch.results[i];
          const maxError = compare(scalar, batched);
          const scalarTop = HW6_RETRIEVAL.rankPropagation(userId, scalar, graph).map(row => row.movieId);
          const batchTop = HW6_RETRIEVAL.rankPropagation(userId, batched, graph).map(row => row.movieId);
          ensure(JSON.stringify(scalarTop) === JSON.stringify(batchTop), `MovieLens Top-10 mismatch at alpha ${alpha}, user ${userId}`);
          liveChecks.push({ alpha, userId, maxAbsScoreDifference: maxError, sameTop10: true,
            scalarIterations: scalar.iterations, batchIterations: batched.iterations,
            scalarResidual: scalar.residual, batchResidual: batched.residual, recommendedIds: batchTop });
        }
        ensure(tf.memory().numTensors === held, "MovieLens batch leaked tensors");
        const scalarMeanMs = scalarResults.reduce((sum, result) => sum + result.runtimeMs, 0) / sampledUsers.length;
        performanceRows.push({ alpha, scalarUsers: sampledUsers.length, scalarMeanUserMs: scalarMeanMs,
          batchUsers: batchUsers.length, batchRuntimeMs: batch.runtimeMs, batchMeanUserMs: batch.runtimeMs / batchUsers.length });
      }
    } finally { engine.dispose(); }
    ensure(tf.memory().numTensors === before, "Equivalence checks retained tensors");
    lastGraph = graph;
    lastReport = { status: "passed", generatedAt: new Date().toISOString(), backend: tf.getBackend(), tensorflowVersion: tf.version.tfjs,
      batchSize: 16, alphas: ALPHA_GRID.slice(), sampledUsers, fixtureChecks, liveChecks, performance: performanceRows,
      independentConvergenceAndCapChecked: true, asynchronousFailureCleanupChecked: true, tieAndIncompleteGridChecked: true,
      tensorsBefore: before, tensorsAfter: tf.memory().numTensors, runtimeMs: performance.now() - started,
      trainingRuns: HW6_MF_ENGINE.trainingRuns, fullTestRecallRun: false };
    return lastReport;
  }
  return { run, get report() { return lastReport; }, matches: graph => lastGraph === graph && lastReport?.backend === tf.getBackend() };
})();
