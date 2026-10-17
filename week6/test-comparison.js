/**
 * One-pass TEST comparison using the already selected validation alpha.
 * Each ranked list supplies hits and popularity together, then is persisted.
 * Saved-result validation never performs propagation or generates new lists.
 */
const HW6_TEST_COMPARISON = (function () {
  let result = null;
  let inFlight = null;
  const methods = ["popularity", "pagerank", "ppr", "mf"];
  const tol = 1e-6;
  const maxIter = 500;
  const batchSize = 16;
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const close = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-12;
  const ensure = (ok, message) => { if (!ok) throw new Error(message); };

  function reset() { result = null; }

  async function fileDigest(file) {
    const response = await fetch(file, { cache: "no-store" });
    if (!response.ok) throw new Error(`Test comparison: required saved file unavailable: ${file} (${response.status})`);
    return HW6_MF_ENGINE.digest(await response.text());
  }

  async function context() {
    await hw6EnsureMf();
    ensure(HW6_ALPHA_SELECTION.result?.status === "complete", "A verified complete validation result is required");
    const alpha = HW6_ALPHA_SELECTION.comparisonAlpha();
    ensure(alpha === 0.50, "This test step requires the verified alpha=0.50");
    const cohort = await hw6SharedCohort("test");
    ensure(cohort.length === 861, `Expected 861 common test users, got ${cohort.length}`);
    const protocol = HW6_RETRIEVAL.protocol("test");
    let graph = getGraph();
    if (!graph) { graph = buildGraph(HW6_SPLIT.train); setGraph(graph); }
    ensure(same(graph.movieIds, HW6_STATE.candidateIds) && same(graph.userIds, HW6_STATE.trainUserIds) &&
      graph.positiveEdges === HW6_STATE.positivePairs, "Active graph must match train only");
    const digest = value => HW6_MF_ENGINE.digest(JSON.stringify(value));
    const alphaSource = { file: "alpha-validation.json", sha256: await fileDigest("alpha-validation.json"),
      selectedAlpha: alpha, objective: "macro validation Recall@10", validationUsers: 846,
      globalPageRankControl: "same PPR-selected alpha; global PageRank was not independently tuned" };
    const mf = { checkpointFile: "mf-checkpoint.json", sha256: await fileDigest("mf-checkpoint.json"),
      sourceCommit: MF_INFO.sourceCommit, sourceBlob: MF_INFO.sourceBlob, runId: MF_INFO.runId,
      checkpoint: MF_INFO.checkpoint, factors: MF_FACTORS, seed: MF_SEED, settings: { ...MF_CONFIG },
      selectionObjective: "clipped validation RMSE; earliest exact tie; selected factors restored",
      selectedValidationRMSE: MF_INFO.validationRMSE, ranking: "unclipped trainMean + dot(user,item)",
      tensorflowVersion: MF_INFO.tensorflowVersion, originalTrainingBackend: MF_INFO.backend };
    const identity = { trainSha256: await digest(HW6_SPLIT.train), validationSha256: await digest(HW6_SPLIT.val),
      testSha256: await digest(HW6_SPLIT.test), candidateIdsSha256: await digest(HW6_STATE.candidateIds),
      cohortSha256: await digest(cohort), candidateCount: graph.nMovies, graphNodes: graph.nodeCount,
      positiveTrainPairs: graph.positiveEdges, alphaSource, mf };
    return { alpha, cohort, protocol, graph, identity, alphaSource, mf };
  }

  function requireConvergence(record) {
    ensure(record && record.converged === true && Number.isInteger(record.iterations) && record.iterations >= 1 &&
      record.iterations <= maxIter && Number.isFinite(record.residual) && record.residual >= 0 && record.residual < tol,
    "Test comparison: nonconvergence is an error, not a zero Recall or user exclusion");
    if (record.mass !== undefined) ensure(Number.isFinite(record.mass) && Math.abs(record.mass - 1) < 1e-5, "Invalid probability mass");
  }

  /** One list, one pair of statistics: no second call for mean popularity. */
  function userRecord(userId, recs, current, convergence = null) {
    ensure(recs.length === 10, `Expected a full Top-10 for test user ${userId}`);
    const relevant = current.protocol.relevantByUser.get(userId);
    const ids = recs.map(rec => rec.movieId);
    const hits = relevant.filter(id => ids.includes(id)).length;
    return { userId, recommendedIds: ids, recommendedScores: recs.map(rec => rec.score),
      trainingPopularities: recs.map(rec => rec.popularity), eligibleRelevantIds: relevant.slice(),
      hits, relevant: relevant.length, recall: hits / relevant.length,
      meanPopularity: recs.reduce((sum, rec) => sum + rec.popularity, 0) / recs.length,
      zeroScoreCount: recs.filter(rec => rec.score === 0).length, convergence };
  }

  function convergenceSummary(users) {
    const values = users.map(user => user.convergence);
    values.forEach(requireConvergence);
    return { convergedUsers: values.length, failedUsers: 0,
      minIterations: Math.min(...values.map(row => row.iterations)),
      maxIterations: Math.max(...values.map(row => row.iterations)),
      maxResidual: Math.max(...values.map(row => row.residual)) };
  }

  function finalizeRow(row) {
    row.recall = row.perUser.reduce((sum, user) => sum + user.recall, 0) / row.cohortSize;
    row.meanPopularity = row.perUser.reduce((sum, user) => sum + user.meanPopularity, 0) / row.cohortSize;
    row.convergence = row.method === "pagerank" || row.method === "ppr" ? convergenceSummary(row.perUser) : null;
    row.status = "complete";
  }

  /** The two-user evidence is extracted from stored lists, not scored again. */
  function twoUserEvidence(rows) {
    const metadata = new Map(movies.map(movie => [movie.id, movie]));
    const lists = [];
    const overlaps = [];
    for (const row of rows) {
      for (const userId of [1, 2]) {
        const user = row.perUser.find(user => user.userId === userId);
        ensure(user, "Users 1 and 2 must both belong to the shared test cohort");
        lists.push({ method: row.method, userId, alpha: row.alpha, split: "test", meanPopularity: user.meanPopularity,
          zeroScoreCount: user.zeroScoreCount, scoreMeaning: row.scoreMeaning,
          recommendations: user.recommendedIds.map((id, rank) => ({ rank: rank + 1, movieId: id,
            title: metadata.get(id).title, genres: metadata.get(id).genres.slice(),
            score: user.recommendedScores[rank], popularity: user.trainingPopularities[rank] })) });
      }
      const pair = row.perUser.filter(user => user.userId === 1 || user.userId === 2);
      const commonMovieIds = pair[0].recommendedIds.filter(id => pair[1].recommendedIds.includes(id)).sort((a, b) => a - b);
      overlaps.push({ method: row.method, users: [1, 2], count: commonMovieIds.length, commonMovieIds });
    }
    return { users: [1, 2], lists, overlaps };
  }

  function publish(value) {
    result = value;
    HW6_STATE.comparisonRows = value?.status === "complete" ? value.rows : null;
    render(value);
  }

  async function perform() {
    publish(null);
    const started = performance.now();
    const output = { schemaVersion: 1, status: "running", split: "test", K: 10,
      generatedAt: new Date().toISOString(), rows: [], tolerance: tol, maxIterations: maxIter, batchSize,
      objective: "macro eligible test Recall@10 on a shared warm-user cohort",
      filtering: "train candidates only; exclude all train+validation ratings, independently of value",
      newTrainingRuns: 0, newAlphaSelectionRuns: 0,
      audit: { rankedLists: 0, globalPageRankRuns: 0, pprBatchCalls: 0,
        recallAndPopularityFromSameLists: true } };
    let engine = null;
    try {
      const current = await context();
      Object.assign(output, { alpha: current.alpha, alphaSource: current.alphaSource, mf: current.mf,
        identity: current.identity, cohort: current.cohort, cohortSize: current.cohort.length,
        exclusions: current.protocol.exclusions, tensorflowVersion: tf.version.tfjs,
        preparationMs: performance.now() - started });
      const baseline = tf.memory().numTensors;
      for (const method of methods) {
        const methodStarted = performance.now();
        const isGraph = method === "pagerank" || method === "ppr";
        const row = { method, status: "running", split: "test", alpha: isGraph ? current.alpha : null,
          alphaSource: isGraph ? "alpha-validation.json" : null, cohortSize: current.cohort.length,
          backend: method === "popularity" ? "javascript" : tf.getBackend(), recall: null, meanPopularity: null,
          scoreMeaning: isGraph ? "long-run graph-walk probability; not a rating" : method === "mf" ?
            "unclipped trainMean + dot product" : "distinct positive training users", perUser: [] };
        output.rows.push(row);
        if (method === "ppr") {
          engine = await HW6_BATCH_PPR.create(current.graph, { maxBatchSize: batchSize });
          for (let start = 0; start < current.cohort.length; start += batchSize) {
            const userIds = current.cohort.slice(start, start + batchSize);
            const batch = await engine.runUsers(userIds, current.alpha, { tol, maxIter });
            output.audit.pprBatchCalls++;
            for (const user of batch.results) {
              requireConvergence(user);
              const recs = HW6_RETRIEVAL.rankPropagation(user.userId, user, current.graph, 10, "test");
              const diagnostics = { backend: batch.backend, iterations: user.iterations, residual: user.residual,
                converged: user.converged, mass: user.mass };
              row.perUser.push(userRecord(user.userId, recs, current, diagnostics));
              output.audit.rankedLists++;
            }
            hw6SetStatus(`Test PPR alpha=${current.alpha}: ${row.perUser.length}/${current.cohort.length} users`);
            await tf.nextFrame();
          }
          engine.dispose(); engine = null;
        } else {
          let sharedScores = null;
          let diagnostics = null;
          if (method === "popularity") sharedScores = { movieIds: HW6_STATE.candidateIds,
            scores: Float64Array.from(HW6_STATE.candidateIds, id => hw6PopularityOf(id)) };
          if (method === "pagerank") {
            const global = await pageRank(current.alpha, { graph: current.graph, tol, maxIter });
            requireConvergence(global); output.audit.globalPageRankRuns++;
            sharedScores = { movieIds: current.graph.movieIds, scores: movieScores(global.p, current.graph), propagation: global };
            diagnostics = { backend: global.backend, iterations: global.iterations, residual: global.residual,
              converged: global.converged, mass: global.p.reduce((sum, score) => sum + score, 0) };
            row.note = "Uses the PPR validation-selected alpha as a shared control; not independently optimized";
          }
          for (let index = 0; index < current.cohort.length; index++) {
            const userId = current.cohort[index];
            const scored = sharedScores || await hw6ScoreVector("mf", userId, current.alpha, "test");
            const recs = HW6_RETRIEVAL.rankScores(userId, scored, 10, "test", method);
            row.perUser.push(userRecord(userId, recs, current, diagnostics));
            output.audit.rankedLists++;
            if (index % 32 === 31) {
              hw6SetStatus(`Test ${method}: ${index + 1}/${current.cohort.length} users`);
              await tf.nextFrame();
            }
          }
          if (method === "mf") row.note = `Saved HW5 MF checkpoint ${MF_INFO.checkpoint}, selected by validation RMSE; test retrieval recomputed`;
        }
        finalizeRow(row);
        row.runtimeMs = performance.now() - methodStarted;
      }
      output.twoUsers = twoUserEvidence(output.rows);
      output.tensorsBefore = baseline; output.tensorsAfter = tf.memory().numTensors;
      ensure(output.tensorsAfter === baseline, "Test evaluation retained propagation tensors");
      output.runtimeMs = performance.now() - started;
      output.status = "complete";
      ensure(HW6_MF_ENGINE.trainingRuns === 0, "MF must only be loaded");
      await verify(output, current);
      publish(output);
      return output;
    } catch (error) {
      output.status = "failed"; output.error = error.message; output.runtimeMs = performance.now() - started;
      publish(output); return output;
    } finally { if (engine) engine.dispose(); }
  }

  /** Fast check: data/settings, filters, order, duplicates, hits and all means. */
  async function verify(value, suppliedContext = null) {
    ensure(value?.schemaVersion === 1 && value.status === "complete" && value.split === "test" && value.K === 10 &&
      value.tolerance === tol && value.maxIterations === maxIter && value.batchSize === batchSize,
    "Incomplete or incompatible test result");
    const current = suppliedContext || await context();
    ensure(same(value.identity, current.identity) && same(value.alphaSource, current.alphaSource) && same(value.mf, current.mf),
      "Saved alpha/MF/data/split settings do not match");
    ensure(value.alpha === current.alpha && same(value.cohort, current.cohort) && value.cohortSize === 861 &&
      same(value.exclusions, current.protocol.exclusions), "Test cohorts or exclusion reasons do not match");
    ensure(same(value.rows.map(row => row.method), methods), "Exactly the four required methods must be present");
    for (const row of value.rows) {
      const graphMethod = row.method === "pagerank" || row.method === "ppr";
      ensure(row.status === "complete" && row.split === "test" && row.cohortSize === 861 &&
        row.alpha === (graphMethod ? current.alpha : null) && row.alphaSource === (graphMethod ? "alpha-validation.json" : null),
      "Method alpha/cohort/settings mismatch");
      ensure(same(row.perUser.map(user => user.userId), current.cohort), "Methods must use identical users, without omissions/duplicates");
      for (const user of row.perUser) {
        const ids = user.recommendedIds; const scores = user.recommendedScores; const pops = user.trainingPopularities;
        ensure(ids.length === 10 && scores.length === 10 && pops.length === 10 && new Set(ids).size === 10, "Missing or duplicate recommendation IDs");
        const excluded = hw6ExclusionSet("test", user.userId);
        for (let rank = 0; rank < ids.length; rank++) {
          ensure(HW6_STATE.candidateSet.has(ids[rank]) && !excluded.has(ids[rank]), "Train candidate/test exclusion filter violated");
          ensure(typeof scores[rank] === "number" && Number.isFinite(scores[rank]), "Invalid saved score");
          ensure(pops[rank] === hw6PopularityOf(ids[rank]), "Saved popularity is not train-only popularity");
          if (row.method === "popularity") ensure(scores[rank] === pops[rank], "Popularity score mismatch");
          if (graphMethod) ensure(scores[rank] >= 0, "Negative walk probability");
          if (rank) ensure(scores[rank - 1] > scores[rank] || (scores[rank - 1] === scores[rank] && ids[rank - 1] < ids[rank]),
            "Saved recommendation order/tie-break is invalid");
        }
        const relevant = current.protocol.relevantByUser.get(user.userId);
        ensure(relevant.length > 0 && same(user.eligibleRelevantIds, relevant) && user.relevant === relevant.length, "Eligible relevant IDs mismatch");
        const hits = relevant.filter(id => ids.includes(id)).length;
        ensure(user.hits === hits && close(user.recall, hits / relevant.length), "Saved hits/user Recall mismatch");
        ensure(close(user.meanPopularity, pops.reduce((sum, count) => sum + count, 0) / 10) &&
          user.zeroScoreCount === scores.filter(score => score === 0).length, "Saved list popularity/zero count mismatch");
        if (graphMethod) {
          requireConvergence(user.convergence);
          ensure(user.convergence.backend === row.backend && Number.isFinite(user.convergence.mass), "Missing graph backend/mass evidence");
          if (row.method === "pagerank") ensure(same(user.convergence, row.perUser[0].convergence), "Global PR must reuse one stationary distribution");
        }
        else ensure(user.convergence === null, "Non-iterative scorers must not claim graph convergence");
      }
      ensure(close(row.recall, row.perUser.reduce((sum, user) => sum + user.recall, 0) / 861), "Macro Recall mismatch");
      ensure(close(row.meanPopularity, row.perUser.reduce((sum, user) => sum + user.meanPopularity, 0) / 861), "Macro list popularity mismatch");
      ensure(same(row.convergence, graphMethod ? convergenceSummary(row.perUser) : null), "Convergence summary mismatch");
      ensure(Number.isFinite(row.runtimeMs) && row.runtimeMs >= 0 && row.backend, "Missing method runtime/backend");
    }
    ensure(same(value.twoUsers, twoUserEvidence(value.rows)), "Final two-user titles/genres/scores/popularity/overlap mismatch");
    ensure(value.newTrainingRuns === 0 && value.newAlphaSelectionRuns === 0 && value.audit.rankedLists === 4 * 861 &&
      value.audit.globalPageRankRuns === 1 && value.audit.pprBatchCalls === Math.ceil(861 / 16) && value.audit.recallAndPopularityFromSameLists,
    "Test evaluation must use one ranking pass and saved MF/alpha");
    return { status: "passed", usersPerMethod: 861, methods: 4, recommendedRowsChecked: 34440,
      macroRecallsRecomputed: true, sharedCohortChecked: true, filtersOrderDuplicatesChecked: true,
      popularityAndTwoUserEvidenceChecked: true, alphaAndMfSettingsChecked: true, propagationRerun: false };
  }

  async function load(value, source = "explicit saved test result") {
    publish(null);
    try { await verify(value); publish({ ...value, loadedFrom: source }); return result; }
    catch (error) { publish({ status: "failed", rows: [], error: error.message }); throw error; }
  }

  async function restore() {
    try {
      const response = await fetch("test-results.json", { cache: "no-store" });
      if (response.status === 404) return null;
      ensure(response.ok, `Saved test result unavailable (${response.status})`);
      return await load(await response.json(), "test-results.json");
    } catch (error) { publish({ status: "failed", rows: [], error: error.message }); return result; }
  }

  function run(options = {}) {
    if (inFlight) return inFlight;
    if (!options.force && result) return Promise.resolve(result);
    inFlight = perform().finally(() => { inFlight = null; });
    return inFlight;
  }

  function render(value = result) {
    const target = document.getElementById("comparison-results");
    const listsTarget = document.getElementById("test-recommendations");
    if (!target) return;
    if (!value) { target.innerHTML = '<p class="empty-state">No saved final test comparison yet.</p>'; if (listsTarget) listsTarget.innerHTML = ''; return; }
    if (value.status !== "complete") { target.innerHTML = '<p class="error">Test comparison failed: ' + hw6EscapeHtml(value.error || value.status) + '. No missing user is counted as zero Recall.</p>'; if (listsTarget) listsTarget.innerHTML = ''; return; }
    target.innerHTML = '<table class="data-table"><thead><tr><th>Method</th><th>Test Recall@10</th><th>Users</th><th>Alpha</th><th>Mean popularity</th><th>Backend</th><th>Seconds</th><th>Convergence</th></tr></thead><tbody>' +
      value.rows.map(row => '<tr><td>' + hw6EscapeHtml(row.method) + '</td><td>' + hw6Fmt(row.recall, 8) + '</td><td>' + row.cohortSize +
        '</td><td>' + (row.alpha === null ? '—' : row.alpha.toFixed(2)) + '</td><td>' + hw6Fmt(row.meanPopularity, 2) +
        '</td><td>' + hw6EscapeHtml(row.backend) + '</td><td>' + hw6Fmt(row.runtimeMs / 1000, 2) + '</td><td>' +
        (row.convergence ? row.convergence.convergedUsers + '/861; max L1 ' + row.convergence.maxResidual.toExponential(8) : 'not iterative') + '</td></tr>').join('') +
      '</tbody></table><p class="hint">Alpha ' + value.alpha + ' comes from saved validation selection, not the example slider. Global PR was not independently tuned. ' +
      'Saved HW5 MF checkpoint ' + value.mf.checkpoint + ' keeps validation-RMSE selection; its test Recall is newly computed under this protocol.</p>' +
      hw6CohortSummaryHtml(value.exclusions);
    if (listsTarget) listsTarget.innerHTML = value.twoUsers.lists.map(list => '<h4>' + hw6EscapeHtml(list.method) + ' — user ' + list.userId + '</h4>' +
      '<p class="hint">Alpha: ' + (list.alpha === null ? 'not applicable' : list.alpha.toFixed(2)) + '; mean training popularity: ' + hw6Fmt(list.meanPopularity, 2) +
      '; zero scores: ' + list.zeroScoreCount + '. ' + hw6EscapeHtml(list.scoreMeaning) + '</p><ol class="recommendation-list">' +
      list.recommendations.map(rec => '<li>' + hw6EscapeHtml(rec.title) + ' — id ' + rec.movieId + '; ' + hw6EscapeHtml(rec.genres.join(', ')) +
        '; score ' + hw6Fmt(rec.score, 8) + '; train popularity ' + rec.popularity + '</li>').join('') + '</ol>').join('') +
      '<p class="hint">Common movies between users 1/2: ' + value.twoUsers.overlaps.map(row => hw6EscapeHtml(row.method) + ': ' + row.count + '/10').join('; ') +
      '. Descriptive overlap only. Graph scores are walk probabilities, not ratings.</p>';
  }

  function csvResults(value = result) {
    const x = value.exclusions;
    return hw6CsvDocument(['method', 'test_recall_at_10', 'cohort_size', 'alpha', 'alpha_source', 'mean_training_popularity', 'backend',
      'runtime_seconds', 'converged_users', 'failed_users', 'min_iterations', 'max_iterations', 'max_residual', 'excluded_unique_users',
      'no_train', 'no_positive_train', 'mf_unavailable', 'no_eligible_relevant', 'overlap_users', 'unseen_relevant_records',
      'mf_checkpoint', 'mf_run_id', 'mf_selection_objective', 'total_runtime_seconds'], value.rows.map(row => [row.method, row.recall,
      row.cohortSize, row.alpha, row.alphaSource, row.meanPopularity, row.backend, row.runtimeMs / 1000,
      row.convergence?.convergedUsers, row.convergence?.failedUsers, row.convergence?.minIterations, row.convergence?.maxIterations,
      row.convergence?.maxResidual, x.excludedTotal, x.excludedNoTrain, x.excludedNoPositiveEdge, x.excludedMfUnavailable,
      x.excludedNoEligible, x.overlapUsers.length, x.excludedUnseen, row.method === 'mf' ? value.mf.checkpoint : null,
      row.method === 'mf' ? value.mf.runId : null, row.method === 'mf' ? value.mf.selectionObjective : null, value.runtimeMs / 1000]));
  }

  function csvRecommendations(value = result) {
    const rows = [];
    for (const list of value.twoUsers.lists) for (const rec of list.recommendations) rows.push([list.userId, list.method, list.alpha,
      rec.rank, rec.movieId, rec.title, rec.genres.join(';'), rec.score, rec.popularity, list.meanPopularity,
      value.twoUsers.overlaps.find(row => row.method === list.method).count, 'test', list.scoreMeaning]);
    return hw6CsvDocument(['user_id', 'method', 'alpha', 'rank', 'movie_id', 'title', 'genres', 'score', 'training_popularity',
      'mean_training_popularity', 'common_movies_between_users', 'split', 'score_meaning'], rows);
  }

  function csvSummary(value = result) {
    return hw6CsvDocument(['method', 'user_id', 'alpha', 'mean_training_popularity', 'zero_score_count', 'common_movies_between_users',
      'common_movie_ids'], value.twoUsers.lists.map(list => { const overlap = value.twoUsers.overlaps.find(row => row.method === list.method);
      return [list.method, list.userId, list.alpha, list.meanPopularity, list.zeroScoreCount, overlap.count, overlap.commonMovieIds.join(';')]; }));
  }

  return { run, verify, load, restore, render, csvResults, csvRecommendations, csvSummary, reset,
    get result() { return result; }, snapshot: () => result, restoreSnapshot: saved => { result = saved; } };
})();
