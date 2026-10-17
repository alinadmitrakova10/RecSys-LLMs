/** Validation-only PPR selection. Example-slider state never stores the winner. */
const HW6_ALPHA_SELECTION = (function () {
  let selected = null;
  let result = null;
  let inFlight = null;
  const batchSize = 16;
  const tol = 1e-6;
  const maxIter = 500;

  function reset() { selected = null; result = null; HW6_STATE.selectedAlpha = null; }
  function publish(value) {
    result = value;
    selected = value && value.status === "complete" ? value.selectedAlpha : null;
    HW6_STATE.selectedAlpha = selected;
    HW6_STATE.alphaSweepRows = value ? value.rows : null;
    HW6_STATE.alphaSweepNote = value && value.error || "";
    render(value);
  }

  async function context() {
    const cohort = await hw6SharedCohort("val");
    if (cohort.length !== 846) throw new Error(`Validation selection: expected 846 users, got ${cohort.length}`);
    const protocol = HW6_RETRIEVAL.protocol("val");
    let graph = getGraph();
    if (!graph) { graph = buildGraph(HW6_SPLIT.train); setGraph(graph); }
    if (graph.movieIds.length !== HW6_STATE.candidateIds.length || graph.movieIds.some((id, i) => id !== HW6_STATE.candidateIds[i]) ||
        graph.positiveEdges !== HW6_STATE.positivePairs) throw new Error("Validation selection: active graph does not match train");
    const digest = value => HW6_MF_ENGINE.digest(JSON.stringify(value));
    const identity = {
      trainSha256: await digest(HW6_SPLIT.train), validationSha256: await digest(HW6_SPLIT.val),
      candidateIdsSha256: await digest(HW6_STATE.candidateIds), cohortSha256: await digest(cohort),
      mfRunId: MF_INFO.runId, mfCheckpoint: MF_INFO.checkpoint, mfSourceCommit: MF_INFO.sourceCommit,
      candidateCount: graph.nMovies, graphNodes: graph.nodeCount, positiveTrainPairs: graph.positiveEdges,
    };
    return { graph, cohort, protocol, identity };
  }

  /** Exact ties use the smaller alpha; incomplete/failed rows cannot win. */
  function choose(rows) {
    if (!Array.isArray(rows) || rows.length !== ALPHA_GRID.length ||
        new Set(rows.map(row => row.alpha)).size !== ALPHA_GRID.length ||
        ALPHA_GRID.some(alpha => !rows.some(row => row.alpha === alpha)) ||
        rows.some(row => row.status !== "complete" || !Number.isFinite(row.recall) || row.convergence?.failedUsers !== 0)) return null;
    return rows.slice().sort((a, b) => a.alpha - b.alpha).reduce((best, row) => !best || row.recall > best.recall ? row : best, null).alpha;
  }

  /** Verify stored per-user evidence and recompute every macro Recall, without PPR. */
  async function validate(value, suppliedContext = null, allowPartial = false) {
    if (!value || value.schemaVersion !== 1 || (value.status !== "complete" && !(allowPartial && value.status === "failed")) || value.split !== "val" ||
        value.tolerance !== tol || value.maxIterations !== maxIter || value.fullTestRecallRun !== false || value.mfTrainingRuns !== 0) {
      throw new Error("Saved alpha selection is incomplete, failed or incompatible");
    }
    const current = suppliedContext || await context();
    if (JSON.stringify(value.identity) !== JSON.stringify(current.identity) || JSON.stringify(value.cohort) !== JSON.stringify(current.cohort)) {
      throw new Error("Saved alpha selection does not match train/validation/cohort/MF checkpoint");
    }
    if (JSON.stringify(value.exclusions) !== JSON.stringify(current.protocol.exclusions)) throw new Error("Saved exclusion reasons do not match validation");
    for (const row of value.rows || []) {
      if (!row.perUser || row.perUser.length !== current.cohort.length || row.cohortSize !== current.cohort.length) throw new Error("Incomplete alpha cohort");
      let sum = 0;
      let failedUsers = 0;
      row.perUser.forEach((user, index) => {
        const relevant = current.protocol.relevantByUser.get(current.cohort[index]);
        if (user.userId !== current.cohort[index] || user.relevant !== relevant.length ||
            JSON.stringify(user.relevantIds) !== JSON.stringify(relevant) ||
            !Number.isInteger(user.iterations) || user.iterations < 1 || user.iterations > maxIter ||
            !Number.isFinite(user.residual) || user.residual < 0 ||
            !Number.isFinite(user.mass) || Math.abs(user.mass - 1) > 1e-5) throw new Error("Invalid per-user convergence/relevance evidence");
        if (typeof user.backend !== "string" || !user.backend) throw new Error("Missing per-user backend provenance");
        if (user.attempts) {
          for (const attempt of user.attempts) {
            if (!attempt.backend || !Number.isInteger(attempt.iterations) || attempt.iterations < 1 || attempt.iterations > maxIter ||
                !Number.isFinite(attempt.residual) || attempt.converged !== (attempt.residual < tol)) throw new Error("Invalid saved solver attempt");
          }
          const accepted = user.attempts[user.attempts.length - 1];
          if (accepted.backend !== user.backend || accepted.iterations !== user.iterations || accepted.residual !== user.residual ||
              accepted.converged !== user.converged) throw new Error("Final result does not match the last retained attempt");
        }
        if (!user.converged) {
          if (!allowPartial || value.status !== "failed" || user.converged !== false || user.residual < tol ||
              user.iterations !== maxIter || user.hits !== null || user.recall !== null || user.recommendedIds.length !== 0) {
            throw new Error("A nonconverged user cannot be accepted as a scored success");
          }
          failedUsers++;
          return;
        }
        if (user.residual >= tol) throw new Error("A converged user must satisfy the strict residual tolerance");
        const excluded = hw6ExclusionSet("val", user.userId);
        if (!Array.isArray(user.recommendedIds) || user.recommendedIds.length !== 10 || new Set(user.recommendedIds).size !== 10 ||
            user.recommendedIds.some(id => !HW6_STATE.candidateSet.has(id) || excluded.has(id))) throw new Error("Invalid saved validation Top-10");
        const hits = relevant.filter(id => user.recommendedIds.includes(id)).length;
        if (user.hits !== hits || Math.abs(user.recall - hits / relevant.length) > 1e-12) throw new Error("Saved hits/Recall mismatch");
        sum += hits / relevant.length;
      });
      if ((failedUsers ? row.recall !== null || row.status !== "failed" : Math.abs(row.recall - sum / current.cohort.length) > 1e-12 || row.status !== "complete") ||
          row.convergence.failedUsers !== failedUsers || row.convergence.convergedUsers !== current.cohort.length - failedUsers ||
          row.convergence.minIterations !== Math.min(...row.perUser.map(user => user.iterations)) ||
          row.convergence.maxIterations !== Math.max(...row.perUser.map(user => user.iterations)) ||
          row.convergence.maxResidual !== Math.max(...row.perUser.map(user => user.residual))) {
        throw new Error("Saved macro Recall or convergence count mismatch");
      }
    }
    if (value.status === "failed" && allowPartial) {
      if (value.selectedAlpha !== null || !value.error) throw new Error("A failed report must explicitly have no winner");
      return value;
    }
    const winner = choose(value.rows);
    if (winner === null || winner !== value.selectedAlpha) throw new Error("Saved winner is inconsistent with the complete validation grid");
    return value;
  }

  async function load(value, source = "explicit validation result", options = {}) {
    publish(null);
    try {
      await validate(value, null, Boolean(options.allowPartial));
      publish({ ...value, loadedFrom: source });
      return result;
    } catch (error) {
      publish({ status: "failed", rows: [], selectedAlpha: null, error: error.message });
      throw error;
    }
  }

  async function restore() {
    try {
      const response = await fetch("alpha-validation.json", { cache: "no-store" });
      if (response.status === 404) return null;
      if (!response.ok) throw new Error(`Validation result fetch failed (${response.status})`);
      return await load(await response.json(), "alpha-validation.json", { allowPartial: true });
    } catch (error) {
      publish({ status: "failed", rows: [], selectedAlpha: null, error: error.message });
      return result;
    }
  }

  function recordFor(user, current) {
    const relevantIds = current.protocol.relevantByUser.get(user.userId);
    const recommendedIds = user.converged ? HW6_RETRIEVAL.rankPropagation(user.userId, user, current.graph, 10, "val").map(rec => rec.movieId) : [];
    const hits = user.converged ? relevantIds.filter(id => recommendedIds.includes(id)).length : null;
    return { userId: user.userId, hits, relevant: relevantIds.length, recall: hits === null ? null : hits / relevantIds.length,
      relevantIds, recommendedIds, iterations: user.iterations, residual: user.residual, converged: user.converged,
      mass: user.mass === undefined ? user.p.reduce((sum, value) => sum + value, 0) : user.mass,
      backend: user.backend || tf.getBackend(),
      selectedSolver: user.converged ? "batch" : null };
  }

  /**
   * A capped batch is not accepted. One independent ordinary-PPR attempt may
   * resolve float32 reduction-order plateaus, with the same alpha/tolerance and
   * 500-update cap. Both attempts remain in the evidence; no threshold is relaxed.
   */
  async function retryCapped(user, alpha, current) {
    const scalar = await personalizedPageRank(current.graph.nodeOfUser(user.userId), alpha,
      { graph: current.graph, tol, maxIter });
    const record = recordFor({ ...scalar, userId: user.userId }, current);
    record.selectedSolver = scalar.converged ? "ordinary-ppr-retry" : null;
    record.attempts = [
      { solver: "batch", backend: user.backend || tf.getBackend(), iterations: user.iterations, residual: user.residual, converged: false },
      { solver: "ordinary-ppr-retry", backend: scalar.backend, iterations: scalar.iterations, residual: scalar.residual, converged: scalar.converged, runtimeMs: scalar.runtimeMs },
    ];
    return record;
  }

  function summarize(row) {
    const failedUsers = row.perUser.filter(user => !user.converged).length;
    row.convergence = { convergedUsers: row.perUser.length - failedUsers, failedUsers,
      minIterations: Math.min(...row.perUser.map(user => user.iterations)),
      maxIterations: Math.max(...row.perUser.map(user => user.iterations)),
      maxResidual: Math.max(...row.perUser.map(user => user.residual)),
      scalarRetries: row.perUser.filter(user => user.attempts).length,
      cpuRetries: row.perUser.filter(user => user.attempts?.some(attempt => attempt.backend === "cpu")).length,
      failedAttempts: row.perUser.reduce((sum, user) => sum + (user.attempts || []).filter(attempt => !attempt.converged).length, 0) };
    row.status = failedUsers ? "failed" : "complete";
    row.recall = failedUsers ? null : row.perUser.reduce((sum, user) => sum + user.recall, 0) / row.cohortSize;
  }

  /**
   * User-requested CPU recovery for the three remaining caps only. First qualify
   * CPU against WebGL on user 1 at the same alpha, then retry 54/310/595. Keep
   * every prior attempt and its backend; never change the cohort or tolerance.
   */
  async function recoverCpu(value) {
    publish(null);
    const started = performance.now();
    const output = structuredClone(value);
    const originalBackend = tf.getBackend();
    let before = null;
    let cpuStarted = false;
    const verification = { generatedAt: new Date().toISOString(), controlUserId: 1, alpha: 0.95,
      tolerance: tol, maxIterations: maxIter, status: "running", fullTestRecallRun: false };
    try {
      const current = await context();
      await validate(output, current, true);
      if (output.status === "complete") { publish(output); return output; }
      if (originalBackend !== "webgl") throw new Error("CPU recovery qualification requires the original WebGL backend");
      const row = output.rows.find(row => row.alpha === 0.95);
      const pendingUsers = row.perUser.filter(user => !user.converged).map(user => user.userId);
      if (JSON.stringify(pendingUsers) !== JSON.stringify([54, 310, 595])) throw new Error("CPU recovery scope must be exactly users 54, 310 and 595");
      for (const alphaRow of output.rows) for (const user of alphaRow.perUser) {
        user.backend = user.backend || output.backend;
        if (user.attempts) user.attempts.forEach(attempt => { attempt.backend = attempt.backend || output.backend; });
      }
      output.cpuVerification = verification;
      output.selectedAlpha = null;
      output.status = "running";
      delete output.error;
      before = tf.memory().numTensors;
      const controlStarted = performance.now();
      hw6SetStatus("Qualifying TensorFlow.js CPU against WebGL on validation user 1...");
      const webgl = await personalizedPageRank(current.graph.nodeOfUser(1), 0.95, { graph: current.graph, tol, maxIter });
      verification.webgl = { backend: webgl.backend, iterations: webgl.iterations, residual: webgl.residual,
        converged: webgl.converged, runtimeMs: webgl.runtimeMs };
      if (!webgl.converged) throw new Error("WebGL control user did not converge");
      if (!await tf.setBackend("cpu")) throw new Error("TensorFlow.js CPU backend could not be selected");
      cpuStarted = true;
      await tf.ready();
      const cpu = await personalizedPageRank(current.graph.nodeOfUser(1), 0.95, { graph: current.graph, tol, maxIter });
      verification.cpu = { backend: cpu.backend, iterations: cpu.iterations, residual: cpu.residual,
        converged: cpu.converged, runtimeMs: cpu.runtimeMs };
      if (!cpu.converged) throw new Error("CPU control user did not converge");
      verification.maxAbsScoreDifference = webgl.p.reduce((max, score, i) => Math.max(max, Math.abs(score - cpu.p[i])), 0);
      verification.webglTop10 = HW6_RETRIEVAL.rankPropagation(1, webgl, current.graph, 10, "val").map(rec => rec.movieId);
      verification.cpuTop10 = HW6_RETRIEVAL.rankPropagation(1, cpu, current.graph, 10, "val").map(rec => rec.movieId);
      verification.sameTop10 = JSON.stringify(verification.webglTop10) === JSON.stringify(verification.cpuTop10);
      if (verification.maxAbsScoreDifference >= 1e-6 || !verification.sameTop10) throw new Error("CPU/WebGL control scores or Top-10 disagree");
      verification.runtimeMs = performance.now() - controlStarted;
      verification.status = "passed";
      const retryStarted = performance.now();
      verification.retries = [];
      for (const userId of pendingUsers) {
        hw6SetStatus(`CPU PPR retry only: alpha=0.95, user=${userId}; tolerance/cap unchanged`);
        const index = row.perUser.findIndex(user => user.userId === userId);
        const previous = row.perUser[index];
        const scalar = await personalizedPageRank(current.graph.nodeOfUser(userId), 0.95, { graph: current.graph, tol, maxIter });
        const replacement = recordFor({ ...scalar, userId }, current);
        replacement.selectedSolver = scalar.converged ? "ordinary-ppr-cpu-retry" : null;
        replacement.attempts = [...previous.attempts, { solver: "ordinary-ppr-cpu-retry", backend: scalar.backend,
          iterations: scalar.iterations, residual: scalar.residual, converged: scalar.converged, runtimeMs: scalar.runtimeMs }];
        row.perUser[index] = replacement;
        verification.retries.push({ userId, ...replacement.attempts[replacement.attempts.length - 1] });
      }
      const retryRuntimeMs = performance.now() - retryStarted;
      row.runtimeMs += retryRuntimeMs;
      row.backendsUsed = ["webgl", "cpu"];
      summarize(row);
      await tf.setBackend(originalBackend); await tf.ready(); cpuStarted = false;
      verification.tensorsBefore = before;
      verification.tensorsAfter = tf.memory().numTensors;
      if (verification.tensorsAfter !== before) throw new Error("CPU recovery leaked tensors");
      output.backendsUsed = ["webgl", "cpu"];
      output.solver.cpuRecovery = "qualified CPU on user 1, then retried only 54/310/595; identical formula, initial v, L1<1e-6 and maxIter=500";
      output.cpuRecoveryRuntimeMs = performance.now() - started;
      output.runtimeMs += output.cpuRecoveryRuntimeMs;
      output.evaluationMs += retryRuntimeMs;
      output.runtimeScope = "Initial validation grid plus WebGL capped-user retry plus qualified CPU recovery; idle time excluded";
      output.completedAt = new Date().toISOString();
      output.selectedAlpha = choose(output.rows);
      if (output.selectedAlpha === null) throw new Error("CPU recovery left nonconverged users; no winner selected");
      output.status = "complete";
      await validate(output, current);
      publish(output);
      return output;
    } catch (error) {
      if (verification.status === "running") { verification.status = "failed"; verification.error = error.message; }
      output.cpuVerification = verification;
      output.status = "failed"; output.selectedAlpha = null; output.error = error.message;
      publish(output); return output;
    } finally {
      if (cpuStarted) { await tf.setBackend(originalBackend); await tf.ready(); }
    }
  }

  /** Resume only capped users from a saved qualified grid; never recompute hits for a failed attempt. */
  async function recover(value) {
    publish(null);
    const started = performance.now();
    const output = structuredClone(value);
    output.selectedAlpha = null;
    output.status = "running";
    delete output.error;
    try {
      const current = await context();
      if (JSON.stringify(output.identity) !== JSON.stringify(current.identity) ||
          output.verification?.status !== "passed" || output.rows.length !== 4) throw new Error("Recovery requires a matching qualified grid");
      const before = tf.memory().numTensors;
      output.recoveryStartedAt = new Date().toISOString();
      output.solver.scalarRetry = "one ordinary PPR attempt after a capped batch; identical alpha, L1<1e-6 and maxIter=500";
      for (const row of output.rows) {
        if (row.status === "complete") continue;
        const alphaStarted = performance.now();
        row.initialBatchConvergence = { ...row.convergence };
        for (let index = 0; index < row.perUser.length; index++) {
          const user = row.perUser[index];
          if (user.converged || user.attempts) continue;
          hw6SetStatus(`Ordinary PPR retry: alpha=${row.alpha}, user=${user.userId}; batch cap remains recorded`);
          row.perUser[index] = await retryCapped(user, row.alpha, current);
        }
        summarize(row);
        row.runtimeMs += performance.now() - alphaStarted;
      }
      output.recoveryRuntimeMs = performance.now() - started;
      output.runtimeMs += output.recoveryRuntimeMs;
      output.evaluationMs += output.recoveryRuntimeMs;
      output.runtimeScope = "Initial measured validation plus capped-user recovery; idle time excluded";
      output.completedAt = new Date().toISOString();
      output.tensorsAfter = tf.memory().numTensors;
      if (output.tensorsAfter !== before) throw new Error("Capped-user recovery leaked tensors");
      output.selectedAlpha = choose(output.rows);
      if (output.selectedAlpha === null) throw new Error("Some users remain nonconverged after the ordinary PPR retry; no winner selected");
      output.status = "complete";
      await validate(output, current);
      publish(output);
      return output;
    } catch (error) {
      output.status = "failed"; output.selectedAlpha = null; output.error = error.message;
      publish(output); return output;
    }
  }

  async function perform(options) {
    publish(null);
    const started = performance.now();
    const output = { schemaVersion: 1, status: "running", split: "val", startedAt: new Date().toISOString(),
      tolerance: tol, maxIterations: maxIter, batchSize, selectedAlpha: null, rows: [],
      backend: tf.getBackend(), tensorflowVersion: tf.version.tfjs, fullTestRecallRun: false,
      objective: "macro validation Recall@10", tieRule: "smaller alpha on exact equality",
      solver: { implementation: "HW6_BATCH_PPR", version: 1, dtype: "float32", rowLayout: "one restart user per row",
        transition: "shared normalized train R and R-transpose blocks", danglingPolicy: "return each row's dangling mass to its own restart distribution",
        frozenAtFirstConvergence: true },
      mfTrainingRuns: HW6_MF_ENGINE.trainingRuns };
    let engine = null;
    try {
      const current = await context();
      output.cohort = current.cohort;
      output.cohortSize = current.cohort.length;
      output.exclusions = current.protocol.exclusions;
      output.identity = current.identity;
      if (!HW6_ALPHA_CHECKS.matches(current.graph)) await HW6_ALPHA_CHECKS.run();
      output.verification = HW6_ALPHA_CHECKS.report;
      output.preparationMs = performance.now() - started;
      output.runtimeScope = "Validation selection call; previously completed qualification checks are recorded separately in verification.runtimeMs";
      const evaluationStarted = performance.now();
      const baseline = tf.memory().numTensors;
      engine = await HW6_BATCH_PPR.create(current.graph, { maxBatchSize: batchSize });
      for (const alpha of ALPHA_GRID) {
        const alphaStarted = performance.now();
        const row = { alpha, status: "running", cohortSize: current.cohort.length, recall: null, perUser: [] };
        output.rows.push(row);
        for (let start = 0; start < current.cohort.length; start += batchSize) {
          const userIds = current.cohort.slice(start, start + batchSize);
          const batch = await engine.runUsers(userIds, alpha, { tol, maxIter });
          for (const user of batch.results) {
            row.perUser.push(user.converged ? recordFor(user, current) : await retryCapped(user, alpha, current));
          }
          hw6SetStatus(`Validation PPR alpha=${alpha.toFixed(2)}: ${row.perUser.length}/${current.cohort.length} users`);
          await tf.nextFrame();
        }
        summarize(row);
        row.runtimeMs = performance.now() - alphaStarted;
        publish({ ...output, selectedAlpha: null });
        if (options.onProgress) await options.onProgress({ ...output, selectedAlpha: null });
      }
      engine.dispose(); engine = null;
      output.tensorsBefore = baseline;
      output.tensorsAfter = tf.memory().numTensors;
      if (output.tensorsAfter !== baseline) throw new Error("Validation selection leaked tensors");
      output.evaluationMs = performance.now() - evaluationStarted;
      output.runtimeMs = performance.now() - started;
      output.completedAt = new Date().toISOString();
      output.selectedAlpha = choose(output.rows);
      if (output.selectedAlpha === null) throw new Error("Validation grid has failed/incomplete/nonconverged runs; no winner selected");
      output.status = "complete";
      await validate(output, current);
      publish(output);
      return output;
    } catch (error) {
      output.status = "failed"; output.selectedAlpha = null; output.error = error.message;
      output.runtimeMs = performance.now() - started;
      publish(output);
      return output;
    } finally { if (engine) engine.dispose(); }
  }

  function run(options = {}) {
    if (inFlight) return inFlight;
    if (!options.force && result && (result.status === "complete" || (result.status === "failed" && result.rows.length === 4))) return Promise.resolve(result);
    inFlight = perform(options).finally(() => { inFlight = null; });
    return inFlight;
  }

  /** Primary comparison must use this stored value, never the demo slider. */
  function comparisonAlpha() {
    if (selected === null || !result || result.status !== "complete" || MF_INFO.runId !== result.identity.mfRunId) {
      throw new Error("A complete compatible validation selection is required for the primary comparison");
    }
    return selected;
  }

  function render(value = result) {
    const target = document.getElementById("alpha-sweep-results");
    if (!target) return;
    if (!value) { target.innerHTML = '<p class="empty-state">No complete validation alpha selection yet.</p>'; return; }
    const body = (value.rows || []).map(row => '<tr><td>' + row.alpha.toFixed(2) + '</td><td>' + hw6Fmt(row.recall, 8) +
      '</td><td>' + row.cohortSize + '</td><td>' + hw6Fmt(row.runtimeMs / 1000, 2) + '</td><td>' +
      hw6EscapeHtml((row.backendsUsed || [value.backend]).join('+')) + '</td><td>' +
      (row.convergence ? row.convergence.convergedUsers + '/' + row.cohortSize + ', max L1 ' + row.convergence.maxResidual.toExponential(8) : 'incomplete') +
      '</td><td>' + (value.status === "complete" && row.alpha === value.selectedAlpha ? 'selected' : hw6EscapeHtml(row.status)) + '</td></tr>').join('');
    target.innerHTML = '<table class="data-table"><thead><tr><th>Alpha</th><th>Validation Recall@10</th><th>Users</th><th>Seconds</th><th>Backend</th><th>Convergence</th><th>Status</th></tr></thead><tbody>' +
      body + '</tbody></table><p class="hint">' + (value.status === "complete" ? 'Selected alpha: ' + value.selectedAlpha +
      '. Validation only; exact ties use the smaller alpha. This value is independent of the example slider.' :
      'No winner selected: ' + hw6EscapeHtml(value.error || value.status)) + '</p>' +
      (value.exclusions ? hw6CohortSummaryHtml(value.exclusions) : '');
  }

  function csv(value) {
    const x = value.exclusions || {};
    const header = ['alpha', 'validation_recall_at_10', 'cohort_size', 'runtime_seconds', 'backend', 'batch_size',
      'converged_users', 'failed_users', 'min_iterations', 'max_iterations', 'max_residual', 'selected', 'status',
      'excluded_unique_users', 'no_train', 'no_positive_train', 'mf_unavailable', 'no_eligible_relevant', 'overlap_users', 'unseen_relevant_records',
      'total_runtime_seconds', 'evaluation_runtime_seconds', 'preparation_seconds', 'scalar_retry_users', 'cpu_retry_users', 'failed_attempts_retained'];
    const rows = value.rows.map(row => [row.alpha, row.recall, row.cohortSize, row.runtimeMs / 1000, (row.backendsUsed || [value.backend]).join('+'),
      value.batchSize, row.convergence?.convergedUsers, row.convergence?.failedUsers, row.convergence?.minIterations,
      row.convergence?.maxIterations, row.convergence?.maxResidual, value.status === 'complete' && row.alpha === value.selectedAlpha,
      row.status, x.excludedTotal, x.excludedNoTrain, x.excludedNoPositiveEdge, x.excludedMfUnavailable,
      x.excludedNoEligible, x.overlapUsers?.length, x.excludedUnseen, value.runtimeMs / 1000, value.evaluationMs / 1000, value.preparationMs / 1000,
      row.convergence?.scalarRetries || 0, row.convergence?.cpuRetries || 0, row.convergence?.failedAttempts || 0]);
    const users = [];
    for (const row of value.rows) for (const user of row.perUser) users.push([row.alpha, user.userId, user.hits, user.relevant,
      user.recall, user.iterations, user.residual, user.converged, user.mass, user.recommendedIds.join(';'), user.relevantIds.join(';'),
      user.selectedSolver || (user.converged ? "batch" : ""), user.backend || value.backend, JSON.stringify(user.attempts || [])]);
    return { summary: hw6CsvDocument(header, rows), users: hw6CsvDocument(['alpha', 'user_id', 'hits', 'relevant', 'recall',
      'iterations', 'residual', 'converged', 'mass', 'recommended_movie_ids', 'eligible_relevant_movie_ids', 'selected_solver', 'backend', 'solver_attempts'], users) };
  }

  return { run, recover, recoverCpu, restore, load, validate, choose, render, csv, comparisonAlpha, reset,
    get selectedAlpha() { return selected; }, get result() { return result; },
    snapshot: () => ({ selected, result }), restoreSnapshot: saved => { selected = saved.selected; result = saved.result; } };
})();
