/**
 * Shared HW6 retrieval protocol. Classic script; application globals are read
 * lazily. No training, alpha search or automatic population Recall runs here.
 * Missing/non-finite scores are errors, never zero-score graph candidates.
 */
const HW6_RETRIEVAL = (function () {
  const methods = ["popularity", "pagerank", "ppr", "mf"];
  let coverage = null;
  let coveragePromise = null;

  function splitName(name) {
    if (name !== "val" && name !== "test") throw new Error("Retrieval: split must be val or test");
    return name;
  }

  function reset() { coverage = null; coveragePromise = null; }

  function coverageReady() {
    return coverage && coverage.model === HW6_MF_MODEL && coverage.candidates === HW6_STATE.candidateIds &&
      coverage.users === HW6_STATE.trainUserIds;
  }

  /** Verify actual scorer output for every train user and every train candidate. */
  async function prepareCoverage() {
    if (coverageReady()) return coverage;
    if (coveragePromise) return coveragePromise;
    coveragePromise = Promise.resolve().then(async () => {
      if (HW6_STATE.trainUserIds.length) await hw6EnsureMf();
      const model = HW6_MF_MODEL;
      const candidates = HW6_STATE.candidateIds;
      const trainUsers = HW6_STATE.trainUserIds;
      const users = Array.from(new Set([...trainUsers, ...HW6_STATE.gradesBySplit.val.keys(),
        ...HW6_STATE.gradesBySplit.test.keys()])).sort((a, b) => a - b);
      const available = new Set();
      const unavailable = new Map();
      for (let index = 0; index < users.length; index++) {
        const userId = users[index];
        try {
          const scored = mfScores(userId);
          const bad = candidates.filter(id => !Object.prototype.hasOwnProperty.call(scored, id) ||
            typeof scored[id] !== "number" || !Number.isFinite(scored[id]));
          if (bad.length) unavailable.set(userId, { reason: "missing or invalid MF scores", movieIds: bad });
          else available.add(userId);
        } catch (error) {
          unavailable.set(userId, { reason: error.message, movieIds: [] });
        }
        if (index % 32 === 31 && typeof tf !== "undefined") await tf.nextFrame();
      }
      if (model !== HW6_MF_MODEL || candidates !== HW6_STATE.candidateIds || trainUsers !== HW6_STATE.trainUserIds) {
        throw new Error("Retrieval: model/data changed during MF coverage verification");
      }
      coverage = { model, candidates, users: trainUsers, available, unavailable };
      return coverage;
    }).finally(() => { coveragePromise = null; });
    return coveragePromise;
  }

  /**
   * Count exclusions using independent reason sets and their unique union.
   * The population is train users UNION users observed in the evaluation split.
   * allowUnverified is ONLY for provisional selectors before loading MF; it must
   * never supply a primary comparison or Recall report.
   */
  function protocol(name, allowUnverified = false) {
    splitName(name);
    const verified = Boolean(coverageReady());
    if (!verified && !allowUnverified) throw new Error("Retrieval: MF coverage has not been verified");
    const trainUsers = new Set(HW6_STATE.trainUserIds);
    const grades = HW6_STATE.gradesBySplit[name];
    const population = Array.from(new Set([...trainUsers, ...grades.keys()])).sort((a, b) => a - b);
    const reasonUsers = { noTrain: [], noPositiveTrain: [], mfUnavailable: [], noEligibleRelevant: [] };
    const excludedUsers = [];
    const cohort = [];
    const relevantByUser = new Map();
    const unseenIds = new Set();
    let unseenRelevant = 0;
    let previouslyRatedRelevant = 0;
    for (const userId of population) {
      const excluded = hw6ExclusionSet(name, userId);
      const relevant = [];
      const rows = grades.get(userId) || new Map();
      rows.forEach((rating, movieId) => {
        if (rating < 4) return;
        if (!HW6_STATE.candidateSet.has(movieId)) { unseenRelevant++; unseenIds.add(movieId); }
        else if (excluded.has(movieId)) previouslyRatedRelevant++;
        else relevant.push(movieId);
      });
      relevant.sort((a, b) => a - b);
      relevantByUser.set(userId, relevant);
      const reasons = [];
      if (!trainUsers.has(userId)) reasons.push("noTrain");
      if (!HW6_STATE.usersWithPositiveTrain.has(userId)) reasons.push("noPositiveTrain");
      if (verified && !coverage.available.has(userId)) reasons.push("mfUnavailable");
      if (!relevant.length) reasons.push("noEligibleRelevant");
      reasons.forEach(reason => reasonUsers[reason].push(userId));
      if (!reasons.length) cohort.push(userId);
      else excludedUsers.push({ userId, reasons });
    }
    const overlapUsers = excludedUsers.filter(row => row.reasons.length > 1).map(row => row.userId);
    const exclusions = {
      split: name, verified, populationUsers: population.length, candidateUsers: population.length,
      cohortSize: cohort.length, excludedTotal: excludedUsers.length,
      excludedNoTrain: reasonUsers.noTrain.length,
      excludedNoPositiveEdge: reasonUsers.noPositiveTrain.length,
      excludedMfUnavailable: verified ? reasonUsers.mfUnavailable.length : null,
      excludedNoEligible: reasonUsers.noEligibleRelevant.length,
      excludedUnseen: unseenRelevant, unseenIds: Array.from(unseenIds).sort((a, b) => a - b),
      unseenDistinctMovies: unseenIds.size, previouslyRatedRelevant,
      reasonUsers, excludedUsers, overlapUsers,
      mfFailures: verified ? Array.from(coverage.unavailable, ([userId, detail]) => ({ userId, ...detail })) : [],
      countingNote: "Reason counts overlap; excludedTotal counts their union, not their sum. Unseen counts user-movie relevance records.",
    };
    return { cohort, relevantByUser, exclusions, verified };
  }

  async function sharedCohort(name) { await prepareCoverage(); return protocol(splitName(name)).cohort; }

  /** Movie-only output aligned exactly with graph.movieIds; zero stays zero. */
  function movieScores(p, graph) {
    if (!graph || !Array.isArray(graph.userIds) || !Array.isArray(graph.movieIds) ||
        !p || p.length !== hw6NodeCount(graph)) throw new Error("movieScores: invalid graph/vector dimensions");
    const nodes = graph.movieNodes || graph.movieIds.map((_, i) => graph.userIds.length + i);
    if (nodes.length !== graph.movieIds.length || new Set(nodes).size !== nodes.length) {
      throw new Error("movieScores: invalid movie-node mapping");
    }
    return Float64Array.from(nodes, node => {
      if (!Number.isInteger(node) || node < graph.userIds.length || node >= p.length) {
        throw new Error("movieScores: movie mapping points outside the movie block");
      }
      if (typeof p[node] !== "number" || !Number.isFinite(p[node]) || p[node] < 0) {
        throw new Error(`movieScores: invalid probability at node ${node}`);
      }
      return p[node];
    });
  }

  /** Filter before ranking. Invalid eligible scores fail explicitly; never pad. */
  async function recommend(method, userId, K = 10, name = "test", options = {}) {
    splitName(name);
    if (!methods.includes(method) || !Number.isFinite(K) || K <= 0) return [];
    if (!Number.isInteger(K)) throw new Error("recommendTopK: K must be an integer");
    if (!HW6_STATE.trainUserIds.includes(userId)) return [];
    const alpha = options.alpha === undefined ? hw6ReadAlpha() : options.alpha;
    const candidates = HW6_STATE.candidateIds;
    const excluded = hw6ExclusionSet(name, userId);
    const eligible = candidates.filter(id => !excluded.has(id));
    if (!eligible.length) return [];
    if (method === "mf") await hw6EnsureMf();
    const scored = await hw6ScoreVector(method, userId, alpha, name);
    if (candidates !== HW6_STATE.candidateIds || scored.movieIds !== candidates || scored.scores.length !== candidates.length) {
      throw new Error("recommendTopK: candidate alignment changed or is invalid");
    }
    return rankScores(userId, scored, K, name, method);
  }

  /** Shared deterministic ranking for scalar and batch propagation results. */
  function rankScores(userId, scored, K, name, method = "ppr") {
    splitName(name);
    const candidates = HW6_STATE.candidateIds;
    const excluded = hw6ExclusionSet(name, userId);
    if (scored.movieIds.length !== candidates.length || scored.movieIds.some((id, i) => id !== candidates[i]) ||
        scored.scores.length !== candidates.length) throw new Error("Ranking: misaligned candidate scores");
    if (scored.propagation && (!scored.propagation.converged || !(scored.propagation.residual < 1e-6))) {
      throw new Error("recommendTopK: propagation did not converge");
    }
    const metadata = new Map(movies.map(movie => [movie.id, movie]));
    const recs = [];
    candidates.forEach((movieId, index) => {
      if (excluded.has(movieId)) return;
      const score = scored.scores[index];
      if (typeof score !== "number" || !Number.isFinite(score)) {
        throw new Error(`recommendTopK: missing or invalid ${method} score for movie ${movieId}; it is not a zero score`);
      }
      const movie = metadata.get(movieId);
      if (!movie) throw new Error(`recommendTopK: missing metadata for train movie ${movieId}`);
      recs.push({ movieId, title: movie.title, genres: movie.genres.slice(), score, popularity: hw6PopularityOf(movieId) });
    });
    recs.sort((a, b) => b.score - a.score || a.movieId - b.movieId);
    return recs.slice(0, K);
  }

  /** Consume verified propagation without re-running PPR or changing filtering. */
  function rankPropagation(userId, result, graph, K = 10, name = "val") {
    return rankScores(userId, { movieIds: graph.movieIds, scores: movieScores(result.p, graph), propagation: result }, K, name);
  }

  /** Macro Recall: one shared intersection per split, not a method-specific set. */
  async function recall(method, K = 10, name = "test", options = {}) {
    splitName(name);
    if (!methods.includes(method)) throw new Error("recallAtK: unknown method");
    if (!Number.isInteger(K) || K < 0) throw new Error("recallAtK: K must be a nonnegative integer");
    await prepareCoverage();
    const { cohort, relevantByUser, exclusions } = protocol(name);
    const alpha = options.alpha === undefined ? hw6ReadAlpha() : options.alpha;
    const perUser = [];
    for (const userId of cohort) {
      const relevant = relevantByUser.get(userId);
      const recs = await recommendTopK(method, userId, K, name, { alpha });
      const excluded = hw6ExclusionSet(name, userId);
      const expectedLength = Math.min(K, HW6_STATE.candidateIds.filter(id => !excluded.has(id)).length);
      if (recs.length !== expectedLength || new Set(recs.map(row => row.movieId)).size !== recs.length) {
        throw new Error("recallAtK: incomplete or duplicate recommendation list");
      }
      const retrieved = new Set(recs.map(row => row.movieId));
      const hit = relevant.filter(id => retrieved.has(id)).length;
      perUser.push({ userId, relevant: relevant.length, hit, recall: hit / relevant.length });
    }
    return { method, split: name, K, alpha: method === "ppr" || method === "pagerank" ? alpha : null,
      recall: perUser.length ? perUser.reduce((sum, row) => sum + row.recall, 0) / perUser.length : 0,
      cohort, excludedUnseen: exclusions.excludedUnseen, excludedNoEligible: exclusions.excludedNoEligible,
      exclusions, perUser };
  }

  /** Eight fixed-alpha TEST lists; eligibility is checked but no Recall is run. */
  async function preview(userIds = [1, 2], alpha = 0.85) {
    const cohort = await sharedCohort("test");
    if (userIds.length !== 2 || userIds[0] === userIds[1] || userIds.some(id => !cohort.includes(id))) {
      throw new Error("Preview: choose two different users from the verified common test cohort");
    }
    const lists = [];
    const overlap = [];
    for (const method of methods) {
      const pair = [];
      for (const userId of userIds) {
        const recs = await recommendTopK(method, userId, 10, "test", { alpha });
        if (recs.length !== 10) throw new Error("Preview: expected ten eligible scored movies");
        const settingAlpha = method === "ppr" || method === "pagerank" ? alpha : null;
        const list = { userId, method, alpha: settingAlpha, split: "test", recommendations: recs,
          meanPopularity: hw6MeanPopularity(recs), zeroScoreCount: recs.filter(rec => rec.score === 0).length,
          scoreMeaning: method === "ppr" || method === "pagerank" ? "long-run graph-walk probability" :
            method === "mf" ? "unclipped trainMean + dot product" : "distinct positive training users" };
        lists.push(list); pair.push(list);
      }
      const first = new Set(pair[0].recommendations.map(rec => rec.movieId));
      const commonMovieIds = pair[1].recommendations.filter(rec => first.has(rec.movieId)).map(rec => rec.movieId).sort((a, b) => a - b);
      overlap.push({ method, users: userIds.slice(), count: commonMovieIds.length, commonMovieIds });
    }
    const result = { generatedAt: new Date().toISOString(), split: "test", users: userIds.slice(), alpha,
      alphaStatus: `fixed ${alpha} preview, not validation-selected`, candidateCount: HW6_STATE.candidateIds.length,
      exclusions: { val: protocol("val").exclusions, test: protocol("test").exclusions },
      mf: { sourceCommit: MF_INFO.sourceCommit, runId: MF_INFO.runId, checkpoint: MF_INFO.checkpoint,
        loadedFrom: MF_INFO.loadedFrom, trainingRuns: HW6_MF_ENGINE.trainingRuns },
      lists, overlap, fullRecallRun: false, alphaSweepRun: false };
    // Export only after all eight lists have been produced successfully.
    lists.forEach(list => hw6RecordRecommendationRows(list.userId, list.method, list.alpha, list.recommendations));
    HW6_STATE.demo = result;
    return result;
  }

  return { methods, splitName, reset, coverageReady, prepareCoverage, protocol, sharedCohort,
    movieScores, recommend, recall, preview, rankScores, rankPropagation,
    snapshot: () => ({ coverage, coveragePromise }),
    restore: saved => { coverage = saved.coverage; coveragePromise = saved.coveragePromise; } };
})();
