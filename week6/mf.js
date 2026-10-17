/**
 * Seeded retraining of our HW5 MF method from Git commit 40a20cc.
 * Classic script, loaded before script.js; application globals are read lazily.
 * No retrieval metric, test-based selection or PageRank calculation lives here.
 * Checkpoints contain selected float32 factors, provenance, data/split hashes,
 * training history and raw score probes. IndexedDB survives page reloads; the
 * optional mf-checkpoint.json artifact also survives a new browser profile.
 */
const HW6_MF_ENGINE = (function () {
  const sourceCommit = "40a20cc59e6ce017f59a33ec18b5b224af875ce2";
  const sourceBlob = "2d9bfb017b38bebcf7420b8e890709a3c2c358af";
  const modelId = "hw5-40a20cc-seeded-retraining-v1";
  let trainingRuns = 0;
  let lastFailure = null;

  /** Exact scoring/loss/selection definitions; seeds are the intentional addition. */
  function spec() {
    return {
      modelId, sourceCommit, sourceBlob, sourceFile: "week5/script.js",
      factors: MF_FACTORS, formula: "trainMean + dot(userFactors, itemFactors)",
      optimizer: "tf.train.adam(lr); other arguments use TensorFlow.js defaults", ...MF_CONFIG,
      regularization: "reg * (mean(square(gatheredUsers)) + mean(square(gatheredItems)))",
      initialization: { distribution: "tf.randomNormal", mean: 0, stddev: 0.05,
        userSeed: MF_SEED, itemSeed: MF_SEED + 1 },
      shuffle: { algorithm: "mulberry32 + Fisher-Yates", seed: MF_SEED + 2, resetOrderEachEpoch: true },
      checkpointRule: "minimum validation RMSE; exact ties keep earliest epoch; restore factors",
      rmseClip: [1, 5], rankingClip: null,
      runKind: "seeded-retraining-not-original-hw5-weights",
    };
  }

  function stableJson(value) {
    if (Array.isArray(value)) return "[" + value.map(stableJson).join(",") + "]";
    if (value && typeof value === "object") {
      return "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + stableJson(value[key])).join(",") + "}";
    }
    return JSON.stringify(value);
  }

  /** SHA-256 of UTF-8 text or bytes; requires a secure context (localhost works). */
  async function digest(value) {
    const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
    const result = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(result), byte => byte.toString(16).padStart(2, "0")).join("");
  }

  /** Identity includes every loaded record and the exact ordered split arrays. */
  async function describeData() {
    if (!HW6_STATE.splitReady) throw new Error("MF: data/split are not ready");
    const splitHashes = {};
    const splitCounts = {};
    for (const name of ["train", "val", "test"]) {
      splitCounts[name] = HW6_SPLIT[name].length;
      splitHashes[name] = await digest(JSON.stringify(HW6_SPLIT[name]));
    }
    const descriptor = {
      dataset: "MovieLens 100K", sourceFiles: ["week6/u.data", "week6/u.item"],
      hashFormat: "SHA-256 of UTF-8 JSON.stringify ordered parsed records",
      users: new Set(ratings.map(row => row.userId)).size, movies: movies.length, ratings: ratings.length,
      ratingsSha256: await digest(JSON.stringify(ratings)), moviesSha256: await digest(JSON.stringify(movies)),
      splitRule: "per user: timestamp asc, movieId asc; floor(0.8*n), floor(0.1*n), remainder test; skip n<5",
      splitCounts, splitHashes,
      candidateCount: HW6_STATE.candidateIds.length,
      candidateIdsSha256: await digest(JSON.stringify(HW6_STATE.candidateIds)),
      userShape: [numUsers + 1, MF_FACTORS], itemShape: [HW6_MAX_MOVIE_ID + 1, MF_FACTORS],
      indexing: "raw userId/movieId factor rows; reserved row 0 is not a candidate or graph node",
      trainMean: HW6_SPLIT.train.reduce((sum, row) => sum + row.rating, 0) / HW6_SPLIT.train.length,
    };
    return { ...descriptor, signature: await digest(stableJson(descriptor)) };
  }

  /** Own both variables; tf.tidy also releases the initialization tensors. */
  function initialize(train, userRows = numUsers + 1, itemRows = HW6_MAX_MOVIE_ID + 1) {
    const globalMean = train.length ? train.reduce((sum, row) => sum + row.rating, 0) / train.length : 0;
    return tf.tidy(() => ({
      userFactors: tf.variable(tf.randomNormal([userRows, MF_FACTORS], 0, 0.05, "float32", MF_SEED)),
      itemFactors: tf.variable(tf.randomNormal([itemRows, MF_FACTORS], 0, 0.05, "float32", MF_SEED + 1)),
      globalMean,
    }));
  }

  /** HW5 scalar prediction: mean fallback for unknown rows, clipping for ratings. */
  function predictRating(userId, movieId, model) {
    if (!model) return 0;
    if (!Number.isInteger(userId) || !Number.isInteger(movieId) || userId < 0 || movieId < 0 ||
        userId >= model.userFactors.shape[0] || movieId >= model.itemFactors.shape[0]) return model.globalMean;
    return tf.tidy(() => {
      const f = model.userFactors.shape[1];
      const dot = model.userFactors.slice([userId, 0], [1, f])
        .mul(model.itemFactors.slice([movieId, 0], [1, f])).sum().dataSync()[0];
      return Math.min(5, Math.max(1, model.globalMean + dot));
    });
  }

  /** Batched computeRMSEFor from HW5: clipped predictions, mean for unknown IDs. */
  function rmse(split, model) {
    if (!split || !split.length) return 0;
    if (!model) return NaN;
    return tf.tidy(() => {
      const userRows = model.userFactors.shape[0];
      const itemRows = model.itemFactors.shape[0];
      const uid = Int32Array.from(split, row => row.userId);
      const iid = Int32Array.from(split, row => row.movieId);
      const valid = Float32Array.from(split, row => row.userId >= 0 && row.userId < userRows && row.movieId >= 0 && row.movieId < itemRows ? 1 : 0);
      const u = tf.gather(model.userFactors, tf.tensor1d(uid, "int32").clipByValue(0, userRows - 1));
      const i = tf.gather(model.itemFactors, tf.tensor1d(iid, "int32").clipByValue(0, itemRows - 1));
      const pred = u.mul(i).sum(1).mul(tf.tensor1d(valid)).add(model.globalMean).clipByValue(1, 5);
      return Math.sqrt(pred.sub(tf.tensor1d(Float32Array.from(split, row => row.rating))).square().mean().dataSync()[0]);
    });
  }

  /** Unclipped HW5 ranking scores. IDs come from the shared HW6 train pool. */
  function scores(userId, candidateIds, model) {
    if (!model || !Number.isInteger(userId) || userId < 0 || userId >= model.userFactors.shape[0]) return {};
    if (!candidateIds.length) return {};
    if (candidateIds.some(id => !Number.isInteger(id) || id <= 0 || id >= model.itemFactors.shape[0])) {
      throw new Error("MF: candidate ID has no factor row");
    }
    return tf.tidy(() => {
      const items = tf.gather(model.itemFactors, tf.tensor1d(Int32Array.from(candidateIds), "int32"));
      const user = model.userFactors.slice([userId, 0], [1, model.userFactors.shape[1]]);
      const values = tf.matMul(items, user, false, true).reshape([candidateIds.length]).add(model.globalMean).dataSync();
      const result = {};
      candidateIds.forEach((id, index) => { result[id] = values[index]; });
      return result;
    });
  }

  /** Caller owns the returned scalar; gathered rows, not whole-matrix norms. */
  function batchLoss(model, userIdx, itemIdx, targets, reg) {
    return tf.tidy(() => {
      const u = tf.gather(model.userFactors, userIdx);
      const i = tf.gather(model.itemFactors, itemIdx);
      const pred = u.mul(i).sum(1).add(model.globalMean);
      const mse = tf.losses.meanSquaredError(targets, pred);
      return mse.add(u.square().mean().add(i.square().mean()).mul(reg));
    });
  }

  /** Exact HW5 minimum rule, including earliest epoch on an exact tie. */
  function selectBest(history) {
    let best = null;
    for (const row of history || []) {
      if (!Number.isFinite(row.valRMSE)) throw new Error("MF: non-finite validation RMSE");
      if (!best || row.valRMSE < best.valRMSE || (row.valRMSE === best.valRMSE && row.epoch < best.epoch)) best = row;
    }
    return best;
  }

  /**
   * HW5 mini-batch Adam loop with seeded Fisher-Yates instead of Math.random.
   * Resets the index array each epoch, keeps the first exact minimum and restores
   * its cloned variables. No test split is read. Model is updated in place.
   * Optional config/onEpoch exist for small regression fixtures, not tuning.
   */
  async function fit(model, train, val, config = MF_CONFIG, shuffleSeed = MF_SEED + 2, onEpoch = () => {}) {
    if (!train.length || !val.length) throw new Error("MF: training and validation must be nonempty");
    const optimizer = tf.train.adam(config.lr);
    const rand = hw6Mulberry32(shuffleSeed);
    const history = [];
    let best = null;
    let bestU = null;
    let bestI = null;
    try {
      for (let epoch = 1; epoch <= config.epochs; epoch++) {
        const indices = Array.from({ length: train.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        for (let start = 0; start < indices.length; start += config.batchSize) {
          const batch = indices.slice(start, start + config.batchSize).map(index => train[index]);
          // The outer tidy owns index/target tensors as well as gradient intermediates.
          tf.tidy(() => {
            const u = tf.tensor1d(Int32Array.from(batch, row => row.userId), "int32");
            const i = tf.tensor1d(Int32Array.from(batch, row => row.movieId), "int32");
            const targets = tf.tensor1d(Float32Array.from(batch, row => row.rating));
            optimizer.minimize(() => batchLoss(model, u, i, targets, config.reg), false,
              [model.userFactors, model.itemFactors]);
          });
        }
        const row = { epoch, trainRMSE: rmse(train, model), valRMSE: rmse(val, model) };
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) throw new Error("MF: non-finite epoch metric");
        history.push(row);
        if (!best || row.valRMSE < best.valRMSE) {
          tf.dispose([bestU, bestI].filter(Boolean));
          best = row;
          bestU = model.userFactors.clone();
          bestI = model.itemFactors.clone();
        }
        await onEpoch(row);
        await tf.nextFrame();
      }
      if (!best) throw new Error("MF: no validation checkpoint selected");
      model.userFactors.assign(bestU);
      model.itemFactors.assign(bestI);
      return { history, best };
    } finally {
      tf.dispose([bestU, bestI].filter(Boolean));
      optimizer.dispose();
    }
  }

  /** One full run per page session, with fixed settings; selection never sees test. */
  async function retrain(train) {
    if (trainingRuns || MF_INFO.checkpoint !== null) throw new Error("MF: refusing an automatic second training run");
    if (train !== HW6_SPLIT.train) throw new Error("MF: full retraining requires the current training split");
    trainingRuns++;
    await tf.ready();
    const data = await describeData();
    const started = performance.now();
    hw6DisposeMf();
    HW6_MF_MODEL = initialize(train);
    MF_INFO.history = [];
    try {
      const result = await fit(HW6_MF_MODEL, train, HW6_SPLIT.val, MF_CONFIG, MF_SEED + 2, row => {
        MF_INFO.history.push(row);
        hw6SetStatus(`Retraining HW5 MF (new seeded run): epoch ${row.epoch}/${MF_CONFIG.epochs}, validation RMSE ${hw6Fmt(row.valRMSE)}`);
      });
      // Only now, after best-factor restoration, may test be measured.
      const testRMSE = rmse(HW6_SPLIT.test, HW6_MF_MODEL);
      Object.assign(MF_INFO, {
        sourceCommit, sourceBlob, modelId, runKind: spec().runKind,
        runId: `${modelId}-${Date.now()}`, backend: tf.getBackend(), tensorflowVersion: tf.version.tfjs,
        checkpoint: result.best.epoch, validationRMSE: result.best.valRMSE, testRMSE,
        trainedAt: new Date().toISOString(), runtimeMs: performance.now() - started,
        history: result.history, datasetSignature: data.signature, loadedFrom: "new seeded retraining", error: null,
      });
      return HW6_MF_MODEL;
    } catch (error) {
      hw6DisposeMf();
      throw error;
    }
  }

  /** Transaction resolves only after IndexedDB commit, not request success alone. */
  async function storage(action, key = modelId, payload = null) {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open("hw6-mf-checkpoints", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("models");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    try {
      return await new Promise((resolve, reject) => {
        const transaction = db.transaction("models", action === "get" ? "readonly" : "readwrite");
        const store = transaction.objectStore("models");
        const request = action === "get" ? store.get(key) : action === "put" ? store.put(payload, key) : store.delete(key);
        transaction.oncomplete = () => resolve(request.result || null);
        transaction.onerror = () => reject(transaction.error || request.error);
        transaction.onabort = () => reject(transaction.error || new Error("MF: IndexedDB transaction aborted"));
      });
    } finally { db.close(); }
  }

  /** Selected factors + metadata + full train-candidate score probes for two users. */
  async function pack() {
    if (!HW6_MF_MODEL || MF_INFO.checkpoint === null) throw new Error("MF: no selected model to save");
    const [userValues, itemValues] = await Promise.all([
      HW6_MF_MODEL.userFactors.data(), HW6_MF_MODEL.itemFactors.data(),
    ]);
    const data = await describeData();
    const probeUsers = Array.from(new Set([HW6_STATE.trainUserIds[0], HW6_STATE.trainUserIds[HW6_STATE.trainUserIds.length - 1]]));
    const probes = probeUsers.map(userId => {
      const scored = scores(userId, HW6_STATE.candidateIds, HW6_MF_MODEL);
      return { userId, scores: HW6_STATE.candidateIds.map(id => scored[id]) };
    });
    const info = JSON.parse(JSON.stringify(MF_INFO));
    delete info.loadedFrom;
    delete info.loadVerification;
    delete info.error;
    const payload = { schemaVersion: 1, spec: spec(), data, info,
      globalMean: HW6_MF_MODEL.globalMean,
      weights: { userShape: HW6_MF_MODEL.userFactors.shape, itemShape: HW6_MF_MODEL.itemFactors.shape,
        userValues: Array.from(userValues), itemValues: Array.from(itemValues) }, probes };
    return { ...payload, checksum: await digest(stableJson(payload)) };
  }

  /**
   * Reject incompatible/corrupt checkpoints before replacing the current model.
   * On a compatible load verify raw scores against saved probes (absolute 1e-5,
   * allowing float32 kernel differences), and recompute clipped validation RMSE.
   */
  async function load(payload, loadedFrom = "explicit checkpoint") {
    await tf.ready();
    if (!payload || payload.schemaVersion !== 1 || stableJson(payload.spec) !== stableJson(spec())) {
      throw new Error("MF: incompatible checkpoint schema, provenance or settings");
    }
    const { checksum, ...body } = payload;
    if (checksum !== await digest(stableJson(body))) throw new Error("MF: checkpoint checksum mismatch");
    const current = await describeData();
    if (stableJson(payload.data) !== stableJson(current)) throw new Error("MF: checkpoint data/split/candidates do not match");
    const weights = payload.weights;
    for (const [kind, expected] of [["user", current.userShape], ["item", current.itemShape]]) {
      const values = weights[kind + "Values"];
      if (stableJson(weights[kind + "Shape"]) !== stableJson(expected) || !Array.isArray(values) ||
          values.length !== expected[0] * expected[1] || values.some(value => !Number.isFinite(value))) {
        throw new Error(`MF: invalid ${kind} factor dimensions/values`);
      }
    }
    const info = payload.info;
    const best = selectBest(info.history);
    if (!best || info.history.length !== MF_CONFIG.epochs || best.epoch !== info.checkpoint ||
        best.valRMSE !== info.validationRMSE || info.history.some((row, i) => row.epoch !== i + 1 || !Number.isFinite(row.trainRMSE)) ||
        !Number.isFinite(info.testRMSE) || !info.trainedAt || info.runKind !== spec().runKind ||
        payload.globalMean !== current.trainMean || info.datasetSignature !== current.signature) {
      throw new Error("MF: invalid checkpoint selection or run metadata");
    }
    let candidate = tf.tidy(() => ({
      userFactors: tf.variable(tf.tensor2d(weights.userValues, weights.userShape)),
      itemFactors: tf.variable(tf.tensor2d(weights.itemValues, weights.itemShape)),
      globalMean: payload.globalMean,
    }));
    try {
      let maxScoreError = 0;
      if (!Array.isArray(payload.probes) || payload.probes.length < 1) throw new Error("MF: missing score probes");
      for (const probe of payload.probes) {
        if (!HW6_STATE.trainUserIds.includes(probe.userId) || probe.scores.length !== HW6_STATE.candidateIds.length) {
          throw new Error("MF: invalid score probe IDs/dimensions");
        }
        const actual = scores(probe.userId, HW6_STATE.candidateIds, candidate);
        HW6_STATE.candidateIds.forEach((id, i) => {
          const error = Math.abs(actual[id] - probe.scores[i]);
          if (!Number.isFinite(error) || error > 1e-5) throw new Error(`MF: loaded score mismatch for user ${probe.userId}, movie ${id}`);
          maxScoreError = Math.max(maxScoreError, error);
        });
      }
      const validationRMSE = rmse(HW6_SPLIT.val, candidate);
      if (Math.abs(validationRMSE - info.validationRMSE) > 1e-5) throw new Error("MF: loaded validation RMSE mismatch");
      hw6DisposeMf();
      HW6_MF_MODEL = candidate;
      candidate = null;
      Object.assign(MF_INFO, info, { loadedFrom, error: null, loadVerification: { maxScoreError,
        scoreCount: payload.probes.length * HW6_STATE.candidateIds.length, validationRMSE, tolerance: 1e-5 } });
      lastFailure = null;
      return HW6_MF_MODEL;
    } finally {
      if (candidate) tf.dispose([candidate.userFactors, candidate.itemFactors]);
    }
  }

  async function save(key = modelId) {
    const payload = await pack();
    await storage("put", key, payload);
    lastFailure = null;
    MF_INFO.error = null;
    return payload;
  }

  /** Memory -> IndexedDB -> portable artifact -> one fixed retraining, in that order. */
  function ensure() {
    if (HW6_MF_TRAINING) return HW6_MF_TRAINING;
    if (lastFailure) return Promise.reject(lastFailure);
    if (HW6_MF_MODEL && MF_INFO.checkpoint !== null) return Promise.resolve(HW6_MF_MODEL);
    HW6_MF_TRAINING = Promise.resolve().then(async () => {
      if (typeof tf === "undefined" || !HW6_STATE.splitReady) throw new Error("MF: runtime/data are not ready");
      await tf.ready();
      const cached = await storage("get");
      if (cached) return load(cached, "IndexedDB");
      const response = await fetch("mf-checkpoint.json", { cache: "no-store" });
      if (response.ok) {
        const payload = await response.json();
        await load(payload, "mf-checkpoint.json");
        await storage("put", modelId, payload);
        return HW6_MF_MODEL;
      }
      if (response.status !== 404) throw new Error(`MF: checkpoint fetch failed (${response.status})`);
      await retrain(HW6_SPLIT.train);
      await save();
      return HW6_MF_MODEL;
    }).catch(error => {
      lastFailure = error;
      MF_INFO.error = error.message;
      throw error;
    }).finally(() => { HW6_MF_TRAINING = null; });
    return HW6_MF_TRAINING;
  }

  /** Called when indices are rebuilt, so an old in-memory model cannot be reused. */
  function invalidateData() {
    if (HW6_MF_TRAINING) throw new Error("MF: cannot replace the dataset while preparing the model");
    hw6DisposeMf();
    Object.assign(MF_INFO, { checkpoint: null, validationRMSE: null, testRMSE: null,
      trainedAt: null, loadedFrom: null, history: [], loadVerification: null, error: null });
    lastFailure = null;
  }

  return { spec, digest, describeData, initialize, predictRating, rmse, scores, batchLoss,
    selectBest, fit, retrain, storage, pack, load, save, ensure, invalidateData,
    get trainingRuns() { return trainingRuns; } };
})();
