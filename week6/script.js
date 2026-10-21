/**
 * week6/script.js — HW6 application layer.
 *
 * Graph-based recommendations with global PageRank, personalized PageRank
 * (PPR) and a matrix-factorization (MF) reference, compared under one shared
 * retrieval protocol: validation-selected alpha, one shared evaluation cohort,
 * Recall@10 and explicit exclusions.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads, in
 * order: the TensorFlow.js CDN runtime, `vendor/d3.min.js`, `data.js`,
 * The graph/batch, MF, retrieval/checks and saved-experiment controllers load
 * before this file, as listed explicitly in index.html.
 *
 * Globals provided by `data.js` (read, never redefined here):
 *   `movies`   `[{ id, title, genres: string[] }]`
 *   `ratings`  `[{ userId, movieId, rating, timestamp }]`
 *   `numUsers` `numMovies` `ratingMatrix` `genreNames` `SPLIT_EXCLUSIONS`
 *   `loadData()`  `splitByTimestamp()` -> `{ train, val, test }`
 *
 * Globals provided by `graph.js` (read, never redefined here):
 *   `REFERENCE_GRAPH`, `buildGraph`, `rowNormalize`, `pageRank`,
 *   `personalizedPageRank`, `GRAPH`, `setGraph`, `getGraph`,
 *   `renderGraphView`, `fixtureStep`, `fixtureReset`, `GRAPH_VIEW_STATE`
 *
 * IMPLEMENTED RETRIEVAL (shared protocol in retrieval.js):
 *   1. `movieScores(p, graph)`                   — movie-node scores, `graph.movieIds` order.
 *   2. `recommendTopK(method, userId, K, split)` — candidate pool, exclusions, tie-breaks.
 *   3. `recallAtK(method, K, split)`             — shared-cohort Recall@K plus exclusions.
 *
 * APPLICATION SUPPORT:
 *   - `hw6InitTf`, the formatting/CSV/exporter helpers, the training-split indices,
 *     `popularityScores`, `mfScores` and the seeded HW5 MF runtime,
 *     every panel renderer, the four-method / alpha-sweep / convergence /
 *     two-user-demo runners, the self-check harness and the UI wiring.
 *   - MF reuses our final HW5 method from 40a20cc. Its seeded retraining and
 *     persisted checkpoint are recorded explicitly; original HW5 weights are unavailable.
 *
 * Design notes:
 *   - The graph itself (transition matrix, PageRank, PPR, dangling-mass policy)
 *     lives in `graph.js`. This file never re-implements propagation for the
 *     product path; it calls `pageRank` / `personalizedPageRank` through the
 *     adapter below. Propagation errors and interface mismatches are failures,
 *     never PENDING. Primary test evaluation uses the saved validation alpha.
 *   - PPR scores are long-run graph-walk probabilities. They are NOT purchase
 *     probabilities, NOT predicted ratings and NOT comparable in scale to MF
 *     scores. No rating RMSE is ever computed for PPR.
 *   - `alpha` always means the link-following probability; `1 - alpha` restarts.
 *   - Alpha is selected on validation only. Test outcomes never choose settings.
 *
 * @module week6/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} RecommendedMovie
 * @property {number} movieId      Raw MovieLens movie id.
 * @property {string} title        Display title.
 * @property {string[]} genres     Genre names.
 * @property {number} score        Method score (PPR scores are walk probabilities).
 * @property {number} popularity   Distinct users with a positive training rating.
 */

/**
 * @typedef {Object} PropagationResult
 * @property {ArrayLike<number>} scores  Node-probability vector, length = node count.
 * @property {number|null} iterations    Iterations performed, when reported.
 * @property {number|null} residual      Final L1 difference, when reported.
 * @property {boolean|null} converged    `true` only when the tolerance was met.
 * @property {string|null} backend TensorFlow.js backend actually used.
 * @property {number|null} runtimeMs Runtime of the underlying calculation.
 * @property {string} source             `"graph.js"` or `"reference"` (self-check path).
 */

/**
 * @typedef {Object} RecallReport
 * @property {number} recall              Macro-average Recall@K over the shared cohort.
 * @property {number[]} cohort            Sorted user ids in the shared cohort.
 * @property {number} excludedUnseen      Relevant movies absent from the training pool.
 * @property {number} excludedNoEligible  Users with no eligible relevant movie.
 * @property {Array<{userId:number, relevant:number, hit:number}>} perUser
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Alpha grid swept on validation for PPR (readme section 6.1). */
const ALPHA_GRID = [0.50, 0.70, 0.85, 0.95];

/** Rank cut-off used throughout the assignment. */
const HW6_K = 10;

/** Final TEST action is enabled; compatible saved results are always preferred. */
const HW6_FULL_EXPERIMENTS_ENABLED = true;

/**
 * Legacy marker understood by the error renderer. All self-check exceptions,
 * including ones with this marker, are failures rather than PENDING.
 */
const HW6_TODO_MARKER = "TODO(hw6)";

/** Marker used for integration gaps; the harness reports them as failures. */
const HW6_INTERFACE_MARK = "HW6_INTERFACE";

/** Iteration cap and tolerance reported by the convergence panel (readme 3.4). */
const HW6_MAX_ITERATIONS = 500;
const HW6_TOLERANCE = 1e-6;

/** HW5 model from commit 40a20cc; this run is a seeded retraining. */
const MF_FACTORS = 16;

/** Base seed: user initialization, item initialization +1, shuffle +2. */
const MF_SEED = 20261013;

/** Training settings, keeping the previous assignment's convention. */
const MF_CONFIG = Object.freeze({ epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64 });

/**
 * Public description of the HW5 MF retraining, filled in when the
 * model is trained. The report must state the model and checkpoint actually
 * used, so these fields are the record.
 *
 * `trainedAt` is the lazy-training marker: it stays `null` until the reference
 * has actually been trained, so "not trained yet" is a truthful, checkable flag
 * rather than an inference from a zero score. `checkpoint` stays `null` while
 * untrained.
 *
 * @type {{factors:number, epochs:number, learningRate:number,
 *         regularization:number, batchSize:number, seed:number,
 *         backend:(string|null), checkpoint:(number|null),
 *         validationRMSE:(number|null), trainedAt:(string|null),
 *         history:Array<Object>}}
 */
const MF_INFO = {
  factors: MF_FACTORS,
  epochs: MF_CONFIG.epochs,
  learningRate: MF_CONFIG.lr,
  regularization: MF_CONFIG.reg,
  batchSize: MF_CONFIG.batchSize,
  seed: MF_SEED,
  backend: null,
  checkpoint: null,
  validationRMSE: null,
  testRMSE: null,
  trainedAt: null,
  loadedFrom: null,
  runKind: "seeded-retraining-not-original-hw5-weights",
  history: [],
};

/**
 * @typedef {Object} MfModel
 * @property {tf.Variable} userFactors Dense `[numUsers + 1, MF_FACTORS]`, row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense `[maxMovieId + 1, MF_FACTORS]`, row index = raw `movieId`.
 * @property {number} globalMean        Mean training rating.
 */

/**
 * HW5 MF model, or null until a compatible checkpoint is loaded or a new seeded
 * run completes. Loading a checkpoint never counts as another training run.
 *
 * @type {MfModel|null}
 */
let HW6_MF_MODEL = null;

/**
 * In-flight MF training promise. A second need while training is running
 * awaits the same promise instead of starting a second training pass.
 *
 * @type {Promise<MfModel|null>|null}
 */
let HW6_MF_TRAINING = null;

/** Highest raw movie id in the dataset; cached for factor allocation. */
let HW6_MAX_MOVIE_ID = 0;

/** Chronological 80/10/10 split, set once by the bootstrap. */
const HW6_SPLIT = { train: [], val: [], test: [] };

/** Mutable application state (all names prefixed so `graph.js` stays untouched). */
const HW6_STATE = {
  backend: null,
  loaded: false,
  splitReady: false,

  /** Sorted distinct movie ids observed in the training split (readme 5 candidate pool). */
  candidateIds: [],
  /** `Set` view of `candidateIds`. */
  candidateSet: new Set(),
  /** Sorted distinct user ids observed in the training split. */
  trainUserIds: [],
  /** `movieId -> count of distinct users with a positive (>= 4.0) training rating`. */
  popularity: new Map(),
  /** `userId -> Set(movieId)` per split; drives the previously-rated exclusion rules. */
  pairsBySplit: { train: new Map(), val: new Map(), test: new Map() },
  /** `userId -> Map(movieId -> rating)` per split; drives relevance. */
  gradesBySplit: { train: new Map(), val: new Map(), test: new Map() },
  /** Users with at least one positive training edge (readme 7.1). */
  usersWithPositiveTrain: new Set(),
  /** Count of distinct positive training (user, movie) pairs. */
  positivePairs: 0,

  /** Cached propagation results, keyed by alpha / seed. */
  pageRank: null,
  ppr: null,
  /** Convergence records for the panel: `{ pagerank, ppr }`. */
  convergence: { pagerank: null, ppr: null },

  /** Last runner outputs (also the exporters' sources). */
  comparisonRows: null,
  alphaSweepRows: null,
  comparisonNote: null,
  alphaSweepNote: null,
  demo: null,
  recallReport: null,
  recommendationRows: [],
  /** `Set` of recommendation row keys already recorded, to avoid duplicate CSV rows. */
  recommendationKeys: new Set(),

  /** Last self-check run, reused by `serializeVerification()`. */
  lastVerification: null,
};

// ---------------------------------------------------------------------------
// Small helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to fixed decimals; non-finite and missing values render as an
 * em dash so tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] Decimal places.
 * @returns {string}
 */
function hw6Fmt(x, d) {
  const places = d === undefined ? 4 : d;
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(places);
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {*} text
 * @returns {string}
 */
function hw6EscapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Look up a movie record by id.
 *
 * @param {number|string} id
 * @returns {Movie|undefined}
 */
function hw6MovieById(id) {
  const numericId = Number(id);
  for (let i = 0; i < movies.length; i++) {
    if (movies[i].id === numericId) return movies[i];
  }
  return undefined;
}

/**
 * Look up a movie title with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function hw6Title(id) {
  const movie = hw6MovieById(id);
  return movie ? movie.title : "Movie #" + id;
}

/**
 * Genre list for a movie id (empty array when unknown).
 *
 * @param {number|string} id
 * @returns {string[]}
 */
function hw6Genres(id) {
  const movie = hw6MovieById(id);
  return movie && Array.isArray(movie.genres) ? movie.genres : [];
}

/**
 * Build a diagnostic error for an invalid graph.js interface shape.
 *
 * @param {string} message
 * @returns {Error}
 */
function hw6InterfaceError(message) {
  const error = new Error(HW6_INTERFACE_MARK + ": " + message);
  error.hw6Interface = true;
  return error;
}

/**
 * Read a numeric DOM value with a fallback.
 *
 * @param {string} id
 * @param {number} fallback
 * @returns {number}
 */
function hw6ReadNumber(id, fallback) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  const raw = el.value;
  // An empty select / number input must fall back rather than read as 0.
  if (raw === undefined || raw === null || String(raw).trim() === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Read the selected method, falling back to `popularity` for unknown values.
 *
 * @returns {"popularity"|"pagerank"|"ppr"|"mf"}
 */
function hw6ReadMethod() {
  const el = document.getElementById("method");
  const value = el ? String(el.value) : "popularity";
  return value === "pagerank" || value === "ppr" || value === "mf" ? value : "popularity";
}

/**
 * Read the alpha slider as the link-following probability.
 *
 * @returns {number} Alpha clamped into `[0.50, 0.95]`.
 */
function hw6ReadAlpha() {
  const value = hw6ReadNumber("alpha", 0.85);
  return Math.min(0.95, Math.max(0.50, value));
}

/**
 * Set the alpha slider, refresh the live readout, and invalidate cached results.
 * The slider must never dispatch an event here (that would recurse).
 *
 * @param {number} value
 * @returns {void}
 */
function hw6SetAlpha(value) {
  const slider = document.getElementById("alpha");
  const readout = document.getElementById("alpha-value");
  const numeric = Number(value);
  const clamped = Math.min(0.95, Math.max(0.50, Number.isFinite(numeric) ? numeric : 0.85));
  if (slider) slider.value = String(clamped);
  if (readout) readout.textContent = clamped.toFixed(2);
  hw6InvalidateCaches();
}

/**
 * Drop cached PageRank / PPR / recommendation results because a control changed.
 * `graph.js` owns the graph and any cache of its own; if it exposes a `stale`
 * flag we set it, but no field is invented here.
 *
 * @returns {void}
 */
function hw6InvalidateCaches() {
  HW6_STATE.pageRank = null;
  HW6_STATE.ppr = null;
  HW6_STATE.convergence = { pagerank: null, ppr: null };
  HW6_STATE.lastVerification = null;
  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  if (viewState && typeof viewState === "object" && "stale" in viewState) {
    viewState.stale = true;
  }
}

/**
 * Write the status line without ever throwing.
 *
 * @param {string} message
 * @returns {void}
 */
function hw6SetStatus(message) {
  const status = document.getElementById("status");
  if (status) status.textContent = message;
}

/**
 * Render a readable "not available yet" state that distinguishes an
 * unimplemented stub from a genuine failure from an interface gap.
 *
 * @param {string} what Human label, e.g. "Recommendations".
 * @param {*} error     Thrown value (consumed defensively).
 * @returns {string} HTML for a panel body.
 */
function hw6UnavailableHtml(what, error) {
  const message = String(error && error.message ? error.message : error);
  const head = '<p class="empty-state">' + hw6EscapeHtml(what);
  const tail = '<p class="hint"><code>' + hw6EscapeHtml(message) + "</code></p>";
  if (message.indexOf(HW6_TODO_MARKER) !== -1) {
    return head + " not available yet \u2014 implement the <code>TODO(hw6)</code> function, then run again.</p>" + tail;
  }
  if (message.indexOf(HW6_INTERFACE_MARK) !== -1) {
    return head + " unavailable \u2014 the <code>graph.js</code> interface did not match this page's adapter.</p>" + tail;
  }
  return head + " failed \u2014 the page stays usable; details below.</p>" + tail;
}

/**
 * Convert a value into a valid CSV field (RFC 4180 quoting).
 *
 * @param {*} value
 * @returns {string}
 */
function hw6CsvField(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}

/**
 * Join one CSV row.
 *
 * @param {Array<*>} values
 * @returns {string}
 */
function hw6CsvRow(values) {
  return values.map(hw6CsvField).join(",");
}

/**
 * Build a CSV document from a header and rows.
 *
 * @param {string[]} header
 * @param {Array<Array<*>>} rows
 * @returns {string}
 */
function hw6CsvDocument(header, rows) {
  const lines = [hw6CsvRow(header)];
  for (let i = 0; i < rows.length; i++) lines.push(hw6CsvRow(rows[i]));
  return lines.join("\r\n") + "\r\n";
}

/**
 * Trigger a client-side download. Uses a `Blob` plus `URL.createObjectURL` and
 * an `<a download>` element, which works when the page is opened from `file://`
 * as well as over HTTP.
 *
 * @param {string} filename Suggested file name.
 * @param {string} text     File body.
 * @param {string} [mime]   MIME type; a `;charset=utf-8` suffix is appended.
 * @returns {boolean} `true` when the download was triggered.
 */
function download(filename, text, mime) {
  const type = (mime || "text/plain") + ";charset=utf-8";
  try {
    const blob = new Blob([text], { type: type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    window.setTimeout(function () {
      if (anchor.parentNode) anchor.parentNode.removeChild(anchor);
      URL.revokeObjectURL(url);
    }, 1500);
    return true;
  } catch (error) {
    console.error("Download failed:", error);
    return false;
  }
}

// ---------------------------------------------------------------------------
// TensorFlow.js backend bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the available TensorFlow.js backend, report it, and warn on CPU.
 *
 * Writes `TensorFlow.js backend: <name>` into `#tf-status` and an explanatory
 * line into `#backend-note`. When the CDN script failed, `tf` is undefined and
 * this renders a readable message and returns `null` instead of throwing.
 *
 * @returns {Promise<string|null>} Selected backend name, or `null` when TF.js is missing.
 */
async function hw6InitTf() {
  const statusEl = document.getElementById("tf-status");
  const noteEl = document.getElementById("backend-note");

  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check the CDN script and your network connection, then reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    }
    if (noteEl) noteEl.textContent = "No backend available; the dataset summary stays readable.";
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    try {
      await tf.setBackend("cpu");
    } catch (fallbackError) {
      // Leave the default in place; `tf.getBackend()` below reports what it is.
    }
  }
  await tf.ready();

  const backend = tf.getBackend();
  HW6_STATE.backend = backend;
  MF_INFO.backend = backend;
  if (statusEl) statusEl.textContent = "TensorFlow.js backend: " + backend;
  if (noteEl) {
    noteEl.textContent =
      backend === "cpu"
        ? "CPU backend selected (WebGL unavailable). Runs are slower; backend and runtime are recorded, not compared across devices."
        : "Backend and runtime are recorded, not compared across devices.";
  }
  return backend;
}

// ---------------------------------------------------------------------------
// Training-split indices and provided scoring helpers
// ---------------------------------------------------------------------------

/**
 * Build every training-split lookup used by the provided runners and by
 * `recommendTopK` / `recallAtK`.
 *
 * Indexing assumptions (readme section 3.1 and 5):
 *   - The candidate pool is every movie observed in the training split, sorted
 *     ascending by raw movie id. A zero in `R` means "no positive training
 *     edge", never a negative preference.
 *   - Popularity counts distinct users whose training rating for the movie is
 *     `>= 4.0`; a repeated (user, movie) pair is counted once.
 *   - `pairsBySplit` stores the raw rated-movie sets so the exclusion rules can
 *     ignore the rating value ("rated in training or validation regardless of
 *     the rating value").
 *
 * @param {{train: Rating[], val: Rating[], test: Rating[]}} split
 * @returns {void}
 */
function hw6BuildSplitIndices(split) {
  if (typeof HW6_PPR_PROFILES !== "undefined") HW6_PPR_PROFILES.reset();
  if (typeof HW6_TEST_COMPARISON !== "undefined") HW6_TEST_COMPARISON.reset();
  if (typeof HW6_ALPHA_SELECTION !== "undefined") HW6_ALPHA_SELECTION.reset();
  if (typeof HW6_RETRIEVAL !== "undefined") HW6_RETRIEVAL.reset();
  if (typeof HW6_MF_ENGINE !== "undefined") HW6_MF_ENGINE.invalidateData();
  HW6_STATE.recommendationRows = [];
  HW6_STATE.recommendationKeys = new Set();
  HW6_STATE.demo = null;
  HW6_STATE.recallReport = null;
  HW6_STATE.comparisonRows = null;
  HW6_STATE.alphaSweepRows = null;
  // Replace all split-specific maps: rebuilding must not retain old users/items.
  HW6_STATE.pairsBySplit = { train: new Map(), val: new Map(), test: new Map() };
  HW6_STATE.gradesBySplit = { train: new Map(), val: new Map(), test: new Map() };
  hw6InvalidateCaches();
  if (typeof setGraph === "function") setGraph(null);
  const names = ["train", "val", "test"];
  const movieIds = new Set();
  const userIds = new Set();
  const positiveUsersPerMovie = new Map();
  const positiveUsers = new Set();
  let positivePairs = 0;

  for (let s = 0; s < names.length; s++) {
    const name = names[s];
    const list = split[name] || [];
    for (let i = 0; i < list.length; i++) {
      const row = list[i];
      const pairs = HW6_STATE.pairsBySplit[name];
      const grades = HW6_STATE.gradesBySplit[name];
      let pairSet = pairs.get(row.userId);
      if (!pairSet) {
        pairSet = new Set();
        pairs.set(row.userId, pairSet);
      }
      pairSet.add(row.movieId);
      let gradeMap = grades.get(row.userId);
      if (!gradeMap) {
        gradeMap = new Map();
        grades.set(row.userId, gradeMap);
      }
      gradeMap.set(row.movieId, row.rating);
    }
  }

  const train = split.train || [];
  for (let i = 0; i < train.length; i++) {
    const row = train[i];
    movieIds.add(row.movieId);
    userIds.add(row.userId);
    if (row.rating >= 4.0) {
      let users = positiveUsersPerMovie.get(row.movieId);
      if (!users) {
        users = new Set();
        positiveUsersPerMovie.set(row.movieId, users);
      }
      if (!users.has(row.userId)) {
        users.add(row.userId);
        positivePairs++;
      }
      positiveUsers.add(row.userId);
    }
  }

  HW6_STATE.candidateIds = Array.from(movieIds).sort(function (a, b) {
    return a - b;
  });
  HW6_STATE.candidateSet = movieIds;
  HW6_STATE.trainUserIds = Array.from(userIds).sort(function (a, b) {
    return a - b;
  });
  HW6_STATE.popularity = new Map();
  positiveUsersPerMovie.forEach(function (users, movieId) {
    HW6_STATE.popularity.set(movieId, users.size);
  });
  HW6_STATE.usersWithPositiveTrain = positiveUsers;
  HW6_STATE.positivePairs = positivePairs;
  HW6_STATE.splitReady = true;
}

/**
 * Training popularity of one movie (readme section 7.3).
 *
 * @param {number} movieId
 * @returns {number} Distinct users with a positive training rating; `0` when unobserved.
 */
function hw6PopularityOf(movieId) {
  const value = HW6_STATE.popularity.get(movieId);
  return value === undefined ? 0 : value;
}

/**
 * Provided popularity scorer: distinct users with a training rating `>= 4.0`.
 *
 * @returns {Object<number, number>} Map-like object keyed by raw movie id.
 */
function popularityScores() {
  const out = {};
  HW6_STATE.popularity.forEach(function (count, movieId) {
    out[movieId] = count;
  });
  return out;
}

/**
 * Movies the user rated in the splits that must be excluded for an evaluation
 * split (readme section 5).
 *
 *   - `"test"` excludes movies rated in training OR validation, regardless of
 *     the rating value.
 *   - `"val"`  excludes movies rated in training only.
 *
 * @param {string} splitName `"val"` or `"test"`.
 * @param {number} userId
 * @returns {Set<number>}
 */
function hw6ExclusionSet(splitName, userId) {
  const out = new Set();
  const addFrom = function (name) {
    const set = HW6_STATE.pairsBySplit[name].get(userId);
    if (set) set.forEach(function (movieId) { out.add(movieId); });
  };
  addFrom("train");
  if (splitName === "test") addFrom("val");
  return out;
}

/**
 * The shared evaluation cohort for a split (readme section 7.1).
 *
 * A user is evaluable when the user appears in training with at least one
 * positive training edge and at least one relevant movie remains eligible in
 * the split. Relevant means: rating `>= 4.0` in the split, the movie belongs to
 * the training candidate pool, and it passes that split's previously-rated
 * exclusion rule. Actual MF scores for all train candidates must be available.
 *
 * All four methods must use exactly this cohort for the primary comparison.
 *
 * @param {string} splitName Exactly `"val"` or `"test"`; other values throw.
 * @returns {Promise<number[]>} Verified ascending user ids, identical for all methods.
 */
async function hw6SharedCohort(splitName) {
  return HW6_RETRIEVAL.sharedCohort(splitName);
}

/**
 * Count users and relevant movies that the cohort rule set aside, so the recall
 * number is interpretable (readme section 7.2).
 *
 * @param {string} splitName
 * @returns {{candidateUsers:number, cohortSize:number, excludedUnseen:number,
 *            excludedNoEligible:number, excludedNoPositiveEdge:number,
 *            unseenIds:number[]}}
 */
function hw6CohortExclusions(splitName) {
  return HW6_RETRIEVAL.protocol(splitName).exclusions;
}

// ---------------------------------------------------------------------------
// HW5 MF adapters. The implementation in mf.js is derived from Git commit
// 40a20cc, not the neighboring worktree or the instructor's former HW6 model.
// Fixed seeds enable a new repeatable run on a fixed backend; they do not
// reconstruct the unavailable historical HW5 factors.
// ---------------------------------------------------------------------------

/**
 * Deterministic 32-bit PRNG (mulberry32). Returns a function producing values
 * in `[0, 1)`.
 *
 * @param {number} seed
 * @returns {() => number}
 */
function hw6Mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Release in-memory MF variables. Persisted factors remain available.
 *
 * @returns {void}
 */
function hw6DisposeMf() {
  if (HW6_MF_MODEL) tf.dispose([HW6_MF_MODEL.userFactors, HW6_MF_MODEL.itemFactors]);
  HW6_MF_MODEL = null;
}

/**
 * HW5 batched RMSE: predictions are clipped to [1,5], ranking scores are not.
 *
 * @param {Rating[]} split
 * @returns {number} RMSE, or `NaN` when the model is unavailable.
 */
function hw6MfRmse(split) {
  return HW6_MF_ENGINE.rmse(split, HW6_MF_MODEL);
}

/** UI readiness only: do not enable native controls before handlers are wired. */
function hw6SetReadiness(stage, message) {
  HW6_STATE.uiStage = stage;
  HW6_STATE.loaded = stage === "ready";
  HW6_STATE.uiLoadError = stage === "error" ? message : null;
  const gate = document.getElementById("application-gate");
  if (gate) {
    gate.disabled = stage !== "ready";
    gate.setAttribute("aria-busy", String(stage === "loading"));
  }
  const readiness = document.getElementById("app-readiness");
  if (readiness) {
    readiness.textContent = message;
    readiness.className = stage === "error" ? "error" : "status-line";
    readiness.setAttribute("role", stage === "error" ? "alert" : "status");
  }
  const status = document.getElementById("status");
  if (status) status.className = stage === "error" ? "status error" : "status";
  hw6SetStatus(message);
  const summary = document.getElementById("tests-summary");
  if (summary) summary.textContent = stage === "ready" ? "Ready to run tests." : stage === "error"
    ? "Tests unavailable: application loading failed." : "Waiting for the application to be ready.";
  if (stage === "error") {
    const dataset = document.getElementById("dataset-summary-body");
    if (dataset) dataset.textContent = "Application not ready. Follow the loading error above, then reload.";
    const details = document.getElementById("self-checks");
    if (details) details.textContent = "Tests were not run. " + message;
  }
  if (stage === "ready") {
    const label = document.getElementById("saved-alpha-label");
    if (label) label.textContent = typeof HW6_ALPHA_SELECTION.selectedAlpha === "number"
      ? HW6_ALPHA_SELECTION.selectedAlpha.toFixed(2) : "not loaded";
  }
}

/**
 * Load a compatible saved checkpoint only. Retrieval never starts training.
 * Concurrent/repeated callers reuse the same promise/model. Errors are exposed.
 * @returns {Promise<MfModel>} Selected model, never placeholder scores.
 */
function hw6EnsureMf() {
  if (HW6_MF_TRAINING) return HW6_MF_TRAINING;
  if (HW6_MF_MODEL && MF_INFO.checkpoint !== null) return Promise.resolve(HW6_MF_MODEL);
  HW6_MF_TRAINING = Promise.resolve().then(async () => {
    if (!HW6_STATE.splitReady) throw new Error("MF: data/split are not ready");
    const key = HW6_MF_ENGINE.spec().modelId;
    let payload = await HW6_MF_ENGINE.storage("get", key);
    let source = "IndexedDB";
    if (!payload) {
      const response = await fetch("mf-checkpoint.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`MF: saved checkpoint unavailable (${response.status}); retraining is disabled`);
      payload = await response.json();
      source = "mf-checkpoint.json";
    }
    const model = await HW6_MF_ENGINE.load(payload, source);
    if (source !== "IndexedDB") await HW6_MF_ENGINE.storage("put", key, payload);
    return model;
  }).finally(() => { HW6_MF_TRAINING = null; });
  return HW6_MF_TRAINING;
}

/**
 * Training entry point is disabled during retrieval preparation. The historical
 * fixed-run implementation is retained unchanged in mf.js for provenance.
 * @param {Rating[]} train Training rows.
 * @returns {Promise<MfModel|null>}
 */
async function hw6TrainMf(train) {
  throw new Error("MF training is disabled at this stage; load the saved checkpoint");
}

/**
 * Load/reuse saved HW5 MF; absent weights are an error, never a training trigger.
 * Report failures without silently substituting another model.
 *
 * @returns {Promise<MfModel|null>}
 */
async function hw6EnsureMfForNeed() {
  if (typeof tf === "undefined") return null;
  if (!HW6_STATE.splitReady) return null;
  hw6SetStatus("Loading saved HW5 MF checkpoint (retraining disabled)...");
  let model;
  try { model = await hw6EnsureMf(); }
  catch (error) {
    hw6SetStatus("HW5 MF failed: " + error.message);
    throw error;
  }
  if (model && MF_INFO.checkpoint !== null) {
    hw6SetStatus(
      "HW5 MF seeded retraining (original weights unavailable): " + MF_INFO.factors +
      " factors, checkpoint epoch " + MF_INFO.checkpoint + ", validation RMSE " +
      hw6Fmt(MF_INFO.validationRMSE) + ", loaded from " + MF_INFO.loadedFrom + ".",
    );
  } else {
    hw6SetStatus("HW5 MF is unavailable; no placeholder scores are reported.");
  }
  return model;
}

/**
 * HW5 raw MF scorer restricted to the shared training-movie candidate pool.
 *
 * Scores are dot products plus the global training mean, on the same scale as
 * the previous assignment's model. They are NOT comparable to PPR walk
 * probabilities.
 *
 * @param {number} userId Raw user id.
 * @returns {Object<number, number>} `movieId -> score`; empty when the model is
 *   untrained or the user row does not exist.
 */
function mfScores(userId) {
  if (!HW6_MF_MODEL) return {};
  return HW6_MF_ENGINE.scores(userId, HW6_STATE.candidateIds, HW6_MF_MODEL);
}

// ---------------------------------------------------------------------------
// Shared retrieval entry points (implementation in retrieval.js).
//
// Provided helpers you will want:
//   `HW6_STATE.candidateIds`  ascending training-movie ids (the candidate pool)
//   `HW6_STATE.candidateSet`  `Set` view of the pool
//   `hw6PopularityOf(id)`     training popularity of one movie
//   `hw6ExclusionSet(split, userId)` movies to exclude for that split
//   `hw6SharedCohort(split)`  the shared evaluation cohort for a split
//   `hw6ScoreVector(method, userId, alpha, split)` candidate-aligned scores
// ---------------------------------------------------------------------------

/**
 * Extract movie-node scores from a full propagation vector.
 *
 * Readme section 5 requires movie-node scores only.
 *
 * Contract:
 *  - `p` is the node-probability vector produced by `pageRank` /
 *    `personalizedPageRank`, indexed by graph node index (length = node count).
 *  - Return a `Float64Array` of length `graph.movieIds.length`, where entry `k`
 *    is the score of the movie node whose raw id is `graph.movieIds[k]`. Do NOT
 *    return user-node scores, and do NOT change the order.
 *  - Prefer `graph.movieNodes` (node indices for the movies, same order) when it
 *    exists; otherwise the convention is that user nodes occupy indices
 *    `[0, graph.userIds.length)` and movie nodes follow, so entry `k` is
 *    `p[userNodeCount + k]`.
 *  - Never add a padding node: the output length is exactly
 *    `graph.movieIds.length`.
 *  - Invalid dimensions, mappings or movie probabilities throw; zero stays zero.
 *
 * @param {ArrayLike<number>} p Node probabilities, indexed by graph node index.
 * @param {Object} graph        Graph object from `graph.js`.
 * @returns {Float64Array} Movie scores in `graph.movieIds` order.
 */
function movieScores(p, graph) {
  return HW6_RETRIEVAL.movieScores(p, graph);
}

/**
 * Recommend the top-K movies for a user with one of the four methods.
 *
 * Readme section 5 defines the rules; invalid eligible scores throw explicitly.
 *
 * Contract:
 *  - `method` is one of `"popularity"`, `"pagerank"`, `"ppr"`, `"mf"`.
 *    `"pagerank"` and `"ppr"` must use the alpha currently selected in the
 *    `#alpha` slider and the cached propagation results returned by
 *    `hw6RunPageRank` / `hw6RunPPR` (or `pageRank` / `personalizedPageRank`
 *    directly through the adapter). Do not re-run a sweep here.
 *  - Candidate pool: every movie observed in the training split, i.e.
 *    `HW6_STATE.candidateIds`. Movies outside it can never be recommended.
 *  - Exclusion: for `split === "test"` drop movies the user rated in training or
 *    validation regardless of the rating value; for `split === "val"` drop
 *    training-rated movies only. Use `hw6ExclusionSet(split, userId)`.
 *  - Ordering: score descending, and ties broken by ASCENDING raw movie id.
 *  - Return `RecommendedMovie[]` of length `<= K` (fewer when the pool is
 *    smaller) with `{ movieId, title, genres, score, popularity }`.
 *  - When fewer than `K` eligible candidates exist, still return what is
 *    available; the renderer reports the shortfall. Note in the panel that
 *    zero-score candidates included only to fill the list have not demonstrated
 *    relevance.
 *  - Return `[]` (never throw) for an unknown method, `K <= 0`, or a user with
 *    no eligible candidate.
 *  - PPR scores are walk probabilities: never present them as ratings.
 *
 * @param {"popularity"|"pagerank"|"ppr"|"mf"} method
 * @param {number} userId Raw user id.
 * @param {number} [K=10]
 * @param {"val"|"test"} [split="test"]
 * @param {{alpha?:number}} [options] Explicit alpha, otherwise captured from UI.
 * @returns {Promise<RecommendedMovie[]>}
 */
async function recommendTopK(method, userId, K = 10, split = "test", options = {}) {
  return HW6_RETRIEVAL.recommend(method, userId, K, split, options);
}

/**
 * Recall@K for one method over the shared evaluation cohort.
 *
 * Readme sections 7.1 and 7.2 define the protocol.
 *
 * Contract:
 *  - Cohort: use `hw6SharedCohort(split)` so all four methods report the exact
 *    same users (the shared intersection). Do not build a per-method cohort.
 *  - A movie is relevant when its rating in the evaluation split is `>= 4.0`,
 *    it belongs to the training candidate pool, and it passes that split's
 *    previously-rated exclusion rule.
 *  - Compute the ranked lists with `recommendTopK(method, userId, K, split)`.
 *    For `split === "val"` the candidate lists must be computed FRESH with the
 *    validation exclusion rule; never reuse the test lists.
 *  - Return `{ recall, cohort, excludedUnseen, excludedNoEligible, perUser }`:
 *      `recall` macro-average of `hit / relevant` over the cohort (`0` when the
 *      cohort is empty; never divide by zero);
 *      `cohort` ascending user ids;
 *      `excludedUnseen` count of relevant movies absent from the training pool;
 *      `excludedNoEligible` count of users with no eligible relevant movie;
 *      `perUser` `[{ userId, relevant, hit }]` for cohort users only.
 *  - Never compute rating RMSE here: PPR outputs are not ratings.
 *
 * @param {"popularity"|"pagerank"|"ppr"|"mf"} method
 * @param {number} [K=10]
 * @param {"val"|"test"} [split="test"]
 * @param {{alpha?:number}} [options] Captured run settings, independent of UI changes.
 * @returns {Promise<RecallReport>} Includes detailed overlapping exclusion reasons.
 */
async function recallAtK(method, K = 10, split = "test", options = {}) {
  return HW6_RETRIEVAL.recall(method, K, split, options);
}

// ---------------------------------------------------------------------------
// graph.js integration adapter
//
// Async graph.js adapters use explicit graph snapshots, not global graph swaps.
// Exceptions are preserved; invalid result shapes fail without signature retries.
// ---------------------------------------------------------------------------

/**
 * The currently active graph, or `null`.
 *
 * Prefers `getGraph()` when available, then the `GRAPH` global.
 *
 * @returns {Object|null}
 */
function hw6ActiveGraph() {
  if (typeof getGraph === "function") {
    try {
      const graph = getGraph();
      if (graph) return graph;
    } catch (error) {
      // Fall through to the global below.
    }
  }
  return typeof GRAPH !== "undefined" && GRAPH ? GRAPH : null;
}

/**
 * Node count of a graph object, or `0` when unknown.
 *
 * Reads, in order: `n`, `nodeCount`, `numNodes`, then `userIds.length +
 * movieIds.length`.
 *
 * @param {Object} graph
 * @returns {number}
 */
function hw6NodeCount(graph) {
  if (!graph) return 0;
  if (Number.isFinite(graph.n)) return graph.n;
  if (Number.isFinite(graph.nodeCount)) return graph.nodeCount;
  if (Number.isFinite(graph.numNodes)) return graph.numNodes;
  const users = Array.isArray(graph.userIds) ? graph.userIds.length : 0;
  const moviesArr = Array.isArray(graph.movieIds) ? graph.movieIds.length : 0;
  if (users || moviesArr) return users + moviesArr;
  if (Array.isArray(graph.nodes)) return graph.nodes.length;
  return 0;
}

/**
 * Raw user ids of a graph object, or `[]`.
 *
 * @param {Object} graph
 * @returns {number[]}
 */
function hw6UserIds(graph) {
  if (!graph) return [];
  if (Array.isArray(graph.userIds)) return graph.userIds;
  if (Array.isArray(graph.nodes)) {
    return graph.nodes
      .filter(function (node) { return node && (node.kind === "user" || node.type === "user"); })
      .map(function (node) { return node.id !== undefined ? node.id : node.index; });
  }
  return [];
}

/**
 * Raw movie ids of a graph object, or `[]`.
 *
 * @param {Object} graph
 * @returns {number[]}
 */
function hw6MovieIds(graph) {
  if (!graph) return [];
  if (Array.isArray(graph.movieIds)) return graph.movieIds;
  if (Array.isArray(graph.nodes)) {
    return graph.nodes
      .filter(function (node) { return node && (node.kind === "movie" || node.type === "movie"); })
      .map(function (node) { return node.id !== undefined ? node.id : node.index; });
  }
  return [];
}

/**
 * Build `movieId -> local movie index` for a graph.
 *
 * This is the index space of the array `movieScores` must return: entry `k` is
 * the score of `graph.movieIds[k]`, i.e. a movie-node-relative index, NOT a
 * global node index. Prefers graph.js's `movieIndexOf`; otherwise the position
 * inside `graph.movieIds`.
 *
 * @param {Object} graph
 * @returns {Map<number, number>}
 */
function hw6MovieLocalIndex(graph) {
  const index = new Map();
  if (!graph) return index;
  if (graph.movieIndexOf instanceof Map) {
    graph.movieIndexOf.forEach(function (localIndex, movieId) {
      index.set(Number(movieId), Number(localIndex));
    });
    return index;
  }
  const ids = hw6MovieIds(graph);
  for (let i = 0; i < ids.length; i++) index.set(Number(ids[i]), i);
  return index;
}

/**
 * Build `movieId -> global node index` for a graph, for the graph-view score
 * accessor (node order is users first, then movies).
 *
 * @param {Object} graph
 * @returns {Map<number, number>}
 */
function hw6MovieNodeIndex(graph) {
  const index = new Map();
  if (!graph) return index;
  const offset = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : hw6UserIds(graph).length;
  hw6MovieLocalIndex(graph).forEach(function (localIndex, movieId) {
    index.set(movieId, offset + localIndex);
  });
  return index;
}

/**
 * Normalise whatever a propagation function returned into a `PropagationResult`.
 *
 * Requires an object carrying scores/p/probabilities, measured iteration count,
 * residual and a consistent convergence flag at the default 1e-6 tolerance.
 * Returns `null` when the shape or the numbers cannot be trusted, so the caller
 * raises an interface error instead of computing with garbage.
 *
 * @param {*} out
 * @param {number} n Expected node count (`<= 0` to accept the vector length).
 * @returns {PropagationResult|null}
 */
function hw6NormalizePropagation(out, n) {
  if (!out) return null;
  let vector = null;
  let iterations = null;
  let residual = null;
  let converged = null;

  if (Array.isArray(out) || ArrayBuffer.isView(out)) {
    vector = out;
  } else if (typeof out === "object") {
    vector = out.scores || out.p || out.probabilities || out.vector || null;
    const itRaw = out.iterations !== undefined ? out.iterations : out.iters;
    const resRaw = out.residual !== undefined ? out.residual : out.delta !== undefined ? out.delta : out.l1;
    iterations = itRaw !== null && itRaw !== undefined && Number.isFinite(Number(itRaw)) ? Number(itRaw) : null;
    residual = resRaw !== null && resRaw !== undefined && Number.isFinite(Number(resRaw)) ? Number(resRaw) : null;
    converged = typeof out.converged === "boolean" ? out.converged : null;
  }
  if (!vector || typeof vector.length !== "number") return null;

  const size = n > 0 ? n : vector.length;
  if (vector.length !== size) return null;
  for (let i = 0; i < size; i++) {
    if (!Number.isFinite(Number(vector[i])) || Number(vector[i]) < 0) return null;
  }
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > HW6_MAX_ITERATIONS ||
      residual === null || residual < 0 || converged !== (residual < HW6_TOLERANCE)) return null;
  const mass = Array.from(vector).reduce(function (sum, value) { return sum + Number(value); }, 0);
  if (Math.abs(mass - 1) > 1e-5) return null;
  return {
    scores: vector,
    iterations: iterations,
    residual: residual,
    converged: converged,
    backend: out.backend || null,
    runtimeMs: Number.isFinite(out.runtimeMs) ? out.runtimeMs : null,
    source: "graph.js",
  };
}

/**
 * Run `fn` with `graph` installed as the active graph, then restore the previous
 * one. Used only when a propagation function reads the module-level `GRAPH`
 * instead of taking a graph argument.
 *
 * @param {Object} graph
 * @param {() => any} fn
 * @returns {Promise<any>}
 */
async function hw6WithActiveGraph(graph, fn) {
  const canSwap = typeof setGraph === "function" && typeof getGraph === "function";
  const previous = canSwap ? getGraph() : null;
  if (canSwap && previous !== graph) setGraph(graph);
  try {
    return await fn();
  } finally {
    if (canSwap && previous !== graph) {
      setGraph(previous);
    }
  }
}

/**
 * Invoke `pageRank` for a specific graph and alpha.
 *
 * Pass an explicit graph to avoid global-state races during asynchronous reads.
 * Preserve original errors; never retry a different alpha or call signature.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @returns {Promise<PropagationResult>}
 * @throws {Error} Solver errors or invalid output, both reported as FAIL.
 */
async function hw6AdapterPageRank(graph, alpha) {
  if (typeof pageRank !== "function") throw hw6InterfaceError("graph.js `pageRank` is unavailable");
  const result = hw6NormalizePropagation(await pageRank(alpha, { graph }), hw6NodeCount(graph));
  if (!result) throw hw6InterfaceError("pageRank returned invalid probabilities or convergence metadata");
  return result;
}

/**
 * Invoke `personalizedPageRank` for a graph, seed node and alpha.
 *
 * The solver accepts a NODE INDEX. Raw user IDs are mapped by hw6RunPPR first.
 * An optional raw ID is checked against that mapping, never passed as an index.
 *
 * @param {Object} graph
 * @param {number} seedNode Seed node index within the graph.
 * @param {number} alpha
 * @param {number} [seedUserId] Raw user id, when known.
 * @returns {Promise<PropagationResult>}
 * @throws {Error} Solver errors or an inconsistent user/index mapping.
 */
async function hw6AdapterPPR(graph, seedNode, alpha, seedUserId) {
  if (typeof personalizedPageRank !== "function") {
    throw hw6InterfaceError("graph.js `personalizedPageRank` is unavailable");
  }
  if (seedUserId !== undefined && hw6UserNodeIndex(graph, seedUserId) !== seedNode) {
    throw new Error("PPR: raw user ID does not match its restart node index");
  }
  const result = hw6NormalizePropagation(await personalizedPageRank(seedNode, alpha, { graph }), hw6NodeCount(graph));
  if (!result) throw hw6InterfaceError("PPR returned invalid probabilities or convergence metadata");
  return result;
}

/**
 * Node index of a raw user id.
 *
 * Prefers graph.js's own mapping (`nodeOfUser`, then `userIndexOf` or the
 * `userIndex` alias), then the position inside `graph.userIds`. There is no
 * numeric-ID fallback: an unknown raw ID is never interpreted as an index.
 *
 * @param {Object} graph
 * @param {number} userId
 * @returns {number} Node index, or `-1`.
 */
function hw6UserNodeIndex(graph, userId) {
  if (!graph) return -1;
  if (typeof graph.nodeOfUser === "function") {
    try {
      const node = Number(graph.nodeOfUser(userId));
      if (Number.isFinite(node) && node >= 0) return node;
    } catch (error) {
      // Fall through to the map / array lookups below.
    }
  }
  const maps = [graph.userIndexOf, graph.userIndex];
  for (let i = 0; i < maps.length; i++) {
    if (maps[i] instanceof Map) {
      const mapped = maps[i].get(userId);
      if (mapped !== undefined) return Number(mapped);
    }
  }
  const users = hw6UserIds(graph);
  const position = users.indexOf(userId);
  if (position !== -1) return position;
  return -1;
}

/**
 * Cached global PageRank for the active graph at `alpha`.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @returns {Promise<PropagationResult>}
 */
async function hw6RunPageRank(graph, alpha) {
  const cached = HW6_STATE.pageRank;
  if (cached && cached.graph === graph && cached.alpha === alpha) return cached;
  const result = await hw6AdapterPageRank(graph, alpha);
  HW6_STATE.pageRank = { ...result, graph: graph, alpha: alpha };
  return HW6_STATE.pageRank;
}

/**
 * Cached PPR for the active graph, restart user and `alpha`. The cache key
 * includes alpha: changing alpha must never reuse a stale walk.
 *
 * @param {Object} graph
 * @param {number} userId
 * @param {number} alpha
 * @returns {Promise<PropagationResult>}
 */
async function hw6RunPPR(graph, userId, alpha) {
  const cached = HW6_STATE.ppr;
  if (cached && cached.graph === graph && cached.userId === userId && cached.alpha === alpha) return cached;
  const seedNode = hw6UserNodeIndex(graph, userId);
  if (seedNode < 0) throw hw6InterfaceError("cannot map user " + userId + " to a graph node");
  const result = await hw6AdapterPPR(graph, seedNode, alpha, userId);
  HW6_STATE.ppr = { ...result, graph: graph, userId: userId, alpha: alpha };
  return HW6_STATE.ppr;
}

/**
 * Candidate-aligned score vector for one method (provided plumbing).
 *
 * Returns `{ movieIds, scores, propagation }` where `scores` is a `Float64Array`
 * aligned to HW6_STATE.candidateIds. Invalid/missing scores become NaN, never 0;
 * recommendTopK rejects them explicitly if the candidate is eligible.
 *
 * @param {string} method
 * @param {number} userId
 * @param {number} alpha
 * @param {"val"|"test"} split
 * @returns {Promise<{movieIds:number[], scores:Float64Array, propagation:(PropagationResult|null)}>}
 */
async function hw6ScoreVector(method, userId, alpha, split) {
  const ids = HW6_STATE.candidateIds;
  const scores = new Float64Array(ids.length);
  if (!ids.length) return { movieIds: ids, scores: scores, propagation: null };

  if (method === "popularity") {
    for (let i = 0; i < ids.length; i++) scores[i] = hw6PopularityOf(ids[i]);
    return { movieIds: ids, scores: scores, propagation: null };
  }

  if (method === "mf") {
    const scored = mfScores(userId);
    for (let i = 0; i < ids.length; i++) {
      const value = scored[ids[i]];
      scores[i] = typeof value === "number" && Number.isFinite(value) ? value : NaN;
    }
    return { movieIds: ids, scores: scores, propagation: null };
  }

  const graph = hw6ActiveGraph();
  if (!graph) throw hw6InterfaceError("no graph is active; press \"Build graph\" first");
  const propagation = await (method === "ppr" ? hw6RunPPR(graph, userId, alpha) : hw6RunPageRank(graph, alpha));
  // `movieScores` returns movie-node-relative entries, so the lookup here must
  // be the LOCAL movie index; using the global node index would skip the second
  // half of the movie array.
  const perMovie = movieScores(propagation.scores, graph);
  const index = hw6MovieLocalIndex(graph);
  for (let i = 0; i < ids.length; i++) {
    const position = index.get(ids[i]);
    const value = position === undefined ? NaN : perMovie[position];
    scores[i] = typeof value === "number" && Number.isFinite(value) ? value : NaN;
  }
  return { movieIds: ids, scores: scores, propagation: propagation };
}

// ---------------------------------------------------------------------------
// Provided runners
// ---------------------------------------------------------------------------

/**
 * Mean training popularity of the movies in one recommended list (readme 7.3).
 *
 * @param {RecommendedMovie[]} recs
 * @returns {number|null} Mean popularity, or `null` for an empty list.
 */
function hw6MeanPopularity(recs) {
  if (!recs || recs.length === 0) return null;
  let sum = 0;
  for (let i = 0; i < recs.length; i++) {
    const value = Number(recs[i].popularity);
    sum += Number.isFinite(value) ? value : hw6PopularityOf(recs[i].movieId);
  }
  return sum / recs.length;
}

/**
 * Record recommended rows for the recommendations CSV, de-duplicating on
 * `(userId, method, alpha, rank)` so repeated clicks do not duplicate rows.
 *
 * @param {number} userId
 * @param {string} method
 * @param {number|null} alpha
 * @param {RecommendedMovie[]} recs
 * @returns {void}
 */
function hw6RecordRecommendationRows(userId, method, alpha, recs) {
  // Replace the whole list, including stale ranks from a previously longer list.
  const alphaValue = alpha === null || alpha === undefined ? "" : Number(alpha).toFixed(2);
  HW6_STATE.recommendationRows = HW6_STATE.recommendationRows.filter(row =>
    !(row.userId === userId && row.method === method && row.alpha === alphaValue));
  for (let i = 0; i < recs.length; i++) {
    const rec = recs[i];
    const alphaKey = alpha === null || alpha === undefined ? "na" : Number(alpha).toFixed(2);
    const key = userId + "|" + method + "|" + alphaKey + "|" + (i + 1);
    HW6_STATE.recommendationKeys.add(key);
    HW6_STATE.recommendationRows.push({
      userId: userId,
      method: method,
      alpha: alpha === null || alpha === undefined ? "" : Number(alpha).toFixed(2),
      rank: i + 1,
      movieId: rec.movieId,
      title: rec.title,
      genres: Array.isArray(rec.genres) ? rec.genres.join(";") : "",
      score: rec.score,
      trainingPopularity: Number.isFinite(Number(rec.popularity)) ? rec.popularity : hw6PopularityOf(rec.movieId),
      split: "test",
      meanPopularity: hw6MeanPopularity(recs),
    });
  }
}

/**
 * Run one propagation pass and store its convergence record for the panel.
 *
 * @param {"pagerank"|"ppr"} kind
 * @param {Object} graph
 * @param {number} alpha
 * @param {number} [userId] Seed user for PPR.
 * @returns {Promise<PropagationResult>}
 */
async function hw6RunAndRecordConvergence(kind, graph, alpha, userId) {
  const result = await (kind === "ppr" ? hw6RunPPR(graph, userId, alpha) : hw6RunPageRank(graph, alpha));
  HW6_STATE.convergence[kind] = {
    alpha: alpha,
    userId: kind === "ppr" ? userId : null,
    iterations: result.iterations,
    residual: result.residual,
    converged: result.converged,
    source: result.source,
    backend: result.backend,
    runtimeMs: result.runtimeMs,
    at: new Date().toISOString(),
  };
  return result;
}

/**
 * Derive a convergence verdict only when the propagation function did not
 * report one. An iteration cap is never treated as proof of convergence.
 *
 * @param {{iterations:(number|null), residual:(number|null), converged:(boolean|null)}} record
 * @returns {{value:boolean|null, derived:boolean}} Verdict and whether it was derived here.
 */
function hw6ConvergenceVerdict(record) {
  if (!record) return { value: null, derived: false };
  if (Number.isFinite(record.residual) && record.residual >= 0) {
    return { value: record.residual < HW6_TOLERANCE, derived: typeof record.converged !== "boolean" };
  }
  return { value: null, derived: true };
}

/**
 * Four-method comparison runner on the shared cohort (readme sections 6 and 7).
 *
 * Recall@10 for popularity, global PageRank, PPR and MF, plus cohort size,
 * exclusion counts and mean recommended-list popularity. The alpha recorded for
 * PageRank and PPR is the selected validation alpha; MF keeps its own
 * validation-RMSE checkpoint, so the validation objectives differ and the
 * report must say so.
 *
 * HW5 MF and validation-selected alpha are loaded only. All methods share one
 * list-generation pass for Recall and popularity; errors fail the comparison.
 *
 * @param {"test"|"val"} [split="test"] Evaluation split.
 * @returns {Promise<Object>} Complete/failed test experiment, or its saved copy.
 */
async function hw6RunComparison(split) {
  if (split && split !== "test") throw new Error("The final comparison evaluates test only; validation selection is already saved");
  return HW6_TEST_COMPARISON.run();
}

/**
 * Read the saved macro list popularity; never regenerate recommendations.
 *
 * @param {string} method
 * @param {number[]} cohort
 * @param {"test"|"val"} split
 * @param {number} alpha
 * @returns {Promise<number>} Throws if a compatible one-pass result is absent.
 */
async function hw6MeanListPopularityForCohort(method, cohort, split, alpha) {
  const saved = HW6_TEST_COMPARISON.result;
  if (saved?.status !== "complete" || split !== "test" || JSON.stringify(cohort) !== JSON.stringify(saved.cohort)) {
    throw new Error("Use the one-pass test comparison before requesting aggregate popularity");
  }
  if ((method === "ppr" || method === "pagerank") && alpha !== saved.alpha) throw new Error("Stored comparison alpha mismatch");
  return saved.rows.find(row => row.method === method).meanPopularity;
}

/**
 * Validation-only alpha sweep for PPR (readme section 6.1).
 *
 * Recall@10 is measured on the validation split for each alpha in `ALPHA_GRID`.
 * The selected alpha maximises validation recall; exact ties are broken toward
 * the SMALLER alpha. Test outcomes are never consulted here.
 *
 * @returns {Promise<{rows:Array<Object>, best:(number|null), note:string}>}
 */
async function hw6RunAlphaSweep(options = {}) {
  return HW6_ALPHA_SELECTION.run(options);
}

/**
 * Run `fn` with the alpha slider temporarily set to `value`, restoring the
 * previous value afterwards. The slider's own `input` event is never dispatched,
 * so this cannot recurse.
 *
 * @param {number} value
 * @param {() => any} fn
 * @returns {Promise<any>}
 */
async function hw6WithAlpha(value, fn) {
  const slider = document.getElementById("alpha");
  const previous = slider ? Number(slider.value) : 0.85;
  hw6SetAlpha(value);
  try {
    return await fn();
  } finally {
    hw6SetAlpha(previous);
  }
}

/**
 * Two-user, one-factor-at-a-time PPR demonstration (readme section 6.2).
 *
 * Current scope is a fixed-alpha preview for users 1 and 2 with all methods.
 * The later controlled alpha-change experiment is not run by this entry point.
 * @returns {Promise<Object>} Eight lists, popularity and overlap evidence.
 */
async function hw6RunTwoUserDemo() {
  return HW6_RETRIEVAL.preview([1, 2], 0.85);
}

/**
 * Recommendations for the current controls (method, user, split), recorded for
 * export. Always uses the TEST split for the on-screen panel.
 *
 * @returns {Promise<{recs:RecommendedMovie[]}|{error:string}>}
 */
async function hw6CurrentRecommendations() {
  const method = hw6ReadMethod();
  const userId = hw6ReadNumber("user-select", Number.NaN);
  const alpha = hw6ReadAlpha();
  if (!Number.isFinite(userId)) return { error: "No target user selected." };
  try {
    const cohort = await hw6SharedCohort("test");
    if (!cohort.includes(userId)) throw new Error("Selected user is outside the verified common test cohort");
    const recs = await recommendTopK(method, userId, HW6_K, "test", { alpha });
    hw6RecordRecommendationRows(userId, method, method === "pagerank" || method === "ppr" ? alpha : null, recs);
    return { recs: recs };
  } catch (error) {
    return { error: String(error && error.message ? error.message : error) };
  }
}

// ---------------------------------------------------------------------------
// Panel renderers (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset and split summary into `#dataset-summary-body`.
 *
 * Reports users, movies, ratings, sparsity, positive training interactions,
 * zero-degree users and movies, and the split record counts. Missing ratings are
 * unobserved, never zeros.
 *
 * @returns {void}
 */
function hw6RenderDatasetSummary() {
  const target = document.getElementById("dataset-summary-body");
  if (!target) return;
  if (!HW6_STATE.splitReady) {
    target.innerHTML = '<p class="empty-state">Loading dataset\u2026</p>';
    return;
  }

  const train = HW6_SPLIT.train;
  const val = HW6_SPLIT.val;
  const test = HW6_SPLIT.test;
  const users = HW6_STATE.trainUserIds.length;
  const movieCount = HW6_STATE.candidateIds.length;
  const loadedUsers = new Set(ratings.map(row => row.userId)).size;
  const loadedMovies = new Set(movies.map(movie => movie.id)).size;
  const cells = loadedUsers * loadedMovies;
  const sparsity = cells > 0 ? 1 - ratings.length / cells : 0;
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;

  let zeroDegreeUsers = 0;
  for (let i = 0; i < HW6_STATE.trainUserIds.length; i++) {
    if (!HW6_STATE.usersWithPositiveTrain.has(HW6_STATE.trainUserIds[i])) zeroDegreeUsers++;
  }
  let zeroDegreeMovies = 0;
  for (let i = 0; i < HW6_STATE.candidateIds.length; i++) {
    if (hw6PopularityOf(HW6_STATE.candidateIds[i]) === 0) zeroDegreeMovies++;
  }

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>Users observed in training</dt><dd>" + users.toLocaleString("en-US") + "</dd>" +
    "<dt>Movies observed in training</dt><dd>" + movieCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Ratings loaded</dt><dd>" + ratings.length.toLocaleString("en-US") + "</dd>" +
    "<dt>Split: train / validation / test</dt><dd>" +
    train.length.toLocaleString("en-US") + " / " + val.length.toLocaleString("en-US") + " / " + test.length.toLocaleString("en-US") +
    "</dd>" +
    "<dt>Positive training interactions</dt><dd>" + HW6_STATE.positivePairs.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroDegreeUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroDegreeMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Matrix sparsity (loaded users × movies, no padding)</dt><dd>" + (sparsity * 100).toFixed(2) + "%</dd>" +
    "</dl>" +
    '<p class="hint">' +
    "Data validation passed: no invalid records, duplicate movie IDs or duplicate rating pairs. " +
    "Loaded universe: " + loadedUsers + " users × " + loadedMovies + " movies. " +
    exclusions.length + " user(s) skipped while splitting. The split is chronological within each user, not globally time-consistent; " +
    "MovieLens records rating times, not viewing sessions. An unobserved rating is missing data, not a zero." +
    "</p>";
}

/**
 * Render the graph panel into `#graph-summary-body`.
 *
 * Ownership: `graph.js` owns the `Graph` shape and renders this panel itself
 * whenever a graph is installed (`setGraph` -> its own summary renderer), so
 * once a graph is active this function leaves that render to stand and does not
 * overwrite it with a second, differently-derived set of counts.
 *
 * Before a graph exists (and if `graph.js` is unavailable) it renders the node
 * universe the training split implies - observed users, observed movies,
 * positive training edges and zero-degree counts - which is exactly the
 * expectation the built graph must match.
 *
 * @returns {void}
 */
function hw6RenderGraphSummary() {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  const graph = hw6ActiveGraph();
  if (graph && typeof setGraph === "function") return;

  const observedUsers = HW6_STATE.trainUserIds.length;
  const observedMovies = HW6_STATE.candidateIds.length;
  if (!HW6_STATE.splitReady) {
    target.innerHTML = '<p class="empty-state">Loading dataset\u2026</p>';
    return;
  }
  let zeroDegreeUsers = 0;
  for (let i = 0; i < HW6_STATE.trainUserIds.length; i++) {
    if (!HW6_STATE.usersWithPositiveTrain.has(HW6_STATE.trainUserIds[i])) zeroDegreeUsers++;
  }
  let zeroDegreeMovies = 0;
  for (let i = 0; i < HW6_STATE.candidateIds.length; i++) {
    if (hw6PopularityOf(HW6_STATE.candidateIds[i]) === 0) zeroDegreeMovies++;
  }

  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press \u201cBuild graph\u201d to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>" +
      '<p class="hint">Expected node universe from the training split: ' +
      (observedUsers + observedMovies).toLocaleString("en-US") +
      " nodes (" + observedUsers.toLocaleString("en-US") + " users + " + observedMovies.toLocaleString("en-US") + " movies), " +
      HW6_STATE.positivePairs.toLocaleString("en-US") + " positive training edge(s), " +
      zeroDegreeUsers.toLocaleString("en-US") + " zero-degree user(s), " +
      zeroDegreeMovies.toLocaleString("en-US") + " zero-degree movie(s). No padding node.</p>";
    return;
  }

  const nodes = hw6NodeCount(graph) || hw6UserIds(graph).length + hw6MovieIds(graph).length;
  const positiveEdges = Number.isFinite(graph.positiveEdges) ? Number(graph.positiveEdges) : HW6_STATE.positivePairs;
  const edgesText =
    (positiveEdges * 2).toLocaleString("en-US") + " directed (" +
    positiveEdges.toLocaleString("en-US") + " positive bipartite edge(s))";
  const zeroUsers = Array.isArray(graph.zeroDegreeUsers) ? graph.zeroDegreeUsers.length : zeroDegreeUsers;
  const zeroMovies = Array.isArray(graph.zeroDegreeMovies) ? graph.zeroDegreeMovies.length : zeroDegreeMovies;

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>Graph nodes</dt><dd>" + nodes.toLocaleString("en-US") + "</dd>" +
    "<dt>Graph edges</dt><dd>" + edgesText + "</dd>" +
    "<dt>User nodes (training)</dt><dd>" + observedUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes (training)</dt><dd>" + observedMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training interactions</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>" +
    '<p class="hint">The graph is built from positive training ratings only; users and movies with no positive ' +
    "training edge are zero-degree nodes. The dangling-mass policy is the same rule for global PageRank and PPR; " +
    "no padding node is added.</p>";
}

/**
 * Render the convergence panel (readme section 3.4) into `#convergence-results`.
 *
 * Reports iterations, the final L1 residual and whether convergence was reached
 * for global PageRank and PPR. Never claims convergence when the iteration cap
 * was hit. `graph.js`'s 4-node teaching demo also writes this panel from
 * `fixtureStep` / `fixtureReset`; its current state is included as a separate,
 * clearly-labelled row so that information is not lost when this renderer runs.
 *
 * @returns {void}
 */
function hw6RenderConvergence() {
  const target = document.getElementById("convergence-results");
  if (!target) return;
  const rows = [];

  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  const fixture = viewState && viewState.fixture ? viewState.fixture : null;
  if (fixture) {
    const residual = Number.isFinite(Number(fixture.residual)) ? hw6Fmt(fixture.residual, 10) : "not run";
    rows.push(
      "<tr>" +
      "<td>Fixture demo (4-node reference graph)</td>" +
      '<td class="num">' + hw6Fmt(fixture.alpha === undefined ? hw6ReadAlpha() : fixture.alpha, 2) + "</td>" +
      '<td class="num">' + (fixture.iteration === undefined || fixture.iteration === null ? "\u2014" : fixture.iteration) + "</td>" +
      '<td class="num">' + residual + "</td>" +
      "<td>\u2014 (teaching demo)</td>" +
      '<td class="num">\u2014</td>' +
      "</tr>",
    );
  }

  const keys = ["pagerank", "ppr"];
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const record = HW6_STATE.convergence[key];
    if (!record) {
      rows.push("<tr><td>" + (key === "ppr" ? "PPR" : "global PageRank") + "</td><td colspan=\"5\">not run yet</td></tr>");
      continue;
    }
    const verdict = hw6ConvergenceVerdict(record);
    const convergedText =
      verdict.value === null
        ? "not reported"
        : verdict.value
          ? "yes" + (verdict.derived ? " (derived from residual)" : "")
          : "no \u2014 cap or tolerance not met" + (verdict.derived ? " (derived)" : "");
    rows.push(
      "<tr>" +
      "<td>" + (key === "ppr" ? "PPR (user " + record.userId + ")" : "global PageRank") + "</td>" +
      '<td class="num">' + hw6Fmt(record.alpha, 2) + "</td>" +
      '<td class="num">' + (record.iterations === null ? "\u2014" : record.iterations) + "</td>" +
      '<td class="num">' + (record.residual === null ? "\u2014" : hw6Fmt(record.residual, 10)) + "</td>" +
      "<td>" + convergedText + "</td>" +
      '<td class="num">' + (record.iterations === null ? "\u2014" : record.iterations + " / " + HW6_MAX_ITERATIONS) + "</td>" +
      "</tr>",
    );
  }

  target.innerHTML =
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Run</th><th scope=\"col\">Alpha</th><th scope=\"col\">Iterations</th>" +
    "<th scope=\"col\">Final L1 residual</th><th scope=\"col\">Converged</th><th scope=\"col\">Cap</th></tr></thead>" +
    "<tbody>" + rows.join("") + "</tbody></table>" +
    '<p class="hint">Tolerance ' + HW6_TOLERANCE + ", cap " + HW6_MAX_ITERATIONS +
    " iterations. Reaching the cap is not proof of convergence. PageRank is an iterative calculation, not neural-network training.</p>" +
    '<p class="hint">' + keys.filter(key => HW6_STATE.convergence[key]).map(key => {
      const record = HW6_STATE.convergence[key];
      return hw6EscapeHtml(key) + ": backend " + hw6EscapeHtml(record.backend || "unknown") +
        ", runtime " + hw6Fmt(record.runtimeMs, 1) + " ms";
    }).join("; ") + "</p>";
}

/**
 * Render the alpha sweep panel (readme section 6.1).
 *
 * @param {{rows:Array<Object>, best:(number|null), note:string}} [sweep]
 * @returns {void}
 */
function hw6RenderAlphaSweep(sweep) {
  HW6_ALPHA_SELECTION.render(sweep);
}

/**
 * Render the four-method comparison panel.
 *
 * @param {Array<Object>} [rows]
 * @returns {void}
 */
function hw6RenderComparison(rows) {
  if (typeof HW6_TEST_COMPARISON !== "undefined") { HW6_TEST_COMPARISON.render(); return; }
  const target = document.getElementById("comparison-results");
  if (!target) return;
  const data = rows || HW6_STATE.comparisonRows;
  if (!data || data.length === 0) {
    target.innerHTML =
      '<p class="empty-state">Recall@10 for the shared cohort across popularity, global PageRank, PPR and MF is ' +
      "compared here.</p>";
    return;
  }
  const failedRow = data.find(function (row) { return !!row.error; });
  if (failedRow) {
    target.innerHTML = hw6UnavailableHtml("Method comparison", failedRow.error);
    return;
  }

  const body = data
    .map(function (row) {
      return (
        "<tr>" +
        "<td>" + hw6EscapeHtml(row.method) +
        (row.note ? '<br><span class="hint">' + hw6EscapeHtml(row.note) + "</span>" : "") +
        "</td>" +
        '<td class="num">' + (row.alpha === null ? "\u2014" : hw6Fmt(row.alpha, 2)) + "</td>" +
        '<td class="num">' + hw6Fmt(row.recall) + "</td>" +
        '<td class="num">' + row.cohortSize.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + row.excludedUnseen.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + row.excludedNoEligible.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + hw6Fmt(row.meanPopularity, 2) + "</td>" +
        "</tr>"
      );
    })
    .join("");

  const notes = data
    .filter(function (row) { return !!row.note; })
    .map(function (row) { return row.method + ": " + row.note; })
    .join(" ");

  target.innerHTML =
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Method</th><th scope=\"col\">Alpha</th><th scope=\"col\">Recall@10</th>" +
    "<th scope=\"col\">Cohort</th><th scope=\"col\">Unseen excluded</th><th scope=\"col\">No eligible user</th>" +
    "<th scope=\"col\">Mean popularity</th></tr></thead>" +
    "<tbody>" + body + "</tbody></table>" +
    (notes ? '<p class="hint">' + hw6EscapeHtml(notes) + "</p>" : "") +
    '<p class="hint">All four methods use the same candidate rules and the same users. PPR scores are long-run ' +
    "graph-walk probabilities, not purchase probabilities and not predicted ratings.</p>";
}

/**
 * HTML body for the readme section 6.2 two-user demonstration.
 *
 * Pure renderer: it renders the demo object it is given (or the last cached one)
 * and never runs the demo itself, so nothing is written outside the dedicated
 * `#two-user-demo-results` container.
 *
 * @param {Object} [demo] Result of `hw6RunTwoUserDemo()`; defaults to the cached one.
 * @returns {string} HTML for the demo body (`""` when there is nothing to show).
 */
function hw6RunTwoUserDemoHtml(demo) {
  const data = demo || HW6_STATE.demo;
  if (!data) return "";
  if (data.lists) {
    return '<p class="hint">Users 1 and 2; fixed alpha 0.85, not validation-selected. ' +
      'Graph scores are long-run walk probabilities, not ratings.</p>' +
      data.lists.map(list => '<h4>' + hw6EscapeHtml(list.method) + ' — user ' + list.userId + '</h4>' +
        '<p class="hint">Mean training popularity: ' + hw6Fmt(list.meanPopularity, 2) +
        '; zero-score movies: ' + list.zeroScoreCount + '</p><ol class="recommendation-list">' +
        list.recommendations.map(rec => '<li>' + hw6EscapeHtml(rec.title) + ' — id ' + rec.movieId +
          '; ' + hw6EscapeHtml(rec.genres.join(', ')) + '; score ' + hw6Fmt(rec.score, 8) +
          '; train popularity ' + rec.popularity + '</li>').join('') + '</ol>').join('') +
      '<p class="hint">Common movies between users: ' + data.overlap.map(row =>
        hw6EscapeHtml(row.method) + ': ' + row.count + '/10').join('; ') + '</p>' +
      hw6CohortSummaryHtml(data.exclusions.test);
  }
  if (data.error) {
    return hw6UnavailableHtml("Two-user demo", data.error);
  }
  if (!data.settings || data.settings.length === 0) {
    return '<p class="empty-state">Need two users in the shared test cohort before the demonstration can run.</p>';
  }

  const columns = data.settings
    .map(function (setting) {
      const items =
        setting.recs && setting.recs.length
          ? setting.recs
              .map(function (rec, i) {
                return (
                  "<li>" +
                  hw6EscapeHtml(rec.title) +
                  ' <span class="rec-score">id ' + rec.movieId +
                  ", score " + hw6Fmt(rec.score, 4) +
                  ", pop " + hw6PopularityOf(rec.movieId) +
                  (i === 0 ? "" : "") +
                  "</span></li>"
                );
              })
              .join("")
          : '<li class="empty-state">no list produced</li>';
      const count = setting.recs ? setting.recs.length : 0;
      return (
        "<div>" +
        "<h4>" + hw6EscapeHtml(setting.label) + "</h4>" +
        '<p class="hint">user <strong>' + setting.user + "</strong>, alpha <strong>" + hw6Fmt(setting.alpha, 2) +
        "</strong>; changed: " + hw6EscapeHtml(setting.changed) + "</p>" +
        '<ol class="recommendation-list">' + items + "</ol>" +
        (count < HW6_K
          ? '<p class="hint">Only ' + count + " eligible candidates (fewer than " + HW6_K + ").</p>"
          : "") +
        "</div>"
      );
    })
    .join("");

  const common = data.common
    .map(function (row) { return row.pair + ": " + row.shared + " common of top " + row.of; })
    .join("; ");

  return (
    "<h3>One-factor-at-a-time demonstration (readme 6.2)</h3>" +
    '<p class="hint">Users fixed and shown: ' + data.users.join(", ") +
    ". Exactly one factor changes per comparison \u2014 never attribute a difference to a factor that was also changed.</p>" +
    '<div class="control-grid control-grid--two">' + columns + "</div>" +
    '<p class="hint">Common movies between lists: ' + hw6EscapeHtml(common) +
    ". Descriptive evidence only, not proof of better personalization.</p>"
  );
}

/**
 * Render the Top-10 recommendations table for the current controls.
 *
 * Verify the common cohort and await ranking. MF is loaded only from saved
 * factors; any load, score or integration failure is reported explicitly.
 *
 * @returns {Promise<void>}
 */
async function hw6RenderRecommendations() {
  const target = document.getElementById("recommendations");
  if (!target) return;
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  try {
    const method = hw6ReadMethod();
    const alpha = hw6ReadAlpha();
    hw6SetStatus("Preparing recommendations and verifying the common cohort (no Recall run)...");
    const outcome = await hw6CurrentRecommendations();
    if (outcome.error) throw new Error(outcome.error);
    hw6PopulateUserSelects();
    const recs = outcome.recs;
    if (!recs || recs.length === 0) {
      target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
      hw6SetStatus("No eligible recommendations for this user.");
      return;
    }

    const body = recs.map(function (rec, i) {
      const genres = rec.genres.length ? rec.genres.join(", ") : "\u2014";
      return "<tr>" +
        '<td class="num">' + (i + 1) + "</td>" +
        '<td class="num">' + rec.movieId + "</td>" +
        "<td>" + hw6EscapeHtml(rec.title) + "</td>" +
        "<td>" + hw6EscapeHtml(genres) + "</td>" +
        '<td class="num">' + hw6Fmt(rec.score, 8) + "</td>" +
        '<td class="num">' + rec.popularity + "</td></tr>";
    }).join("");
    const zeroScore = recs.some(rec => rec.score === 0);

    target.innerHTML =
      '<p class="hint">Method <strong>' + hw6EscapeHtml(method) + "</strong>, alpha <strong>" + hw6Fmt(alpha, 2) +
      "</strong>, test-split exclusions (training + validation rated movies).</p>" +
      '<table class="data-table">' +
      "<thead><tr><th scope=\"col\">Rank</th><th scope=\"col\">Movie id</th><th scope=\"col\">Title</th>" +
      "<th scope=\"col\">Genres</th><th scope=\"col\">Score</th><th scope=\"col\">Training popularity</th></tr></thead>" +
      "<tbody>" + body + "</tbody></table>" +
      (recs.length < HW6_K
        ? '<p class="hint">Fewer than ' + HW6_K + " eligible candidates exist for this user (" + recs.length + ").</p>"
        : "") +
      (zeroScore
        ? '<p class="hint">Some listed candidates have a zero score: they were included only to fill the list and have not demonstrated relevance.</p>'
        : "") +
      (method === "ppr" || method === "pagerank"
        ? '<p class="hint">These are graph-walk probabilities, not predicted ratings.</p>'
        : "") + '<p class="hint">Mean training popularity: ' + hw6Fmt(hw6MeanPopularity(recs), 2) + '</p>' +
        hw6CohortSummaryHtml(hw6CohortExclusions("test"));
    hw6SetStatus("Top-10 ready; MF loaded from checkpoint; no full Recall or alpha selection run.");
  } catch (error) {
    target.innerHTML = hw6UnavailableHtml("Recommendations", error);
    hw6SetStatus("Recommendations failed: " + error.message);
  } finally { unlock(); }
}

/** Reason counts are not additive; show their unique union explicitly. */
function hw6CohortSummaryHtml(exclusions) {
  return '<p class="hint">Verified common ' + hw6EscapeHtml(exclusions.split) + ' cohort: ' + exclusions.cohortSize +
    '. Excluded users (unique union): ' + exclusions.excludedTotal +
    '; no train: ' + exclusions.excludedNoTrain + '; no positive train edge: ' + exclusions.excludedNoPositiveEdge +
    '; unavailable MF scores: ' + exclusions.excludedMfUnavailable +
    '; no eligible relevant movies: ' + exclusions.excludedNoEligible +
    '. Users with overlapping reasons: ' + exclusions.overlapUsers.length +
    '. Reasons overlap and must not be summed. Relevant user-movie records outside train: ' +
    exclusions.excludedUnseen + ' (' + exclusions.unseenDistinctMovies + ' distinct movies).</p>';
}

/**
 * Render the retrieval-evaluation panel from a recall report.
 *
 * @param {RecallReport} report
 * @param {string} method
 * @param {string} split
 * @returns {void}
 */
function hw6RenderRecall(report, method, split) {
  const target = document.getElementById("recall-results");
  if (!target) return;
  const perUser = report.perUser || [];
  const rows = perUser
    .slice(0, 50)
    .map(function (row) {
      return (
        "<tr>" +
        "<td>" + row.userId + "</td>" +
        '<td class="num">' + row.relevant + "</td>" +
        '<td class="num">' + row.hit + "</td>" +
        '<td class="num">' + (row.relevant ? hw6Fmt(row.hit / row.relevant) : "\u2014") + "</td>" +
        "</tr>"
      );
    })
    .join("");

  target.innerHTML =
    "<p><strong>Recall@" + HW6_K + " (" + hw6EscapeHtml(method) + ", " + hw6EscapeHtml(split) + "):</strong> " +
    '<span class="num">' + hw6Fmt(report.recall) + "</span> over " + (report.cohort ? report.cohort.length : 0) + " user(s).</p>" +
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">User</th><th scope=\"col\">Relevant</th><th scope=\"col\">Hits</th>" +
    "<th scope=\"col\">Recall</th></tr></thead>" +
    "<tbody>" + (rows || '<tr><td colspan="4">No evaluable users.</td></tr>') + "</tbody></table>" +
    (perUser.length > 50 ? '<p class="hint">Showing the first 50 of ' + perUser.length + " cohort users.</p>" : "") +
    '<p class="hint">Excluded: ' + (report.excludedUnseen || 0) + " relevant movie(s) absent from the training candidate pool; " +
    (report.excludedNoEligible || 0) + " user(s) without an eligible relevant movie. This is a warm-user evaluation and " +
    "does not establish cold-start performance. No rating RMSE is computed for PPR.</p>" +
    hw6CohortSummaryHtml(report.exclusions);
}

/**
 * Populate `#user-select` and `#seed-user-select` from the shared test cohort.
 *
 * @returns {void}
 */
function hw6PopulateUserSelects() {
  const protocol = HW6_RETRIEVAL.protocol("test", true);
  const cohort = protocol.cohort;
  const options = cohort.length
    ? cohort.map(function (id) { return '<option value="' + id + '">User ' + id +
      (protocol.verified ? '' : ' (MF eligibility pending)') + "</option>"; }).join("")
    : '<option value="">No cohort users</option>';
  const userSelect = document.getElementById("user-select");
  if (userSelect) {
    const previous = Number(userSelect.value);
    userSelect.innerHTML = options;
    if (cohort.includes(previous)) userSelect.value = String(previous);
  }
  const seedSelect = document.getElementById("seed-user-select");
  if (seedSelect) {
    const previous = Number(seedSelect.value);
    seedSelect.innerHTML = options;
    if (cohort.includes(previous)) seedSelect.value = String(previous);
    else if (cohort.length > 1) seedSelect.value = String(cohort[1]);
  }
}

// ---------------------------------------------------------------------------
// Verification helpers for the self-checks
//
// These helpers exist only to verify the readme's stated propagation properties
// against `REFERENCE_GRAPH`. The product path never uses them; it calls
// `pageRank` / `personalizedPageRank` through the adapter above.
// ---------------------------------------------------------------------------

/**
 * Assert that block normalization is available. Failures are never converted
 * into readiness=false/PENDING. Dispose the probe's owned tensors immediately.
 *
 * @returns {boolean}
 */
function hw6GraphJsReady() {
  if (typeof rowNormalize !== "function") throw new Error("rowNormalize is unavailable");
  const result = rowNormalize(hw6MakeSyntheticGraph([[1, 1]]));
  try {
    if (!result.userToMovie || !result.movieToUser || result.nodeCount !== 3) {
      throw new Error("rowNormalize returned invalid tensor blocks");
    }
    return true;
  } finally {
    result.dispose();
  }
}

/**
 * Run `fn` with a synthetic bipartite graph installed as the active graph.
 *
 * The real `pageRank` / `personalizedPageRank` read the module-level `GRAPH`, so
 * verifying the student's implementation on a small, cheap graph means swapping
 * the active graph for the duration and restoring it afterwards.
 *
 * @param {number[][]} R `R[u][i]` 0/1.
 * @param {() => any} fn
 * @returns {Promise<any>}
 */
async function hw6WithSyntheticGraph(R, fn) {
  if (typeof setGraph !== "function" || typeof getGraph !== "function") {
    throw hw6InterfaceError("graph.js `setGraph` / `getGraph` are unavailable");
  }
  const previous = getGraph();
  setGraph(hw6MakeSyntheticGraph(R));
  try {
    return await fn();
  } finally {
    setGraph(previous);
  }
}

/**
 * Iterate `graph.js`'s provided 4-node fixture propagation to convergence.
 *
 * Kept for the provided step demo / diagnostics. The readme section 4.1 harness
 * checks NO LONGER call this: they exercise the student's own `pageRank` /
 * `personalizedPageRank` through `REFERENCE_GRAPH.transition`, so an untouched
 * starter cannot pass a check it did not implement.
 *
 * Uses the provided `graphFixturePropagate(alpha, restart, pPrev)` helper when
 * present (fast, no rendering) and otherwise the public fixture demo
 * (`GRAPH_VIEW_STATE.fixture.restart` + `fixtureReset` / `fixtureStep`).
 *
 * @param {string} restartName `"uniform"` or a node label such as `"A"`.
 * @param {number} alpha
 * @returns {PropagationResult|null}
 */
function hw6FixtureIterate(restartName, alpha) {
  const nodeCount = 4;
  if (typeof graphFixturePropagate === "function") {
    try {
      let p = [0.25, 0.25, 0.25, 0.25];
      if (restartName !== "uniform" && typeof graphFixtureRestartVector === "function") {
        p = Array.from(graphFixtureRestartVector(restartName)).map(Number);
      } else if (restartName !== "uniform") {
        const order = typeof REFERENCE_GRAPH !== "undefined" && Array.isArray(REFERENCE_GRAPH.nodeOrder)
          ? REFERENCE_GRAPH.nodeOrder
          : ["A", "B", "C", "D"];
        const seed = order.indexOf(restartName);
        p = [0, 0, 0, 0];
        if (seed >= 0) p[seed] = 1;
      }
      let iterations = 0;
      let residual = Infinity;
      let converged = false;
      while (iterations < HW6_MAX_ITERATIONS) {
        const step = graphFixturePropagate(alpha, restartName, p);
        if (!step || !step.p || step.p.length !== nodeCount) return null;
        let l1 = 0;
        for (let i = 0; i < nodeCount; i++) l1 += Math.abs(Number(step.p[i]) - p[i]);
        p = Array.from(step.p).map(Number);
        iterations++;
        residual = l1;
        if (l1 < HW6_TOLERANCE) {
          converged = true;
          break;
        }
      }
      return { scores: p, iterations: iterations, residual: residual, converged: converged, source: "graph.js graphFixturePropagate" };
    } catch (error) {
      // Fall through to the public fixture demo below.
    }
  }

  if (typeof fixtureReset !== "function" || typeof fixtureStep !== "function") return null;
  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  if (!viewState || !viewState.fixture) return null;
  const fixture = viewState.fixture;
  const previousRestart = fixture.restart;
  const previousMode = viewState.mode;
  const previousAlpha = hw6ReadAlpha();
  try {
    fixture.restart = restartName;
    hw6SetAlpha(alpha);
    fixtureReset();
    let iterations = 0;
    let residual = Infinity;
    let converged = false;
    while (iterations < HW6_MAX_ITERATIONS) {
      const before = Array.from(fixture.p || []).map(Number);
      fixtureStep();
      const after = Array.from(fixture.p || []).map(Number);
      if (after.length !== nodeCount || before.length !== nodeCount) return null;
      let l1 = 0;
      for (let i = 0; i < nodeCount; i++) l1 += Math.abs(after[i] - before[i]);
      iterations++;
      residual = l1;
      if (l1 < HW6_TOLERANCE) {
        converged = true;
        break;
      }
    }
    const scores = Array.from(fixture.p || []).map(Number);
    if (scores.length !== nodeCount) return null;
    return { scores: scores, iterations: iterations, residual: residual, converged: converged, source: "graph.js fixtureStep" };
  } catch (error) {
    return null;
  } finally {
    try {
      fixture.restart = previousRestart;
      fixtureReset();
    } catch (error) {
      // Restoring the demo is best-effort.
    }
    viewState.mode = previousMode;
    hw6SetAlpha(previousAlpha);
  }
}

/**
 * Read `graph.js`'s row-normalized transition for a graph object and convert it
 * to the `{ rows, degrees }` form the verification step uses.
 *
 * `degrees[i]` is 0 exactly when `danglingMask[i]` is true, so the readme's
 * dangling-mass correction `alpha * (Q^T p + d * v) + (1 - alpha) * v` applies
 * unchanged.
 *
 * @param {Object} graph
 * Only small verification fixtures (at most 32 nodes) may be densified here.
 * MovieLens propagation never calls this independent checking helper.
 * @returns {Promise<Object>} Dense rows for one-extra-update verification.
 */
async function hw6NormalizedFromGraph(graph) {
  if (!graph || typeof rowNormalize !== "function") throw new Error("Missing graph or rowNormalize");
  const n = hw6NodeCount(graph);
  if (!n || n > 32) throw new Error("Dense verification is limited to small nonempty fixtures");
  const result = rowNormalize(graph);
  try {
    const [um, mu] = await Promise.all([result.userToMovie.array(), result.movieToUser.array()]);
    const rows = Array.from({ length: n }, () => new Float64Array(n));
    for (let u = 0; u < graph.nUsers; u++) {
      for (let i = 0; i < graph.nMovies; i++) {
        rows[u][graph.nUsers + i] = um[u][i];
        rows[graph.nUsers + i][u] = mu[i][u];
      }
    }
    return { rows, degrees: Float64Array.from(graph.degree), danglingMask: result.danglingMask };
  } finally {
    result.dispose();
  }
}

/**
 * Build a minimal bipartite graph in `graph.js`'s documented `Graph` shape, so a
 * check can exercise the real `rowNormalize` on a fixture with known zero-degree
 * nodes instead of on the readme reference graph only.
 *
 * Node order is users first, then movies (ungrouped `nUsers` / `nMovies`).
 *
 * @param {number[][]} R `R[u][i]` 0/1, shape `[nUsers][nMovies]`.
 * @returns {Object} A `Graph`-shaped object sufficient for `rowNormalize`.
 */
function hw6MakeSyntheticGraph(R) {
  const nUsers = R.length;
  const nMovies = nUsers > 0 ? R[0].length : 0;
  const nodeCount = nUsers + nMovies;
  const userIds = [];
  const movieIds = [];
  for (let u = 0; u < nUsers; u++) userIds.push(u + 1);
  for (let i = 0; i < nMovies; i++) movieIds.push(i + 1);
  const userIndexOf = new Map();
  userIds.forEach(function (id, u) { userIndexOf.set(id, u); });
  const movieIndexOf = new Map();
  movieIds.forEach(function (id, i) { movieIndexOf.set(id, i); });

  const degree = new Float64Array(nodeCount);
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < nMovies; i++) {
      if (R[u][i]) {
        degree[u]++;
        degree[nUsers + i]++;
      }
    }
  }
  const zeroDegreeUsers = [];
  for (let u = 0; u < nUsers; u++) if (degree[u] === 0) zeroDegreeUsers.push(userIds[u]);
  const zeroDegreeMovies = [];
  for (let i = 0; i < nMovies; i++) if (degree[nUsers + i] === 0) zeroDegreeMovies.push(movieIds[i]);
  let positiveEdges = 0;
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < nMovies; i++) if (R[u][i]) positiveEdges++;
  }

  return {
    nUsers: nUsers,
    nMovies: nMovies,
    userIds: userIds,
    movieIds: movieIds,
    userIndexOf: userIndexOf,
    movieIndexOf: movieIndexOf,
    nodeCount: nodeCount,
    nodeOfUser: function (userId) {
      const local = userIndexOf.get(Number(userId));
      return local === undefined ? -1 : local;
    },
    movieOfNode: function (node) {
      const local = Number(node) - nUsers;
      return local >= 0 && local < nMovies ? movieIds[local] : -1;
    },
    isUserNode: function (node) { return Number(node) < nUsers; },
    degree: degree,
    zeroDegreeUsers: zeroDegreeUsers,
    zeroDegreeMovies: zeroDegreeMovies,
    positiveEdges: positiveEdges,
    R: R,
  };
}

/**
 * One readme-formula update: `p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v`.
 *
 * @param {number} alpha
 * @param {{rows:Array<Float64Array>, degrees:Float64Array}} normalized
 * @param {ArrayLike<number>} v Restart distribution.
 * @param {ArrayLike<number>} p Current distribution.
 * @returns {Float64Array}
 */
function hw6Step(alpha, normalized, v, p) {
  const n = p.length;
  const out = new Float64Array(n);
  let dangling = 0;
  for (let i = 0; i < n; i++) {
    if (normalized.degrees[i] <= 0) dangling += p[i];
  }
  for (let i = 0; i < n; i++) {
    if (normalized.degrees[i] <= 0) continue;
    const row = normalized.rows[i];
    const weighted = alpha * p[i];
    for (let j = 0; j < n; j++) {
      if (row[j] !== 0) out[j] += weighted * row[j];
    }
  }
  const base = 1 - alpha + alpha * dangling;
  for (let j = 0; j < n; j++) out[j] += base * v[j];
  return out;
}

/**
 * Reference power iteration on a normalized dense matrix (readme section 3.4).
 *
 * @param {{rows:Array<Float64Array>, degrees:Float64Array}} normalized
 * @param {number} alpha
 * @param {ArrayLike<number>} restart
 * @returns {PropagationResult}
 */
function hw6ReferenceIterate(normalized, alpha, restart) {
  const n = restart.length;
  let p = Float64Array.from(restart);
  let iterations = 0;
  let residual = Infinity;
  let converged = false;
  while (iterations < HW6_MAX_ITERATIONS) {
    const next = hw6Step(alpha, normalized, restart, p);
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - p[i]);
    p = next;
    iterations++;
    residual = l1;
    if (l1 < HW6_TOLERANCE) {
      converged = true;
      break;
    }
  }
  return { scores: p, iterations: iterations, residual: residual, converged: converged, source: "reference" };
}

/**
 * Uniform restart vector over `n` nodes.
 *
 * @param {number} n
 * @returns {Float64Array}
 */
function hw6UniformRestart(n) {
  const v = new Float64Array(n);
  if (n <= 0) return v;
  for (let i = 0; i < n; i++) v[i] = 1 / n;
  return v;
}

/**
 * One-hot restart vector at `seed`.
 *
 * @param {number} n
 * @param {number} seed
 * @returns {Float64Array}
 */
function hw6OneHotRestart(n, seed) {
  const v = new Float64Array(n);
  if (seed >= 0 && seed < n) v[seed] = 1;
  return v;
}

/**
 * Node index of a label inside a graph object, or `-1`.
 *
 * Reads `graph.nodeOrder` (the array `REFERENCE_GRAPH` uses) and then the
 * `label` field of `graph.nodes`.
 *
 * @param {Object} graph
 * @param {string} label
 * @returns {number}
 */
function hw6LabelIndex(graph, label) {
  if (!graph) return -1;
  if (Array.isArray(graph.nodeOrder)) {
    const position = graph.nodeOrder.indexOf(label);
    if (position !== -1) return position;
  }
  if (Array.isArray(graph.nodes)) {
    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i];
      if (!node) continue;
      if (node.label === label) return Number.isFinite(node.index) ? node.index : i;
    }
  }
  return -1;
}

/**
 * Run the STUDENT's `pageRank` on the 4-node reference fixture through the
 * frozen `opts.transition` interface.
 *
 * This is what makes the readme section 4.1 checks exercise the student's own
 * implementation instead of a provided fixture propagation. The provided
 * `graph.js` fixture propagation stays available for the step demo, but the
 * harness must not use it here.
 *
 * Solver failures and invalid interfaces are reported as FAIL.
 *
 * @param {number} alpha Link-following probability.
 * @returns {Promise<PropagationResult>} Length-4 vector, source names the solver.
 */
async function hw6StudentReferencePageRank(alpha) {
  if (typeof pageRank !== "function") throw hw6InterfaceError("graph.js `pageRank` is unavailable");
  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
  if (!reference.transition) throw hw6InterfaceError("REFERENCE_GRAPH.transition is not available yet");
  const out = await pageRank(alpha, { transition: reference.transition });
  const result = hw6NormalizePropagation(out, 4);
  if (!result) throw hw6InterfaceError("graph.js `pageRank` did not return a length-4 vector for REFERENCE_GRAPH.transition");
  result.source = "student pageRank(alpha, {transition: REFERENCE_GRAPH.transition})";
  return result;
}

/**
 * Run the STUDENT's `personalizedPageRank` on the 4-node reference fixture
 * through the frozen `opts.transition` interface, seeding the given node index.
 *
 * @param {number} seedNode Seed node index (A = 0).
 * @param {number} alpha    Link-following probability.
 * @returns {Promise<PropagationResult>} Length-4 vector, source names the solver.
 */
async function hw6StudentReferencePPR(seedNode, alpha) {
  if (typeof personalizedPageRank !== "function") {
    throw hw6InterfaceError("graph.js `personalizedPageRank` is unavailable");
  }
  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
  if (!reference.transition) throw hw6InterfaceError("REFERENCE_GRAPH.transition is not available yet");
  const out = await personalizedPageRank(seedNode, alpha, { transition: reference.transition });
  const result = hw6NormalizePropagation(out, 4);
  if (!result) {
    throw hw6InterfaceError("graph.js `personalizedPageRank` did not return a length-4 vector for REFERENCE_GRAPH.transition");
  }
  result.source = "student personalizedPageRank(seedNode, alpha, {transition: REFERENCE_GRAPH.transition})";
  return result;
}

/**
 * Run the automated self-checks and write the log into `#self-checks`.
 *
 * Known-answer retrieval fixtures do not evaluate MovieLens Recall. All errors,
 * missing interfaces and invalid output report FAIL; no PENDING is accepted.
 *
 * @param {HTMLElement|null} [logElement] Element that receives the text log.
 * @returns {Promise<{passed:number, failed:number, pending:number, checks:Array<Object>}>}
 */
async function runSelfChecks(logElement) {
  const checks = [];
  const jobs = [];
  const close = function (a, b, eps) {
    const tolerance = eps === undefined ? 1e-6 : eps;
    return Math.abs(Number(a) - Number(b)) < tolerance;
  };

  const pass = function (name, detail) { checks.push({ name: name, status: "PASS", detail: detail }); };
  const fail = function (name, detail) { checks.push({ name: name, status: "FAIL", detail: detail }); };

  const check = function (name, fn) { jobs.push({ name, fn }); };

  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  const UNIFORM_REF = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375];
  const RESTART_A_REF = [0.4522328999, 0.1921989825, 0.3555681176, 0];

  // 1. Reference graph, uniform restart (readme 4.1) — the STUDENT's pageRank.
  check("reference graph: uniform alpha=0.85 matches readme 4.1 values", async function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const readme = reference.reference && reference.reference.uniform ? reference.reference.uniform : null;
    const expected = readme && Array.isArray(readme.p) ? readme.p : UNIFORM_REF;
    const expectedRanking = readme && Array.isArray(readme.ranking) ? readme.ranking.join(",") : "2,0,1,3";
    // Exercise the student's own implementation through the frozen interface.
    // There is deliberately NO fixture fallback here: a provided propagation
    // would let an untouched starter pass a readme 4.1 check it did not implement.
    const result = await hw6StudentReferencePageRank(0.85);

    for (let i = 0; i < expected.length; i++) {
      if (!close(result.scores[i], expected[i], 1e-6)) {
        throw new Error("node " + i + " = " + Number(result.scores[i]).toFixed(10) + ", expected " + expected[i]);
      }
    }
    const order = [0, 1, 2, 3].slice().sort(function (a, b) { return Number(result.scores[b]) - Number(result.scores[a]); });
    if (order.join(",") !== expectedRanking) throw new Error("ranking = " + order.join(",") + ", expected " + expectedRanking);
    return "C>A>B>D; produced by " + result.source + ", iterations " + result.iterations;
  });

  // 2. Reference graph, restart at A (readme 4.1) — the STUDENT's personalizedPageRank.
  check("reference graph: restart at A alpha=0.85 matches readme 4.1 values", async function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const readme = reference.reference && reference.reference.restartAtA ? reference.reference.restartAtA : null;
    const expected = readme && Array.isArray(readme.p) ? readme.p : RESTART_A_REF;
    const expectedRanking = readme && Array.isArray(readme.ranking) ? readme.ranking.join(",") : "0,2,1,3";
    const seed = hw6LabelIndex(reference, "A");
    if (seed < 0) throw hw6InterfaceError("node index of A is unavailable on REFERENCE_GRAPH");
    const result = await hw6StudentReferencePPR(seed, 0.85);

    for (let i = 0; i < expected.length; i++) {
      if (!close(result.scores[i], expected[i], 1e-6)) {
        throw new Error("node " + i + " = " + Number(result.scores[i]).toFixed(10) + ", expected " + expected[i]);
      }
    }
    const order = [0, 1, 2, 3].slice().sort(function (a, b) { return Number(result.scores[b]) - Number(result.scores[a]); });
    if (order.join(",") !== expectedRanking) throw new Error("ranking = " + order.join(",") + ", expected " + expectedRanking);
    return "A>C>B>D (D stays 0); produced by " + result.source;
  });

  // 3. Verify the normalized tensor blocks on small fixtures only.
  check("row normalization: every nonzero-degree row sums to 1", async function () {
    await tf.ready();
    // Two fixtures: a star with no zero-degree nodes, and one with two of them.
    const fixtures = [
      { R: [[1, 1]], dangling: [false, false, false] },
      { R: [[1, 1, 0], [0, 0, 0]], dangling: [false, true, false, false, true] },
    ];
    let positive = 0;
    let dangling = 0;
    for (let f = 0; f < fixtures.length; f++) {
      const synthetic = hw6MakeSyntheticGraph(fixtures[f].R);
      const result = await hw6NormalizedFromGraph(synthetic);
      const n = synthetic.nodeCount;
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += result.rows[i][j];
        if (!!result.danglingMask[i] !== fixtures[f].dangling[i]) {
          throw new Error("danglingMask[" + i + "] = " + result.danglingMask[i] + ", expected " + fixtures[f].dangling[i]);
        }
        if (fixtures[f].dangling[i]) {
          if (!close(sum, 0, 1e-12)) throw new Error("dangling row " + i + " is not zero (sum " + sum + ")");
          dangling++;
        } else {
          if (!close(sum, 1, 1e-9)) throw new Error("row " + i + " sums to " + sum);
          positive++;
        }
      }
    }
    return positive + " positive-degree row(s) sum to 1; " + dangling + " dangling row(s) are zero";
  });

  // 4. Restart distributions.
  check("restart distributions: nonnegative and summing to 1", function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const n = hw6NodeCount(reference);
    const vectors = [hw6UniformRestart(n), hw6OneHotRestart(n, 0), hw6OneHotRestart(n, n - 1)];
    for (let k = 0; k < vectors.length; k++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        if (vectors[k][i] < 0) throw new Error("restart vector " + k + " has a negative entry");
        sum += vectors[k][i];
      }
      if (!close(sum, 1, 1e-12)) throw new Error("restart vector " + k + " sums to " + sum);
    }
    return "uniform + one-hot over " + n + " nodes";
  });

  // 5. Probability preservation, on the application's own pageRank.
  check("iterations preserve total probability (sum approx 1)", async function () {
    // 1 user x 2 movies: user 1 rates both movies; both movie nodes point back.
    const result = await pageRank(0.85, { graph: hw6MakeSyntheticGraph([[1, 1]]), trace: true });
    for (const step of result.trace) {
      if (!close(step.mass, 1, 1e-6)) throw new Error("iteration " + step.iteration + " mass = " + step.mass);
    }
    return result.trace.length + " updates preserve mass, not just the final vector";
  });

  // 6. Nonnegativity, on the application's own pageRank.
  check("stationary scores are nonnegative", async function () {
    const result = await hw6AdapterPageRank(hw6MakeSyntheticGraph([[1, 1]]), 0.85);
    for (let i = 0; i < result.scores.length; i++) {
      if (!(Number(result.scores[i]) >= 0)) throw new Error("node " + i + " = " + result.scores[i]);
    }
    return result.scores.length + " scores nonnegative (" + result.source + ")";
  });

  // 7. Stationarity of one extra update, on the application's own pageRank.
  check("one extra update produces approximately the same distribution", async function () {
    const synthetic = hw6MakeSyntheticGraph([[1, 1]]);
    const normalized = await hw6NormalizedFromGraph(synthetic);
    const result = await hw6AdapterPageRank(synthetic, 0.85);
    const n = synthetic.nodeCount;
    const next = hw6Step(0.85, normalized, hw6UniformRestart(n), result.scores);
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - Number(result.scores[i]));
    if (!(l1 < 1e-6)) throw new Error("L1 change after one update = " + l1);
    return "L1 = " + l1.toExponential(3);
  });

  // 8. Zero-degree node fixture preserves mass.
  check("zero-degree node fixture preserves probability mass", async function () {
    // 2 users x 3 movies: user 1 rates movies 1 and 2; user 2 has no positives;
    // movie 3 has no positives. Node order users-then-movies gives exactly two
    // zero-degree nodes (node 1 and node 4) and no padding node.
    const R = [[1, 1, 0], [0, 0, 0]];
    const synthetic = hw6MakeSyntheticGraph(R);
    const normalized = await hw6NormalizedFromGraph(synthetic);
    const expectedDangling = [false, true, false, false, true];
    for (let i = 0; i < synthetic.nodeCount; i++) {
      if (!!normalized.danglingMask[i] !== expectedDangling[i]) {
        throw new Error("danglingMask[" + i + "] = " + normalized.danglingMask[i] + ", expected " + expectedDangling[i]);
      }
    }
    // The application's own PageRank and PPR must keep the total mass at 1 even
    // when a zero-degree node is seeded.
    const outcome = {
      global: await hw6AdapterPageRank(synthetic, 0.85),
      seeded: await hw6AdapterPPR(synthetic, 1, 0.85, 2),
    };
    if (!close(outcome.seeded.scores[1], 1, 1e-6)) throw new Error("isolated seed must retain all mass");
    let globalSum = 0;
    let seededSum = 0;
    for (let i = 0; i < synthetic.nodeCount; i++) {
      globalSum += Number(outcome.global.scores[i]);
      seededSum += Number(outcome.seeded.scores[i]);
    }
    if (!close(globalSum, 1, 1e-6)) throw new Error("uniform-restart total mass = " + globalSum);
    if (!close(seededSum, 1, 1e-6)) throw new Error("zero-degree-seed total mass = " + seededSum);
    return "danglingMask [" + expectedDangling.join(",") + "]; mass " + globalSum.toFixed(8) + " / " + seededSum.toFixed(8);
  });

  // 9. Matrix orientation / graph direction.
  check("matrix orientation: probability collects at the pointed-to node, not the source", async function () {
    // 1 user x 2 movies, both movies pointed to by the user and both pointing
    // back to it. The user node is the hub, so P^T must give it the largest
    // stationary mass; an untransposed update would make the movies win.
    const synthetic = hw6MakeSyntheticGraph([[1, 1]]);
    const normalized = await hw6NormalizedFromGraph(synthetic);
    const n = synthetic.nodeCount;
    const fromApp = Array.from((await hw6AdapterPageRank(synthetic, 0.85)).scores);
    const fromReference = Array.from(hw6ReferenceIterate(normalized, 0.85, hw6UniformRestart(n)).scores);
    const pairs = [fromApp, fromReference];
    for (let k = 0; k < pairs.length; k++) {
      const p = pairs[k];
      if (!(p[0] > p[1])) {
        throw new Error(
          (k === 0 ? "pageRank" : "reference") + " hub mass " + p[0].toFixed(6) +
          " should exceed movie mass " + p[1].toFixed(6),
        );
      }
      if (!close(p[1], p[2], 1e-9)) throw new Error("the two symmetric movies must tie: " + p[1] + " vs " + p[2]);
    }
    let sum = 0;
    for (let i = 0; i < n; i++) sum += fromApp[i];
    if (!close(sum, 1, 1e-6)) throw new Error("total mass = " + sum);
    return "p = [" + fromApp.map(function (x) { return x.toFixed(6); }).join(", ") + "]";
  });

  // 10. Known answers only: never evaluate MovieLens Recall through self-checks.
  check("retrieval: known-answer extraction, ranking, macro Recall and exclusions", async function () {
    const result = await HW6_RETRIEVAL_CHECKS.run();
    return result.checks.join("; ");
  });

  // 11. No padding node.
  check("graph mapping: node count equals observed users plus observed movies", function () {
    const graph = hw6ActiveGraph();
    if (!graph) throw hw6InterfaceError("no graph is active; press \"Build graph\" first");
    if (!HW6_STATE.splitReady) throw hw6InterfaceError("the training split is not available");
    const observedUsers = HW6_STATE.trainUserIds.length;
    const observedMovies = HW6_STATE.candidateIds.length;
    const observed = observedUsers + observedMovies;
    const nodes = hw6NodeCount(graph);
    if (!nodes) throw hw6InterfaceError("the graph does not report a node count");
    if (nodes > observed) {
      throw new Error("graph reports " + nodes + " nodes but observed users + movies = " + observed + " (a padding node?)");
    }
    if (nodes < observed) {
      // A dataset mapping check must fail on a smaller universe as well.
      throw new Error("active graph has " + nodes + " nodes, expected " + observed);
    }
    const graphMovies = hw6MovieIds(graph).length;
    const graphUsers = hw6UserIds(graph).length;
    if (graphMovies && graphUsers && (graphMovies !== observedMovies || graphUsers !== observedUsers)) {
      throw new Error(
        "graph reports " + graphUsers + " users / " + graphMovies + " movies but the training split observed " +
        observedUsers + " / " + observedMovies,
      );
    }
    return observed + " nodes = " + observedUsers + " users + " + observedMovies + " movies";
  });

  // 12. Public surface.
  check("public surface: every required name is defined", function () {
    const required = [
      "movieScores", "recommendTopK", "recallAtK", "popularityScores", "mfScores",
      "serializeVerification", "serializeResults", "serializeRecommendations", "download", "runSelfChecks",
    ];
    const missing = required.filter(function (name) { return typeof window[name] !== "function"; });
    if (missing.length) throw new Error("missing top-level functions: " + missing.join(", "));
    if (!Array.isArray(ALPHA_GRID) || ALPHA_GRID.length !== 4) throw new Error("ALPHA_GRID is not the four-value grid");
    if (!MF_INFO || typeof MF_INFO !== "object") throw new Error("MF_INFO is not exposed");
    return required.length + " functions + ALPHA_GRID + MF_INFO";
  });

  for (const job of jobs) {
    try {
      const outcome = await job.fn();
      if (outcome && outcome.pending) throw new Error(outcome.pending);
      pass(job.name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      const message = String(error && error.message ? error.message : error);
      fail(job.name, message);
    }
  }

  const passed = checks.filter(function (c) { return c.status === "PASS"; }).length;
  const failed = checks.filter(function (c) { return c.status === "FAIL"; }).length;
  const pendingCount = checks.filter(function (c) { return c.status === "PENDING"; }).length;

  const lines = [
    "HW6 self-checks \u2014 pass " + passed + ", fail " + failed + ", pending " + pendingCount + " (of " + checks.length + ")",
    "backend: " + (HW6_STATE.backend || "unknown"),
    "HW5 MF (seeded retraining): " + (MF_INFO.checkpoint === null ? "not loaded/trained yet" : "factors " + MF_INFO.factors + ", checkpoint epoch " + MF_INFO.checkpoint + ", val RMSE " + hw6Fmt(MF_INFO.validationRMSE)),
    "",
  ];
  checks.forEach(function (c) {
    lines.push(c.status + "  " + c.name + (c.detail ? " \u2014 " + c.detail : ""));
  });
  const text = lines.join("\n");

  const target = logElement || document.getElementById("self-checks");
  if (target) target.textContent = text;

  const summary = { passed: passed, failed: failed, pending: pendingCount, checks: checks };
  HW6_STATE.lastVerification = summary;
  return summary;
}

// ---------------------------------------------------------------------------
// Exporters (readme section 9)
// ---------------------------------------------------------------------------

/**
 * Machine-readable self-check export.
 *
 * @returns {Promise<string>} JSON text with `backend`, `generated_at`, `checks`,
 *   `passed`, `failed` and `pending`. Runs the harness first when it has not
 *   been run in this session, so the exported file is never empty.
 */
async function serializeVerification() {
  const result = HW6_STATE.lastVerification || await runSelfChecks();
  let backend = HW6_STATE.backend;
  if (!backend && typeof tf !== "undefined" && typeof tf.getBackend === "function") {
    try {
      backend = tf.getBackend();
    } catch (error) {
      backend = null;
    }
  }
  const payload = {
    backend: backend,
    generated_at: new Date().toISOString(),
    checks: result.checks.map(function (c) {
      return { name: c.name, status: c.status, detail: c.detail };
    }),
    passed: result.passed,
    failed: result.failed,
    pending: result.pending,
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Exporter for the retrieval metrics, as CSV.
 *
 * Columns, in order: `method, alpha, recall_at_10, cohort_size,
 * excluded_unseen_ids, excluded_no_eligible_user, mean_popularity, iterations,
 * residual, converged`, then two extra evidence columns (`split`,
 * `excluded_unseen_id_list`) that do not disturb the required prefix.
 *
 * Rows come from whatever has actually been run: the four-method TEST
 * comparison, the PPR VALIDATION alpha sweep, and the PageRank / PPR
 * convergence records. An unrun panel simply contributes no row - it is never
 * filled with invented numbers.
 *
 * @returns {string} CSV text (RFC 4180 quoting).
 */
function serializeResults() {
  if (typeof HW6_TEST_COMPARISON !== "undefined" && HW6_TEST_COMPARISON.result?.status === "complete") return HW6_TEST_COMPARISON.csvResults();
  if (HW6_ALPHA_SELECTION.result && !HW6_STATE.comparisonRows) return HW6_ALPHA_SELECTION.csv(HW6_ALPHA_SELECTION.result).summary;
  const header = [
    "method", "alpha", "recall_at_10", "cohort_size", "excluded_unseen_ids",
    "excluded_no_eligible_user", "mean_popularity", "iterations", "residual",
    "converged", "split", "excluded_unseen_id_list",
  ];
  const rows = [];

  const exclusions = HW6_STATE.splitReady && HW6_RETRIEVAL.coverageReady() ? hw6CohortExclusions("test") : null;
  const unseenList = exclusions && exclusions.unseenIds ? exclusions.unseenIds.join(";") : "";

  (HW6_STATE.comparisonRows || []).forEach(function (row) {
    rows.push([
      row.method,
      row.alpha === null ? "" : Number(row.alpha).toFixed(2),
      row.recall === null ? "" : row.recall,
      row.cohortSize,
      row.excludedUnseen,
      row.excludedNoEligible,
      row.meanPopularity === null ? "" : row.meanPopularity,
      row.iterations === null ? "" : row.iterations,
      row.residual === null ? "" : row.residual,
      row.converged === null ? "" : String(row.converged),
      row.split,
      unseenList,
    ]);
  });

  (HW6_STATE.alphaSweepRows || []).forEach(function (row) {
    rows.push([
      "ppr",
      Number(row.alpha).toFixed(2),
      row.recall === null ? "" : row.recall,
      row.cohortSize,
      "",
      "",
      "",
      "",
      "",
      "",
      "val",
      "",
    ]);
  });

  ["pagerank", "ppr"].forEach(function (key) {
    const record = HW6_STATE.convergence[key];
    if (!record) return;
    const verdict = hw6ConvergenceVerdict(record);
    rows.push([
      key,
      hw6Fmt(record.alpha, 2),
      "",
      exclusions ? exclusions.cohortSize : "",
      "",
      "",
      "",
      record.iterations === null ? "" : record.iterations,
      record.residual === null ? "" : record.residual,
      verdict.value === null ? "" : String(verdict.value),
      "test",
      "",
    ]);
  });

  return hw6CsvDocument(header, rows);
}

/**
 * Exporter for the recommendation lists, as CSV.
 *
 * Columns: `user_id, method, alpha, rank, movie_id, title, genres, score,
 * training_popularity`. Rows are recorded whenever a list is produced on screen
 * (the Top-10 panel or the two-user demonstration), so the exported file matches
 * the last tables actually rendered.
 *
 * @returns {string} CSV text (RFC 4180 quoting; commas inside titles are quoted).
 */
function serializeRecommendations(options = {}) {
  if (!options.exampleOnly && typeof HW6_TEST_COMPARISON !== "undefined" && HW6_TEST_COMPARISON.result?.status === "complete") {
    return HW6_TEST_COMPARISON.csvRecommendations();
  }
  const header = [
    "user_id", "method", "alpha", "rank", "movie_id", "title", "genres",
    "score", "training_popularity",
    "split", "mean_training_popularity",
  ];
  const rows = HW6_STATE.recommendationRows.map(function (row) {
    return [
      row.userId,
      row.method,
      row.alpha,
      row.rank,
      row.movieId,
      row.title,
      row.genres,
      row.score,
      row.trainingPopularity,
      row.split,
      row.meanPopularity,
    ];
  });
  return hw6CsvDocument(header, rows);
}

/**
 * Wire the three exporter buttons to `download`.
 *
 * @returns {void}
 */
function hw6WireExports() {
  const verification = document.getElementById("export-verification");
  if (verification) {
    verification.addEventListener("click", async function () {
      download("verification.json", await serializeVerification(), "application/json");
    });
  }
  const results = document.getElementById("export-results");
  if (results) {
    results.addEventListener("click", function () {
      download("results.csv", serializeResults(), "text/csv");
    });
  }
  const recommendations = document.getElementById("export-recommendations");
  if (recommendations) {
    recommendations.addEventListener("click", function () {
      download("recommendations.csv", serializeRecommendations(), "text/csv");
    });
  }
}

// ---------------------------------------------------------------------------
// graph.js call sites (provided)
// ---------------------------------------------------------------------------

/**
 * Build the graph from the training split, then refresh the graph summary and
 * view.
 *
 * The real `graph.js` signature is `buildGraph(train)`, so the training-record
 * array is passed; the arity dispatch below keeps a 2-argument
 * `(train, movies)` or 0-argument form working too. `setGraph` is preferred over
 * assigning `GRAPH`, because `graph.js` re-renders the summary and the graph
 * view from inside it.
 *
 * @returns {void}
 */
function hw6HandleBuildGraph() {
  if (typeof buildGraph !== "function") {
    hw6SetStatus("graph.js `buildGraph` is not available on this page.");
    hw6RenderGraphSummary();
    return;
  }
  try {
    const arity = buildGraph.length;
    let graph;
    if (arity >= 2) graph = buildGraph(HW6_SPLIT.train, movies);
    else if (arity === 1) graph = buildGraph(HW6_SPLIT.train);
    else graph = buildGraph();
    if (graph && typeof setGraph === "function") {
      setGraph(graph);
    }
    hw6InvalidateCaches();
    hw6RenderConvergence();
    hw6SafeRenderGraphView();
    const nodes = hw6NodeCount(graph);
    hw6SetStatus("Graph built: " + (nodes ? nodes.toLocaleString("en-US") + " nodes reported." : "node count not reported."));
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    hw6SetStatus("Graph build failed: " + message);
    if (message.indexOf(HW6_TODO_MARKER) === -1) {
      const target = document.getElementById("graph-summary-body");
      if (target) target.innerHTML = hw6UnavailableHtml("Graph summary", error);
    } else {
      hw6RenderGraphSummary();
    }
  }
}

/**
 * Candidate node-score vector for the graph view's "score" colouring.
 *
 * Returns a `Float64Array` indexed by graph node index (users first, then
 * movies): user nodes contribute 0, movie nodes carry the selected method's
 * score. Propagation is never started by rendering: only a matching completed
 * cache may supply PageRank/PPR scores. Otherwise return null for degree colour.
 *
 * @param {Object} graph
 * @param {number} seedUserId
 * @returns {Float64Array|null}
 */
function hw6NodeScoresForView(graph, seedUserId) {
  if (!graph) return null;
  const n = hw6NodeCount(graph);
  if (!n) return null;
  const method = hw6ReadMethod();
  const alpha = hw6ReadAlpha();
  try {
    if (method === "pagerank" || method === "ppr") {
      const cached = method === "ppr" ? HW6_STATE.ppr : HW6_STATE.pageRank;
      if (!cached || cached.graph !== graph || cached.alpha !== alpha ||
          (method === "ppr" && cached.userId !== seedUserId)) return null;
      return Float64Array.from(cached.scores);
    }
    const vector = new Float64Array(n);
    // The view accessor takes a GLOBAL node index.
    const index = hw6MovieNodeIndex(graph);
    if (method === "popularity") {
      index.forEach(function (node, movieId) { vector[node] = hw6PopularityOf(movieId); });
      return vector;
    }
    const scored = mfScores(seedUserId);
    index.forEach(function (node, movieId) {
      const value = scored[movieId];
      vector[node] = value === undefined ? 0 : value;
    });
    return vector;
  } catch (error) {
    return null;
  }
}

/**
 * Render the graph view with the current controls.
 *
 * Uses the real `graph.js` options object (`renderGraphView({ mode, seedUserId,
 * colorBy, maxNodes, getScore })`): after a graph is built the view switches to
 * the seed user's ego subgraph coloured by the selected method's scores. Falls
 * back to a bare call if that shape is rejected.
 *
 * @returns {void}
 */
function hw6SafeRenderGraphView() {
  if (typeof renderGraphView !== "function") return;
  const maxNodes = Math.max(10, Math.round(hw6ReadNumber("max-nodes", 120)));
  const graph = hw6ActiveGraph();
  const seedUserId = hw6ReadNumber("seed-user-select", hw6ReadNumber("user-select", Number.NaN));

  const opts = { maxNodes: maxNodes };
  let scores = null;
  if (graph && Number.isFinite(seedUserId)) {
    scores = hw6NodeScoresForView(graph, seedUserId);
    opts.mode = "ego";
    opts.seedUserId = seedUserId;
    if (scores) {
      opts.colorBy = "score";
      opts.getScore = function (nodeIndex) {
        const value = Number(scores[nodeIndex]);
        return Number.isFinite(value) ? value : 0;
      };
    } else {
      opts.colorBy = "degree";
    }
  }

  try {
    renderGraphView(opts);
    return;
  } catch (error) {
    // Fall through to the no-argument form.
  }
  try {
    renderGraphView();
  } catch (error) {
    hw6SetStatus("Graph view could not be rendered: " + String(error && error.message ? error.message : error));
  }
}

/**
 * Disable existing controls during an async calculation, preserving their
 * previous disabled states. This prevents double clicks and graph/seed changes
 * while a run is in progress; no layout or styling is changed.
 * @returns {function():void|null} Unlock callback, or null if already busy.
 */
function hw6LockPropagationControls() {
  if (!HW6_STATE.loaded) {
    hw6SetStatus(HW6_STATE.uiLoadError || "The application is still loading. Wait for the Ready message before running an action.");
    return null;
  }
  if (HW6_STATE.propagationBusy) return null;
  HW6_STATE.propagationBusy = true;
  const gate = document.getElementById("application-gate");
  const gateWasDisabled = gate ? gate.disabled : false;
  if (gate) gate.disabled = true;
  const controls = Array.from(document.querySelectorAll("button, input, select"));
  const disabled = controls.map(control => control.disabled);
  controls.forEach(control => { control.disabled = true; });
  return function () {
    controls.forEach((control, index) => { control.disabled = disabled[index]; });
    if (gate) gate.disabled = gateWasDisabled || !HW6_STATE.loaded;
    HW6_STATE.propagationBusy = false;
  };
}

/**
 * Run global PageRank at the selected alpha and record convergence.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleRunPageRank() {
  const graph = hw6ActiveGraph();
  const target = document.getElementById("convergence-results");
  if (!graph) {
    hw6SetStatus("Build the graph before running PageRank.");
    if (target) target.innerHTML = '<p class="empty-state">No graph yet \u2014 press \u201cBuild graph\u201d first.</p>';
    return;
  }
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  try {
    hw6SetStatus("Running global PageRank with TensorFlow.js...");
    await hw6RunAndRecordConvergence("pagerank", graph, hw6ReadAlpha());
    hw6RenderConvergence();
    hw6SafeRenderGraphView();
    const record = HW6_STATE.convergence.pagerank;
    const verdict = hw6ConvergenceVerdict(record);
    hw6SetStatus(
      "Global PageRank: " + record.iterations + " iteration(s), final L1 " + hw6Fmt(record.residual, 8) +
      ", converged " + (verdict.value ? "yes" : "no") + (verdict.derived ? " (derived)" : "") + ".",
    );
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Convergence", error);
    hw6SetStatus("PageRank unavailable: " + String(error && error.message ? error.message : error));
  } finally {
    unlock();
  }
}

/**
 * Run PPR from the restart seed user at the selected alpha and record convergence.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleRunPPR() {
  const graph = hw6ActiveGraph();
  const target = document.getElementById("convergence-results");
  if (!graph) {
    hw6SetStatus("Build the graph before running PPR.");
    if (target) target.innerHTML = '<p class="empty-state">No graph yet \u2014 press \u201cBuild graph\u201d first.</p>';
    return;
  }
  const seedUserId = hw6ReadNumber("seed-user-select", hw6ReadNumber("user-select", Number.NaN));
  if (!Number.isFinite(seedUserId)) {
    hw6SetStatus("No restart seed user selected.");
    return;
  }
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  try {
    hw6SetStatus("Running PPR for user " + seedUserId + " with TensorFlow.js...");
    await hw6RunAndRecordConvergence("ppr", graph, hw6ReadAlpha(), seedUserId);
    hw6RenderConvergence();
    hw6SafeRenderGraphView();
    const record = HW6_STATE.convergence.ppr;
    const verdict = hw6ConvergenceVerdict(record);
    hw6SetStatus(
      "PPR from user " + seedUserId + ": " + record.iterations + " iteration(s), final L1 " + hw6Fmt(record.residual, 8) +
      ", converged " + (verdict.value ? "yes" : "no") + (verdict.derived ? " (derived)" : "") + ".",
    );
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Convergence", error);
    hw6SetStatus("PPR unavailable: " + String(error && error.message ? error.message : error));
  } finally {
    unlock();
  }
}

/**
 * Show a saved final TEST comparison, or perform one explicit one-pass run when
 * absent. Details are read from the same per-user records, never recalculated.
 * This action cannot start alpha selection or MF training. Fixed-alpha examples
 * keep their separate button and output container.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleRecall() {
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  try {
    hw6SetStatus("Preparing test comparison from saved alpha/MF; existing results are reused...");
    const comparison = await hw6RunComparison("test");
    if (comparison.status !== "complete") throw new Error(comparison.error);
    const row = comparison.rows.find(row => row.method === hw6ReadMethod());
    const report = { recall: row.recall, cohort: comparison.cohort, exclusions: comparison.exclusions,
      excludedUnseen: comparison.exclusions.excludedUnseen, excludedNoEligible: comparison.exclusions.excludedNoEligible,
      perUser: row.perUser.map(user => ({ userId: user.userId, relevant: user.relevant, hit: user.hits })) };
    HW6_STATE.recallReport = report;
    hw6RenderRecall(report, row.method, "test");
    hw6SetStatus("Test comparison ready at validation-selected alpha " + comparison.alpha + "; no retraining or alpha sweep.");
  } catch (error) {
    hw6SetStatus("Test comparison failed: " + error.message);
  } finally { unlock(); }
}

/**
 * Run the validation-only PPR alpha sweep (readme section 6.1) and render it
 * into `#alpha-sweep-results`, reporting the selected alpha (exact ties toward
 * the smaller alpha). Degrades to a readable explanation while the stubs throw.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleAlphaSweep() {
  const target = document.getElementById("alpha-sweep-results");
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  let sweep = null;
  try {
    sweep = await hw6RunAlphaSweep();
    hw6RenderAlphaSweep(sweep);
    hw6SetStatus(sweep.status === "complete" ? "Validation alpha selection complete: alpha " + sweep.selectedAlpha +
      ". Example slider unchanged; full test Recall has not been run." : "Validation selection failed: " + sweep.error + ". No winner selected.");
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Alpha sweep", error);
    hw6SetStatus("Alpha sweep unavailable: " + String(error && error.message ? error.message : error));
  } finally {
    unlock();
  }
}

/**
 * Display exactly three saved PPR examples and train-only profiles. No
 * propagation, final metric calculation, model training or alpha search.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleTwoUserDemo() {
  const target = document.getElementById("ppr-settings-results");
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  try {
    hw6SetStatus("Loading three cached PPR settings and train-only profiles...");
    await HW6_PPR_PROFILES.ensureSaved();
    HW6_PPR_PROFILES.render();
    hw6SetStatus("Three PPR settings ready; two baseline comparisons; profiles use train only. No algorithms or metrics rerun.");
  } catch (error) {
    if (target) target.textContent = "PPR settings/profile check failed: " + error.message;
    hw6SetStatus("PPR settings/profile check failed: " + error.message);
  } finally {
    unlock();
  }
}

/**
 * Run the self-checks and report the summary in the status line.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleRunTests() {
  const unlock = hw6LockPropagationControls();
  if (!unlock) return;
  const button = document.getElementById("run-tests");
  const originalLabel = button ? button.textContent : "Run tests";
  const summaryEl = document.getElementById("tests-summary");
  const details = document.getElementById("self-checks");
  if (button) { button.textContent = "Running tests…"; button.setAttribute("aria-busy", "true"); }
  if (summaryEl) { summaryEl.textContent = "Running tests… Please wait."; summaryEl.setAttribute("aria-busy", "true"); }
  if (details) details.textContent = "Running self-checks…";
  try {
    hw6SetStatus("Running asynchronous self-checks...");
    // Give the busy state a paint before starting the existing test harness.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const summary = await runSelfChecks();
    if (summaryEl) summaryEl.textContent = summary.passed + " passed, " + summary.failed + " failed, " + summary.pending + " pending";
    hw6SetStatus(
      "Self-checks: pass " + summary.passed + ", fail " + summary.failed + ", pending " + summary.pending +
      ". All implementation errors are reported as FAIL, never PENDING.",
    );
  } catch (error) {
    if (summaryEl) summaryEl.textContent = "Tests could not complete: " + graphErrorMessage(error);
    if (details) details.textContent = "Self-checks failed: " + graphErrorMessage(error);
    hw6SetStatus("Self-checks failed: " + graphErrorMessage(error));
  } finally {
    if (button) { button.textContent = originalLabel; button.setAttribute("aria-busy", "false"); }
    if (summaryEl) summaryEl.setAttribute("aria-busy", "false");
    unlock();
  }
}

// ---------------------------------------------------------------------------
// UI wiring (provided)
// ---------------------------------------------------------------------------

/**
 * Wire every control once. Safe to call a single time from the bootstrap; all
 * lookups are null-guarded so a missing element cannot throw at load time.
 *
 * @returns {void}
 */
function hw6WireControls() {
  const alpha = document.getElementById("alpha");
  if (alpha) {
    alpha.addEventListener("input", function () {
      const readout = document.getElementById("alpha-value");
      const value = hw6ReadAlpha();
      if (readout) readout.textContent = value.toFixed(2);
      hw6InvalidateCaches();
      hw6RenderConvergence();
      hw6SetStatus("Alpha changed to " + value.toFixed(2) + " \u2014 cached PageRank and PPR results are stale; run again.");
    });
  }

  const method = document.getElementById("method");
  if (method) {
    // Selecting MF loads saved weights only; retrieval cannot initiate training.
    method.addEventListener("change", async function () {
      const value = hw6ReadMethod();
      if (value === "mf") {
        try { await hw6EnsureMfForNeed(); }
        catch (error) { hw6SetStatus("HW5 MF failed: " + error.message); return; }
        return;
      }
      hw6SetStatus("Method set to " + value + ". Press \u201cRecommend Top-10\u201d to refresh the list.");
    });
  }

  const buildButton = document.getElementById("build-graph");
  if (buildButton) buildButton.addEventListener("click", hw6HandleBuildGraph);

  const pagerankButton = document.getElementById("run-pagerank");
  if (pagerankButton) pagerankButton.addEventListener("click", hw6HandleRunPageRank);

  const pprButton = document.getElementById("run-ppr");
  if (pprButton) pprButton.addEventListener("click", hw6HandleRunPPR);

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) recommendButton.addEventListener("click", hw6RenderRecommendations);

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) {
    recallButton.disabled = !HW6_FULL_EXPERIMENTS_ENABLED;
    recallButton.title = "Show saved final test results; calculate once only if absent";
    recallButton.addEventListener("click", hw6HandleRecall);
  }

  const alphaSweepButton = document.getElementById("alpha-sweep-btn");
  if (alphaSweepButton) {
    alphaSweepButton.disabled = false;
    alphaSweepButton.title = "Validation-only selection; a compatible saved result is reused";
    alphaSweepButton.addEventListener("click", hw6HandleAlphaSweep);
  }

  const twoUserDemoButton = document.getElementById("two-user-demo-btn");
  if (twoUserDemoButton) twoUserDemoButton.addEventListener("click", hw6HandleTwoUserDemo);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", hw6HandleRunTests);

  hw6WireExports();
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the backend, load the data, split it chronologically, build the
 * provided indices, render the summaries and wire the controls.
 *
 * MF is loaded lazily from a compatible saved checkpoint; absence is an error.
 * Bootstrap itself does not train or claim MF eligibility has been verified.
 *
 * Resolves without throwing; every failure is reported in the status line.
 *
 * @returns {Promise<void>}
 */
async function hw6Bootstrap() {
  hw6SetReadiness("loading", "Loading runtime, data and saved results. Controls will be enabled when the application is ready.");
  if (window.location.protocol === "file:") {
    hw6SetReadiness("error", "This app cannot load data from file://. Open GitHub Pages, or run python -m http.server 8000 --directory week6 from the repository root and visit http://localhost:8000/index.html. Double-clicking index.html does not work.");
    const tfStatus = document.getElementById("tf-status");
    if (tfStatus) tfStatus.textContent = "Initialization skipped: open this app over HTTP or GitHub Pages.";
    return;
  }
  try {
    const backend = await hw6InitTf();
    if (!backend) throw new Error("TensorFlow.js could not start. Check the CDN/network connection and reload.");
    hw6SetStatus("Loading data and checking saved results…");
    await loadData();
    const split = splitByTimestamp();
    HW6_SPLIT.train = split.train;
    HW6_SPLIT.val = split.val;
    HW6_SPLIT.test = split.test;
    HW6_MAX_MOVIE_ID = movies.reduce(function (max, movie) { return Math.max(max, movie.id); }, 0);
    hw6BuildSplitIndices(HW6_SPLIT);
    hw6PopulateUserSelects();
    hw6RenderDatasetSummary();
    hw6RenderGraphSummary();
    const alphaResult = await HW6_ALPHA_SELECTION.restore();
    if (alphaResult?.status === "failed") throw new Error("Saved validation result failed to load: " + alphaResult.error);
    hw6RenderAlphaSweep();
    const testResult = await HW6_TEST_COMPARISON.restore();
    if (testResult?.status === "failed") throw new Error("Saved test result failed to load: " + testResult.error);
    await HW6_PPR_PROFILES.restore();
    hw6WireControls();
    hw6SetReadiness("ready",
      "Ready. Data and saved results loaded: " + HW6_STATE.trainUserIds.length.toLocaleString("en-US") + " training users, " +
      HW6_STATE.candidateIds.length.toLocaleString("en-US") + " candidate movies on " + backend +
      ". HW5 MF loads its saved checkpoint on first use; retraining is disabled. " +
      "Choose an action; Run tests is now available.",
    );
  } catch (error) {
    console.error("Initialization failed:", error);
    hw6SetReadiness("error", "Application could not finish loading: " + graphErrorMessage(error) +
      " Check that the HTTP server is running with all week6 files, then reload. Use GitHub Pages or a local HTTP server, not file://.");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", function () {
    hw6Bootstrap();
  });
} else {
  hw6Bootstrap();
}

