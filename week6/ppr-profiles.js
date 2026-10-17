/**
 * Exactly three cached PPR settings and train-only user profiles. No propagation,
 * model training, alpha selection or population evaluation is performed here.
 */
const HW6_PPR_PROFILES = (function () {
  let result = null;
  let inFlight = null;
  const settings = [
    { id: "baseline", userId: 1, alpha: 0.85, changedFactor: "none" },
    { id: "change-user", userId: 2, alpha: 0.85, changedFactor: "target user" },
    { id: "change-alpha", userId: 1, alpha: 0.50, changedFactor: "alpha" },
  ];
  const sampleIds = { 1: [181, 127, 14, 204, 56, 216], 2: [50, 100, 127, 258, 275, 283] };
  const genreCounting = "Each distinct movieId with a train rating >=4 contributes once to every listed genre, including unknown. Multi-genre films therefore contribute to several counts; shares use positive-film count and need not sum to 100%. Different movie IDs are kept even when titles coincide.";
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const ensure = (ok, message) => { if (!ok) throw new Error(message); };
  const digest = value => HW6_MF_ENGINE.digest(JSON.stringify(value));

  function reset() { result = null; }

  /** Full history is selected from train only, never from a full rating matrix. */
  function trainHistory(userId) {
    const metadata = new Map(movies.map(movie => [movie.id, movie]));
    return HW6_SPLIT.train.filter(row => row.userId === userId)
      .slice().sort((a, b) => a.timestamp - b.timestamp || a.movieId - b.movieId)
      .map(row => {
        const movie = metadata.get(row.movieId);
        ensure(movie, "Missing metadata in train history");
        return { userId, movieId: row.movieId, title: movie.title, genres: movie.genres.slice(),
          rating: row.rating, timestamp: row.timestamp, positive: row.rating >= 4, split: "train" };
      });
  }

  /** Metadata describes the profile; genres are not PPR graph features. */
  async function trainingProfile(userId) {
    const history = trainHistory(userId);
    const positives = history.filter(row => row.positive);
    const positiveMovies = new Map(positives.map(row => [row.movieId, row]));
    const genreCounts = new Map();
    for (const row of positiveMovies.values()) for (const genre of row.genres) {
      genreCounts.set(genre, (genreCounts.get(genre) || 0) + 1);
    }
    const genreDistribution = Array.from(genreCounts, ([genre, count]) => ({ genre, count,
      shareOfPositiveFilms: count / positiveMovies.size })).sort((a, b) => b.count - a.count || (a.genre < b.genre ? -1 : 1));
    const ratingDistribution = Object.fromEntries([1, 2, 3, 4, 5].map(rating => [rating, history.filter(row => row.rating === rating).length]));
    const examples = sampleIds[userId].map(id => {
      const row = history.find(row => row.movieId === id);
      ensure(row, `Requested profile example ${id} is not in user ${userId}'s train history`);
      return { ...row };
    });
    return { userId, sourceSplit: "train", ratingCount: history.length, positiveRatingCount: positives.length,
      positiveMovieCount: positiveMovies.size, genreAssignments: genreDistribution.reduce((sum, row) => sum + row.count, 0),
      genreDistribution, ratingDistribution, examples, historySha256: await digest(history) };
  }

  async function sourceFile(file) {
    const response = await fetch(file, { cache: "no-store" });
    ensure(response.ok, `Required cached source unavailable: ${file}`);
    const text = await response.text();
    return { file, sha256: await HW6_MF_ENGINE.digest(text), data: JSON.parse(text) };
  }

  function normalizeList(setting, list, source) {
    ensure(list && list.method === "ppr" && list.userId === setting.userId && list.alpha === setting.alpha,
      "Saved source does not contain the exact PPR setting");
    ensure(list.recommendations.length === 10, "Each example must have ten movies");
    const metadata = new Map(movies.map(movie => [movie.id, movie]));
    const excluded = hw6ExclusionSet("test", setting.userId);
    const ids = new Set();
    const recommendations = list.recommendations.map((rec, index) => {
      ensure(!ids.has(rec.movieId) && HW6_STATE.candidateSet.has(rec.movieId) && !excluded.has(rec.movieId),
        "Duplicate/out-of-train/previously rated recommendation");
      ids.add(rec.movieId);
      const movie = metadata.get(rec.movieId);
      ensure(movie && rec.title === movie.title && same(rec.genres, movie.genres), "Cached title/genre mismatch");
      ensure(typeof rec.score === "number" && Number.isFinite(rec.score) && rec.score >= 0,
        "Missing/invalid PPR score is not a zero score");
      ensure(rec.popularity === hw6PopularityOf(rec.movieId), "Popularity must come from positive train users");
      if (index) {
        const previous = list.recommendations[index - 1];
        ensure(previous.score > rec.score || (previous.score === rec.score && previous.movieId < rec.movieId),
          "Cached Top-10 order is invalid");
      }
      return { rank: index + 1, movieId: rec.movieId, title: rec.title, genres: rec.genres.slice(),
        score: rec.score, popularity: rec.popularity };
    });
    return { ...setting, split: "test", sourceFile: source.file, sourceSha256: source.sha256,
      scoreMeaning: "long-run graph-walk probability, not a rating or purchase probability",
      recommendations, meanPopularity: recommendations.reduce((sum, rec) => sum + rec.popularity, 0) / 10,
      zeroScoreCount: recommendations.filter(rec => rec.score === 0).length,
      genreCounts: recommendations.reduce((counts, rec) => {
        rec.genres.forEach(genre => { counts[genre] = (counts[genre] || 0) + 1; }); return counts;
      }, {}) };
  }

  /** Only two comparisons, both against baseline; never compare both-changed settings. */
  function compare(baseline, changed) {
    const oldIds = baseline.recommendations.map(rec => rec.movieId);
    const newIds = changed.recommendations.map(rec => rec.movieId);
    return { baselineId: "baseline", changedId: changed.id, changedFactor: changed.changedFactor,
      commonIds: oldIds.filter(id => newIds.includes(id)), commonCount: oldIds.filter(id => newIds.includes(id)).length,
      appearedIds: newIds.filter(id => !oldIds.includes(id)), disappearedIds: oldIds.filter(id => !newIds.includes(id)),
      baselineMeanPopularity: baseline.meanPopularity, changedMeanPopularity: changed.meanPopularity,
      meanPopularityDelta: (changed.recommendations.reduce((sum, rec) => sum + rec.popularity, 0) -
        baseline.recommendations.reduce((sum, rec) => sum + rec.popularity, 0)) / 10,
      baselineZeroScoreCount: baseline.zeroScoreCount, changedZeroScoreCount: changed.zeroScoreCount };
  }

  function observations(examples, comparisons, profiles) {
    const byUser = new Map(profiles.map(profile => [profile.userId, profile]));
    const genre = (userId, label) => byUser.get(userId).genreDistribution.find(row => row.genre === label)?.count || 0;
    const sample = (userId, id) => {
      const row = byUser.get(userId).examples.find(row => row.movieId === id);
      ensure(row, "Observation must be grounded in a displayed train example");
      return `${row.title} (${row.rating}/5 in train)`;
    };
    const alpha = comparisons.find(row => row.changedId === "change-alpha");
    const seed = comparisons.find(row => row.changedId === "change-user");
    return [
      `Changing the target from user 1 to user 2 at alpha=0.85 keeps ${seed.commonCount}/10 movies. The mean training popularity changes from ${seed.baselineMeanPopularity} to ${seed.changedMeanPopularity}. This describes these two lists, not a quality improvement.`,
      `The target-user change also changes which rated IDs are filtered, under the same train+validation exclusion rule. For example, ${sample(1, 181)} cannot appear for user 1, while Return of the Jedi appears for user 2. ${sample(2, 100)} and ${sample(2, 258)} are filtered for user 2 regardless of rating. These filtering facts alone can create list differences.`,
      `User 1 has ${byUser.get(1).ratingCount} train ratings, ${byUser.get(1).positiveRatingCount} of them positive. Drama has ${genre(1, "Drama")} positive films, Comedy ${genre(1, "Comedy")}, Action ${genre(1, "Action")} and Sci-Fi ${genre(1, "Sci-Fi")}. Contact and Star Trek: First Contact share Sci-Fi with ${sample(1, 204)}; Fargo shares Crime/Drama with ${sample(1, 127)} and ${sample(1, 56)}.`,
      `User 2 has ${byUser.get(2).ratingCount} train ratings, ${byUser.get(2).positiveRatingCount} positive. Drama has ${genre(2, "Drama")} positive films, Romance ${genre(2, "Romance")} and Comedy ${genre(2, "Comedy")}. Return of the Jedi and The Empire Strikes Back have a concrete title/genre connection to ${sample(2, 50)}; Pulp Fiction shares Crime/Drama with ${sample(2, 100)} and ${sample(2, 127)}.`,
      `Changing only alpha for user 1 from 0.85 to 0.50 retains ${alpha.commonCount}/10 movies. Trainspotting (Drama), Sense and Sensibility (Drama/Romance), and E.T. (Children's/Drama/Fantasy/Sci-Fi) appear; Scream, Air Force One and Titanic disappear. Drama tags increase from ${examples[0].genreCounts.Drama} to ${examples[2].genreCounts.Drama}. The mean training popularity decreases from ${alpha.baselineMeanPopularity} to ${alpha.changedMeanPopularity}.`,
      `Sense and Sensibility's Drama/Romance tags are already represented by ${sample(1, 14)} and ${sample(1, 216)}. These are descriptive genre connections, not traced causal explanations of the walks: PPR uses positive rating edges, not genre features.`,
      `Zero-score counts (baseline, change-user, change-alpha): ${examples.map(example => example.zeroScoreCount).join(", ")}. In general, a tie-broken zero-score movie has not demonstrated relevance. Different lists do not prove better quality, and lower training popularity alone does not establish a useful discovery.`,
      `The three settings are fixed controlled examples. They do not choose or replace the separately saved final alpha=0.50. Only train records explain the profiles; held-out ratings are used for exclusion checks, not profile evidence.`,
    ];
  }

  async function construct() {
    const [legacy, final] = await Promise.all([sourceFile("recommendations-alpha085.json"), sourceFile("test-results.json")]);
    ensure(legacy.data.alpha === 0.85 && legacy.data.split === "test" && same(legacy.data.users, [1, 2]) &&
      legacy.data.candidateCount === HW6_STATE.candidateIds.length && final.data.status === "complete" &&
      final.data.split === "test" && final.data.alpha === 0.50 && same(final.data.twoUsers.users, [1, 2]),
    "Incompatible saved example/final settings");
    ensure(legacy.data.mf.runId === final.data.mf.runId, "Example/final MF provenance must refer to the same verified data setup");
    const trainSha256 = await digest(HW6_SPLIT.train);
    const candidateIdsSha256 = await digest(HW6_STATE.candidateIds);
    ensure(trainSha256 === final.data.identity.trainSha256 && candidateIdsSha256 === final.data.identity.candidateIdsSha256,
      "Saved examples must use the current train graph and candidates");
    ensure(final.data.identity.graphNodes === HW6_STATE.trainUserIds.length + HW6_STATE.candidateIds.length &&
      final.data.identity.positiveTrainPairs === HW6_STATE.positivePairs, "Saved examples must use the same train-node/edge universe");
    const examples = settings.map(setting => {
      const source = setting.alpha === 0.85 ? legacy : final;
      const lists = setting.alpha === 0.85 ? source.data.lists : source.data.twoUsers.lists;
      const matches = lists.filter(list => list.method === "ppr" && list.userId === setting.userId && list.alpha === setting.alpha);
      ensure(matches.length === 1, "Expected one and only one cached source list per setting");
      return normalizeList(setting, matches[0], source);
    });
    const profiles = await Promise.all([trainingProfile(1), trainingProfile(2)]);
    const comparisons = [compare(examples[0], examples[1]), compare(examples[0], examples[2])];
    return { schemaVersion: 1, generatedAt: new Date().toISOString(),
      sources: { legacy: { file: legacy.file, sha256: legacy.sha256 }, final: { file: final.file, sha256: final.sha256 },
        trainSha256, candidateIdsSha256, candidateCount: HW6_STATE.candidateIds.length,
        graphNodes: final.data.identity.graphNodes, positiveTrainPairs: final.data.identity.positiveTrainPairs },
      historyRule: "Only HW6_SPLIT.train; timestamp asc then movieId asc. All train ratings retained; positive means rating>=4.",
      genreCounting, examples, comparisons, profiles, observations: observations(examples, comparisons, profiles),
      profileEvidenceSplit: "train", propagatedAgain: false, populationMetricsRecomputed: false, alphaSelectedAgain: false,
      separateFinalSelectedAlpha: 0.50 };
  }

  async function verify(value) {
    ensure(value?.schemaVersion === 1 && value.profileEvidenceSplit === "train" && value.propagatedAgain === false &&
      value.populationMetricsRecomputed === false && value.alphaSelectedAgain === false && value.separateFinalSelectedAlpha === 0.50,
    "Invalid controlled-example scope/provenance");
    const expected = await construct();
    for (const key of ["sources", "historyRule", "genreCounting", "examples", "comparisons", "profiles", "observations"]) {
      ensure(same(value[key], expected[key]), `Saved controlled examples mismatch: ${key}`);
    }
    const trainKeys = new Set(HW6_SPLIT.train.map(row => `${row.userId}:${row.movieId}`));
    const heldoutKeys = new Set([...HW6_SPLIT.val, ...HW6_SPLIT.test].map(row => `${row.userId}:${row.movieId}`));
    const histories = [1, 2].flatMap(trainHistory);
    ensure(histories.every(row => row.split === "train" && trainKeys.has(`${row.userId}:${row.movieId}`) &&
      !heldoutKeys.has(`${row.userId}:${row.movieId}`)), "Profile history includes held-out evidence");
    ensure(value.examples.length === 3 && value.comparisons.length === 2 &&
      value.comparisons.every(row => row.baselineId === "baseline"), "Exactly three settings and two baseline comparisons are required");
    return { status: "passed", settings: settings.map(({ userId, alpha }) => ({ userId, alpha })),
      recommendationRowsChecked: 30, uniqueWithinEachList: true, baselineComparisons: 2, trainHistoryRows: histories.length,
      trainOnlyProfilesChecked: true, cachedSourcesMatched: true, noPropagationOrEvaluation: true };
  }

  async function build() { const value = await construct(); await verify(value); result = value; render(); return value; }
  async function load(value) { await verify(value); result = value; render(); return value; }
  async function restore() {
    try {
      const response = await fetch("ppr-settings-examples.json", { cache: "no-store" });
      if (response.status === 404) return null;
      ensure(response.ok, `Saved PPR examples unavailable (${response.status})`);
      return await load(await response.json());
    } catch (error) {
      result = null;
      const target = document.getElementById("ppr-settings-results");
      if (target) target.textContent = "Saved PPR settings/profile check failed: " + error.message;
      throw error;
    }
  }
  function ensureSaved() {
    if (result) return Promise.resolve(result);
    if (!inFlight) inFlight = restore().then(value => value || build()).finally(() => { inFlight = null; });
    return inFlight;
  }

  function movieLabels(value, ids) {
    const records = new Map(value.examples.flatMap(example => example.recommendations).map(rec => [rec.movieId, rec]));
    return ids.length ? ids.map(id => `${records.get(id).title} (#${id})`).join(", ") : "none";
  }

  function markdown(value = result) {
    const lines = ["# PPR settings and train-only profiles", "", "These are fixed examples, separate from validation alpha selection and final test metrics.", "",
      "| Setting | User | Alpha | Mean train popularity | Zero scores |", "|---|---:|---:|---:|---:|"];
    value.examples.forEach(example => lines.push(`| ${example.id} | ${example.userId} | ${example.alpha.toFixed(2)} | ${example.meanPopularity} | ${example.zeroScoreCount} |`));
    lines.push("", "## Train-only summaries", "", value.genreCounting, "");
    value.profiles.forEach(profile => {
      lines.push(`- **User ${profile.userId}:** ${profile.ratingCount} train ratings; ${profile.positiveRatingCount} positive ratings. ` +
        profile.genreDistribution.map(row => `${row.genre}: ${row.count}`).join("; ") + ".");
      lines.push("  Examples: " + profile.examples.map(row => `${row.title} (#${row.movieId}), ${row.rating}/5`).join("; ") + ".");
    });
    lines.push("", "## Two comparisons with the same baseline", "");
    value.comparisons.forEach(comparison => {
      lines.push(`### ${comparison.changedId}`, `- Common: ${comparison.commonCount}/10 — ${movieLabels(value, comparison.commonIds)}.`,
        `- Appeared: ${movieLabels(value, comparison.appearedIds)}.`, `- Disappeared: ${movieLabels(value, comparison.disappearedIds)}.`,
        `- Mean training popularity: ${comparison.baselineMeanPopularity} -> ${comparison.changedMeanPopularity}; zero scores: ${comparison.baselineZeroScoreCount} -> ${comparison.changedZeroScoreCount}.`, "");
    });
    lines.push("## Observations", "", ...value.observations.map(text => "- " + text), "",
      "## Checkable files", "", "See `ppr-settings-examples.csv` for the three full Top-10 lists and `ppr-settings-comparisons.csv` for deltas. " +
      "Full histories are in `ppr-train-histories.csv`; counts and genre distributions are in `ppr-train-summary.csv` and `ppr-train-genres.csv`. " +
      "JSON records source hashes and history hashes. No MovieLens propagation, MF training or population metric was rerun.", "");
    return lines.join("\n");
  }

  function csv(value = result) {
    const lists = [];
    for (const example of value.examples) for (const rec of example.recommendations) lists.push([example.id, example.changedFactor,
      example.userId, example.alpha, rec.rank, rec.movieId, rec.title, rec.genres.join(';'), rec.score, rec.popularity,
      example.meanPopularity, example.zeroScoreCount, 'test', example.sourceFile]);
    const histories = [1, 2].flatMap(trainHistory);
    const genres = value.profiles.flatMap(profile => profile.genreDistribution.map(row => [profile.userId, row.genre, row.count,
      profile.positiveMovieCount, row.shareOfPositiveFilms, 'train']));
    return {
      lists: hw6CsvDocument(['setting', 'changed_factor', 'user_id', 'alpha', 'rank', 'movie_id', 'title', 'genres', 'ppr_probability',
        'training_popularity', 'mean_training_popularity', 'zero_score_count', 'exclusion_split', 'source_file'], lists),
      comparisons: hw6CsvDocument(['baseline', 'changed', 'changed_factor', 'common_count', 'common_ids', 'appeared_ids', 'disappeared_ids',
        'baseline_mean_popularity', 'changed_mean_popularity', 'mean_popularity_delta', 'baseline_zero_scores', 'changed_zero_scores'],
      value.comparisons.map(row => [row.baselineId, row.changedId, row.changedFactor, row.commonCount, row.commonIds.join(';'),
        row.appearedIds.join(';'), row.disappearedIds.join(';'), row.baselineMeanPopularity, row.changedMeanPopularity,
        row.meanPopularityDelta, row.baselineZeroScoreCount, row.changedZeroScoreCount])),
      histories: hw6CsvDocument(['user_id', 'movie_id', 'title', 'genres', 'rating', 'timestamp', 'positive', 'split'],
        histories.map(row => [row.userId, row.movieId, row.title, row.genres.join(';'), row.rating, row.timestamp, row.positive, row.split])),
      profiles: hw6CsvDocument(['user_id', 'train_ratings', 'positive_train_ratings', 'positive_movie_ids', 'genre_assignments', 'history_sha256', 'source_split'],
        value.profiles.map(profile => [profile.userId, profile.ratingCount, profile.positiveRatingCount, profile.positiveMovieCount,
          profile.genreAssignments, profile.historySha256, 'train'])),
      genres: hw6CsvDocument(['user_id', 'genre', 'positive_film_count', 'positive_films_denominator', 'share_of_positive_films', 'source_split'], genres),
    };
  }

  function render(value = result) {
    const target = document.getElementById("ppr-settings-results");
    if (!target) return;
    if (!value) { target.innerHTML = '<p class="empty-state">Load the saved controlled examples and train-only summaries.</p>'; return; }
    const escape = hw6EscapeHtml;
    target.innerHTML = '<p class="hint">Exactly three PPR settings; all use the same train graph, train candidates and test exclusion rule. ' +
      'These examples do not change the final validation-selected alpha. Scores are walk probabilities, not ratings.</p>' +
      value.examples.map(example => '<h3>' + escape(example.id) + ': user ' + example.userId + ', alpha ' + example.alpha.toFixed(2) + '</h3>' +
        '<p class="hint">Mean train popularity: ' + example.meanPopularity + '; zero scores: ' + example.zeroScoreCount + '.</p>' +
        '<table class="data-table"><thead><tr><th>Rank</th><th>ID</th><th>Title</th><th>Genres</th><th>PPR probability</th><th>Train popularity</th></tr></thead><tbody>' +
        example.recommendations.map(rec => '<tr><td>' + rec.rank + '</td><td>' + rec.movieId + '</td><td>' + escape(rec.title) +
          '</td><td>' + escape(rec.genres.join(', ')) + '</td><td>' + hw6Fmt(rec.score, 9) + '</td><td>' + rec.popularity + '</td></tr>').join('') + '</tbody></table>').join('') +
      '<h3>Only comparisons against baseline</h3>' + value.comparisons.map(row => '<h4>' + escape(row.changedId) + '</h4><p>Common: ' + row.commonCount + '/10 — ' +
        escape(movieLabels(value, row.commonIds)) + '.</p><p>Appeared: ' + escape(movieLabels(value, row.appearedIds)) +
        '.</p><p>Disappeared: ' + escape(movieLabels(value, row.disappearedIds)) + '.</p><p>Mean popularity: ' + row.baselineMeanPopularity + ' → ' + row.changedMeanPopularity +
        '; zeros: ' + row.baselineZeroScoreCount + ' → ' + row.changedZeroScoreCount + '.</p>').join('') +
      '<h3>Train-only profile summaries</h3><p class="hint">' + escape(value.genreCounting) + '</p>' + value.profiles.map(profile =>
        '<h4>User ' + profile.userId + ': ' + profile.ratingCount + ' train ratings, ' + profile.positiveRatingCount + ' positive</h4><p>' +
        escape(profile.genreDistribution.map(row => row.genre + ': ' + row.count).join('; ')) + '.</p><ul>' +
        profile.examples.map(row => '<li>' + escape(row.title) + ' (#' + row.movieId + '), rating ' + row.rating + '/5 — ' + escape(row.genres.join(', ')) + '</li>').join('') + '</ul>').join('') +
      '<h3>Observations</h3><ul>' + value.observations.map(text => '<li>' + escape(text) + '</li>').join('') + '</ul>' +
      '<p><a href="ppr-settings-observations.md">English observations</a> · <a href="ppr-settings-examples.csv">Three example lists</a> · ' +
      '<a href="ppr-train-histories.csv">Full train histories</a></p>';
  }

  return { build, load, restore, ensureSaved, verify, csv, markdown, render, reset, trainHistory, trainingProfile,
    get result() { return result; }, snapshot: () => result, restoreSnapshot: saved => { result = saved; } };
})();
