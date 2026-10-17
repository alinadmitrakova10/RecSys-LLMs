// ---------------------------------------------------------------------------
// week6/data.js — MovieLens 100k data layer (classic <script>, no modules)
//
// Loads u.item (movies) and u.data (ratings), builds a sparse-friendly
// rating matrix, and splits every user's ratings chronologically 80/10/10.
//
// Genre alignment note (fixes the Week-2 off-by-one):
//   A u.item row has 24 '|'-separated fields:
//     0 id | 1 title | 2 release | 3 video | 4 url | 5 unknown |
//     6 Action | 7 Adventure | ... | 23 Western
//   Keep unknown separately from the 18 named flags at indices 6..23.
//   Invalid records and duplicate IDs/pairs fail explicitly with line numbers.
// ---------------------------------------------------------------------------

// Global variables for storing movie and rating data
let movies = [];          // [{ id, title, genres: string[] }]
let ratings = [];         // [{ userId, movieId, rating, timestamp }]

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;         // highest user id found in u.data
let numMovies = 0;        // number of parsed movies
let ratingMatrix = null;  // (numUsers + 1) x (maxMovieId + 1); 0 = unobserved

// Genre names for the 18 real flags (u.item field indices 6..23)
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Users skipped while splitting (e.g. too few ratings), collected by
// splitByTimestamp() for reporting.
let SPLIT_EXCLUSIONS = [];

/**
 * Fetch and validate both files before replacing the loaded dataset.
 * MovieLens u.item uses a Latin-1-compatible single-byte encoding, not UTF-8.
 * A failed fetch or parse leaves the previous dataset intact. Successful calls
 * replace arrays, dimensions and the matrix, so reloads never append records.
 * @returns {Promise<void>}
 */
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = new TextDecoder('iso-8859-1').decode(
            await moviesResponse.arrayBuffer()
        );
        const nextMovies = readItemRecords(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        const nextRatings = readRatingRecords(
            ratingsText, new Set(nextMovies.map(movie => movie.id))
        );

        movies = nextMovies;
        ratings = nextRatings;
        SPLIT_EXCLUSIONS = [];
        // Raw-ID matrix dimensions are retained for the existing MF interface.
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('result');
        if (errorTarget) {
            errorTarget.textContent =
                `Error: ${error.message}. Please make sure u.item and u.data are in the correct location.`;
            errorTarget.className = 'error';
        }
        throw error; // Re-throw so the page script can handle the error
    }
}

/**
 * Parse every physical record; allow one final line terminator, not blank rows.
 * Collect an error count and the first five line-specific diagnostics. Nothing
 * is committed when any row fails, and an empty dataset is rejected.
 * @param {string} text Decoded file contents.
 * @param {string} source File name used in errors.
 * @param {function(string, number):Object} parseRow Validates one record.
 * @returns {Object[]} Valid records only when the entire file is valid.
 */
function readMovieLensRecords(text, source, parseRow) {
    const lines = text.split(/\r?\n/);
    if (lines[lines.length - 1] === '') lines.pop();
    const records = [];
    const errors = [];
    let errorCount = 0;
    lines.forEach((line, index) => {
        try {
            if (!line.trim()) throw new Error('blank record');
            records.push(parseRow(line, index + 1));
        } catch (error) {
            errorCount++;
            if (errors.length < 5) errors.push(`line ${index + 1}: ${error.message}`);
        }
    });
    if (errorCount) {
        throw new Error(`${source}: ${errorCount} invalid record(s); ${errors.join('; ')}`);
    }
    if (!records.length) throw new Error(`${source}: no records`);
    return records;
}

/** @returns {boolean} Whether a field is a positive, safe decimal integer. */
function isMovieLensInteger(value) {
    return /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) && Number(value) > 0;
}

/** Parse metadata without mutating globals; duplicate movie IDs are errors. */
function readItemRecords(text) {
    const firstLineById = new Map();
    return readMovieLensRecords(text, 'u.item', (line, lineNumber) => {
        const fields = line.split('|');
        if (fields.length !== 24) throw new Error('expected exactly 24 fields');
        if (!isMovieLensInteger(fields[0])) throw new Error('invalid movie ID');
        const id = Number(fields[0]);
        if (!fields[1].trim()) throw new Error('missing title');
        if (fields.slice(5).some(flag => flag !== '0' && flag !== '1')) {
            throw new Error('genre flags must be 0 or 1');
        }
        if (firstLineById.has(id)) {
            throw new Error(`duplicate movie ID ${id} (first at line ${firstLineById.get(id)})`);
        }
        firstLineById.set(id, lineNumber);
        const genres = genreNames.filter((_, index) => fields[index + 6] === '1');
        if (fields[5] === '1') genres.unshift('unknown');
        return { id, title: fields[1], genres };
    });
}

/**
 * Parse ratings without mutating globals. Duplicate pairs are rejected rather
 * than choosing a winner. Optional metadata IDs enable referential validation.
 * @param {string} text Decoded u.data contents.
 * @param {Set<number>|null} [movieIds=null] Known movie IDs, when available.
 * @returns {Object[]} Valid rating records in original file order.
 */
function readRatingRecords(text, movieIds = null) {
    const firstLineByPair = new Map();
    return readMovieLensRecords(text, 'u.data', (line, lineNumber) => {
        const fields = line.split('\t');
        if (fields.length !== 4) throw new Error('expected exactly 4 fields');
        if (!fields.every(isMovieLensInteger)) throw new Error('expected positive integer fields');
        const [userId, movieId, rating, timestamp] = fields.map(Number);
        if (rating > 5) throw new Error('rating must be in [1, 5]');
        if (movieIds && !movieIds.has(movieId)) throw new Error(`unknown movie ID ${movieId}`);
        const pair = `${userId}:${movieId}`;
        if (firstLineByPair.has(pair)) {
            throw new Error(`duplicate pair ${pair} (first at line ${firstLineByPair.get(pair)})`);
        }
        firstLineByPair.set(pair, lineNumber);
        return { userId, movieId, rating, timestamp };
    });
}

/** Replace metadata only after successful validation; never append records. */
function parseItemData(text) {
    movies = readItemRecords(text);
}

/** Replace ratings only after successful validation; never append records. */
function parseRatingData(text) {
    ratings = readRatingRecords(text, movies.length ? new Set(movies.map(movie => movie.id)) : null);
}

// Build the sparse-friendly rating matrix.
// Shape: (numUsers + 1) x (maxMovieId + 1), indexed by raw ids, so that
//   ratingMatrix[userId][movieId] === rating
// and an unobserved pair is 0. MovieLens ratings are 1-5, so 0 is unambiguous.
function buildRatingMatrix() {
    const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);

    ratingMatrix = [];
    for (let u = 0; u <= numUsers; u++) {
        ratingMatrix.push(new Array(maxMovieId + 1).fill(0));
    }

    for (const r of ratings) {
        if (r.userId <= numUsers && r.movieId <= maxMovieId) {
            ratingMatrix[r.userId][r.movieId] = r.rating;
        }
    }
}

// Split EACH user's ratings by ascending timestamp into 80/10/10.
// Returns { train, val, test }, each an array of the same
// { userId, movieId, rating, timestamp } objects.
//
// Rules:
//   - counts are deterministic: nTrain = floor(0.8 * n), nVal = floor(0.1 * n),
//     nTest = n - nTrain - nVal (the remainder)
//   - users with fewer than 5 ratings are skipped and recorded in
//     SPLIT_EXCLUSIONS
//   - timestamp ties are broken by movieId so the split is deterministic
function splitByTimestamp() {
    const MIN_RATINGS = 5;

    const byUser = new Map();
    for (const r of ratings) {
        if (!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push(r);
    }

    const train = [];
    const val = [];
    const test = [];

    SPLIT_EXCLUSIONS = [];

    for (const [userId, userRatings] of byUser) {
        const n = userRatings.length;
        if (n < MIN_RATINGS) {
            SPLIT_EXCLUSIONS.push({
                userId,
                count: n,
                reason: `fewer than ${MIN_RATINGS} ratings`
            });
            continue;
        }

        const sorted = userRatings.slice().sort((a, b) =>
            a.timestamp - b.timestamp || a.movieId - b.movieId
        );

        const nTrain = Math.floor(0.8 * n);
        const nVal = Math.floor(0.1 * n);

        for (let i = 0; i < n; i++) {
            if (i < nTrain) train.push(sorted[i]);
            else if (i < nTrain + nVal) val.push(sorted[i]);
            else test.push(sorted[i]);
        }
    }

    return { train, val, test };
}
