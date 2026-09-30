// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Lookup structures for collaborative filtering
let userRatings = new Map();      // userId -> Map<movieId, rating>
let movieRatings = new Map();     // movieId -> Map<userId, rating>
let userMeans = new Map();        // userId -> mean rating
let movieMeans = new Map();       // movieId -> mean rating

// Movie data structure: { id: number, title: string, year: number }
// Rating data structure: { userId: number, movieId: number, rating: number }

async function loadData() {
    try {
        // Load movie data with windows-1252 encoding
        const movieResponse = await fetch('u.item');
        const movieBuffer = await movieResponse.arrayBuffer();
        const movieText = new TextDecoder('windows-1252').decode(movieBuffer);
        movies = parseItemData(movieText);
        numMovies = movies.length;

        // Load rating data
        const ratingResponse = await fetch('u.data');
        const ratingText = await ratingResponse.text();
        ratings = parseRatingData(ratingText);

        // Build lookup structures
        buildLookupStructures();

        // Calculate number of unique users
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        numUsers = uniqueUsers.size;

        console.log(`Loaded ${movies.length} movies, ${ratings.length} ratings from ${numUsers} users`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function buildLookupStructures() {
    // Build user -> movie ratings and movie -> user ratings
    for (const r of ratings) {
        // User ratings
        if (!userRatings.has(r.userId)) {
            userRatings.set(r.userId, new Map());
        }
        userRatings.get(r.userId).set(r.movieId, r.rating);

        // Movie ratings
        if (!movieRatings.has(r.movieId)) {
            movieRatings.set(r.movieId, new Map());
        }
        movieRatings.get(r.movieId).set(r.userId, r.rating);
    }

    // Calculate user means
    for (const [userId, movieMap] of userRatings) {
        let sum = 0;
        for (const rating of movieMap.values()) {
            sum += rating;
        }
        userMeans.set(userId, sum / movieMap.size);
    }

    // Calculate movie means
    for (const [movieId, userMap] of movieRatings) {
        let sum = 0;
        for (const rating of userMap.values()) {
            sum += rating;
        }
        movieMeans.set(movieId, sum / userMap.size);
    }
}

function parseItemData(text) {
    const lines = text.split('\n');
    const movieData = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('|');
        if (parts.length >= 2) {
            const id = parseInt(parts[0]);
            // Extract title and year from the title field (format: "Title (Year)")
            const titleMatch = parts[1].match(/(.+)\s+\((\d{4})\)$/);
            let title = parts[1];
            let year = null;
            
            if (titleMatch) {
                title = titleMatch[1].trim();
                year = parseInt(titleMatch[2]);
            }
            
            movieData.push({
                id: id,
                title: title,
                year: year
            });
        }
    }
    
    return movieData;
}

function parseRatingData(text) {
    const lines = text.split('\n');
    const ratingData = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('\t');
        if (parts.length >= 4) {
            ratingData.push({
                userId: parseInt(parts[0]),
                movieId: parseInt(parts[1]),
                rating: parseFloat(parts[2]),
                timestamp: parseInt(parts[3])
            });
        }
    }
    
    return ratingData;
}

// ===== Helper functions for Collaborative Filtering =====

// Get all ratings for a specific user: returns Map<movieId, rating> or empty Map
function getUserRatings(userId) {
    return userRatings.get(userId) || new Map();
}

// Get all ratings for a specific movie: returns Map<userId, rating> or empty Map
function getMovieRatings(movieId) {
    return movieRatings.get(movieId) || new Map();
}

// Get average rating of a user
function getUserMean(userId) {
    return userMeans.get(userId) || 0;
}

// Get average rating of a movie
function getMovieMean(movieId) {
    return movieMeans.get(movieId) || 0;
}

// Get rating of specific user for specific movie, or null if not rated
function getRating(userId, movieId) {
    const userMap = userRatings.get(userId);
    if (!userMap) return null;
    return userMap.get(movieId) || null;
}

// Get all user IDs
function getAllUserIds() {
    return Array.from(userRatings.keys()).sort((a, b) => a - b);
}

// Get all movie IDs
function getAllMovieIds() {
    return Array.from(movieRatings.keys()).sort((a, b) => a - b);
}

// Get movie title by ID
function getMovieTitle(movieId) {
    const movie = movies.find(m => m.id === movieId);
    return movie ? movie.title : `Movie ${movieId}`;
}