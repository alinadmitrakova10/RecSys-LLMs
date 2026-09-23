// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in the u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data with windows-1252 encoding
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesArrayBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('windows-1252').decode(moviesArrayBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Compute popularity metrics
        computeMoviePopularity();
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('|');
        if (fields.length < 24) continue; // Skip invalid lines
        
        const id = parseInt(fields[0]);
        const title = fields[1];
        
        // Extract genres: skip field 5 (unknown), take fields 6-23 (18 genres: Action to Western)
        const genreVector = fields.slice(6, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreVector[index] === 1);
        
        movies.push({ id, title, genres, genreVector });
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines
        
        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);
        
        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// Compute rating count per movie and attach to movie objects
function computeMoviePopularity() {
    const ratingCounts = {};
    for (const r of ratings) {
        ratingCounts[r.itemId] = (ratingCounts[r.itemId] || 0) + 1;
    }
    for (const m of movies) {
        m.ratingCount = ratingCounts[m.id] || 0;
    }
}
