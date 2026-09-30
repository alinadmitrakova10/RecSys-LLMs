// Selected IDs for CF prediction
let selectedUserId = null;
let selectedMovieId = null;

// Initialize application when window loads
window.onload = async function() {
    try {
        updateStatus('Loading data', false);
        await loadData();
        setupUserSearch();
        setupMovieSearch();
        updatePredictButton();
        updateStatusLoaded();
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message, true);
    }
};

// ===== User search: "1", "25", "User 25", "user 25" =====
function setupUserSearch() {
    const input = document.getElementById('user-input');
    const dropdown = document.getElementById('user-suggestions');
    const clearBtn = document.getElementById('user-clear');

    input.addEventListener('input', () => {
        selectedUserId = null;
        updatePredictButton();
        const query = input.value.trim().toLowerCase();
        if (!query) {
            hideDropdown(dropdown);
            return;
        }
        const matches = [];
        for (let id = 1; id <= numUsers; id++) {
            const label = `User ${id}`;
            if (label.toLowerCase().includes(query) || String(id).includes(query)) {
                matches.push(id);
                if (matches.length >= 20) break;
            }
        }
        renderUserSuggestions(matches, dropdown, input);
    });

    input.addEventListener('focus', () => {
        if (input.value.trim() === '') {
            const initial = [];
            for (let id = 1; id <= Math.min(numUsers, 20); id++) initial.push(id);
            renderUserSuggestions(initial, dropdown, input);
        }
    });

    clearBtn.addEventListener('click', () => {
        input.value = '';
        selectedUserId = null;
        hideDropdown(dropdown);
        updatePredictButton();
        input.focus();
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) hideDropdown(dropdown);
    });
}

function renderUserSuggestions(ids, dropdown, input) {
    dropdown.innerHTML = '';
    if (ids.length === 0) {
        hideDropdown(dropdown);
        return;
    }
    ids.forEach((id) => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<div class="suggestion-main">User ${id}</div>`;
        div.addEventListener('click', () => {
            selectedUserId = id;
            input.value = `User ${id}`;
            hideDropdown(dropdown);
            updatePredictButton();
        });
        dropdown.appendChild(div);
    });
    dropdown.classList.add('show');
}

// ===== Movie search: id or any part of title, case-insensitive =====
function setupMovieSearch() {
    const input = document.getElementById('movie-input');
    const dropdown = document.getElementById('movie-suggestions');
    const clearBtn = document.getElementById('movie-clear');

    input.addEventListener('input', () => {
        selectedMovieId = null;
        updatePredictButton();
        const query = input.value.trim().toLowerCase();
        if (!query) {
            hideDropdown(dropdown);
            return;
        }
        const matches = [];
        for (const m of movies) {
            const titleLower = m.title.toLowerCase();
            if (titleLower.includes(query) || String(m.id).includes(query)) {
                matches.push(m);
                if (matches.length >= 20) break;
            }
        }
        renderMovieSuggestions(matches, dropdown, input);
    });

    input.addEventListener('focus', () => {
        if (input.value.trim() === '') {
            renderMovieSuggestions(movies.slice(0, 20), dropdown, input);
        }
    });

    clearBtn.addEventListener('click', () => {
        input.value = '';
        selectedMovieId = null;
        hideDropdown(dropdown);
        updatePredictButton();
        input.focus();
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) hideDropdown(dropdown);
    });
}

function renderMovieSuggestions(list, dropdown, input) {
    dropdown.innerHTML = '';
    if (list.length === 0) {
        hideDropdown(dropdown);
        return;
    }
    list.forEach((m) => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        const fullTitle = m.year ? `${m.title} (${m.year})` : m.title;
        div.innerHTML = `<div class="suggestion-main">${fullTitle}</div><div class="suggestion-sub">ID: ${m.id}</div>`;
        div.addEventListener('click', () => {
            selectedMovieId = m.id;
            input.value = fullTitle;
            hideDropdown(dropdown);
            updatePredictButton();
        });
        dropdown.appendChild(div);
    });
    dropdown.classList.add('show');
}

function hideDropdown(dropdown) {
    dropdown.classList.remove('show');
}

function updatePredictButton() {
    document.getElementById('predict-btn').disabled = !(selectedUserId && selectedMovieId);
}

// ===== Predict with both CF approaches =====
function predictRating() {
    if (!selectedUserId || !selectedMovieId) {
        showError('Please select both a user and a movie.');
        return;
    }
    try {
        const userPred = predictUserBased(selectedUserId, selectedMovieId);
        const itemPred = predictItemBased(selectedUserId, selectedMovieId);
        showPredictions(userPred, itemPred);
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Error making prediction: ' + error.message);
    }
}

function formatPrediction(value, emptyMessage) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return `<span class="no-prediction">${emptyMessage}</span>`;
    }
    return `<span class="prediction-value">${value.toFixed(2)}</span><span class="prediction-label">predicted</span>`;
}

function showPredictions(userPred, itemPred) {
    const resultElement = document.getElementById('result');
    resultElement.className = 'result';
    resultElement.innerHTML = `
        <div class="result-card">
            <div class="pred">
                <h4>User-Based CF:</h4>
                <div class="method-desc">Based on ratings from similar users</div>
                <div>${formatPrediction(userPred, 'Not enough similar users')}</div>
            </div>
            <div class="divider"></div>
            <div class="pred">
                <h4>Item-Based CF:</h4>
                <div class="method-desc">Based on movies similar to the ones you rated</div>
                <div>${formatPrediction(itemPred, 'Not enough similar movies')}</div>
            </div>
        </div>`;
}

function showError(message) {
    const resultElement = document.getElementById('result');
    resultElement.className = 'result error';
    resultElement.textContent = message;
}

// UI helper functions
let statusHideTimer = null;

function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    if (statusHideTimer) {
        clearTimeout(statusHideTimer);
        statusHideTimer = null;
    }
    statusElement.className = 'status visible' + (isError ? ' error' : '');
    statusElement.textContent = message;
}

function updateStatusLoaded() {
    const statusElement = document.getElementById('status');
    if (statusHideTimer) {
        clearTimeout(statusHideTimer);
        statusHideTimer = null;
    }
    statusElement.className = 'status visible success';
    statusElement.textContent = '✓ Data loaded';
    statusHideTimer = setTimeout(() => {
        statusElement.classList.add('hidden');
        statusHideTimer = setTimeout(() => {
            statusElement.className = 'status';
            statusElement.textContent = '';
        }, 400);
    }, 2000);
}

// ===== Collaborative Filtering Similarity Functions =====

// Pearson correlation between two users
// Returns { similarity, commonCount }
function getUserSimilarity(userId1, userId2) {
    const ratings1 = getUserRatings(userId1);
    const ratings2 = getUserRatings(userId2);
    
    if (ratings1.size === 0 || ratings2.size === 0) {
        return { similarity: 0, commonCount: 0 };
    }
    
    // Find common movies
    const commonMovies = [];
    for (const [movieId, rating1] of ratings1) {
        const rating2 = ratings2.get(movieId);
        if (rating2 !== undefined) {
            commonMovies.push({ movieId, rating1, rating2 });
        }
    }
    
    const commonCount = commonMovies.length;
    if (commonCount < 3) {
        return { similarity: 0, commonCount };
    }
    
    // Get user means
    const mean1 = getUserMean(userId1);
    const mean2 = getUserMean(userId2);
    
    // Calculate Pearson correlation
    let num = 0;
    let den1 = 0;
    let den2 = 0;
    
    for (const { rating1, rating2 } of commonMovies) {
        const diff1 = rating1 - mean1;
        const diff2 = rating2 - mean2;
        num += diff1 * diff2;
        den1 += diff1 * diff1;
        den2 += diff2 * diff2;
    }
    
    if (den1 === 0 || den2 === 0) {
        return { similarity: 0, commonCount };
    }
    
    const similarity = num / (Math.sqrt(den1) * Math.sqrt(den2));
    return { similarity, commonCount };
}

// Pearson correlation between two movies (items)
// Returns { similarity, commonCount }
function getMovieSimilarity(movieId1, movieId2) {
    const ratings1 = getMovieRatings(movieId1);
    const ratings2 = getMovieRatings(movieId2);
    
    if (ratings1.size === 0 || ratings2.size === 0) {
        return { similarity: 0, commonCount: 0 };
    }
    
    // Find common users
    const commonUsers = [];
    for (const [userId, rating1] of ratings1) {
        const rating2 = ratings2.get(userId);
        if (rating2 !== undefined) {
            commonUsers.push({ userId, rating1, rating2 });
        }
    }
    
    const commonCount = commonUsers.length;
    if (commonCount < 3) {
        return { similarity: 0, commonCount };
    }
    
    // Get movie means
    const mean1 = getMovieMean(movieId1);
    const mean2 = getMovieMean(movieId2);
    
    // Calculate Pearson correlation
    let num = 0;
    let den1 = 0;
    let den2 = 0;
    
    for (const { rating1, rating2 } of commonUsers) {
        const diff1 = rating1 - mean1;
        const diff2 = rating2 - mean2;
        num += diff1 * diff2;
        den1 += diff1 * diff1;
        den2 += diff2 * diff2;
    }
    
    if (den1 === 0 || den2 === 0) {
        return { similarity: 0, commonCount };
    }
    
    const similarity = num / (Math.sqrt(den1) * Math.sqrt(den2));
    return { similarity, commonCount };
}

// ===== Collaborative Filtering Prediction Functions =====

// Item-based CF prediction
// Formula: movieMean + Σ(sim * (userRating - neighborMovieMean)) / Σ|sim|
// Returns predicted rating (1-5) or null if no neighbors
function predictItemBased(userId, movieId) {
    const userRatingsMap = getUserRatings(userId);
    if (userRatingsMap.size === 0) return null;
    
    const targetMovieMean = getMovieMean(movieId);
    if (targetMovieMean === 0) return null;
    
    let weightedSum = 0;
    let similaritySum = 0;
    
    // For each movie the user has rated
    for (const [ratedMovieId, userRating] of userRatingsMap) {
        if (ratedMovieId === movieId) continue; // Skip the target movie itself
        
        const { similarity } = getMovieSimilarity(movieId, ratedMovieId);
        if (similarity === 0) continue; // Skip only zero similarity
        
        const neighborMovieMean = getMovieMean(ratedMovieId);
        const deviation = userRating - neighborMovieMean;
        
        weightedSum += similarity * deviation;
        similaritySum += Math.abs(similarity);
    }
    
    if (similaritySum === 0) return null;
    
    const prediction = targetMovieMean + weightedSum / similaritySum;
    return Math.max(1, Math.min(5, prediction));
}

// User-based CF prediction
// Formula: userMean + Σ(sim * (neighborRating - neighborMean)) / Σ|sim|
// Returns predicted rating (1-5) or null if no neighbors
function predictUserBased(userId, movieId) {
    const movieRatingsMap = getMovieRatings(movieId);
    if (movieRatingsMap.size === 0) return null;
    
    const targetUserMean = getUserMean(userId);
    if (targetUserMean === 0) return null;
    
    let weightedSum = 0;
    let similaritySum = 0;
    
    // For each user who rated the target movie
    for (const [otherUserId, neighborRating] of movieRatingsMap) {
        if (otherUserId === userId) continue; // Skip the target user
        
        const { similarity } = getUserSimilarity(userId, otherUserId);
        if (similarity === 0) continue; // Skip only zero similarity
        
        const neighborMean = getUserMean(otherUserId);
        const deviation = neighborRating - neighborMean;
        
        weightedSum += similarity * deviation;
        similaritySum += Math.abs(similarity);
    }
    
    if (similaritySum === 0) return null;
    
    const prediction = targetUserMean + weightedSum / similaritySum;
    return Math.max(1, Math.min(5, prediction));
}
