// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate comboboxes and set up event listeners
        populateComboboxes();
        setupComboboxEvents();
        updateComboboxAvailability();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Sorted movie list for comboboxes
let sortedMovies = [];

// Populate all comboboxes with sorted movie titles
function populateComboboxes() {
    // Only include recommendable movies in the dropdown
    sortedMovies = [...movies].filter(m => m.isRecommendable).sort((a, b) => a.displayTitle.localeCompare(b.displayTitle));
    
    const comboboxConfigs = [
        { id: 'movie-select', inputId: 'movie-select-input', dropdownId: 'combobox-1' },
        { id: 'movie-select-2', inputId: 'movie-select-2-input', dropdownId: 'combobox-2' },
        { id: 'movie-select-3', inputId: 'movie-select-3-input', dropdownId: 'combobox-3' }
    ];
    
    comboboxConfigs.forEach(config => {
        const hiddenInput = document.getElementById(config.id);
        const dropdown = document.querySelector(`#${config.dropdownId} .combobox-dropdown`);
        renderComboboxOptions(dropdown, sortedMovies, hiddenInput.value, new Set());
    });
}

// Render options in a combobox dropdown
function renderComboboxOptions(dropdown, moviesList, currentValue, disabledIds = new Set()) {
    dropdown.innerHTML = '';
    
    if (moviesList.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'combobox-empty';
        emptyDiv.textContent = 'No movies found';
        dropdown.appendChild(emptyDiv);
        return;
    }
    
    moviesList.forEach(movie => {
        const option = document.createElement('div');
        option.className = 'combobox-option';
        option.dataset.value = movie.id;
        option.textContent = movie.displayTitle;
        option.setAttribute('role', 'option');
        
        if (movie.id == currentValue) {
            option.classList.add('highlighted');
        }
        
        if (disabledIds.has(movie.id)) {
            option.classList.add('disabled');
        }
        
        dropdown.appendChild(option);
    });
}

// Helper to get currently selected movie IDs in other comboboxes
function getDisabledIds(currentHiddenInputId) {
    const disabledIds = new Set();
    ['movie-select', 'movie-select-2', 'movie-select-3'].forEach(id => {
        if (id !== currentHiddenInputId) {
            const val = parseInt(document.getElementById(id).value);
            if (!isNaN(val)) disabledIds.add(val);
        }
    });
    return disabledIds;
}

// Set up event listeners for all comboboxes
function setupComboboxEvents() {
    const comboboxConfigs = [
        { id: 'movie-select', inputId: 'movie-select-input', dropdownId: 'combobox-1', clearId: '#combobox-1 .combobox-clear' },
        { id: 'movie-select-2', inputId: 'movie-select-2-input', dropdownId: 'combobox-2', clearId: '#combobox-2 .combobox-clear' },
        { id: 'movie-select-3', inputId: 'movie-select-3-input', dropdownId: 'combobox-3', clearId: '#combobox-3 .combobox-clear' }
    ];
    
    comboboxConfigs.forEach(config => {
        const hiddenInput = document.getElementById(config.id);
        const textInput = document.getElementById(config.inputId);
        const dropdown = document.querySelector(`#${config.dropdownId} .combobox-dropdown`);
        const clearBtn = document.querySelector(config.clearId);
        const combobox = document.getElementById(config.dropdownId);
        
        let highlightedIndex = -1;
        let isFocused = false;
        
        // Filter options on input
        textInput.addEventListener('input', () => {
            const query = textInput.value.toLowerCase().trim();
            const filtered = query === '' 
                ? sortedMovies 
                : sortedMovies.filter(m => m.title.toLowerCase().includes(query) || m.displayTitle.toLowerCase().includes(query));
            const disabledIds = getDisabledIds(config.id);
            
            renderComboboxOptions(dropdown, filtered, hiddenInput.value, disabledIds);
            highlightedIndex = -1;
            dropdown.hidden = false;
            updateClearButton(clearBtn, hiddenInput.value);
        });
        
        // Show dropdown on focus
        textInput.addEventListener('focus', () => {
            isFocused = true;
            const query = textInput.value.toLowerCase().trim();
            const filtered = query === '' 
                ? sortedMovies 
                : sortedMovies.filter(m => m.title.toLowerCase().includes(query) || m.displayTitle.toLowerCase().includes(query));
            const disabledIds = getDisabledIds(config.id);
            
            renderComboboxOptions(dropdown, filtered, hiddenInput.value, disabledIds);
            highlightedIndex = -1;
            dropdown.hidden = false;
            updateClearButton(clearBtn, hiddenInput.value);
        });
        
        // Hide dropdown on blur (with delay to allow click)
        textInput.addEventListener('blur', () => {
            isFocused = false;
            setTimeout(() => {
                if (!isFocused) {
                    dropdown.hidden = true;
                }
            }, 150);
        });
        
        // Keyboard navigation
        textInput.addEventListener('keydown', (e) => {
            const options = dropdown.querySelectorAll('.combobox-option:not(.disabled)');
            if (options.length === 0) return;
            
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                highlightedIndex = Math.min(highlightedIndex + 1, options.length - 1);
                updateHighlight(options, highlightedIndex);
                scrollIntoView(options[highlightedIndex], dropdown);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                highlightedIndex = Math.max(highlightedIndex - 1, -1);
                updateHighlight(options, highlightedIndex);
                if (highlightedIndex >= 0) scrollIntoView(options[highlightedIndex], dropdown);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (highlightedIndex >= 0 && options[highlightedIndex]) {
                    selectOption(options[highlightedIndex], hiddenInput, textInput, dropdown, clearBtn);
                }
            } else if (e.key === 'Escape') {
                dropdown.hidden = true;
                textInput.blur();
            }
        });
        
        // Click on option
        dropdown.addEventListener('mousedown', (e) => {
            const option = e.target.closest('.combobox-option');
            if (option && !option.classList.contains('disabled')) {
                e.preventDefault(); // Prevent blur from hiding dropdown before click
                selectOption(option, hiddenInput, textInput, dropdown, clearBtn);
            }
        });
        
        // Clear button
        clearBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            clearSelection(hiddenInput, textInput, dropdown, clearBtn);
            updateComboboxAvailability();
        });
        
        // Hidden input change event for cross-combobox constraints
        hiddenInput.addEventListener('change', updateComboboxAvailability);
    });
    
    // Click outside to close all dropdowns
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.combobox')) {
            document.querySelectorAll('.combobox-dropdown').forEach(d => d.hidden = true);
        }
    });
}

function updateHighlight(options, index) {
    options.forEach((opt, i) => {
        opt.classList.toggle('highlighted', i === index);
    });
}

function scrollIntoView(element, container) {
    const rect = element.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    if (rect.bottom > containerRect.bottom || rect.top < containerRect.top) {
        element.scrollIntoView({ block: 'nearest' });
    }
}

function selectOption(option, hiddenInput, textInput, dropdown, clearBtn) {
    const value = option.dataset.value;
    const title = option.textContent;
    
    hiddenInput.value = value;
    textInput.value = title;
    dropdown.hidden = true;
    
    // Trigger change event for cross-combobox constraints
    hiddenInput.dispatchEvent(new Event('change'));
    
    updateClearButton(clearBtn, value);
}

function clearSelection(hiddenInput, textInput, dropdown, clearBtn) {
    hiddenInput.value = '';
    textInput.value = '';
    dropdown.hidden = true;
    hiddenInput.dispatchEvent(new Event('change'));
    updateClearButton(clearBtn, '');
}

function updateClearButton(clearBtn, value) {
    if (value) {
        clearBtn.hidden = false;
    } else {
        clearBtn.hidden = true;
    }
}

// Update which options are disabled across all comboboxes
// Prevents selecting the same movie in multiple comboboxes
function updateComboboxAvailability() {
    const comboboxConfigs = [
        { id: 'movie-select', inputId: 'movie-select-input', dropdownId: 'combobox-1' },
        { id: 'movie-select-2', inputId: 'movie-select-2-input', dropdownId: 'combobox-2' },
        { id: 'movie-select-3', inputId: 'movie-select-3-input', dropdownId: 'combobox-3' }
    ];
    
    // Collect currently selected movie IDs
    const selectedIds = new Set();
    comboboxConfigs.forEach(config => {
        const val = parseInt(document.getElementById(config.id).value);
        if (!isNaN(val)) selectedIds.add(val);
    });
    
    // Update each combobox dropdown if visible: re-render with new disabled state
    comboboxConfigs.forEach(config => {
        const dropdown = document.querySelector(`#${config.dropdownId} .combobox-dropdown`);
        // Only re-render if dropdown is currently visible
        if (!dropdown.hidden) {
            const currentVal = parseInt(document.getElementById(config.id).value);
            const textInput = document.getElementById(config.inputId);
            const query = textInput.value.toLowerCase().trim();
            const filtered = query === '' 
                ? sortedMovies 
                : sortedMovies.filter(m => m.title.toLowerCase().includes(query) || m.displayTitle.toLowerCase().includes(query));
            const disabledIds = getDisabledIds(config.id);
            renderComboboxOptions(dropdown, filtered, currentVal, disabledIds);
        }
    });
}

// Cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Build user profile vector from multiple movie IDs
// Returns average genre vector (18 dims, values 0.0 to 1.0)
function buildUserProfile(movieIds) {
    const selectedMovies = movies.filter(m => movieIds.includes(m.id));
    if (selectedMovies.length === 0) return new Array(18).fill(0);
    
    const profileVector = new Array(18).fill(0);
    for (const m of selectedMovies) {
        for (let i = 0; i < 18; i++) {
            profileVector[i] += m.genreVector[i];
        }
    }
    for (let i = 0; i < 18; i++) {
        profileVector[i] /= selectedMovies.length;
    }
    return profileVector;
}

// Helper to render recommendation list HTML
function renderRecList(recommendations, titlePrefix) {
    return recommendations.map((movie, idx) => {
        const genresStr = movie.genres.join(', ') || '(no genres)';
        return `<div class="rec-item">
            <span class="rec-rank">${idx + 1}.</span>
            <span class="rec-title">${movie.displayTitle}</span>
            <span class="rec-score">score: ${movie.score.toFixed(4)}</span>
            <span class="rec-genres">${genresStr}</span>
            <span class="rec-ratings">${movie.ratingCount} ratings</span>
        </div>`;
    }).join('');
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input - main movie (required)
        const mainSelect = document.getElementById('movie-select');
        const mainMovieId = parseInt(mainSelect.value);
        
        if (isNaN(mainMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Get additional movies (optional)
        const add1Select = document.getElementById('movie-select-2');
        const add2Select = document.getElementById('movie-select-3');
        const add1Id = parseInt(add1Select.value);
        const add2Id = parseInt(add2Select.value);
        const hasAdd1 = !isNaN(add1Id);
        const hasAdd2 = !isNaN(add2Id);
        
        // Step 2: Find the main movie
        const mainMovie = movies.find(movie => movie.id === mainMovieId);
        if (!mainMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // ITEM-TO-ITEM RECOMMENDATIONS (always calculated)
                const likedVector = mainMovie.genreVector;
                const candidateMovies = movies.filter(movie => movie.id !== mainMovieId && movie.isRecommendable);
                
                const scoredMovies = candidateMovies.map(candidate => {
                    const score = cosineSimilarity(likedVector, candidate.genreVector);
                    return { ...candidate, score: score };
                });
                
                scoredMovies.sort((a, b) => b.score - a.score || a.displayTitle.localeCompare(b.displayTitle));
                const itemTop5 = scoredMovies.slice(0, 5);
                
                const mainGenresStr = mainMovie.genres.join(', ') || '(no genres)';
                const itemRecHtml = renderRecList(itemTop5, 'item');
                
                const itemSection = `
                    <div class="rec-header" style="margin-top: 24px; padding-top: 16px; border-top: 2px solid #2c3e50;">
                        <div class="rec-liked-title">Item-to-item: Because you liked <strong>${mainMovie.displayTitle}</strong></div>
                        <div class="rec-liked-genres">Genres: ${mainGenresStr}</div>
                    </div>
                    <div class="rec-list">${itemRecHtml}</div>
                `;
                
                let finalHtml = '';
                
                // PROFILE-BASED RECOMMENDATIONS (only if both additional movies selected)
                if (hasAdd1 && hasAdd2) {
                    const add1Movie = movies.find(m => m.id === add1Id);
                    const add2Movie = movies.find(m => m.id === add2Id);
                    
                    if (add1Movie && add2Movie) {
                        // Build profile from all 3 movies
                        const profileVector = buildUserProfile([mainMovieId, add1Id, add2Id]);
                        
                        // Candidates: exclude all 3 selected movies AND only recommendable
                        const excludedIds = new Set([mainMovieId, add1Id, add2Id]);
                        const profileCandidates = movies.filter(m => !excludedIds.has(m.id) && m.isRecommendable);
                        
                        const profileScored = profileCandidates.map(candidate => {
                            const score = cosineSimilarity(profileVector, candidate.genreVector);
                            return { ...candidate, score: score };
                        });
                        
                        profileScored.sort((a, b) => b.score - a.score || a.displayTitle.localeCompare(b.displayTitle));
                        const profileTop5 = profileScored.slice(0, 5);
                        
                        const profileGenres = [];
                        for (let i = 0; i < 18; i++) {
                            if (profileVector[i] > 0) profileGenres.push(`${genreNames[i]}: ${profileVector[i].toFixed(2)}`);
                        }
                        
                        const profileRecHtml = renderRecList(profileTop5, 'profile');
                        
                        // Profile-based FIRST
                        finalHtml += `
                            <div class="rec-header">
                                <div class="rec-liked-title">Profile-based: Your profile (${mainMovie.displayTitle}, ${add1Movie.displayTitle}, ${add2Movie.displayTitle})</div>
                                <div class="rec-liked-genres">Profile genre weights: ${profileGenres.join(', ')}</div>
                            </div>
                            <div class="rec-list">${profileRecHtml}</div>
                        `;
                        
                        // Item-to-item SECOND
                        finalHtml += itemSection;
                    }
                } else if (hasAdd1 || hasAdd2) {
                    // Only one additional movie selected - hint FIRST
                    finalHtml += `
                        <div class="rec-header" style="padding-top: 0; border-top: none;">
                            <div class="rec-liked-title" style="color: #f39c12;">Profile-based recommendations</div>
                            <div class="rec-liked-genres">Select one more additional movie to build a profile from 3 movies.</div>
                        </div>
                    `;
                    
                    // Item-to-item SECOND
                    finalHtml += itemSection;
                } else {
                    // Only main movie selected - just item-to-item
                    finalHtml = `
                        <div class="rec-header">
                            <div class="rec-liked-title">Item-to-item: Because you liked <strong>${mainMovie.displayTitle}</strong></div>
                            <div class="rec-liked-genres">Genres: ${mainGenresStr}</div>
                        </div>
                        <div class="rec-list">${itemRecHtml}</div>
                    `;
                }
                
                resultElement.innerHTML = finalHtml;
                resultElement.className = 'success';
                
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
