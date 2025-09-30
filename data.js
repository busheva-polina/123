// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// MovieLens dataset URLs
const MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-public-data/u.item';
const RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-public-data/u.data';

async function loadData() {
    try {
        console.log('Loading movie data...');
        const moviesResponse = await fetch(MOVIES_URL);
        const moviesText = await moviesResponse.text();
        movies = parseItemData(moviesText);
        numMovies = movies.length;
        
        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        ratings = parseRatingData(ratingsText);
        
        // Find the maximum user ID to determine number of users
        const maxUserId = Math.max(...ratings.map(r => r.userId));
        numUsers = maxUserId + 1; // User IDs are 0-indexed in the model
        
        console.log(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    const lines = text.split('\n');
    const movies = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]) - 1; // Convert to 0-based index
            const title = parts[1];
            
            movies[movieId] = {
                id: movieId,
                title: title,
                originalId: parseInt(parts[0]) // Keep original ID for reference
            };
        }
    }
    
    return movies.filter(movie => movie !== undefined);
}

function parseRatingData(text) {
    const lines = text.split('\n');
    const ratings = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('\t');
        if (parts.length >= 3) {
            // Convert to 0-based indices for TensorFlow.js
            const userId = parseInt(parts[0]) - 1;
            const movieId = parseInt(parts[1]) - 1;
            const rating = parseFloat(parts[2]);
            
            ratings.push({
                userId: userId,
                movieId: movieId,
                rating: rating
            });
        }
    }
    
    return ratings;
}
