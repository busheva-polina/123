// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Using a smaller subset of data for faster training
const MOVIES_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-public-data/u.item';
const RATINGS_URL = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/recommendation-public-data/u.data';

async function loadData() {
    try {
        console.log('Loading movie data...');
        const moviesResponse = await fetch(MOVIES_URL);
        const moviesText = await moviesResponse.text();
        movies = parseItemData(moviesText);
        
        // Use only first 500 movies for faster training
        movies = movies.slice(0, 500);
        numMovies = movies.length;
        
        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        ratings = parseRatingData(ratingsText);
        
        // Filter ratings to only include our selected movies and first 2000 users
        ratings = ratings.filter(r => r.movieId < numMovies && r.userId < 2000);
        
        // Find the maximum user ID to determine number of users
        const maxUserId = Math.max(...ratings.map(r => r.userId));
        numUsers = maxUserId + 1;
        
        console.log(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        // Fallback: create dummy data for demonstration
        console.log('Using fallback dummy data...');
        createDummyData();
        return { movies, ratings, numUsers, numMovies };
    }
}

function createDummyData() {
    // Create some dummy movies
    movies = [];
    for (let i = 0; i < 100; i++) {
        movies.push({
            id: i,
            title: `Movie ${i + 1}`,
            originalId: i + 1
        });
    }
    numMovies = movies.length;
    
    // Create dummy ratings
    ratings = [];
    numUsers = 50;
    
    for (let userId = 0; userId < numUsers; userId++) {
        for (let movieId = 0; movieId < 20; movieId++) {
            // Create some realistic rating patterns
            const baseRating = 3 + Math.sin(userId * 0.5) * 0.5 + Math.cos(movieId * 0.3) * 0.5;
            const rating = Math.max(1, Math.min(5, Math.round(baseRating + (Math.random() - 0.5) * 2)));
            
            ratings.push({
                userId: userId,
                movieId: movieId,
                rating: rating
            });
        }
    }
    
    console.log(`Dummy data created: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
}

function parseItemData(text) {
    const lines = text.split('\n');
    const movies = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]) - 1;
            const title = parts[1];
            
            movies[movieId] = {
                id: movieId,
                title: title,
                originalId: parseInt(parts[0])
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
