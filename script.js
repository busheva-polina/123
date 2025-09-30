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
        
        if (movies.length === 0) {
            throw new Error('No movies loaded from URL');
        }
        
        // Use only first 200 movies for faster training (increased from 0 for safety)
        const maxMovies = Math.min(200, movies.length);
        movies = movies.slice(0, maxMovies);
        numMovies = movies.length;
        
        console.log('Loading rating data...');
        const ratingsResponse = await fetch(RATINGS_URL);
        const ratingsText = await ratingsResponse.text();
        let allRatings = parseRatingData(ratingsText);
        
        if (allRatings.length === 0) {
            throw new Error('No ratings loaded from URL');
        }
        
        // Filter ratings to only include our selected movies and first 500 users
        ratings = allRatings.filter(r => {
            return r.movieId < numMovies && r.userId < 500 && 
                   !isNaN(r.userId) && !isNaN(r.movieId) && !isNaN(r.rating);
        });
        
        if (ratings.length === 0) {
            throw new Error('No valid ratings after filtering');
        }
        
        // Find the maximum user ID to determine number of users
        const maxUserId = Math.max(...ratings.map(r => r.userId));
        numUsers = maxUserId + 1; // User IDs are 0-indexed
        
        console.log(`Data loaded successfully: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
        
    } catch (error) {
        console.error('Error loading data from URL:', error);
        console.log('Using fallback dummy data...');
        return createDummyData();
    }
}

function createDummyData() {
    console.log('Creating realistic dummy data...');
    
    // Create realistic dummy movies
    movies = [
        { id: 0, title: "The Shawshank Redemption", originalId: 1 },
        { id: 1, title: "The Godfather", originalId: 2 },
        { id: 2, title: "The Dark Knight", originalId: 3 },
        { id: 3, title: "Pulp Fiction", originalId: 4 },
        { id: 4, title: "Forrest Gump", originalId: 5 },
        { id: 5, title: "Inception", originalId: 6 },
        { id: 6, title: "The Matrix", originalId: 7 },
        { id: 7, title: "Goodfellas", originalId: 8 },
        { id: 8, title: "The Silence of the Lambs", originalId: 9 },
        { id: 9, title: "Star Wars: A New Hope", originalId: 10 },
        { id: 10, title: "The Lord of the Rings: Fellowship", originalId: 11 },
        { id: 11, title: "Fight Club", originalId: 12 },
        { id: 12, title: "The Avengers", originalId: 13 },
        { id: 13, title: "The Social Network", originalId: 14 },
        { id: 14, title: "The Lion King", originalId: 15 }
    ];
    numMovies = movies.length;
    
    // Create realistic dummy ratings with patterns
    ratings = [];
    numUsers = 50;
    
    // Create user preferences (action lovers, drama lovers, etc.)
    const userPreferences = [];
    for (let userId = 0; userId < numUsers; userId++) {
        userPreferences[userId] = {
            action: Math.random() * 2 - 1, // -1 to 1
            drama: Math.random() * 2 - 1,
            comedy: Math.random() * 2 - 1,
            scifi: Math.random() * 2 - 1
        };
    }
    
    // Movie genres (simplified)
    const movieGenres = [
        { action: 0.9, drama: 0.8, comedy: 0.1, scifi: 0.2 }, // Shawshank
        { action: 0.7, drama: 0.9, comedy: 0.3, scifi: 0.1 }, // Godfather
        { action: 0.95, drama: 0.6, comedy: 0.2, scifi: 0.3 }, // Dark Knight
        { action: 0.8, drama: 0.7, comedy: 0.8, scifi: 0.1 }, // Pulp Fiction
        { action: 0.3, drama: 0.9, comedy: 0.7, scifi: 0.1 }, // Forrest Gump
        { action: 0.8, drama: 0.6, comedy: 0.3, scifi: 0.95 }, // Inception
        { action: 0.9, drama: 0.5, comedy: 0.2, scifi: 0.95 }, // Matrix
        { action: 0.7, drama: 0.8, comedy: 0.4, scifi: 0.1 }, // Goodfellas
        { action: 0.6, drama: 0.9, comedy: 0.2, scifi: 0.1 }, // Silence Lambs
        { action: 0.8, drama: 0.5, comedy: 0.4, scifi: 0.9 }, // Star Wars
        { action: 0.85, drama: 0.7, comedy: 0.3, scifi: 0.8 }, // Lord of Rings
        { action: 0.7, drama: 0.6, comedy: 0.5, scifi: 0.1 }, // Fight Club
        { action: 0.95, drama: 0.4, comedy: 0.6, scifi: 0.9 }, // Avengers
        { action: 0.2, drama: 0.8, comedy: 0.5, scifi: 0.1 }, // Social Network
        { action: 0.3, drama: 0.7, comedy: 0.8, scifi: 0.2 }  // Lion King
    ];
    
    // Generate ratings based on user preferences and movie genres
    for (let userId = 0; userId < numUsers; userId++) {
        for (let movieId = 0; movieId < numMovies; movieId++) {
            // Calculate affinity score based on dot product of preferences and genres
            const prefs = userPreferences[userId];
            const genres = movieGenres[movieId];
            let affinity = 0;
            
            affinity += prefs.action * genres.action;
            affinity += prefs.drama * genres.drama;
            affinity += prefs.comedy * genres.comedy;
            affinity += prefs.scifi * genres.scifi;
            
            // Convert to rating (1-5 scale)
            let rating = 3 + affinity * 2; // Center at 3, scale by 2
            
            // Add some randomness
            rating += (Math.random() - 0.5) * 1.5;
            
            // Clamp to 1-5 range and round
            rating = Math.max(1, Math.min(5, rating));
            rating = Math.round(rating * 2) / 2; // Round to nearest 0.5
            
            // Only include some ratings (sparse matrix simulation)
            if (Math.random() > 0.3) { // 70% density
                ratings.push({
                    userId: userId,
                    movieId: movieId,
                    rating: rating
                });
            }
        }
    }
    
    console.log(`Dummy data created: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
    
    return { movies, ratings, numUsers, numMovies };
}

function parseItemData(text) {
    const lines = text.split('\n');
    const movies = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]);
            if (isNaN(movieId)) continue;
            
            const title = parts[1];
            const zeroBasedId = movieId - 1; // Convert to 0-based index
            
            movies[zeroBasedId] = {
                id: zeroBasedId,
                title: title,
                originalId: movieId
            };
        }
    }
    
    // Remove undefined entries and return
    return movies.filter(movie => movie !== undefined);
}

function parseRatingData(text) {
    const lines = text.split('\n');
    const ratings = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('\t');
        if (parts.length >= 3) {
            const userId = parseInt(parts[0]);
            const movieId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            
            // Validate all values
            if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating) && 
                userId > 0 && movieId > 0 && rating >= 1 && rating <= 5) {
                
                ratings.push({
                    userId: userId - 1, // Convert to 0-based
                    movieId: movieId - 1, // Convert to 0-based
                    rating: rating
                });
            }
        }
    }
    
    return ratings;
}
