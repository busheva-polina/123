// Global variables to store parsed data and dimensions
let movies = [];
let ratings = [];
let numUsers = 50; // Fixed number for demo
let numMovies = 100; // Fixed number for demo

/**
 * Loads mock data instantly - no network requests
 */
async function loadData() {
    try {
        console.log('Loading mock data...');
        
        // Generate mock movies instantly
        movies = generateMockMovies(100);
        
        // Generate mock ratings instantly
        ratings = generateMockRatings(50, 100, 1000); // 50 users, 100 movies, 1000 ratings
        
        console.log(`Mock data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading mock data:', error);
        throw error;
    }
}

/**
 * Generates mock movie data
 */
function generateMockMovies(count) {
    const movieTitles = [
        "The Matrix", "Inception", "Pulp Fiction", "The Godfather", "Forrest Gump",
        "The Dark Knight", "Fight Club", "Goodfellas", "The Shawshank Redemption", "Star Wars",
        "Avatar", "Titanic", "Jurassic Park", "The Avengers", "Black Panther",
        "Interstellar", "The Departed", "Gladiator", "The Prestige", "Django Unchained"
    ];
    
    const genres = ["Action", "Drama", "Comedy", "Sci-Fi", "Thriller", "Romance", "Horror", "Adventure"];
    
    const movies = [];
    for (let i = 1; i <= count; i++) {
        const randomTitle = movieTitles[Math.floor(Math.random() * movieTitles.length)];
        const randomGenre = genres[Math.floor(Math.random() * genres.length)];
        const year = 1980 + Math.floor(Math.random() * 40);
        
        movies.push({
            id: i,
            title: `${randomTitle} ${i}`,
            releaseDate: `01-Jan-${year}`,
            genre: randomGenre
        });
    }
    return movies;
}

/**
 * Generates mock rating data
 */
function generateMockRatings(userCount, movieCount, ratingCount) {
    const ratings = [];
    
    for (let i = 0; i < ratingCount; i++) {
        const userId = Math.floor(Math.random() * userCount) + 1;
        const movieId = Math.floor(Math.random() * movieCount) + 1;
        // Generate ratings with some distribution (more 3-5 ratings)
        const rating = Math.random() < 0.7 ? 
            (3 + Math.random() * 2).toFixed(1) : // 70% between 3-5
            (1 + Math.random() * 2).toFixed(1);  // 30% between 1-3
        
        ratings.push({
            userId: userId,
            movieId: movieId,
            rating: parseFloat(rating),
            timestamp: Date.now()
        });
    }
    return ratings;
}

/**
 * Mock parsing functions (keep interface same)
 */
function parseItemData(text) {
    return generateMockMovies(100);
}

function parseRatingData(text) {
    return generateMockRatings(50, 100, 1000);
}
