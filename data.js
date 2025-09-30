// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

async function loadData() {
    try {
        console.log('Creating optimized dummy data for fast training...');
        return createDummyData();
        
    } catch (error) {
        console.error('Error creating data:', error);
        return createDummyData(); // Fallback to dummy data
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
        { id: 9, title: "Star Wars: A New Hope", originalId: 10 }
    ];
    numMovies = movies.length;
    
    // Create realistic dummy ratings
    ratings = [];
    numUsers = 30; // Fixed number of users
    
    // Generate ratings
    for (let userId = 0; userId < numUsers; userId++) {
        for (let movieId = 0; movieId < numMovies; movieId++) {
            // Base rating with some patterns
            let baseRating = 3.0;
            
            // User preferences simulation
            if (userId % 3 === 0) baseRating += 0.5; // Some users rate higher
            if (userId % 5 === 0) baseRating -= 0.5; // Some users rate lower
            
            // Movie quality simulation
            if (movieId === 0 || movieId === 1) baseRating += 1.0; // Popular movies
            if (movieId === 9) baseRating += 0.5; // Star Wars
            
            // Add some randomness
            const randomVariation = (Math.random() - 0.5) * 1.5;
            let rating = baseRating + randomVariation;
            
            // Clamp to 1-5 range
            rating = Math.max(1.0, Math.min(5.0, rating));
            
            // Only include 60% of possible ratings (sparse matrix)
            if (Math.random() < 0.6) {
                ratings.push({
                    userId: userId,
                    movieId: movieId,
                    rating: rating
                });
            }
        }
    }
    
    console.log(`Dummy data created: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
    
    // Set global variables explicitly
    window.movies = movies;
    window.ratings = ratings;
    window.numUsers = numUsers;
    window.numMovies = numMovies;
    
    return { movies, ratings, numUsers, numMovies };
}
