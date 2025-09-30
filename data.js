// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Sample data for demonstration
const SAMPLE_MOVIES = [
    { id: 1, title: "Toy Story (1995)" },
    { id: 2, title: "GoldenEye (1995)" },
    { id: 3, title: "Four Rooms (1995)" },
    { id: 4, title: "Get Shorty (1995)" },
    { id: 5, title: "Copycat (1995)" },
    { id: 6, title: "Shanghai Triad (1995)" },
    { id: 7, title: "Twelve Monkeys (1995)" },
    { id: 8, title: "Babe (1995)" },
    { id: 9, title: "Dead Man Walking (1995)" },
    { id: 10, title: "Richard III (1995)" }
];

const SAMPLE_RATINGS = [
    { userId: 1, movieId: 1, rating: 5 }, { userId: 1, movieId: 2, rating: 3 }, { userId: 1, movieId: 3, rating: 4 },
    { userId: 2, movieId: 1, rating: 4 }, { userId: 2, movieId: 4, rating: 5 }, { userId: 2, movieId: 5, rating: 3 },
    { userId: 3, movieId: 2, rating: 4 }, { userId: 3, movieId: 6, rating: 5 }, { userId: 3, movieId: 7, rating: 4 },
    { userId: 4, movieId: 3, rating: 3 }, { userId: 4, movieId: 8, rating: 5 }, { userId: 4, movieId: 9, rating: 4 },
    { userId: 5, movieId: 4, rating: 4 }, { userId: 5, movieId: 10, rating: 3 }, { userId: 5, movieId: 1, rating: 5 },
    { userId: 6, movieId: 5, rating: 3 }, { userId: 6, movieId: 2, rating: 4 }, { userId: 6, movieId: 6, rating: 5 },
    { userId: 7, movieId: 6, rating: 4 }, { userId: 7, movieId: 7, rating: 3 }, { userId: 7, movieId: 8, rating: 5 },
    { userId: 8, movieId: 7, rating: 4 }, { userId: 8, movieId: 9, rating: 5 }, { userId: 8, movieId: 10, rating: 3 },
    { userId: 9, movieId: 8, rating: 5 }, { userId: 9, movieId: 1, rating: 4 }, { userId: 9, movieId: 2, rating: 3 },
    { userId: 10, movieId: 9, rating: 4 }, { userId: 10, movieId: 3, rating: 5 }, { userId: 10, movieId: 4, rating: 4 }
];

async function loadData() {
    console.log('Loading data...');
    
    try {
        // Try to load from external URLs first
        const moviesResponse = await fetch('https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/uci-iris-mlens-u.item');
        const ratingsResponse = await fetch('https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/multivariate-linear-regression/data/uci-iris-mlens-u.data');
        
        if (moviesResponse.ok && ratingsResponse.ok) {
            const moviesText = await moviesResponse.text();
            const ratingsText = await ratingsResponse.text();
            
            movies = parseItemData(moviesText);
            ratings = parseRatingData(ratingsText);
        } else {
            throw new Error('External data not available, using sample data');
        }
    } catch (error) {
        console.log('Using sample data:', error.message);
        // Use sample data
        movies = SAMPLE_MOVIES;
        ratings = SAMPLE_RATINGS;
    }

    // Calculate number of unique users and movies
    const uniqueUsers = new Set(ratings.map(r => r.userId));
    const uniqueMovies = new Set(ratings.map(r => r.movieId));
    
    numUsers = Math.max(...uniqueUsers) + 1; // +1 because IDs are 1-indexed
    numMovies = Math.max(...uniqueMovies) + 1;
    
    console.log(`Data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
    
    return { movies, ratings, numUsers, numMovies };
}

function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const movies = [];
    
    for (const line of lines) {
        const parts = line.split('|');
        if (parts.length >= 2) {
            const id = parseInt(parts[0]);
            const title = parts[1];
            if (!isNaN(id) && title) {
                movies.push({ id, title });
            }
        }
    }
    
    return movies;
}

function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const ratings = [];
    
    for (const line of lines) {
        const parts = line.split('\t');
        if (parts.length >= 3) {
            const userId = parseInt(parts[0]);
            const movieId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            
            if (!isNaN(userId) && !isNaN(movieId) && !isNaN(rating)) {
                ratings.push({ userId, movieId, rating });
            }
        }
    }
    
    return ratings;
}
