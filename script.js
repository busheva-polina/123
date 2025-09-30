// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
    try {
        updateStatus('Loading dataset...');
        
        // Load data and wait for it to complete
        const data = await loadData();
        
        // Set global variables explicitly
        window.movies = data.movies;
        window.ratings = data.ratings;
        window.numUsers = data.numUsers;
        window.numMovies = data.numMovies;
        
        console.log('Data loaded:', {
            users: window.numUsers,
            movies: window.numMovies,
            ratings: window.ratings.length
        });
        
        // Validate data
        if (window.numUsers <= 0 || window.numMovies <= 0 || window.ratings.length === 0) {
            throw new Error('Invalid data loaded');
        }
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Use the global numUsers variable
    for (let i = 0; i < window.numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i + 1}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Use the global movies variable
    window.movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 8) { // Reduced to 8 for even faster training
    console.log(`Creating model with: ${numUsers} users, ${numMovies} movies`);
    
    // Input layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding Layers
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Flatten embeddings
    const userFlat = tf.layers.flatten().apply(userEmbedding);
    const movieFlat = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product for prediction
    const dot = tf.layers.dot({axes: 1}).apply([userFlat, movieFlat]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dot,
        name: 'SimpleMatrixFactorization'
    });
    
    return model;
}

async function trainModel() {
    try {
        isTraining = true;
        updateStatus('Creating model...');
        
        // Use global variables
        const numUsers = window.numUsers;
        const numMovies = window.numMovies;
        const ratings = window.ratings;
        
        console.log('Training with:', {numUsers, numMovies, ratingsCount: ratings.length});
        
        // Create model
        model = createModel(numUsers, numMovies, 8);
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.1), // Higher learning rate
            loss: 'meanSquaredError'
        });
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => r.movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        updateStatus('Training model (fast - 3 epochs)...');
        
        // Train for only 3 epochs
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 3,
            batchSize: 32,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const currentEpoch = epoch + 1;
                    updateStatus(`Training ${currentEpoch}/3 - Loss: ${logs.loss.toFixed(4)}`);
                    
                    // Enable predictions after first epoch
                    if (epoch === 0) {
                        document.getElementById('predict-btn').disabled = false;
                        updateStatus(`Training ${currentEpoch}/3 - Loss: ${logs.loss.toFixed(4)} - Ready for predictions!`);
                    }
                }
            }
        });
        
        // Clean up
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        isTraining = false;
        updateStatus('Training completed! Select user and movie to predict ratings.');
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Training failed: ' + error.message);
        isTraining = false;
    }
}

async function predictRating() {
    if (!model || isTraining) {
        updateResult('Model not ready. Please wait...');
        return;
    }
    
    try {
        const userSelect = document.getElementById('user-select');
        const movieSelect = document.getElementById('movie-select');
        
        const userId = parseInt(userSelect.value);
        const movieId = parseInt(movieSelect.value);
        
        if (isNaN(userId) || isNaN(movieId)) {
            updateResult('Please select both user and movie.');
            return;
        }
        
        const selectedMovie = window.movies.find(m => m.id === movieId);
        
        // Create tensors for prediction
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Predict
        const prediction = model.predict([userTensor, movieTensor]);
        const ratingValue = await prediction.data();
        let predictedRating = ratingValue[0];
        
        // Scale and clamp the rating (since our model outputs dot product)
        predictedRating = (predictedRating + 3) / 2; // Simple scaling
        predictedRating = Math.max(1, Math.min(5, predictedRating));
        
        // Clean up
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        displayRatingResult(predictedRating, selectedMovie.title);
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Prediction error: ' + error.message);
    }
}

function displayRatingResult(rating, movieTitle) {
    const resultDiv = document.getElementById('result');
    
    const clampedRating = Math.max(0, Math.min(5, rating));
    const percentage = (clampedRating / 5) * 100;
    
    let ratingText = '';
    if (clampedRating >= 4.0) ratingText = 'Excellent!';
    else if (clampedRating >= 3.0) ratingText = 'Good';
    else if (clampedRating >= 2.0) ratingText = 'Fair';
    else ratingText = 'Poor';
    
    resultDiv.innerHTML = `
        <div class="rating-display" style="color: ${getRatingColor(clampedRating)};">
            ${clampedRating.toFixed(1)}
        </div>
        <div class="rating-bar">
            <div class="rating-fill" style="width: ${percentage}%"></div>
        </div>
        <div class="rating-text">
            Predicted rating for <strong>"${movieTitle}"</strong><br>
            ${ratingText} • ${clampedRating.toFixed(1)} out of 5 stars
        </div>
    `;
}

function getRatingColor(rating) {
    if (rating >= 4.0) return '#2ecc71';
    if (rating >= 3.0) return '#f39c12';
    if (rating >= 2.0) return '#e67e22';
    return '#e74c3c';
}

function updateStatus(message) {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<span class="loading"></span>${message}`;
}

function updateResult(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<p>${message}</p>`;
}
