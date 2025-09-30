// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
    try {
        // Update status
        updateStatus('Loading dataset (optimized for speed)...');
        
        // Load data
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training with smaller model
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error initializing application: ' + error.message);
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Add users (limited for demo)
    for (let i = 0; i < Math.min(50, numUsers); i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i + 1}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Add movies (limited for faster loading)
    movies.slice(0, 100).forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title.length > 50 ? movie.title.substring(0, 50) + '...' : movie.title;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 16) { // Reduced from 64 to 16
    console.log(`Creating optimized model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding Layers - smaller dimensions for speed
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
    
    // Latent Vectors
    const userLatent = tf.layers.flatten().apply(userEmbedding);
    const movieLatent = tf.layers.flatten().apply(movieEmbedding);
    
    // Prediction - simplified architecture
    const dotProduct = tf.layers.dot({axes: 1}).apply([userLatent, movieLatent]);
    
    // Simple bias terms
    const userBias = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: 1,
        name: 'userBias'
    }).apply(userInput);
    
    const movieBias = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: 1,
        name: 'movieBias'
    }).apply(movieInput);
    
    const flattenedUserBias = tf.layers.flatten().apply(userBias);
    const flattenedMovieBias = tf.layers.flatten().apply(movieBias);
    
    // Combine and scale to 1-5 range
    const combined = tf.layers.add().apply([dotProduct, flattenedUserBias, flattenedMovieBias]);
    
    // Simple scaling to rating range (1-5)
    const ratingPrediction = tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'ones',
        biasInitializer: 'zeros'
    }).apply(combined);

    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: ratingPrediction,
        name: 'FastMatrixFactorization'
    });
    
    return model;
}

async function trainModel() {
    try {
        isTraining = true;
        updateStatus('Creating optimized model for fast training...');
        
        // Create smaller model
        model = createModel(numUsers, numMovies, 16); // Small latent dimension
        
        // Compile with higher learning rate for faster convergence
        model.compile({
            optimizer: tf.train.adam(0.01), // Increased learning rate
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Use smaller subset of data for training
        const trainingRatings = ratings.slice(0, 2000); // Use only 2000 ratings
        
        const userIds = trainingRatings.map(r => r.userId);
        const movieIds = trainingRatings.map(r => r.movieId);
        const ratingValues = trainingRatings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        // Train for fewer epochs with larger batches
        updateStatus('Fast training in progress...');
        
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 4, // Reduced from 8 to 4
            batchSize: 256, // Larger batches
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    updateStatus(`Fast training ${epoch + 1}/4 - loss: ${logs.loss.toFixed(4)}`);
                    
                    // Enable prediction button after first epoch for immediate use
                    if (epoch === 0) {
                        document.getElementById('predict-btn').disabled = false;
                        updateStatus(`Fast training ${epoch + 1}/4 - loss: ${logs.loss.toFixed(4)} - Predictions available!`);
                    }
                }
            }
        });
        
        // Clean up
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        document.getElementById('predict-btn').disabled = false;
        isTraining = false;
        
        updateStatus('Fast training completed! Ready for predictions.');
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message);
        isTraining = false;
    }
}

async function predictRating() {
    if (!model) {
        updateResult('Model is not ready yet. Please wait...');
        return;
    }
    
    try {
        const userSelect = document.getElementById('user-select');
        const movieSelect = document.getElementById('movie-select');
        
        const userId = parseInt(userSelect.value);
        const movieId = parseInt(movieSelect.value);
        
        if (isNaN(userId) || isNaN(movieId)) {
            updateResult('Please select both a user and a movie.');
            return;
        }
        
        const selectedMovie = movies.find(m => m.id === movieId);
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]]);
        const movieTensor = tf.tensor2d([[movieId]]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        let predictedRating = rating[0];
        
        // Ensure rating is in reasonable range
        predictedRating = Math.max(0.5, Math.min(5, predictedRating));
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result
        displayRatingResult(predictedRating, selectedMovie.title);
        
    } catch (error) {
        console.error('Prediction error:', error);
        updateResult('Error making prediction: ' + error.message);
    }
}

function displayRatingResult(rating, movieTitle) {
    const resultDiv = document.getElementById('result');
    
    const clampedRating = Math.max(0, Math.min(5, rating));
    const percentage = (clampedRating / 5) * 100;
    
    let ratingText = '';
    if (clampedRating >= 4.5) ratingText = 'Excellent!';
    else if (clampedRating >= 4.0) ratingText = 'Very Good';
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
