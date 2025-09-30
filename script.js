// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
    try {
        // Update status
        updateStatus('Loading MovieLens dataset...');
        
        // Load data
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error initializing application: ' + error.message);
    }
};

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Add some sample users (first 100 users for demo)
    for (let i = 0; i < Math.min(100, numUsers); i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i + 1}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    // Add all movies
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

function createModel(numUsers, numMovies, latentDim = 64) {
    console.log(`Creating model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input layers
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding Layers:
    // Create embedding layers that map user/movie IDs to latent vectors
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
    
    // Latent Vectors: 
    // Reshape embeddings to remove the sequence dimension
    const userLatent = tf.layers.flatten().apply(userEmbedding);
    const movieLatent = tf.layers.flatten().apply(movieEmbedding);
    
    // Prediction: 
    // Dot product of user and movie latent vectors
    const dotProduct = tf.layers.dot({axes: 1}).apply([userLatent, movieLatent]);
    
    // Add bias terms
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
    
    // Combine dot product with biases
    const prediction = tf.layers.add().apply([
        dotProduct, 
        flattenedUserBias, 
        flattenedMovieBias
    ]);
    
    // Scale output to rating range (1-5)
    const scaledPrediction = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: 'zeros',
        biasInitializer: tf.initializers.constant({value: 3.0})
    }).apply(prediction);
    
    const finalPrediction = tf.layers.multiply()
        .apply([scaledPrediction, tf.layers.constant({value: tf.scalar(4)})]);
    const ratingPrediction = tf.layers.add()
        .apply([finalPrediction, tf.layers.constant({value: tf.scalar(1)})]);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: ratingPrediction,
        name: 'MatrixFactorization'
    });
    
    return model;
}

async function trainModel() {
    try {
        isTraining = true;
        updateStatus('Creating and training model...');
        
        // Create model with latent dimension of 64
        model = createModel(numUsers, numMovies, 64);
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Prepare training data
        const userIds = ratings.map(r => r.userId);
        const movieIds = ratings.map(r => movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        // Train model
        updateStatus('Training model (this may take a few minutes)...');
        
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 2,
            batchSize: 128,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
                    updateStatus(`Training epoch ${epoch + 1}/2 - loss: ${logs.loss.toFixed(4)}`);
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        // Enable prediction button
        document.getElementById('predict-btn').disabled = false;
        isTraining = false;
        
        updateStatus('Model training completed! Select a user and movie to predict ratings.');
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error training model: ' + error.message);
        isTraining = false;
    }
}

async function predictRating() {
    if (!model || isTraining) {
        updateResult('Model is not ready yet. Please wait for training to complete.');
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
        const predictedRating = rating[0];
        
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
    
    // Clamp rating between 0 and 5
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
    if (rating >= 4.0) return '#2ecc71'; // Green
    if (rating >= 3.0) return '#f39c12'; // Orange
    if (rating >= 2.0) return '#e67e22'; // Dark orange
    return '#e74c3c'; // Red
}

function updateStatus(message) {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<span class="loading"></span>${message}`;
}

function updateResult(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<p>${message}</p>`;
}
