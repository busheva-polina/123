// Global variables
let model;
let isTraining = false;

// Initialize when window loads
window.onload = async function() {
    console.log('Initializing application...');
    updateTrainingStatus('Loading data...', 'loading');
    
    try {
        // Load data
        await loadData();
        updateTrainingStatus('Data loaded successfully!', 'success');
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Start training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateTrainingStatus('Error during initialization: ' + error.message, 'error');
    }
};

// Update training status in UI
function updateTrainingStatus(message, type = 'loading') {
    const statusElement = document.getElementById('training-status');
    statusElement.textContent = message;
    statusElement.className = type;
}

// Populate user dropdown
function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';
    
    // Create options for users
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'Select a user';
    userSelect.appendChild(option);
    
    for (let i = 1; i < numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

// Populate movie dropdown
function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '';
    
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'Select a movie';
    movieSelect.appendChild(option);
    
    // Use available movies from data
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

// Model Definition Function - Fixed version without lambda layers
function createModel(numUsers, numMovies, latentDim = 8) {
    console.log(`Creating model with ${numUsers} users, ${numMovies} movies, latent dimension: ${latentDim}`);
    
    // Input Layers
    const userInput = tf.input({ shape: [1], name: 'userInput' });
    const movieInput = tf.input({ shape: [1], name: 'movieInput' });
    
    // Embedding Layers for latent factors
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
    
    // Embedding Layers for bias terms
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
    
    // Flatten all embeddings
    const userVector = tf.layers.flatten().apply(userEmbedding);
    const movieVector = tf.layers.flatten().apply(movieEmbedding);
    const userBiasFlat = tf.layers.flatten().apply(userBias);
    const movieBiasFlat = tf.layers.flatten().apply(movieBias);
    
    // Dot product of user and movie vectors
    const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
    
    // Global bias (average rating)
    const globalBias = tf.layers.dense({
        units: 1,
        useBias: true,
        biasInitializer: 'zeros',
        trainable: true,
        name: 'globalBias'
    }).apply(tf.layers.flatten().apply(dotProduct));
    
    // Combine all components: dotProduct + userBias + movieBias + globalBias
    const combined = tf.layers.add().apply([dotProduct, userBiasFlat, movieBiasFlat, globalBias]);
    
    // Apply sigmoid activation and scale to 1-5 range
    const scaledOutput = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: 'zeros',
        biasInitializer: 'zeros'
    }).apply(combined);
    
    // Scale from sigmoid range (0-1) to rating range (1-5)
    const finalOutput = tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: tf.initializers.constant({ value: 4.0 }),
        biasInitializer: tf.initializers.constant({ value: 1.0 }),
        trainable: false  // Fixed scaling, not trainable
    }).apply(scaledOutput);
    
    // Create model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: finalOutput,
        name: 'MatrixFactorization'
    });
    
    return model;
}

// Training Function
async function trainModel() {
    if (isTraining) {
        console.log('Model is already training...');
        return;
    }
    
    isTraining = true;
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    
    try {
        updateTrainingStatus('Creating model architecture...', 'loading');
        
        // Create model
        model = createModel(numUsers, numMovies, 6); // Smaller latent dim for faster training
        
        updateTrainingStatus('Compiling model...', 'loading');
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Prepare training data
        updateTrainingStatus('Preparing training data...', 'loading');
        
        const userIDs = ratings.map(r => r.userId);
        const movieIDs = ratings.map(r => r.movieId);
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor2d(userIDs, [userIDs.length, 1]);
        const movieTensor = tf.tensor2d(movieIDs, [movieIDs.length, 1]);
        const ratingTensor = tf.tensor2d(ratingValues, [ratingValues.length, 1]);
        
        updateTrainingStatus('Starting training... (5 epochs, fast training)', 'loading');
        
        // Train model with minimal epochs for speed
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: 5,
            batchSize: 32,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const status = `Epoch ${epoch + 1}/5 - Loss: ${logs.loss.toFixed(4)}`;
                    updateTrainingStatus(status, 'loading');
                    console.log(status);
                }
            }
        });
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, ratingTensor]);
        
        updateTrainingStatus('✅ Model training completed! Ready for predictions.', 'success');
        predictBtn.disabled = false;
        
    } catch (error) {
        console.error('Training error:', error);
        updateTrainingStatus('❌ Error during training: ' + error.message, 'error');
    } finally {
        isTraining = false;
    }
}

// Prediction Function
async function predictRating() {
    const userSelect = document.getElementById('user-select');
    const movieSelect = document.getElementById('movie-select');
    const resultElement = document.getElementById('result');
    
    const userId = parseInt(userSelect.value);
    const movieId = parseInt(movieSelect.value);
    
    if (!userId || !movieId) {
        resultElement.innerHTML = '<p style="color: #f56565;">Please select both a user and a movie.</p>';
        return;
    }
    
    if (!model) {
        resultElement.innerHTML = '<p style="color: #f56565;">Model is not ready yet. Please wait for training to complete.</p>';
        return;
    }
    
    try {
        resultElement.innerHTML = '<p>Calculating prediction...</p>';
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userId]], [1, 1]);
        const movieTensor = tf.tensor2d([[movieId]], [1, 1]);
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const ratingArray = await prediction.data();
        let predictedRating = ratingArray[0];
        
        // Ensure rating is between 1-5
        predictedRating = Math.max(1, Math.min(5, predictedRating));
        
        // Clean up tensors
        tf.dispose([userTensor, movieTensor, prediction]);
        
        // Display result with stars
        const starCount = Math.round(predictedRating);
        const stars = '★'.repeat(starCount) + '☆'.repeat(5 - starCount);
        const movieTitle = movieSelect.options[movieSelect.selectedIndex].text;
        const userName = userSelect.options[userSelect.selectedIndex].text;
        
        resultElement.innerHTML = `
            <div class="rating-display">${predictedRating.toFixed(1)}</div>
            <div class="rating-stars">${stars}</div>
            <p style="margin-top: 10px; color: #4a5568;">
                Predicted rating for <strong>${movieTitle}</strong> 
                by <strong>${userName}</strong>
            </p>
        `;
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultElement.innerHTML = `<p style="color: #f56565;">Error making prediction: ${error.message}</p>`;
    }
}
