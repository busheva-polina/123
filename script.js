function createModel(numUsers, numMovies, latentDim = 8) {
    console.log(`Creating SIMPLE model with: ${numUsers} users, ${numMovies} movies`);
    
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
    
    // Simple output - we'll handle scaling in post-processing
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dot,
        name: 'SimpleMatrixFactorization'
    });
    
    return model;
}

// Update the predictRating function for the simple model:
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
        
        // Scale the raw dot product to 1-5 range
        // Dot products can be negative, so we scale appropriately
        predictedRating = (predictedRating * 0.5) + 3.0; // Adjust these values as needed
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
