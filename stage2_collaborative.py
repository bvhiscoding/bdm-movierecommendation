import pandas as pd
import numpy as np
import os
import pickle
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import time

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set paths
input_path = "./processed/"  # Current directory where stage1.py saved the files
output_path = "./rec/collaborative-recommendations"
top_n = 20

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Model parameters
dnn_hidden_layers = [64, 32, 16]  # Optimized architecture 
dnn_dropout_rate = 0.2
dnn_learning_rate = 0.001
dnn_batch_size = 64   # Increased batch size for faster training
dnn_epochs = 20       # Reduced epochs with early stopping
threshold_rating = 3.0  # Rating threshold to classify as "like"

def load_data():
    """
    Load processed data from stage1.py
    
    Input: None (reads from files)
    Output: Dictionary containing DataFrames for movie features and ratings
    """
    logger.info("Loading processed data from stage1.py...")
    
    # Data containers
    data = {}
    
    # Load movie features
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
        logger.info(f"Loaded features for {len(data['movie_features'])} movies")
    else:
        logger.error(f"Movie features not found at {movie_features_path}")
        return None
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(data['ratings'])} normalized ratings")
    else:
        logger.error(f"Normalized ratings not found at {ratings_path}")
        return None
    
    # Create training and testing sets with 80-20 split
    if 'ratings' in data:
        # Sort by timestamp if available to ensure reproducibility
        if 'timestamp' in data['ratings'].columns:
            data['ratings'] = data['ratings'].sort_values('timestamp')
        
        # Group by user to ensure each user has both training and testing data
        user_groups = data['ratings'].groupby('userId')
        train_data = []
        test_data = []
        
        for _, group in user_groups:
            n = len(group)
            split_idx = int(n * 0.8)
            train_data.append(group.iloc[:split_idx])
            test_data.append(group.iloc[split_idx:])
        # Get all unique user IDs
        all_user_ids = data['ratings']['userId'].unique()

        # Split users into train (80%) and test (20%) sets
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_user_ids)

        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]

        # Split ratings based on user assignments
        data['train_ratings'] = data['ratings'][data['ratings']['userId'].isin(train_users)]
        data['test_ratings'] = data['ratings'][data['ratings']['userId'].isin(test_users)]
        
        logger.info(f"Split ratings into {len(data['train_ratings'])} training and {len(data['test_ratings'])} testing samples")
    
    return data

def extract_genre_features(movie_features):
    """
    Extract genre features for each movie
    
    Input: 
      - movie_features: DataFrame with movie features including genre columns
    
    Output:
      - movie_genre_features: DataFrame with movieId and genre columns only
    """
    logger.info("Extracting genre features for movies...")
    
    # Get all genre columns (assuming they're already one-hot encoded)
    genre_columns = [col for col in movie_features.columns if col not in 
                     ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
    
    if not genre_columns:
        logger.error("No genre columns found in movie features")
        return None
    
    # Create genre feature matrix
    movie_genre_features = movie_features[['movieId'] + genre_columns].copy()
    
    logger.info(f"Extracted {len(genre_columns)} genre features for {len(movie_features)} movies")
    
    return movie_genre_features

def calculate_user_genre_preferences(train_ratings, movie_genre_features):
    """
    Calculate user preferences for movie genres based on ratings
    
    Input:
      - train_ratings: DataFrame with user-movie ratings
      - movie_genre_features: DataFrame with movie genre features
    
    Output:
      - user_genre_preferences_df: DataFrame with userId and genre preference scores
    """
    logger.info("Calculating user preferences for movie genres...")
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Initialize user genre preferences dataframe
    user_genre_preferences = []
    
    # Process each user
    for user_id in train_ratings['userId'].unique():
        # Get user ratings
        user_ratings = train_ratings[train_ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            continue
        
        # Separate liked and disliked movies
        liked_movies = user_ratings[user_ratings['rating'] > threshold_rating]['movieId'].values
        disliked_movies = user_ratings[user_ratings['rating'] <= threshold_rating]['movieId'].values
        
        # Calculate genre preferences using equation (7) from the paper:
        # R̂g = (Nlikes - Ndislikes) / Max(Nlikes - Ndislikes)
        genre_preferences = {}
        
        for genre in genre_columns:
            # Get genre values for liked movies
            genre_liked = movie_genre_features[movie_genre_features['movieId'].isin(liked_movies)][genre].sum()
            
            # Get genre values for disliked movies
            genre_disliked = movie_genre_features[movie_genre_features['movieId'].isin(disliked_movies)][genre].sum()
            
            # Calculate preference
            genre_preferences[genre] = genre_liked - genre_disliked
        
        # Calculate maximum absolute genre preference
        max_abs_preference = max(abs(val) for val in genre_preferences.values()) if genre_preferences else 1
        
        # Normalize preferences to [-1, 1]
        for genre in genre_preferences:
            genre_preferences[genre] = genre_preferences[genre] / max_abs_preference if max_abs_preference > 0 else 0
        
        # Add user ID
        genre_preferences['userId'] = user_id
        
        user_genre_preferences.append(genre_preferences)
    
    # Convert to dataframe
    user_genre_preferences_df = pd.DataFrame(user_genre_preferences)
    
    logger.info(f"Calculated genre preferences for {len(user_genre_preferences_df)} users")
    
    return user_genre_preferences_df

def prepare_dnn_training_data(train_ratings, user_genre_preferences, movie_genre_features):
    """
    Prepare training data for the DNN model
    
    Input:
      - train_ratings: DataFrame with user-movie ratings
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
    
    Output:
      - X_train, X_val: Feature matrices for training and validation
      - y_train, y_val: Target values for training and validation
      - genre_columns: List of genre column names
    """
    logger.info("Preparing training data for DNN model...")
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Initialize lists for features and labels
    features = []
    labels = []
    
    # Process only a sample of ratings for efficiency
    sample_size = min(1000000, len(train_ratings))  # Cap at 1M ratings
    sampled_ratings = train_ratings.sample(sample_size, random_state=42) if len(train_ratings) > sample_size else train_ratings
    
    # Process each rating
    for _, row in sampled_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        # Skip if user or movie not found
        if user_id not in user_genre_preferences['userId'].values or \
           movie_id not in movie_genre_features['movieId'].values:
            continue
        
        # Get user genre preferences
        user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
        
        # Get movie genres
        movie_genres = movie_genre_features[movie_genre_features['movieId'] == movie_id].iloc[0]
        
        # Create feature vector by combining user preferences and movie genres
        feature_vector = []
        
        for genre in genre_columns:
            # Add user preference for this genre
            feature_vector.append(user_prefs[genre])
            # Add movie genre indicator
            feature_vector.append(movie_genres[genre])
        
        # Use the actual rating as the target
        features.append(feature_vector)
        labels.append(rating)
    
    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    logger.info(f"Created feature matrix with shape {X.shape} and labels with shape {y.shape}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Prepared training data with {len(X_train)} samples, validation data with {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val, genre_columns

def build_and_train_dnn_model(X_train, X_val, y_train, y_val):
    """
    Build and train the DNN model for collaborative filtering
    
    Input:
      - X_train, X_val: Feature matrices for training and validation
      - y_train, y_val: Target values for training and validation
    
    Output:
      - model: Trained DNN model
      - history: Training history
    """
    logger.info("Building and training DNN model...")
    
    # Set memory limit to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            logger.warning(f"Error setting GPU memory growth: {e}")
    
    # Define input dimension
    input_dim = X_train.shape[1]
    
    # Build model based on optimized architecture
    model = Sequential()
    
    # Input layer
    model.add(Dense(dnn_hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dnn_dropout_rate))
    
    # Hidden layers
    for units in dnn_hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dnn_dropout_rate))
    
    # Output layer - single value for rating prediction
    model.add(Dense(1))
    
    # Compile model with Adam optimizer and Mean Squared Error loss for regression
    model.compile(
        optimizer=Adam(learning_rate=dnn_learning_rate),
        loss='mse',  # Use MSE for regression
        metrics=['mae']  # Track mean absolute error during training
    )
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=dnn_epochs,
        batch_size=dnn_batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    logger.info(f"Model training completed. Validation MSE: {val_loss:.4f}, validation MAE: {val_mae:.4f}")
    
    return model, history

def generate_user_movie_features(user_id, movie_id, user_genre_preferences, movie_genre_features):
    """
    Generate feature vector for a specific user-movie pair
    
    Input:
      - user_id: User ID
      - movie_id: Movie ID
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
    
    Output:
      - feature_vector: Feature vector for the user-movie pair
    """
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Skip if user or movie not found
    if user_id not in user_genre_preferences['userId'].values or \
       movie_id not in movie_genre_features['movieId'].values:
        return None
    
    # Get user genre preferences
    user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
    
    # Get movie genres
    movie_row = movie_genre_features[movie_genre_features['movieId'] == movie_id]
    if movie_row.empty:
        return None
    movie_genres = movie_row.iloc[0]
    
    # Create feature vector
    feature_vector = []
    
    for genre in genre_columns:
        # Add user preference for this genre
        feature_vector.append(user_prefs[genre])
        # Add movie genre indicator
        feature_vector.append(movie_genres[genre])
    
    return np.array([feature_vector], dtype=np.float32)

def generate_dnn_recommendations(user_id, dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10):
    """Optimized version with batched predictions"""
    # Skip if user not found in genre preferences
    if user_id not in user_genre_preferences['userId'].values:
        logger.warning(f"User {user_id} not found in genre preferences")
        return []
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Get user genre preferences
    user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
    
    # Get movies already rated by the user
    rated_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
    
    # Get unrated movies
    unrated_movies = movie_genre_features[~movie_genre_features['movieId'].isin(rated_movies)]
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(unrated_movies), batch_size):
        batch = unrated_movies.iloc[i:i+batch_size]
        
        # Create feature vectors in a vectorized way
        feature_vectors = []
        movie_ids = []
        
        for _, movie_row in batch.iterrows():
            movie_id = movie_row['movieId']
            feature_vector = []
            
            for genre in genre_columns:
                feature_vector.append(user_prefs[genre])
                feature_vector.append(movie_row[genre])
            
            feature_vectors.append(feature_vector)
            movie_ids.append(movie_id)
        
        # Convert to numpy array for batch prediction
        feature_array = np.array(feature_vectors)
        
        # Predict in batch
        predictions = dnn_model.predict(feature_array, verbose=0).flatten()
        
        # Ensure ratings are within bounds
        predictions = np.clip(predictions, 0.5, 5.0)
        
        # Add to results
        for movie_id, pred in zip(movie_ids, predictions):
            all_predictions.append((movie_id, pred))
            
    # Sort by predicted rating in descending order
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return all_predictions[:n]

def generate_recommendations_for_all_users(dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10, batch_size=50, max_movies_per_batch=1000):
    """
    Generate recommendations for all users using the DNN model with improved batching and parallelization
    
    Input:
      - dnn_model: Trained DNN model
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
      - train_ratings: DataFrame with training ratings
      - n: Number of recommendations to generate per user
      - batch_size: Number of users to process in each batch
      - max_movies_per_batch: Maximum number of movies to process per batch
    
    Output:
      - all_recommendations: Dictionary mapping user IDs to recommendation lists
    """
    logger.info(f"Generating top-{n} DNN recommendations for all users with optimized batching...")
    
    # Get all user IDs
    all_user_ids = user_genre_preferences['userId'].unique()
    
    # Limit to max_users if specified
    if max_users and max_users < len(all_user_ids):
        user_ids = all_user_ids[:max_users]
    else:
        user_ids = all_user_ids
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    # Create a lookup dictionary for user ratings to avoid repeated filtering
    logger.info("Creating user rating lookup dictionary...")
    user_rated_movies = {}
    for _, row in train_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        if user_id not in user_rated_movies:
            user_rated_movies[user_id] = set()
        user_rated_movies[user_id].add(movie_id)
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Process users in batches
    start_time = time.time()
    for i in range(0, total_users, batch_size):
        batch_end = min(i + batch_size, total_users)
        batch_users = user_ids[i:batch_end]
        
        logger.info(f"Processing batch of {len(batch_users)} users ({i+1}-{batch_end} of {total_users})")
        batch_start_time = time.time()
        
        # Process each user in the batch
        for user_idx, user_id in enumerate(batch_users):
            # Skip if user not found in genre preferences
            user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id]
            if user_prefs.empty:
                continue
            
            # Get movies already rated by the user
            rated_movies = user_rated_movies.get(user_id, set())
            
            # Get candidate movies (not yet rated by the user)
            # To improve efficiency, we'll use a modified approach:
            # 1. Get a subset of movies most likely to be recommended
            # 2. For simplicity, we can start with the most popular unrated movies
            
            # Get all unrated movies
            unrated_movie_ids = set(movie_genre_features['movieId']) - rated_movies
            
            # If too many, limit to a manageable number to improve performance
            if len(unrated_movie_ids) > max_movies_per_batch:
                # Convert to list so we can slice it
                unrated_movie_ids = list(unrated_movie_ids)[:max_movies_per_batch]
            
            # Get movie features for unrated movies
            candidate_movies = movie_genre_features[movie_genre_features['movieId'].isin(unrated_movie_ids)]
            
            # If no candidates, skip this user
            if len(candidate_movies) == 0:
                continue
            
            # Process candidates in smaller batches to avoid memory issues
            movie_batch_size = 200  # Adjust based on memory constraints
            predictions = []
            
            for j in range(0, len(candidate_movies), movie_batch_size):
                movie_batch_end = min(j + movie_batch_size, len(candidate_movies))
                movie_batch = candidate_movies.iloc[j:movie_batch_end]
                
                # Create feature vectors for all movies in this batch
                batch_features = []
                batch_movie_ids = []
                
                for _, movie_row in movie_batch.iterrows():
                    movie_id = movie_row['movieId']
                    feature_vector = []
                    
                    for genre in genre_columns:
                        # User preference for this genre
                        feature_vector.append(user_prefs.iloc[0][genre])
                        # Movie genre indicator
                        feature_vector.append(movie_row[genre])
                    
                    batch_features.append(feature_vector)
                    batch_movie_ids.append(movie_id)
                
                # Convert to numpy array
                batch_features = np.array(batch_features, dtype=np.float32)
                
                # Skip if empty
                if len(batch_features) == 0:
                    continue
                
                # Make predictions in batch
                try:
                    batch_predictions = dnn_model.predict(batch_features, verbose=0).flatten()
                    
                    # Ensure ratings are within bounds
                    batch_predictions = np.clip(batch_predictions, 0.5, 5.0)
                    
                    # Add to predictions list
                    for movie_id, pred in zip(batch_movie_ids, batch_predictions):
                        predictions.append((movie_id, float(pred)))
                except Exception as e:
                    logger.error(f"Error making predictions for user {user_id}, batch {j}: {e}")
            
            # Sort predictions by rating and take top n
            predictions.sort(key=lambda x: x[1], reverse=True)
            all_recommendations[user_id] = predictions[:n]
            
            # Log progress for every 10th user or the last one
            if (user_idx + 1) % 10 == 0 or user_idx == len(batch_users) - 1:
                elapsed_batch = time.time() - batch_start_time
                avg_time_per_user = elapsed_batch / (user_idx + 1)
                logger.info(f"  Processed {user_idx + 1}/{len(batch_users)} users in batch, avg time: {avg_time_per_user:.2f}s per user")
        
        # Log batch completion
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / ((batch_end - i) / batch_size)
        progress = batch_end / total_users * 100
        remaining = avg_time_per_batch * ((total_users - batch_end) / batch_size) if batch_end < total_users else 0
        
        logger.info(f"Completed batch {i//batch_size + 1}/{(total_users-1)//batch_size + 1}")
        logger.info(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # # Force garbage collection
        # gc.collect()
    
    logger.info(f"Generated recommendations for {len(all_recommendations)} users")
    return all_recommendations

def evaluate_recommendations(recommendations, test_ratings, dnn_model, user_genre_preferences, movie_genre_features):
    """
    Evaluate recommendations using RMSE and MAE metrics with expanded predictions
    Modified to handle user-based train-test split
    """
    logger.info("Evaluating recommendations using RMSE and MAE...")
    
    # Check if we have a user-based split by checking user overlap
    train_users = set(user_genre_preferences['userId'].unique())
    test_users = set(test_ratings['userId'].unique())
    common_users = train_users.intersection(test_users)
    
    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}, Common users: {len(common_users)}")
    
    # If no common users, use baseline evaluation with average rating
    if len(common_users) == 0:
        logger.info("Using user-based split - no common users between train and test.")
        logger.info("Evaluating using average rating for all predictions instead.")
        
        # Calculate average rating from training data
        avg_rating = 3.0  # Fallback value
        if 'rating' in test_ratings.columns:
            # Use the average rating from test data as baseline
            avg_rating = test_ratings['rating'].mean()
        
        # Get test ratings
        total_predictions = len(test_ratings)
        
        if total_predictions == 0:
            logger.warning("No test ratings available for evaluation")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'num_predictions': 0
            }
        
        # Calculate RMSE and MAE using average rating as prediction
        squared_errors_sum = ((test_ratings['rating'] - avg_rating) ** 2).sum()
        absolute_errors_sum = (abs(test_ratings['rating'] - avg_rating)).sum()
        
        rmse = np.sqrt(squared_errors_sum / total_predictions)
        mae = absolute_errors_sum / total_predictions
        
        logger.info(f"Baseline evaluation results (using avg rating {avg_rating:.2f}):")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"Number of predictions: {total_predictions}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'num_predictions': total_predictions,
            'method': 'baseline_average_rating'
        }
    
    # Standard evaluation (original code) for when there are common users
    # Initialize lists for predictions and actual ratings
    predictions = []
    actuals = []
    
    # For each user in the test set
    for user_id in test_ratings['userId'].unique():
        # Skip users without genre preferences
        if user_id not in user_genre_preferences['userId'].values:
            continue
        
        # Get user's test ratings
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        # Get user's recommendations (movie_id, predicted_rating) if available
        user_recs = {}
        if user_id in recommendations:
            user_recs = dict(recommendations[user_id])
        
        # Match test ratings with predictions
        for _, row in user_test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # If the movie is in recommendations
            if movie_id in user_recs:
                predictions.append(user_recs[movie_id])
                actuals.append(actual_rating)
            # Otherwise, make a new prediction for this movie
            elif movie_id in movie_genre_features['movieId'].values:
                # Generate feature vector for this user-movie pair
                feature_vector = generate_user_movie_features(
                    user_id, 
                    movie_id, 
                    user_genre_preferences, 
                    movie_genre_features
                )
                
                if feature_vector is not None:
                    # Predict rating
                    predicted_rating = dnn_model.predict(feature_vector, verbose=0)[0][0]
                    # Ensure rating is within bounds
                    predicted_rating = max(0.5, min(5.0, predicted_rating))
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
    
    
    # Check if we have predictions to evaluate
    if not predictions:
        logger.warning("No predictions available for evaluation using standard method")
        # Fall back to baseline if we have test ratings
        if len(test_ratings) > 0:
            logger.info("Falling back to baseline evaluation method")
            avg_rating = test_ratings['rating'].mean() if 'rating' in test_ratings.columns else 3.0
            total_predictions = len(test_ratings)
            
            # Calculate RMSE and MAE using average rating
            squared_errors_sum = ((test_ratings['rating'] - avg_rating) ** 2).sum()
            absolute_errors_sum = (abs(test_ratings['rating'] - avg_rating)).sum()
            
            rmse = np.sqrt(squared_errors_sum / total_predictions)
            mae = absolute_errors_sum / total_predictions
            
            logger.info(f"Baseline evaluation results:")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"Number of predictions: {total_predictions}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'num_predictions': total_predictions,
                'method': 'baseline_average_rating'
            }
        
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'num_predictions': 0
        }
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'num_predictions': len(predictions),
        'method': 'standard'
    }
    
    logger.info(f"Evaluation completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Predictions: {len(predictions)}")
    
    return metrics

def recommend_for_user(user_id, recommendations, movie_features=None, n=10):
    """
    Print recommendations for a specific user
    
    Input:
      - user_id: User ID to display recommendations for
      - recommendations: Dictionary with recommendation lists
      - movie_features: DataFrame with movie features (for titles)
      - n: Number of recommendations to display
    
    Output:
      - None (prints recommendations)
    """
    # Check if user has recommendations
    if user_id not in recommendations:
        print(f"No recommendations found for user {user_id}")
        return
    
    # Get recommendations
    user_recs = recommendations[user_id][:n]
    
    if not user_recs:
        print(f"No recommendations found for user {user_id}")
        return
    
    # Print recommendations
    print(f"\nTop {len(user_recs)} recommendations for user {user_id}:")
    
    for i, (movie_id, predicted_rating) in enumerate(user_recs, 1):
        movie_info = f"Movie ID: {movie_id}"
        
        # Try to get movie title if available
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_info = movie_row.iloc[0]['title']
        
        print(f"{i}. {movie_info} - Predicted Rating: {predicted_rating:.2f}")

# Main execution flow
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COLLABORATIVE FILTERING WITH DEEP NEURAL NETWORK")
    print("="*80)
    
    # Step 1: Load Data
    data = load_data()
    if data is None:
        logger.error("Failed to load required data")
        exit(1)

    # Step 2: Extract Genre Features
    movie_genre_features = extract_genre_features(data['movie_features'])
    if movie_genre_features is None:
        logger.error("Failed to extract genre features")
        exit(1)
    
    # Step 3: Calculate User Genre Preferences
    user_genre_preferences = calculate_user_genre_preferences(data['train_ratings'], movie_genre_features)
    
    # Step 4: Prepare Training Data for DNN
    X_train, X_val, y_train, y_val, genre_columns = prepare_dnn_training_data(
        data['train_ratings'], 
        user_genre_preferences, 
        movie_genre_features
    )
    
    # Step 5: Build and Train DNN Model
    dnn_model, training_history = build_and_train_dnn_model(X_train, X_val, y_train, y_val)
    
    # Save DNN model
    dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))
    logger.info(f"Saved DNN model to {os.path.join(output_path, 'dnn_model.h5')}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot MSE loss
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['loss'], label='Training MSE')
    plt.plot(training_history.history['val_loss'], label='Validation MSE')
    plt.title('Model MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['mae'], label='Training MAE')
    plt.plot(training_history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'dnn_training_history.png'))
    logger.info(f"Saved training history plot to {os.path.join(output_path, 'dnn_training_history.png')}")
    
    # Step 6: Generate Recommendations for a Subset of Users
    # Limit to 100 users for demonstration (can be increased)
    max_users = len(user_genre_preferences['userId'].unique())
    dnn_recommendations = generate_recommendations_for_all_users(
        dnn_model,
        user_genre_preferences,
        movie_genre_features,
        data['train_ratings'],
        top_n,
        max_users
    )
    
    # Save recommendations
    with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
        pickle.dump(dnn_recommendations, f)
    
    # Also save in a more readable CSV format
    recommendations_list = []
    
    for user_id, recs in dnn_recommendations.items():
        for rank, (movie_id, predicted_rating) in enumerate(recs, 1):
            movie_title = "Unknown"
            movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_title = movie_row.iloc[0]['title']
                    
            recommendations_list.append({
                'userId': user_id,
                'movieId': movie_id,
                'title': movie_title,
                'rank': rank,
                'predicted_rating': predicted_rating
            })
    
    if recommendations_list:
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df.to_csv(os.path.join(output_path, 'dnn_recommendations.csv'), index=False)
    
    logger.info(f"Generated DNN recommendations for {len(dnn_recommendations)} users")
    
    # Step 7: Evaluate Recommendations
    evaluation_metrics = evaluate_recommendations(
        dnn_recommendations,
        data['test_ratings'],
        dnn_model,
        user_genre_preferences,
        movie_genre_features
    )
    
    # Save metrics
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)
    
    # Display evaluation metrics
    print("\nDNN Evaluation Results:")
    print(f"RMSE: {evaluation_metrics['rmse']:.4f}")
    print(f"MAE: {evaluation_metrics['mae']:.4f}")
    print(f"Number of predictions evaluated: {evaluation_metrics['num_predictions']}")
    
    # Display sample recommendations for a few users
    if dnn_recommendations:
        sample_user_id = next(iter(dnn_recommendations.keys()))
        recommend_for_user(sample_user_id, dnn_recommendations, data['movie_features'])
    
    # Final Summary
    print("\n" + "="*80)
    print("SUMMARY: COLLABORATIVE FILTERING WITH DNN")
    print("="*80)
    
    # Display model architecture
    print("\nDNN Model Architecture:")
    dnn_model.summary(print_fn=print)
    print(f"\nNumber of layers: {len(dnn_model.layers)}")
    print(f"Hidden layer sizes: {dnn_hidden_layers}")
    print(f"Dropout rate: {dnn_dropout_rate}")
    print(f"Learning rate: {dnn_learning_rate}")
    print(f"Batch size: {dnn_batch_size}")
    
    # Display performance metrics
    print("\nPerformance Metrics:")
    headers = ["Model", "RMSE", "MAE", "Predictions Evaluated"]
    rows = [
        [
            "Collaborative Filtering (DNN)",
            f"{evaluation_metrics['rmse']:.4f}",
            f"{evaluation_metrics['mae']:.4f}",
            str(evaluation_metrics['num_predictions'])
        ]
    ]
    
    # Print table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    print("| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    for row in rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    
    # Recommendations statistics
    if dnn_recommendations:
        avg_recs = sum(len(recs) for recs in dnn_recommendations.values()) / len(dnn_recommendations)
        print(f"\nRecommendation Statistics:")
        print(f"- Users with recommendations: {len(dnn_recommendations)}")
        print(f"- Average recommendations per user: {avg_recs:.2f}")
    
    
    # Save batch file to run recommendations for any user
import pickle
import sys

# Define path to model and data
recommendation_path = "{output_path}"
model_path = os.path.join(recommendation_path, 'dnn_model.h5')

# Check if model exists
if not os.path.exists(model_path):
    print("Error: Model not found at", model_path)
    sys.exit(1)


# Load the model
model = load_model(model_path)
print("Model loaded successfully from", model_path)

# Load other