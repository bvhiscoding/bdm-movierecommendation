import numpy as np
import pandas as pd
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

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set paths
input_path = "./"  # Current directory where stage1.py saved the files
output_path = "./recommendations"
top_n = 10

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Model parameters
dnn_hidden_layers = [32, 18, 9]  # Based on the paper's DNN architecture 
dnn_dropout_rate = 0.2
dnn_learning_rate = 0.001
dnn_batch_size = 32
dnn_epochs = 50
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
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(data['ratings'])} normalized ratings")
    else:
        logger.error(f"Normalized ratings not found at {ratings_path}")
    
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
        
        data['train_ratings'] = pd.concat(train_data).reset_index(drop=True)
        data['test_ratings'] = pd.concat(test_data).reset_index(drop=True)
        
        logger.info(f"Split ratings into {len(data['train_ratings'])} training and {len(data['test_ratings'])} testing samples")
    
    # Try to load content-based recommendations if they exist
    content_recs_path = os.path.join(output_path, 'content_based_recommendations.pkl')
    if os.path.exists(content_recs_path):
        with open(content_recs_path, 'rb') as f:
            data['content_based_recommendations'] = pickle.load(f)
        logger.info(f"Loaded content-based recommendations for {len(data['content_based_recommendations'])} users")
    
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
            liked_genre = movie_genre_features[movie_genre_features['movieId'].isin(liked_movies)][genre].sum()
            
            # Get genre values for disliked movies
            disliked_genre = movie_genre_features[movie_genre_features['movieId'].isin(disliked_movies)][genre].sum()
            
            # Calculate preference
            genre_preferences[genre] = liked_genre - disliked_genre
        
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
    
    # Process each rating
    for _, row in train_ratings.iterrows():
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
            feature_vector.append(user_prefs[genre])
            feature_vector.append(movie_genres[genre])
        
        # Use the actual rating as the target
        features.append(feature_vector)
        labels.append(rating)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
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
    
    # Define input dimension
    input_dim = X_train.shape[1]
    
    # Build model based on paper's architecture
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

def evaluate_model_on_test_set(dnn_model, user_genre_preferences, movie_genre_features, test_ratings):
    """
    Directly evaluate the DNN model on the test set ratings
    
    Input:
      - dnn_model: Trained DNN model
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
      - test_ratings: DataFrame with test ratings
    
    Output:
      - metrics: Dictionary with evaluation metrics (rmse, mae)
    """
    logger.info("Directly evaluating DNN model on test set...")
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Initialize lists for predictions and actual ratings
    predictions = []
    actuals = []
    
    # For each rating in the test set
    for _, row in test_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        # Skip if user or movie not found
        if user_id not in user_genre_preferences['userId'].values or \
           movie_id not in movie_genre_features['movieId'].values:
            continue
        
        # Get user genre preferences
        user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
        
        # Get movie genres
        movie_row = movie_genre_features[movie_genre_features['movieId'] == movie_id]
        if movie_row.empty:
            continue
        movie_genres = movie_row.iloc[0]
        
        # Create feature vector
        feature_vector = []
        
        for genre in genre_columns:
            feature_vector.append(user_prefs[genre])
            feature_vector.append(movie_genres[genre])
        
        # Reshape for prediction
        feature_vector = np.array([feature_vector])
        
        # Predict movie rating
        predicted_rating = dnn_model.predict(feature_vector, verbose=0)[0][0]
        
        # Ensure rating is within bounds
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        predictions.append(predicted_rating)
        actuals.append(actual_rating)
    
    # Check if we have predictions to evaluate
    if not predictions:
        logger.warning("No predictions to evaluate")
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
        'num_predictions': len(predictions)
    }
    
    logger.info(f"Direct evaluation completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Predictions: {len(predictions)}")
    
    return metrics

def generate_dnn_recommendations(user_id, dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10):
    """
    Generate recommendations for a user using the DNN model
    
    Input:
      - user_id: User ID to generate recommendations for
      - dnn_model: Trained DNN model
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
      - train_ratings: DataFrame with training ratings
      - n: Number of recommendations to generate
    
    Output:
      - movie_predictions: List of (movie_id, predicted_rating) tuples
    """
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
    
    # Initialize movie predictions
    movie_predictions = []
    
    # Process each movie
    for _, movie_row in movie_genre_features.iterrows():
        movie_id = movie_row['movieId']
        
        # Skip if movie already rated
        if movie_id in rated_movies:
            continue
        
        # Create feature vector
        feature_vector = []
        
        for genre in genre_columns:
            feature_vector.append(user_prefs[genre])
            feature_vector.append(movie_row[genre])
        
        # Reshape for prediction
        feature_vector = np.array([feature_vector])
        
        # Predict movie rating
        predicted_rating = dnn_model.predict(feature_vector, verbose=0)[0][0]
        
        # Ensure rating is within bounds
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        movie_predictions.append((movie_id, predicted_rating))
    
    # Sort by predicted rating in descending order
    movie_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return movie_predictions[:n]

def generate_recommendations_for_all_users_dnn(dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10):
    """
    Generate recommendations for all users using the DNN model
    
    Input:
      - dnn_model: Trained DNN model
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
      - train_ratings: DataFrame with training ratings
      - n: Number of recommendations to generate per user
    
    Output:
      - all_recommendations: Dictionary mapping user IDs to recommendation lists
    """
    logger.info(f"Generating top-{n} DNN recommendations for all users...")
    
    # Get all user IDs
    user_ids = user_genre_preferences['userId'].unique()
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    # Process each user
    for i, user_id in enumerate(user_ids):
        recommendations = generate_dnn_recommendations(
            user_id, 
            dnn_model, 
            user_genre_preferences, 
            movie_genre_features, 
            train_ratings, 
            n
        )
        
        if recommendations:
            all_recommendations[user_id] = recommendations
        
        # Log progress
        if (i+1) % 100 == 0 or (i+1) == total_users:
            logger.info(f"Generated recommendations for {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%)")
    
    return all_recommendations

def evaluate_recommendations_rmse_mae(recommendations, test_ratings):
    """
    Evaluate recommendations using RMSE and MAE metrics
    
    Input:
      - recommendations: Dictionary mapping user IDs to recommendation lists
      - test_ratings: DataFrame with test ratings
    
    Output:
      - metrics: Dictionary with evaluation metrics (rmse, mae)
    """
    logger.info("Evaluating recommendations using RMSE and MAE...")
    
    # Initialize lists for predictions and actual ratings
    predictions = []
    actuals = []
    
    # For each user in the test set
    for user_id in test_ratings['userId'].unique():
        # Skip users without recommendations
        if user_id not in recommendations:
            continue
        
        # Get user's test ratings
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        # Get user's recommendations (movie_id, predicted_rating)
        user_recs = dict(recommendations[user_id])
        
        # Match test ratings with predictions
        for _, row in user_test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # If the movie is in recommendations
            if movie_id in user_recs:
                predictions.append(user_recs[movie_id])
                actuals.append(actual_rating)
    
    # Check if we have predictions to evaluate
    if not predictions:
        logger.warning("No predictions to evaluate")
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
        'num_predictions': len(predictions)
    }
    
    logger.info(f"Evaluation completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Predictions: {len(predictions)}")
    
    return metrics

def combine_recommendations(content_based_recommendations, dnn_recommendations, alpha=0.5):
    """
    Combine content-based and DNN recommendations with weighted approach
    
    Input:
      - content_based_recommendations: Dictionary with content-based recommendations
      - dnn_recommendations: Dictionary with DNN recommendations
      - alpha: Weight for content-based recommendations (1-alpha for DNN)
    
    Output:
      - combined_recommendations: Dictionary with combined recommendations
    """
    logger.info("Combining content-based and DNN recommendations...")
    
    combined_recommendations = {}
    
    # Get all users from both recommendation sets
    all_users = set(content_based_recommendations.keys()).union(set(dnn_recommendations.keys()))
    
    for user_id in all_users:
        # Initialize combined recommendations dictionary for this user
        user_combined_recs = {}
        
        # Add content-based recommendations if available
        if user_id in content_based_recommendations:
            for movie_id, score in content_based_recommendations[user_id]:
                # Convert similarity score to rating scale (0.5-5.0)
                # Assuming score is in [0,1] range
                rating = 0.5 + 4.5 * score
                user_combined_recs[movie_id] = alpha * rating
        
        # Add DNN recommendations if available
        if user_id in dnn_recommendations:
            for movie_id, rating in dnn_recommendations[user_id]:
                if movie_id in user_combined_recs:
                    user_combined_recs[movie_id] += (1 - alpha) * rating
                else:
                    user_combined_recs[movie_id] = (1 - alpha) * rating
        
        # Sort and convert to list of tuples
        sorted_recs = sorted(user_combined_recs.items(), key=lambda x: x[1], reverse=True)
        
        combined_recommendations[user_id] = sorted_recs[:top_n]
    
    return combined_recommendations

def recommend_for_user_combined(user_id, combined_recommendations, movie_features=None, n=10):
    """
    Generate and print combined recommendations for a specific user
    
    Input:
      - user_id: User ID to generate recommendations for
      - combined_recommendations: Dictionary with combined recommendations
      - movie_features: DataFrame with movie features
      - n: Number of recommendations to display
    
    Output:
      - recommendations: List of recommendations for the user
    """
    # Check if user has recommendations
    if user_id not in combined_recommendations:
        print(f"No combined recommendations found for user {user_id}")
        return None
    
    # Get recommendations
    recommendations = combined_recommendations[user_id][:n]
    
    if not recommendations:
        print(f"No combined recommendations found for user {user_id}")
        return None
    
    # Print recommendations
    print(f"\nTop {len(recommendations)} combined recommendations for user {user_id}:")
    
    for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
        movie_info = f"Movie ID: {movie_id}"
        
        # Try to get movie title if available
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_info = movie_row.iloc[0]['title']
        
        print(f"{i}. {movie_info} - Predicted Rating: {predicted_rating:.2f}")
    
    return recommendations

# Main execution flow
if __name__ == "__main__":
    # Step 1: Load Data
    data = load_data()
    # Output: data dictionary with movie_features, ratings, train_ratings, test_ratings

    # Step 2: Extract Genre Features
    if 'movie_features' in data:
        movie_genre_features = extract_genre_features(data['movie_features'])
        if movie_genre_features is not None:
            data['movie_genre_features'] = movie_genre_features
            # Output: movie_genre_features DataFrame with movieId and genre columns
            
            # Save genre features
            movie_genre_features.to_csv(os.path.join(output_path, 'movie_genre_features.csv'), index=False)
            logger.info(f"Saved genre features to {os.path.join(output_path, 'movie_genre_features.csv')}")

    # Step 3: Calculate User Genre Preferences
    if 'train_ratings' in data and 'movie_genre_features' in data:
        user_genre_preferences = calculate_user_genre_preferences(data['train_ratings'], data['movie_genre_features'])
        data['user_genre_preferences'] = user_genre_preferences
        # Output: user_genre_preferences DataFrame with userId and genre preference scores
        
        # Save user genre preferences
        user_genre_preferences.to_csv(os.path.join(output_path, 'user_genre_preferences.csv'), index=False)
        logger.info(f"Saved user genre preferences to {os.path.join(output_path, 'user_genre_preferences.csv')}")

    # Step 4: Prepare Training Data for DNN
    if all(key in data for key in ['train_ratings', 'user_genre_preferences', 'movie_genre_features']):
        X_train, X_val, y_train, y_val, genre_columns = prepare_dnn_training_data(
            data['train_ratings'], 
            data['user_genre_preferences'], 
            data['movie_genre_features']
        )
        # Output: X_train, X_val (feature matrices), y_train, y_val (target values)

    # Step 5: Build and Train DNN Model
    if 'X_train' in locals() and 'X_val' in locals() and 'y_train' in locals() and 'y_val' in locals():
        dnn_model, training_history = build_and_train_dnn_model(X_train, X_val, y_train, y_val)
        data['dnn_model'] = dnn_model
        data['dnn_training_history'] = training_history
        # Output: dnn_model (trained model), training_history (training metrics)
        
        # Save DNN model
        dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))
        
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
        
        logger.info(f"Saved DNN model to {os.path.join(output_path, 'dnn_model.h5')}")
        logger.info(f"Saved training history plot to {os.path.join(output_path, 'dnn_training_history.png')}")

    # Step 6: Evaluate DNN Model on Test Set
    if all(key in data for key in ['dnn_model', 'user_genre_preferences', 'movie_genre_features', 'test_ratings']):
        dnn_direct_metrics = evaluate_model_on_test_set(
            data['dnn_model'],
            data['user_genre_preferences'],
            data['movie_genre_features'],
            data['test_ratings']
        )
        data['dnn_direct_metrics'] = dnn_direct_metrics
        # Output: dnn_direct_metrics (dictionary with rmse, mae)
        
        # Save direct evaluation metrics
        direct_evaluation_df = pd.DataFrame([dnn_direct_metrics])
        direct_evaluation_df.to_csv(os.path.join(output_path, 'dnn_direct_evaluation.csv'), index=False)
        
        # Display direct evaluation metrics
        print("\nDNN Direct Evaluation Results:")
        print(f"RMSE: {dnn_direct_metrics['rmse']:.4f}")
        print(f"MAE: {dnn_direct_metrics['mae']:.4f}")
        print(f"Number of predictions evaluated: {dnn_direct_metrics['num_predictions']}")

    # Step 7: Generate DNN Recommendations for All Users
    if all(key in data for key in ['dnn_model', 'user_genre_preferences', 'movie_genre_features', 'train_ratings']):
        dnn_recommendations = generate_recommendations_for_all_users_dnn(
            data['dnn_model'],
            data['user_genre_preferences'],
            data['movie_genre_features'],
            data['train_ratings'],
            top_n
        )
        data['dnn_recommendations'] = dnn_recommendations
        # Output: dnn_recommendations (dictionary mapping user IDs to recommendation lists)
        
        # Save recommendations
        with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
            pickle.dump(dnn_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in dnn_recommendations.items():
            for rank, (movie_id, predicted_rating) in enumerate(recs, 1):
                movie_title = "Unknown"
                if 'movie_features' in data:
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

    # Step 8: Evaluate DNN Recommendations
    if 'dnn_recommendations' in data and 'test_ratings' in data:
        dnn_evaluation_metrics = evaluate_recommendations_rmse_mae(
            data['dnn_recommendations'],
            data['test_ratings']
        )
        data['dnn_evaluation_metrics'] = dnn_evaluation_metrics
        # Output: dnn_evaluation_metrics (dictionary with rmse, mae)
        
        # Save metrics
        evaluation_results = pd.DataFrame([dnn_evaluation_metrics])
        evaluation_results.to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)
        
        # Display evaluation metrics
        print("\nDNN Evaluation Results:")
        print(f"RMSE: {dnn_evaluation_metrics['rmse']:.4f}")
        print(f"MAE: {dnn_evaluation_metrics['mae']:.4f}")
        print(f"Number of predictions evaluated: {dnn_evaluation_metrics['num_predictions']}")

    # Step 9: Combine with Content-Based Recommendations (if available)
    if 'content_based_recommendations' in data and 'dnn_recommendations' in data:
        # Try different alpha values to find optimal weight
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_alpha = 0.5  # Default
        best_rmse = float('inf')
        
        alpha_results = []
        
        for alpha in alpha_values:
            combined_recs = combine_recommendations(
                data['content_based_recommendations'],
                data['dnn_recommendations'],
                alpha
            )
            # Output for each alpha: combined_recs (dictionary with combined recommendations)
            
            # Evaluate combined recommendations
            combined_metrics = evaluate_recommendations_rmse_mae(
                combined_recs,
                data['test_ratings']
            )
            
            alpha_results.append({
                'alpha': alpha,
                'rmse': combined_metrics['rmse'],
                'mae': combined_metrics['mae']
            })
            
            # Update best alpha if needed
            if combined_metrics['rmse'] < best_rmse:
                best_rmse = combined_metrics['rmse']
                best_alpha = alpha
        
        # Save alpha comparison results
        alpha_df = pd.DataFrame(alpha_results)
        alpha_df.to_csv(os.path.join(output_path, 'alpha_comparison.csv'), index=False)
        # Output: alpha_df (DataFrame with alpha comparison results)
        
        # Plot alpha comparison
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_df['alpha'], alpha_df['rmse'], 'o-', label='RMSE')
        plt.plot(alpha_df['alpha'], alpha_df['mae'], 's-', label='MAE')
        plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha}')
        plt.xlabel('Alpha (Weight of Content-Based Recommendations)')
        plt.ylabel('Error Metric Value')
        plt.title('Effect of Alpha on Recommendation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'alpha_comparison.png'))
        
        # Generate final combined recommendations with best alpha
        final_combined_recs = combine_recommendations(
            data['content_based_recommendations'],
            data['dnn_recommendations'],
            best_alpha
        )
        data['combined_recommendations'] = final_combined_recs
        # Output: final_combined_recs (dictionary with final combined recommendations)
        
        # Save combined recommendations
        with open(os.path.join(output_path, 'combined_recommendations.pkl'), 'wb') as f:
            pickle.dump(final_combined_recs, f)
        
        # Also save in a more readable CSV format
        combined_recommendations_list = []
        
        for user_id, recs in final_combined_recs.items():
            for rank, (movie_id, predicted_rating) in enumerate(recs, 1):
                movie_title = "Unknown"
                if 'movie_features' in data:
                    movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
                    if not movie_row.empty and 'title' in movie_row.columns:
                        movie_title = movie_row.iloc[0]['title']
                        
                combined_recommendations_list.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'title': movie_title,
                    'rank': rank,
                    'predicted_rating': predicted_rating
                })
        
        if combined_recommendations_list:
            combined_recommendations_df = pd.DataFrame(combined_recommendations_list)
            combined_recommendations_df.to_csv(os.path.join(output_path, 'combined_recommendations.csv'), index=False)
        # Output: combined_recommendations_df (DataFrame with combined recommendations)
        
        # Evaluate final combined recommendations
        combined_evaluation_metrics = evaluate_recommendations_rmse_mae(
            final_combined_recs,
            data['test_ratings']
        )
        data['combined_evaluation_metrics'] = combined_evaluation_metrics
        # Output: combined_evaluation_metrics (dictionary with evaluation metrics)
        
        # Save metrics
        combined_evaluation_df = pd.DataFrame([combined_evaluation_metrics])
        combined_evaluation_df.to_csv(os.path.join(output_path, 'combined_evaluation.csv'), index=False)
        
        logger.info(f"Generated combined recommendations with alpha={best_alpha} for {len(final_combined_recs)} users")
        logger.info(f"Combined recommendations - RMSE: {combined_evaluation_metrics['rmse']:.4f}, MAE: {combined_evaluation_metrics['mae']:.4f}")

    # Step 10: Display Sample Recommendations
    if 'combined_recommendations' in data and 'movie_features' in data:
        print("\n" + "="*80)
        print("COMBINED MODEL RECOMMENDATIONS EXAMPLE")
        print("="*80)
        
        # Find a user with combined recommendations
        if data['combined_recommendations']:
            sample_user_id = next(iter(data['combined_recommendations'].keys()))
            
            print(f"\nGenerating sample combined recommendations for User ID {sample_user_id}:")
            
            # Get user ratings for context
            if 'train_ratings' in data:
                user_ratings = data['train_ratings'][data['train_ratings']['userId'] == sample_user_id]
                
                print(f"\nThis user has rated {len(user_ratings)} movies.")
                if len(user_ratings) > 0:
                    print("Sample of their highest-rated movies:")
                    
                    # Get top 5 highest-rated movies
                    top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
                    
                    for _, row in top_rated.iterrows():
                        movie_info = f"Movie ID: {row['movieId']}"
                        if 'movie_features' in data:
                            movie_row = data['movie_features'][data['movie_features']['movieId'] == row['movieId']]
                            if not movie_row.empty and 'title' in movie_row.columns:
                                movie_info = movie_row.iloc[0]['title']
                        print(f"  {movie_info} - Rating: {row['rating']}")
            
            # Generate combined recommendations
            recommend_for_user_combined(
                sample_user_id, 
                data['combined_recommendations'], 
                data.get('movie_features'),
                n=10
            )
    
    # Final Summary
    print("\n" + "="*80)
    print("SUMMARY: MEMORY-BASED COLLABORATIVE FILTERING WITH DNN")
    print("="*80)
    
    # Display model architecture
    print("\nDNN Model Architecture:")
    if 'dnn_model' in data:
        data['dnn_model'].summary(print_fn=print)
        print(f"\nNumber of layers: {len(data['dnn_model'].layers)}")
        print(f"Hidden layer sizes: {dnn_hidden_layers}")
        print(f"Dropout rate: {dnn_dropout_rate}")
        print(f"Learning rate: {dnn_learning_rate}")
        print(f"Batch size: {dnn_batch_size}")
    
    # User genre preferences
    if 'user_genre_preferences' in data:
        print(f"\nCalculated genre preferences for {len(data['user_genre_preferences'])} users")
        genre_columns = [col for col in data['user_genre_preferences'].columns if col != 'userId']
        print(f"Genre features: {len(genre_columns)} dimensions")
    
    # Display performance metrics
    print("\nPerformance Metrics Comparison:")
    headers = ["Model", "RMSE", "MAE", "Predictions Evaluated"]
    rows = []
    
    # Content-based model metrics (if available)
    if 'content_based_evaluation' in data:
        rows.append([
            "Content-Based (Log-Likelihood + Word2Vec)",
            f"{data['content_based_evaluation']['rmse']:.4f}",
            f"{data['content_based_evaluation']['mae']:.4f}",
            str(data['content_based_evaluation']['num_predictions'])
        ])
    
    # DNN model metrics
    if 'dnn_direct_metrics' in data:
        rows.append([
            "Memory-Based (DNN)",
            f"{data['dnn_direct_metrics']['rmse']:.4f}",
            f"{data['dnn_direct_metrics']['mae']:.4f}",
            str(data['dnn_direct_metrics']['num_predictions'])
        ])
    
    # Combined model metrics
    if 'combined_evaluation_metrics' in data:
        rows.append([
            f"Hybrid (α={best_alpha})",
            f"{data['combined_evaluation_metrics']['rmse']:.4f}",
            f"{data['combined_evaluation_metrics']['mae']:.4f}",
            str(data['combined_evaluation_metrics']['num_predictions'])
        ])
    
    # Print table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    print("| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    for row in rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
    print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    
    # Recommendations statistics
    print("\nRecommendation Statistics:")
    if 'dnn_recommendations' in data:
        avg_dnn_recs = sum(len(recs) for recs in data['dnn_recommendations'].values()) / len(data['dnn_recommendations'])
        print(f"- Average DNN recommendations per user: {avg_dnn_recs:.2f}")
    if 'combined_recommendations' in data:
        avg_combined_recs = sum(len(recs) for recs in data['combined_recommendations'].values()) / len(data['combined_recommendations'])
        print(f"- Average combined recommendations per user: {avg_combined_recs:.2f}")
    
    # Model advantages
    print("\nAdvantages of DNN-based Collaborative Filtering:")
    print("- Captures non-linear relationships between user preferences and movie genres")
    print("- Automatically extracts complex patterns from rating data")
    print("- Effectively addresses the cold-start problem for new users")
    print("- Can model implicit feedback and preference strength")
    print("- High scalability with large datasets")
    
    # Saved files
    print("\nFiles Generated:")
    for file in os.listdir(output_path):
        if file.startswith('dnn_') or file.startswith('combined_'):
            print(f"- {file}")
    
    print("\nMemory-Based Collaborative Filtering with DNN Successfully Implemented!")