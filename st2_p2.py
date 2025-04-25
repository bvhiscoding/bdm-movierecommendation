import pandas as pd
import numpy as np
import os
import pickle
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import time
import gc

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "./processed/"
output_path = "./rec/collaborative-recommendations"
top_n = 20

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Improved hyperparameters for better performance
dnn_hidden_layers = [128, 64, 32]  # Larger network
dnn_dropout_rate = 0.3  # Increased dropout for better regularization
dnn_l2_reg = 0.001  # L2 regularization to prevent overfitting
dnn_learning_rate = 0.001  # Lower learning rate for more stable training
dnn_batch_size = 128  # Larger batch size
dnn_epochs = 50  # More epochs with early stopping
threshold_rating = 3  # Rating threshold for binary classification

def load_data():
    """
    Load and prepare data for model training
    """
    data = {}
    
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
        logger.info(f"Loaded movie features with shape {data['movie_features'].shape}")
    else:
        logger.error(f"File not found: {movie_features_path}")
        return None
    
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
        logger.info(f"Loaded ratings with shape {data['ratings'].shape}")
    else:
        logger.error(f"File not found: {ratings_path}")
        return None
    
    if 'ratings' in data:
        # Create train/test split
        logger.info("Creating train/test split by user")
        all_user_ids = data['ratings']['userId'].unique()
        
        np.random.seed(42)
        np.random.shuffle(all_user_ids)
        
        # Ensure user split has enough users for testing
        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]
        
        # For each test user, split their ratings: 80% train, 20% test
        train_ratings_1 = data['ratings'][data['ratings']['userId'].isin(train_users)]
        
        test_user_ratings = data['ratings'][data['ratings']['userId'].isin(test_users)]
        train_chunks = []
        test_chunks = []
        
        for user_id in test_users:
            user_data = test_user_ratings[test_user_ratings['userId'] == user_id]
            
            # Skip users with very few ratings
            if len(user_data) < 5:
                continue
                
            # Split by time if available, otherwise random
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
            
            split_idx = int(len(user_data) * 0.8)
            train_chunks.append(user_data.iloc[:split_idx])
            test_chunks.append(user_data.iloc[split_idx:])
        
        # Combine all training data
        train_ratings_2 = pd.concat(train_chunks) if train_chunks else pd.DataFrame()
        test_ratings = pd.concat(test_chunks) if test_chunks else pd.DataFrame()
        
        data['train_ratings'] = pd.concat([train_ratings_1, train_ratings_2])
        data['test_ratings'] = test_ratings
        
        logger.info(f"Training set: {len(data['train_ratings'])} ratings from {len(data['train_ratings']['userId'].unique())} users")
        logger.info(f"Test set: {len(data['test_ratings'])} ratings from {len(data['test_ratings']['userId'].unique())} users")
    
    # Check if we have region features
    region_columns = [col for col in data['movie_features'].columns 
                     if col in ['North America', 'Europe', 'East Asia', 'South Asia', 
                               'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                               'Latin America', 'Other']]
    
    if region_columns:
        logger.info(f"Found {len(region_columns)} region features: {region_columns}")
        data['region_columns'] = region_columns
    
    return data

def extract_genre_and_region_features(movie_features):
    """
    Extract both genre and region features from movie data
    """
    genre_columns = [col for col in movie_features.columns if col not in 
                     ['movieId', 'title', 'tokens', 'token_count', 'top_keywords'] and
                     col not in ['North America', 'Europe', 'East Asia', 'South Asia', 
                                'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                                'Latin America', 'Other']]
    
    region_columns = [col for col in movie_features.columns 
                     if col in ['North America', 'Europe', 'East Asia', 'South Asia', 
                               'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                               'Latin America', 'Other']]
    
    if not genre_columns:
        logger.error("No genre columns found in movie features")
        return None
    
    # Extract features
    movie_genre_features = movie_features[['movieId'] + genre_columns].copy()
    
    # Add region features if available
    if region_columns:
        movie_region_features = movie_features[['movieId'] + region_columns].copy()
        # Combine with genre features
        movie_features_combined = pd.merge(
            movie_genre_features,
            movie_region_features,
            on='movieId',
            how='left'
        )
        
        # Fill NaN values with 0
        for col in region_columns:
            if col in movie_features_combined.columns:
                movie_features_combined[col] = movie_features_combined[col].fillna(0).astype(int)
                
        logger.info(f"Extracted {len(genre_columns)} genre features and {len(region_columns)} region features")
        return movie_features_combined, genre_columns, region_columns
    
    logger.info(f"Extracted {len(genre_columns)} genre features (no region features found)")
    return movie_genre_features, genre_columns, []

def calculate_user_preferences(train_ratings, movie_features, feature_columns, rating_threshold=3.5):
    """
    Calculate user preferences based on movie features and ratings
    """
    logger.info(f"Calculating user preferences for {len(feature_columns)} features")
    
    user_preferences = []
    
    total_users = len(train_ratings['userId'].unique())
    processed_users = 0
    start_time = time.time()
    
    for user_id in train_ratings['userId'].unique():
        user_ratings = train_ratings[train_ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            continue
        
        # Split into liked and disliked movies
        liked_movies = user_ratings[user_ratings['rating'] > rating_threshold]['movieId'].values
        disliked_movies = user_ratings[user_ratings['rating'] <= rating_threshold]['movieId'].values
        
        feature_preferences = {}
        
        for feature in feature_columns:
            # Calculate feature preference as weighted difference between liked and disliked
            feature_liked = movie_features[movie_features['movieId'].isin(liked_movies)][feature].sum()
            feature_disliked = movie_features[movie_features['movieId'].isin(disliked_movies)][feature].sum()
            
            # Calculate normalized preference score
            # More weight given to liked items (2/3) than disliked (1/3)
            liked_count = len(liked_movies) if len(liked_movies) > 0 else 1
            disliked_count = len(disliked_movies) if len(disliked_movies) > 0 else 1
            
            # Calculate weighted preference
            liked_weight = 2.0 / 3.0
            disliked_weight = 1.0 / 3.0
            
            # Weighted preference
            feature_preferences[feature] = (
                liked_weight * (feature_liked / liked_count) -
                disliked_weight * (feature_disliked / disliked_count)
            )
        
        # Normalize preferences to -1 to 1 range
        max_abs_preference = max(abs(val) for val in feature_preferences.values()) if feature_preferences else 1
        
        for feature in feature_preferences:
            feature_preferences[feature] = feature_preferences[feature] / max_abs_preference if max_abs_preference > 0 else 0
        
        feature_preferences['userId'] = user_id
        
        user_preferences.append(feature_preferences)
        
        processed_users += 1
        if processed_users % 1000 == 0:
            elapsed = time.time() - start_time
            progress = processed_users / total_users * 100
            remaining = elapsed * (total_users - processed_users) / processed_users
            logger.info(f"Processed {processed_users}/{total_users} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
    
    user_preferences_df = pd.DataFrame(user_preferences)
    logger.info(f"Created preferences for {len(user_preferences_df)} users")
    
    return user_preferences_df

def prepare_dnn_training_data(train_ratings, user_preferences, movie_features, genre_columns, region_columns=None, threshold=3.5, max_samples=1000000):
    """
    Prepare training data for DNN model with both genre and region features
    """
    logger.info("Preparing training data for DNN model")
    
    # Include both genre and region columns for feature vectors
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    features = []
    labels = []
    
    # Limit sample size for memory efficiency
    sample_size = min(max_samples, len(train_ratings))
    sampled_ratings = train_ratings.sample(sample_size, random_state=42) if len(train_ratings) > sample_size else train_ratings
    
    batch_size = 10000
    total_ratings = len(sampled_ratings)
    processed_ratings = 0
    start_time = time.time()
    
    for batch_start in range(0, total_ratings, batch_size):
        batch_end = min(batch_start + batch_size, total_ratings)
        ratings_batch = sampled_ratings.iloc[batch_start:batch_end]
        
        batch_features = []
        batch_labels = []
        
        for _, row in ratings_batch.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']
            
            # Convert rating to binary label - 1 if liked, 0 if not
            binary_label = 1 if rating > threshold else 0
            
            if user_id not in user_preferences['userId'].values or \
               movie_id not in movie_features['movieId'].values:
                continue
            
            user_prefs = user_preferences[user_preferences['userId'] == user_id].iloc[0]
            
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if movie_row.empty:
                continue
                
            movie_features_row = movie_row.iloc[0]
            
            # Create feature vector with user preferences and movie features
            feature_vector = []
            
            # Add normalized user preferences for each feature
            for feature in feature_columns:
                user_pref = user_prefs[feature]
                movie_feat = movie_features_row[feature]
                
                # Add user preference
                feature_vector.append(user_pref)
                
                # Add movie feature
                feature_vector.append(movie_feat)
                
                # Add interaction term (product of user preference and movie feature)
                feature_vector.append(user_pref * movie_feat)
            
            batch_features.append(feature_vector)
            batch_labels.append(binary_label)
        
        features.extend(batch_features)
        labels.extend(batch_labels)
        
        processed_ratings += len(ratings_batch)
        
        if batch_start % (10 * batch_size) == 0:
            elapsed = time.time() - start_time
            progress = processed_ratings / total_ratings * 100
            remaining = elapsed * (total_ratings - processed_ratings) / processed_ratings if processed_ratings > 0 else 0
            logger.info(f"Processed {processed_ratings}/{total_ratings} ratings ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        gc.collect()
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Created training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    logger.info(f"Created validation data: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    
    return X_train, X_val, y_train, y_val, feature_columns

def build_and_train_dnn_model(X_train, X_val, y_train, y_val, learning_rate=0.00001, batch_size=128, epochs=40):
    """
    Build and train an enhanced DNN model with better architecture and regularization
    """
    logger.info("Building and training DNN model")
    
    # Configure GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU config error: {e}")
    
    input_dim = X_train.shape[1]
    
    # Create more sophisticated model with improved architecture
    inputs = Input(shape=(input_dim,))
    
    # First layer with BatchNormalization to stabilize inputs
    x = BatchNormalization()(inputs)
    x = Dense(dnn_hidden_layers[0], activation='relu', kernel_regularizer=l2(dnn_l2_reg))(x)
    x = Dropout(dnn_dropout_rate)(x)
    
    # Hidden layers with residual connections for better gradient flow
    for i, units in enumerate(dnn_hidden_layers[1:]):
        prev_x = x
        x = BatchNormalization()(x)
        x = Dense(units, activation='relu', kernel_regularizer=l2(dnn_l2_reg))(x)
        x = Dropout(dnn_dropout_rate)(x)
        
        # Add residual connection if dimensions match
        if prev_x.shape[-1] == units:
            x = tf.keras.layers.add([x, prev_x])
    
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Adam optimizer with gradient clipping to prevent exploding gradients
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )
    
    # Compile with binary crossentropy loss for binary classification
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # More sophisticated callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    logger.info(f"Training model with {epochs} max epochs, batch size {batch_size}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )
    
    # Evaluate on validation set
    val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=1)
    
    logger.info(f"Model validation metrics:")
    logger.info(f"- Loss: {val_loss:.4f}")
    logger.info(f"- Accuracy: {val_acc:.4f}")
    logger.info(f"- AUC: {val_auc:.4f}")
    logger.info(f"- Precision: {val_precision:.4f}")
    logger.info(f"- Recall: {val_recall:.4f}")
    logger.info(f"- F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'dnn_training_history.png'))
    plt.close()
    
    return model, history

def generate_user_movie_features(user_id, movie_id, user_preferences, movie_features, genre_columns, region_columns=None):
    """
    Generate feature vector for a user-movie pair
    """
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    if user_id not in user_preferences['userId'].values or \
       movie_id not in movie_features['movieId'].values:
        return None
    
    user_prefs = user_preferences[user_preferences['userId'] == user_id].iloc[0]
    
    movie_row = movie_features[movie_features['movieId'] == movie_id]
    if movie_row.empty:
        return None
    
    movie_features_row = movie_row.iloc[0]
    
    feature_vector = []
    
    for feature in feature_columns:
        user_pref = user_prefs[feature]
        movie_feat = movie_features_row[feature]
        
        # Add user preference
        feature_vector.append(user_pref)
        
        # Add movie feature
        feature_vector.append(movie_feat)
        
        # Add interaction term
        feature_vector.append(user_pref * movie_feat)
    
    return np.array([feature_vector], dtype=np.float32)

def generate_dnn_recommendations(user_id, dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, train_ratings=None, n=10):
    """
    Generate movie recommendations for a user using the DNN model
    """
    if user_id not in user_preferences['userId'].values:
        logger.warning(f"User {user_id} not found in preferences")
        return []
    
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    user_prefs = user_preferences[user_preferences['userId'] == user_id].iloc[0]
    
    # Get movies the user has already rated
    rated_movies = set()
    if train_ratings is not None:
        rated_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
    
    # Consider only unrated movies
    unrated_movies = movie_features[~movie_features['movieId'].isin(rated_movies)]
    
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(unrated_movies), batch_size):
        batch = unrated_movies.iloc[i:i+batch_size]
        
        feature_vectors = []
        movie_ids = []
        
        for _, movie_row in batch.iterrows():
            movie_id = movie_row['movieId']
            feature_vector = []
            
            for feature in feature_columns:
                user_pref = user_prefs[feature]
                movie_feat = movie_row[feature]
                
                # Add user preference
                feature_vector.append(user_pref)
                
                # Add movie feature
                feature_vector.append(movie_feat)
                
                # Add interaction term
                feature_vector.append(user_pref * movie_feat)
            
            feature_vectors.append(feature_vector)
            movie_ids.append(movie_id)
        
        feature_array = np.array(feature_vectors)
        
        # Get probability scores from sigmoid output (0-1)
        like_probabilities = dnn_model.predict(feature_array, verbose=0).flatten()
        
        # Convert probabilities to rating-like scores (0.5-5.0) for compatibility
        # Map 0-1 to 0.5-5.0 scale: score = 0.5 + probability * 4.5
        predicted_ratings = 0.5 + like_probabilities * 4.5
        
        for movie_id, pred in zip(movie_ids, predicted_ratings):
            all_predictions.append((movie_id, float(pred)))
    
    # Sort by predicted rating (converted from probability)
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return all_predictions[:n]

def generate_recommendations_for_all_users(dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, train_ratings=None, n=10, batch_size=50, max_users=None):
    """
    Generate recommendations for all users
    """
    all_user_ids = user_preferences['userId'].unique()
    
    if max_users and max_users < len(all_user_ids):
        user_ids = all_user_ids[:max_users]
    else:
        user_ids = all_user_ids
    
    logger.info(f"Generating recommendations for {len(user_ids)} users")
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    # Create a dictionary of rated movies by user for faster lookups
    user_rated_movies = {}
    if train_ratings is not None:
        for _, row in train_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            if user_id not in user_rated_movies:
                user_rated_movies[user_id] = set()
            user_rated_movies[user_id].add(movie_id)
    
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    start_time = time.time()
    for i in range(0, total_users, batch_size):
        batch_end = min(i + batch_size, total_users)
        batch_users = user_ids[i:batch_end]
        
        batch_start_time = time.time()
        
        for user_idx, user_id in enumerate(batch_users):
            user_prefs = user_preferences[user_preferences['userId'] == user_id]
            if user_prefs.empty:
                continue
            
            rated_movies = user_rated_movies.get(user_id, set())
            
            # Consider only unrated movies
            unrated_movie_ids = set(movie_features['movieId']) - rated_movies
            
            # Limit the number of movies to process for efficiency
            max_movies_per_batch = 2000
            if len(unrated_movie_ids) > max_movies_per_batch:
                unrated_movie_ids = list(unrated_movie_ids)
                np.random.shuffle(unrated_movie_ids)
                unrated_movie_ids = unrated_movie_ids[:max_movies_per_batch]
            
            candidate_movies = movie_features[movie_features['movieId'].isin(unrated_movie_ids)]
            
            if len(candidate_movies) == 0:
                continue
            
            movie_batch_size = 200
            predictions = []
            
            for j in range(0, len(candidate_movies), movie_batch_size):
                movie_batch_end = min(j + movie_batch_size, len(candidate_movies))
                movie_batch = candidate_movies.iloc[j:movie_batch_end]
                
                batch_features = []
                batch_movie_ids = []
                
                for _, movie_row in movie_batch.iterrows():
                    movie_id = movie_row['movieId']
                    feature_vector = []
                    
                    for feature in feature_columns:
                        user_pref = user_prefs.iloc[0][feature]
                        movie_feat = movie_row[feature]
                        
                        # Add user preference
                        feature_vector.append(user_pref)
                        
                        # Add movie feature
                        feature_vector.append(movie_feat)
                        
                        # Add interaction term
                        feature_vector.append(user_pref * movie_feat)
                    
                    batch_features.append(feature_vector)
                    batch_movie_ids.append(movie_id)
                
                batch_features = np.array(batch_features, dtype=np.float32)
                
                if len(batch_features) == 0:
                    continue
                
                try:
                    # Get probability scores
                    batch_predictions = dnn_model.predict(batch_features, verbose=0).flatten()
                    
                    # Convert to rating-like scale
                    batch_ratings = 0.5 + (batch_predictions * 4.5)
                    
                    for movie_id, pred in zip(batch_movie_ids, batch_ratings):
                        predictions.append((movie_id, float(pred)))
                except Exception as e:
                    logger.error(f"Error predicting for user {user_id}: {e}")
            
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                all_recommendations[user_id] = predictions[:n]
        
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = elapsed / batch_end * (total_users - batch_end) if batch_end > 0 else 0
        
        logger.info(f"Processed {batch_end}/{total_users} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        gc.collect()
    
    logger.info(f"Generated recommendations for {len(all_recommendations)}/{total_users} users")
    
    return all_recommendations

def evaluate_recommendations(recommendations, test_ratings, dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, threshold=3.5):
    """
    Evaluate recommendation model using classification metrics
    """
    logger.info("Evaluating recommendations")
    
    # Check for common users between training and test sets
    train_users = set(user_preferences['userId'].unique())
    test_users = set(test_ratings['userId'].unique())
    common_users = train_users.intersection(test_users)
    
    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}, Common users: {len(common_users)}")
    
    if len(common_users) == 0:
        logger.warning("No common users between train and test sets, using baseline evaluation")
        
        # For binary classification baseline, predict majority class
        test_positives = (test_ratings['rating'] > threshold).sum()
        test_negatives = len(test_ratings) - test_positives
        majority_class = 1 if test_positives >= test_negatives else 0
        
        # Convert actual ratings to binary
        binary_actuals = (test_ratings['rating'] > threshold).astype(int)
        
        # Calculate metrics
        accuracy = (binary_actuals == majority_class).mean()
        
        # Log-loss (similar to binary cross-entropy)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        prob = majority_class if majority_class == 1 else epsilon
        log_loss = -np.mean(binary_actuals * np.log(prob) + (1 - binary_actuals) * np.log(1 - prob))
        
        logger.info(f"Baseline evaluation results (using majority class: {majority_class}):")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Log Loss: {log_loss:.4f}")
        logger.info(f"Number of predictions: {len(test_ratings)}")
        
        return {
            'accuracy': accuracy,
            'log_loss': log_loss,
            'num_predictions': len(test_ratings),
            'method': 'baseline_majority_class'
        }
    
    # Prepare evaluation data
    predictions = []
    actuals = []
    binary_predictions = []
    binary_actuals = []
    
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    # Track metrics per user
    user_metrics = {}
    
    for user_id in common_users:
        if user_id not in user_preferences['userId'].values:
            continue
        
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        user_recs = {}
        if user_id in recommendations:
            user_recs = dict(recommendations[user_id])
        
        user_preds = []
        user_actuals = []
        user_binary_preds = []
        user_binary_actuals = []
        
        for _, row in user_test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            binary_actual = 1 if actual_rating > threshold else 0
            
            if movie_id in user_recs:
                # Use pre-computed recommendation score
                predicted_rating = user_recs[movie_id]
                binary_prediction = 1 if predicted_rating > 3.0 else 0
                
                user_preds.append(predicted_rating)
                user_actuals.append(actual_rating)
                user_binary_preds.append(binary_prediction)
                user_binary_actuals.append(binary_actual)
                
            elif movie_id in movie_features['movieId'].values:
                # Generate features and predict
                feature_vector = generate_user_movie_features(
                    user_id, 
                    movie_id, 
                    user_preferences, 
                    movie_features, 
                    genre_columns,
                    region_columns
                )
                
                if feature_vector is not None:
                    # Get probability from model
                    like_probability = dnn_model.predict(feature_vector, verbose=0)[0][0]
                    
                    # Convert to rating-like scale
                    predicted_rating = 0.5 + like_probability * 4.5
                    
                    # Binary prediction
                    binary_prediction = 1 if like_probability > 0.5 else 0
                    
                    user_preds.append(predicted_rating)
                    user_actuals.append(actual_rating)
                    user_binary_preds.append(binary_prediction)
                    user_binary_actuals.append(binary_actual)
        
        if user_preds:
            # Add user predictions to global list
            predictions.extend(user_preds)
            actuals.extend(user_actuals)
            binary_predictions.extend(user_binary_preds)
            binary_actuals.extend(user_binary_actuals)
            
            # Calculate per-user metrics
            user_binary_preds_np = np.array(user_binary_preds)
            user_binary_actuals_np = np.array(user_binary_actuals)
            
            # Accuracy
            user_accuracy = np.mean(user_binary_preds_np == user_binary_actuals_np)
            
            # RMSE (on original rating scale)
            user_preds_np = np.array(user_preds)
            user_actuals_np = np.array(user_actuals)
            user_rmse = np.sqrt(np.mean((user_preds_np - user_actuals_np) ** 2))
            
            # Store user metrics
            user_metrics[user_id] = {
                'accuracy': user_accuracy,
                'rmse': user_rmse,
                'num_predictions': len(user_preds)
            }
    
    if not predictions:
        logger.warning("No predictions available for evaluation")
        return {
            'accuracy': 0.0,
            'log_loss': float('inf'),
            'num_predictions': 0
        }
    
    # Convert to numpy arrays for calculations
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    binary_predictions = np.array(binary_predictions)
    binary_actuals = np.array(binary_actuals)
    
    # Calculate binary classification metrics
    accuracy = np.mean(binary_predictions == binary_actuals)
    
    # Calculate confusion matrix elements
    true_positives = np.sum((binary_predictions == 1) & (binary_actuals == 1))
    false_positives = np.sum((binary_predictions == 1) & (binary_actuals == 0))
    true_negatives = np.sum((binary_predictions == 0) & (binary_actuals == 0))
    false_negatives = np.sum((binary_predictions == 0) & (binary_actuals == 1))
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'rmse': rmse,
        'mae': mae,
        'num_predictions': len(predictions),
        'method': 'binary_classification'
    }
    
    logger.info(f"Evaluation completed:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1_score:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Predictions: {len(predictions)}")
    
    return metrics, user_metrics

def recommend_for_user(user_id, recommendations, movie_features=None, n=10):
    """
    Display recommendations for a user
    """
    if user_id not in recommendations:
        logger.warning(f"No recommendations found for user {user_id}")
        return
    
    user_recs = recommendations[user_id][:n]
    
    if not user_recs:
        logger.warning(f"No recommendations found for user {user_id}")
        return
    
    logger.info(f"\nTop {len(user_recs)} recommendations for user {user_id}:")
    
    recs_info = []
    
    for i, (movie_id, predicted_rating) in enumerate(user_recs, 1):
        movie_info = f"Movie ID: {movie_id}"
        
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_info = movie_row.iloc[0]['title']
        
        logger.info(f"{i}. {movie_info} - Predicted Rating: {predicted_rating:.2f}")
        
        recs_info.append({
            'rank': i,
            'movie_id': movie_id,
            'title': movie_info,
            'predicted_rating': predicted_rating
        })
    
    return recs_info

# Main execution flow
logger.info("Starting DNN-based recommendation pipeline")

# Load data
data = load_data()
if data is None:
    logger.error("Failed to load data")
    exit(1)

# Extract genre and region features
movie_features_with_regions, genre_columns, region_columns = extract_genre_and_region_features(data['movie_features'])

if movie_features_with_regions is None:
    logger.error("Failed to extract movie features")
    exit(1)

# Calculate user preferences
user_preferences = calculate_user_preferences(
    data['train_ratings'], 
    movie_features_with_regions,
    genre_columns + region_columns,
    threshold_rating
)

# Save user preferences
user_preferences.to_csv(os.path.join(output_path, 'user_preferences.csv'), index=False)
logger.info(f"Saved user preferences for {len(user_preferences)} users")

# Prepare training data
X_train, X_val, y_train, y_val, feature_columns = prepare_dnn_training_data(
    data['train_ratings'],
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    threshold=threshold_rating
)

# Build and train the model
dnn_model, training_history = build_and_train_dnn_model(
    X_train, 
    X_val, 
    y_train, 
    y_val,
    learning_rate=dnn_learning_rate,
    batch_size=dnn_batch_size,
    epochs=dnn_epochs
)

# Save the trained model
dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))
logger.info("Saved trained DNN model")

# Generate recommendations for users
max_users = 10000000  # Limit for efficiency
dnn_recommendations = generate_recommendations_for_all_users(
    dnn_model,
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    data['train_ratings'],
    top_n,
    batch_size=50,
    max_users=max_users
)

# Save recommendations
with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
    pickle.dump(dnn_recommendations, f)
logger.info(f"Saved recommendations for {len(dnn_recommendations)} users")

# Evaluate the recommendations
logger.info("Evaluating DNN recommendations")
evaluation_metrics, user_metrics = evaluate_recommendations(
    dnn_recommendations,
    data['test_ratings'],
    dnn_model,
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    threshold=threshold_rating
)

# Save evaluation results
if evaluation_metrics:
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)
    logger.info("Saved evaluation metrics")
    
    # Save per-user metrics
    user_metrics_df = pd.DataFrame.from_dict(user_metrics, orient='index')
    user_metrics_df.reset_index(inplace=True)
    user_metrics_df.rename(columns={'index': 'userId'}, inplace=True)
    user_metrics_df.to_csv(os.path.join(output_path, 'dnn_user_metrics.csv'), index=False)
    logger.info(f"Saved per-user metrics for {len(user_metrics)} users")

# Show sample recommendations
logger.info("\nSample recommendation for exploration:")
if dnn_recommendations:
    # Pick a random user with recommendations
    sample_user_id = np.random.choice(list(dnn_recommendations.keys()))
    
    if sample_user_id in user_preferences['userId'].values:
        user_prefs = user_preferences[user_preferences['userId'] == sample_user_id].iloc[0]
        genre_pref_columns = [col for col in user_preferences.columns if col in genre_columns]
        
        logger.info(f"\nUser {sample_user_id} Genre Preferences:")
        user_prefs_list = [(genre, user_prefs[genre]) for genre in genre_pref_columns]
        liked_genres = sorted(user_prefs_list, key=lambda x: x[1], reverse=True)[:3]
        disliked_genres = sorted(user_prefs_list, key=lambda x: x[1])[:3]
        
        logger.info(f"- Most liked genres: {', '.join([f'{g} ({v:.2f})' for g, v in liked_genres])}")
        logger.info(f"- Most disliked genres: {', '.join([f'{g} ({v:.2f})' for g, v in disliked_genres])}")
        
        # Show region preferences if available
        if region_columns:
            region_pref_columns = [col for col in user_preferences.columns if col in region_columns]
            region_prefs_list = [(region, user_prefs[region]) for region in region_pref_columns]
            liked_regions = sorted(region_prefs_list, key=lambda x: x[1], reverse=True)[:3]
            
            logger.info(f"- Preferred regions: {', '.join([f'{r} ({v:.2f})' for r, v in liked_regions if v > 0])}")
    
    recommend_for_user(sample_user_id, dnn_recommendations, data['movie_features'])

logger.info("DNN-based recommendation pipeline completed")