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
import gc

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "./processed/"
output_path = "./rec/collaborative-recommendations"
top_n = 20

if not os.path.exists(output_path):
    os.makedirs(output_path)

dnn_hidden_layers = [64, 32, 16]
dnn_dropout_rate = 0.2
dnn_learning_rate = 0.001
dnn_batch_size = 64
dnn_epochs = 20
threshold_rating = 3.5

def load_data():
    data = {}
    
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
    else:
        return None
    
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
    else:
        return None
    
    if 'ratings' in data:
        all_user_ids = data['ratings']['userId'].unique()

        np.random.seed(42)
        np.random.shuffle(all_user_ids)

        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]

        data['train_ratings'] = data['ratings'][data['ratings']['userId'].isin(train_users)]
        data['test_ratings'] = data['ratings'][data['ratings']['userId'].isin(test_users)]
    
    return data

data = load_data()
if data is None:
    exit(1)

def extract_genre_features(movie_features):
    genre_columns = [col for col in movie_features.columns if col not in 
                     ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
    
    if not genre_columns:
        return None
    
    movie_genre_features = movie_features[['movieId'] + genre_columns].copy()
    
    return movie_genre_features

movie_genre_features = extract_genre_features(data['movie_features'])
if movie_genre_features is None:
    exit(1)

def calculate_user_genre_preferences(train_ratings, movie_genre_features):
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    user_genre_preferences = []
    
    total_users = len(train_ratings['userId'].unique())
    processed_users = 0
    
    for user_id in train_ratings['userId'].unique():
        user_ratings = train_ratings[train_ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            continue
        
        liked_movies = user_ratings[user_ratings['rating'] > threshold_rating]['movieId'].values
        disliked_movies = user_ratings[user_ratings['rating'] <= threshold_rating]['movieId'].values
        
        genre_preferences = {}
        
        for genre in genre_columns:
            genre_liked = movie_genre_features[movie_genre_features['movieId'].isin(liked_movies)][genre].sum()
            
            genre_disliked = movie_genre_features[movie_genre_features['movieId'].isin(disliked_movies)][genre].sum()
            
            genre_preferences[genre] = genre_liked - genre_disliked
        
        max_abs_preference = max(abs(val) for val in genre_preferences.values()) if genre_preferences else 1
        
        for genre in genre_preferences:
            genre_preferences[genre] = genre_preferences[genre] / max_abs_preference if max_abs_preference > 0 else 0
        
        genre_preferences['userId'] = user_id
        
        user_genre_preferences.append(genre_preferences)
        
        processed_users += 1
    
    user_genre_preferences_df = pd.DataFrame(user_genre_preferences)
    
    return user_genre_preferences_df

user_genre_preferences = calculate_user_genre_preferences(data['train_ratings'], movie_genre_features)

def prepare_dnn_training_data(train_ratings, user_genre_preferences, movie_genre_features, threshold=3.5):
    """
    Modified to prepare binary classification data using rating threshold
    """
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    features = []
    labels = []
    
    sample_size = min(1000000, len(train_ratings))
    sampled_ratings = train_ratings.sample(sample_size, random_state=42) if len(train_ratings) > sample_size else train_ratings
    
    batch_size = 10000
    total_ratings = len(sampled_ratings)
    processed_ratings = 0
    
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
            
            if user_id not in user_genre_preferences['userId'].values or \
               movie_id not in movie_genre_features['movieId'].values:
                continue
            
            user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
            
            movie_genres = movie_genre_features[movie_genre_features['movieId'] == movie_id].iloc[0]
            
            feature_vector = []
            
            for genre in genre_columns:
                feature_vector.append(user_prefs[genre])
                feature_vector.append(movie_genres[genre])
            
            batch_features.append(feature_vector)
            batch_labels.append(binary_label)  # Using binary label instead of rating value
        
        features.extend(batch_features)
        labels.extend(batch_labels)
        
        processed_ratings += len(ratings_batch)
        
        gc.collect()
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, genre_columns

X_train, X_val, y_train, y_val, genre_columns = prepare_dnn_training_data(
    data['train_ratings'], 
    user_genre_preferences, 
    movie_genre_features
)

def build_and_train_dnn_model(X_train, X_val, y_train, y_val):
    """
    Modified to use Binary Cross-Entropy loss and sigmoid activation for binary classification
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass
    
    input_dim = X_train.shape[1]
    
    model = Sequential()
    
    # Keep same architecture but modify the output layer and activation
    model.add(Dense(dnn_hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dnn_dropout_rate))
    
    for units in dnn_hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dnn_dropout_rate))
    
    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Use binary_crossentropy loss for binary classification
    model.compile(
        optimizer=Adam(learning_rate=dnn_learning_rate),
        loss='binary_crossentropy',  # Changed from MSE to BCE
        metrics=['accuracy', tf.keras.metrics.AUC()]  # Added accuracy and AUC metrics
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=dnn_epochs,
        batch_size=dnn_batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Evaluate on validation set
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    return model, history

dnn_model, training_history = build_and_train_dnn_model(X_train, X_val, y_train, y_val)

dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))

def generate_user_movie_features(user_id, movie_id, user_genre_preferences, movie_genre_features):
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    if user_id not in user_genre_preferences['userId'].values or \
       movie_id not in movie_genre_features['movieId'].values:
        return None
    
    user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
    
    movie_row = movie_genre_features[movie_genre_features['movieId'] == movie_id]
    if movie_row.empty:
        return None
    movie_genres = movie_row.iloc[0]
    
    feature_vector = []
    
    for genre in genre_columns:
        feature_vector.append(user_prefs[genre])
        feature_vector.append(movie_genres[genre])
    
    return np.array([feature_vector], dtype=np.float32)

def generate_dnn_recommendations(user_id, dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10):
    """
    Modified to work with binary classification model
    """
    if user_id not in user_genre_preferences['userId'].values:
        return []
    
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
    
    rated_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
    
    unrated_movies = movie_genre_features[~movie_genre_features['movieId'].isin(rated_movies)]
    
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(unrated_movies), batch_size):
        batch = unrated_movies.iloc[i:i+batch_size]
        
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

def generate_recommendations_for_all_users(dnn_model, user_genre_preferences, movie_genre_features, train_ratings, n=10, batch_size=50, max_users=None):
    all_user_ids = user_genre_preferences['userId'].unique()
    
    if max_users and max_users < len(all_user_ids):
        user_ids = all_user_ids[:max_users]
    else:
        user_ids = all_user_ids
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    user_rated_movies = {}
    
    for _, row in train_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        if user_id not in user_rated_movies:
            user_rated_movies[user_id] = set()
        user_rated_movies[user_id].add(movie_id)
    
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    start_time = time.time()
    for i in range(0, total_users, batch_size):
        batch_end = min(i + batch_size, total_users)
        batch_users = user_ids[i:batch_end]
        
        batch_start_time = time.time()
        
        for user_idx, user_id in enumerate(batch_users):
            user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id]
            if user_prefs.empty:
                continue
            
            rated_movies = user_rated_movies.get(user_id, set())
            
            unrated_movie_ids = set(movie_genre_features['movieId']) - rated_movies
            
            max_movies_per_batch = 1000
            if len(unrated_movie_ids) > max_movies_per_batch:
                unrated_movie_ids = list(unrated_movie_ids)[:max_movies_per_batch]
            
            candidate_movies = movie_genre_features[movie_genre_features['movieId'].isin(unrated_movie_ids)]
            
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
                    
                    for genre in genre_columns:
                        feature_vector.append(user_prefs.iloc[0][genre])
                        feature_vector.append(movie_row[genre])
                    
                    batch_features.append(feature_vector)
                    batch_movie_ids.append(movie_id)
                
                batch_features = np.array(batch_features, dtype=np.float32)
                
                if len(batch_features) == 0:
                    continue
                
                try:
                    batch_predictions = dnn_model.predict(batch_features, verbose=0).flatten()
                    
                    batch_predictions = np.clip(batch_predictions, 0.5, 5.0)
                    
                    for movie_id, pred in zip(batch_movie_ids, batch_predictions):
                        predictions.append((movie_id, float(pred)))
                except Exception as e:
                    pass
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            all_recommendations[user_id] = predictions[:n]
        
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / ((batch_end - i) / batch_size)
        progress = batch_end / total_users * 100
        remaining = avg_time_per_batch * ((total_users - batch_end) / batch_size) if batch_end < total_users else 0
        
        gc.collect()
    
    return all_recommendations

max_users = 200
dnn_recommendations = generate_recommendations_for_all_users(
    dnn_model,
    user_genre_preferences,
    movie_genre_features,
    data['train_ratings'],
    top_n,
    batch_size=50,
    max_users=max_users
)

with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
    pickle.dump(dnn_recommendations, f)

def evaluate_recommendations(recommendations, test_ratings, dnn_model, user_genre_preferences, movie_genre_features, threshold=3.5):
    """
    Modified to evaluate binary prediction model
    """
    print("Evaluating recommendations using binary classification metrics...")
    
    train_users = set(user_genre_preferences['userId'].unique())
    test_users = set(test_ratings['userId'].unique())
    common_users = train_users.intersection(test_users)
    
    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}, Common users: {len(common_users)}")
    
    if len(common_users) == 0:
        print("Using user-based split - no common users between train and test.")
        print("Evaluating using baseline majority class predictor.")
        
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
        
        print(f"Baseline evaluation results (using majority class: {majority_class}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {log_loss:.4f}")
        print(f"Number of predictions: {len(test_ratings)}")
        
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
    
    for user_id in test_ratings['userId'].unique():
        if user_id not in user_genre_preferences['userId'].values:
            continue
        
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        user_recs = {}
        if user_id in recommendations:
            user_recs = dict(recommendations[user_id])
        
        for _, row in user_test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            binary_actual = 1 if actual_rating > threshold else 0
            
            if movie_id in user_recs:
                predicted_rating = user_recs[movie_id]
                binary_prediction = 1 if predicted_rating > 3.0 else 0  # Converting back to binary
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                binary_predictions.append(binary_prediction)
                binary_actuals.append(binary_actual)
                
            elif movie_id in movie_genre_features['movieId'].values:
                feature_vector = generate_user_movie_features(
                    user_id, 
                    movie_id, 
                    user_genre_preferences, 
                    movie_genre_features
                )
                
                if feature_vector is not None:
                    # Get probability from model
                    like_probability = dnn_model.predict(feature_vector, verbose=0)[0][0]
                    
                    # Convert to rating-like scale for compatibility
                    predicted_rating = 0.5 + like_probability * 4.5
                    
                    # Also store binary prediction
                    binary_prediction = 1 if like_probability > 0.5 else 0
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
                    binary_predictions.append(binary_prediction)
                    binary_actuals.append(binary_actual)
    
    
    if not predictions:
        print("No predictions available for evaluation using standard method")
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
    
    # For backward compatibility, also calculate RMSE and MAE
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
        'rmse': rmse,  # Kept for backward compatibility
        'mae': mae,    # Kept for backward compatibility
        'num_predictions': len(predictions),
        'method': 'binary_classification'
    }
    
    print(f"Evaluation completed:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f} (for backward compatibility)")
    print(f"Predictions: {len(predictions)}")
    
    return metrics

def recommend_for_user(user_id, recommendations, movie_features=None, n=10):
    if user_id not in recommendations:
        print(f"No recommendations found for user {user_id}")
        return
    
    user_recs = recommendations[user_id][:n]
    
    if not user_recs:
        print(f"No recommendations found for user {user_id}")
        return
    
    print(f"\nTop {len(user_recs)} recommendations for user {user_id}:")
    
    for i, (movie_id, predicted_rating) in enumerate(user_recs, 1):
        movie_info = f"Movie ID: {movie_id}"
        
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_info = movie_row.iloc[0]['title']
        
        print(f"{i}. {movie_info} - Predicted Rating: {predicted_rating:.2f}")

print("\nEvaluating DNN recommendations...")
evaluation_metrics = evaluate_recommendations(
    dnn_recommendations,
    data['test_ratings'],
    dnn_model,
    user_genre_preferences,
    movie_genre_features
)

if evaluation_metrics:
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)

print("\nSample recommendation for exploration:")
if dnn_recommendations:
    sample_user_id = np.random.choice(list(dnn_recommendations.keys()))
    
    if sample_user_id in user_genre_preferences['userId'].values:
        user_prefs = user_genre_preferences[user_genre_preferences['userId'] == sample_user_id].iloc[0]
        genre_columns = [col for col in user_genre_preferences.columns if col != 'userId']
        
        print(f"\nUser {sample_user_id} Genre Preferences:")
        user_prefs_list = [(genre, user_prefs[genre]) for genre in genre_columns]
        liked_genres = sorted(user_prefs_list, key=lambda x: x[1], reverse=True)[:3]
        disliked_genres = sorted(user_prefs_list, key=lambda x: x[1])[:3]
        
        print(f"- Most liked genres: {', '.join([f'{g} ({v:.2f})' for g, v in liked_genres])}")
        print(f"- Most disliked genres: {', '.join([f'{g} ({v:.2f})' for g, v in disliked_genres])}")
    
    recommend_for_user(sample_user_id, dnn_recommendations, data['movie_features'])