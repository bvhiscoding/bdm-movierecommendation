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
import gc  # For garbage collection

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("COLLABORATIVE FILTERING WITH DEEP NEURAL NETWORK")
print("="*80)

# Set paths
input_path = "./processed/"  # Current directory where stage1.py saved the files
output_path = "./rec/collaborative-recommendations"
top_n = 20

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Model parameters
dnn_hidden_layers = [128, 64, 32]  # Optimized architecture 
dnn_dropout_rate = 0.2
dnn_learning_rate = 0.001
dnn_batch_size = 64   # Increased batch size for faster training
dnn_epochs = 20       # Reduced epochs with early stopping
threshold_rating = 3  # Rating threshold to classify as "like"

print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

def load_data():
    """
    Load processed data from stage1.py
    
    Input: None (reads from files)
    Output: Dictionary containing DataFrames for movie features and ratings
    """
    print("Loading processed data from stage1.py...")
    
    # Data containers
    data = {}
    
    # Load movie features
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
        print(f"Loaded features for {len(data['movie_features'])} movies")
    else:
        print(f"Movie features not found at {movie_features_path}")
        return None
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
        print(f"Loaded {len(data['ratings'])} normalized ratings")
    else:
        print(f"Normalized ratings not found at {ratings_path}")
        return None
    
    return data

# Load the data
data = load_data()
if data is None:
    print("Failed to load required data")
    exit(1)

# Analyze the loaded data
print("\n" + "-"*50)
print("DATA ANALYSIS: LOADED DATASETS")
print("-"*50)

# Show movie features summary
if 'movie_features' in data:
    print(f"\nMovie Features Summary:")
    print(f"- Total movies: {len(data['movie_features'])}")
    
    # Get genre columns
    genre_columns = [col for col in data['movie_features'].columns if col not in 
                     ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
    
    print(f"- Number of genres: {len(genre_columns)}")
    print(f"- Genre columns: {genre_columns}")
    
    # Show sample movie features
    print("\nSample movie features:")
    print(data['movie_features'][['movieId', 'title'] + genre_columns[:3]].head(3))

# Show ratings summary
if 'ratings' in data:
    print(f"\nRatings Summary:")
    print(f"- Total ratings: {len(data['ratings'])}")
    print(f"- Unique users: {data['ratings']['userId'].nunique()}")
    print(f"- Unique movies: {data['ratings']['movieId'].nunique()}")
    print(f"- Rating range: {data['ratings']['rating'].min()} - {data['ratings']['rating'].max()}")
    print(f"- Average rating: {data['ratings']['rating'].mean():.2f}")
    
    # Show rating distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data=data['ratings'], x='rating', bins=9, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_path, 'rating_distribution.png'))
    print(f"\nRating distribution plot saved to {os.path.join(output_path, 'rating_distribution.png')}")
    plt.close()

# Analyze user rating distribution
ratings_per_user = data['ratings'].groupby('userId').size()

print(f"\nRatings per user:")
print(f"- Average: {ratings_per_user.mean():.2f}")
print(f"- Min: {ratings_per_user.min()}, Max: {ratings_per_user.max()}")

print("\n" + "="*80)
print("STEP 2: MOVIE GENRE FEATURE EXTRACTION")
print("="*80)

def extract_genre_features(movie_features):
    """
    Extract genre features for each movie
    
    Input: 
      - movie_features: DataFrame with movie features including genre columns
    
    Output:
      - movie_genre_features: DataFrame with movieId and genre columns only
    """
    print("Extracting genre features for movies...")
    
    # Get all genre columns (assuming they're already one-hot encoded)
    genre_columns = [col for col in movie_features.columns if col not in 
                     ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
    
    if not genre_columns:
        print("No genre columns found in movie features")
        return None
    
    # Create genre feature matrix
    movie_genre_features = movie_features[['movieId'] + genre_columns].copy()
    
    print(f"Extracted {len(genre_columns)} genre features for {len(movie_features)} movies")
    
    return movie_genre_features

# Extract genre features
movie_genre_features = extract_genre_features(data['movie_features'])
if movie_genre_features is None:
    print("Failed to extract genre features")
    exit(1)

# Analyze the extracted genre features
print("\n" + "-"*50)
print("DATA ANALYSIS: GENRE FEATURES")
print("-"*50)

# Show genre distribution
print("\nGenre Distribution:")
genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
genre_counts = {}

for genre in genre_columns:
    count = movie_genre_features[genre].sum()
    genre_counts[genre] = count
    print(f"- {genre}: {count} movies ({count/len(movie_genre_features)*100:.1f}%)")

# Plot genre distribution
plt.figure(figsize=(12, 6))
sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
genres, counts = zip(*sorted_genres)
plt.bar(genres, counts)
plt.title('Distribution of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'genre_distribution.png'))
print(f"\nGenre distribution plot saved to {os.path.join(output_path, 'genre_distribution.png')}")
plt.close()

# Analyze genre co-occurrence
print("\nGenre Co-occurrence Analysis:")
genre_co_occurrence = pd.DataFrame(0, index=genre_columns, columns=genre_columns)

for _, row in movie_genre_features.iterrows():
    movie_genres = [genre for genre in genre_columns if row[genre] == 1]
    for g1 in movie_genres:
        for g2 in movie_genres:
            genre_co_occurrence.loc[g1, g2] += 1

# Normalize by diagonal for correlation-like measure
for g in genre_columns:
    genre_co_occurrence[g] = genre_co_occurrence[g] / genre_co_occurrence.loc[g, g]

# Display most common genre combinations
print("Most common genre combinations:")
for i, g1 in enumerate(genre_columns[:5]):  # Limit to 5 genres for brevity
    most_common = genre_co_occurrence.loc[g1].sort_values(ascending=False)[1:6]  # Skip self (always 1.0)
    print(f"- {g1} most commonly appears with: {', '.join([f'{g2} ({v:.2f})' for g2, v in most_common.items()])}")

# Save genre co-occurrence matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(genre_co_occurrence, annot=False, cmap='viridis')
plt.title('Genre Co-occurrence Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'genre_co_occurrence.png'))
print(f"\nGenre co-occurrence matrix saved to {os.path.join(output_path, 'genre_co_occurrence.png')}")
plt.close()

# Show sample of genre features
print("\nSample of movie genre features:")
print(movie_genre_features.head(3))

# Save the genre features for later use
movie_genre_features.to_csv(os.path.join(output_path, 'movie_genre_features.csv'), index=False)
print(f"\nSaved movie genre features to {os.path.join(output_path, 'movie_genre_features.csv')}")

print("\n" + "="*80)
print("STEP 3: USER GENRE PREFERENCE CALCULATION")
print("="*80)

def calculate_user_genre_preferences(ratings, movie_genre_features):
    """
    Calculate user preferences for movie genres based on ratings
    
    Input:
      - ratings: DataFrame with user-movie ratings
      - movie_genre_features: DataFrame with movie genre features
    
    Output:
      - user_genre_preferences_df: DataFrame with userId and genre preference scores
    """
    print("Calculating user preferences for movie genres...")
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Initialize user genre preferences dataframe
    user_genre_preferences = []
    
    # Process each user
    total_users = len(ratings['userId'].unique())
    processed_users = 0
    
    for user_id in ratings['userId'].unique():
        # Get user ratings
        user_ratings = ratings[ratings['userId'] == user_id]
        
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
        
        # Update progress
        processed_users += 1
        if processed_users % 100 == 0 or processed_users == total_users:
            print(f"Processed {processed_users}/{total_users} users ({processed_users/total_users*100:.1f}%)")
    
    # Convert to dataframe
    user_genre_preferences_df = pd.DataFrame(user_genre_preferences)
    
    print(f"Calculated genre preferences for {len(user_genre_preferences_df)} users")
    
    return user_genre_preferences_df

# Calculate user genre preferences
user_genre_preferences = calculate_user_genre_preferences(data['ratings'], movie_genre_features)

# Analyze the user genre preferences
print("\n" + "-"*50)
print("DATA ANALYSIS: USER GENRE PREFERENCES")
print("-"*50)

# Show basic statistics of user genre preferences
if not user_genre_preferences.empty:
    print("\nUser Genre Preferences Summary:")
    
    genre_columns = [col for col in user_genre_preferences.columns if col != 'userId']
    
    # Calculate statistics for each genre
    genre_stats = {}
    for genre in genre_columns:
        stats = {
            'mean': user_genre_preferences[genre].mean(),
            'min': user_genre_preferences[genre].min(),
            'max': user_genre_preferences[genre].max(),
            'std': user_genre_preferences[genre].std(),
            'positive': (user_genre_preferences[genre] > 0).sum(),
            'negative': (user_genre_preferences[genre] < 0).sum(),
            'neutral': (user_genre_preferences[genre] == 0).sum()
        }
        genre_stats[genre] = stats
    
    # Display statistics for top genres
    print("\nStatistics for top genres:")
    top_genres = sorted(genre_stats.items(), key=lambda x: x[1]['positive'], reverse=True)[:5]
    
    for genre, stats in top_genres:
        print(f"- {genre}:")
        print(f"  * Mean preference: {stats['mean']:.3f} (std: {stats['std']:.3f})")
        print(f"  * Range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"  * Users with positive preference: {stats['positive']} ({stats['positive']/len(user_genre_preferences)*100:.1f}%)")
        print(f"  * Users with negative preference: {stats['negative']} ({stats['negative']/len(user_genre_preferences)*100:.1f}%)")
    
    # Plot distribution of preferences for top genres
    plt.figure(figsize=(15, 10))
    for i, (genre, _) in enumerate(top_genres):
        plt.subplot(2, 3, i+1)
        sns.histplot(user_genre_preferences[genre], kde=True)
        plt.title(f'Distribution of {genre} Preferences')
        plt.xlabel('Preference Score')
        plt.ylabel('Number of Users')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'user_preference_distributions.png'))
    print(f"\nUser preference distributions saved to {os.path.join(output_path, 'user_preference_distributions.png')}")
    plt.close()
    
    # Create a correlation heatmap of genre preferences
    plt.figure(figsize=(12, 10))
    corr_matrix = user_genre_preferences[genre_columns].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Between Genre Preferences')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'genre_preference_correlation.png'))
    print(f"\nGenre preference correlation matrix saved to {os.path.join(output_path, 'genre_preference_correlation.png')}")
    plt.close()
    
    # Show example users with diverse preferences
    print("\nExample users with diverse preferences:")
    # Calculate preference diversity as standard deviation across genres
    user_genre_preferences['preference_diversity'] = user_genre_preferences[genre_columns].std(axis=1)
    
    # Get top 3 users with highest diversity
    diverse_users = user_genre_preferences.nlargest(3, 'preference_diversity')
    for _, user in diverse_users.iterrows():
        user_id = user['userId']
        print(f"\nUser {user_id} (diversity score: {user['preference_diversity']:.3f}):")
        
        # Show top 3 liked and disliked genres
        user_prefs = [(genre, user[genre]) for genre in genre_columns]
        liked_genres = sorted(user_prefs, key=lambda x: x[1], reverse=True)[:3]
        disliked_genres = sorted(user_prefs, key=lambda x: x[1])[:3]
        
        print(f"- Most liked genres: {', '.join([f'{g} ({v:.2f})' for g, v in liked_genres])}")
        print(f"- Most disliked genres: {', '.join([f'{g} ({v:.2f})' for g, v in disliked_genres])}")
    
    # Remove the temporary column
    user_genre_preferences.drop('preference_diversity', axis=1, inplace=True)
    
    # Show sample of user genre preferences
    print("\nSample of user genre preferences:")
    sample_users = user_genre_preferences.sample(3)
    for _, user in sample_users.iterrows():
        user_id = user['userId']
        print(f"\nUser {user_id} preferences:")
        # Show top 5 genres with non-zero preferences
        user_prefs = [(genre, user[genre]) for genre in genre_columns if user[genre] != 0]
        sorted_prefs = sorted(user_prefs, key=lambda x: abs(x[1]), reverse=True)[:5]
        for genre, value in sorted_prefs:
            print(f"- {genre}: {value:.3f}")

    # Save the user genre preferences for later use
    user_genre_preferences.to_csv(os.path.join(output_path, 'user_genre_preferences.csv'), index=False)
    print(f"\nSaved user genre preferences to {os.path.join(output_path, 'user_genre_preferences.csv')}")

print("\n" + "="*80)
print("STEP 4: DNN TRAINING DATA PREPARATION")
print("="*80)

def prepare_dnn_training_data(ratings, user_genre_preferences, movie_genre_features):
    """
    Prepare training data for the DNN model
    
    Input:
      - ratings: DataFrame with user-movie ratings
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
    
    Output:
      - X_train, X_val: Feature matrices for training and validation
      - y_train, y_val: Target values for training and validation
      - genre_columns: List of genre column names
    """
    print("Preparing training data for DNN model...")
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Initialize lists for features and labels
    features = []
    labels = []
    
    # Process only a sample of ratings for efficiency
    sample_size = min(1000000, len(ratings))  # Cap at 1M ratings
    sampled_ratings = ratings.sample(sample_size, random_state=42) if len(ratings) > sample_size else ratings
    
    print(f"Using {len(sampled_ratings)} ratings to train the DNN model")
    
    # Process each rating in batches to avoid memory issues
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
            batch_features.append(feature_vector)
            batch_labels.append(rating)
        
        # Extend the main lists
        features.extend(batch_features)
        labels.extend(batch_labels)
        
        # Update progress
        processed_ratings += len(ratings_batch)
        print(f"Processed {processed_ratings}/{total_ratings} ratings ({processed_ratings/total_ratings*100:.1f}%)")
        
        # Force garbage collection
        gc.collect()
    
    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Created feature matrix with shape {X.shape} and labels with shape {y.shape}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    print(f"Prepared training data with {len(X_train)} samples, validation data with {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val, genre_columns

# Prepare DNN training data
X_train, X_val, y_train, y_val, genre_columns = prepare_dnn_training_data(
    data['ratings'], 
    user_genre_preferences, 
    movie_genre_features
)

# Analyze the training data
print("\n" + "-"*50)
print("DATA ANALYSIS: DNN TRAINING DATA")
print("-"*50)

# Feature dimension analysis
feature_dim = X_train.shape[1]
print(f"\nFeature Vector Dimension: {feature_dim}")
print(f"Number of genres: {len(genre_columns)}")
print(f"Features per genre: 2 (user preference + movie indicator)")
print(f"Total features: {len(genre_columns) * 2}")

# Analyze distribution of training labels
print("\nTraining Labels Distribution:")
plt.figure(figsize=(10, 5))
sns.histplot(y_train, bins=9, kde=True)
plt.title('Distribution of Training Labels (Ratings)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(os.path.join(output_path, 'training_labels_distribution.png'))
print(f"Training labels distribution saved to {os.path.join(output_path, 'training_labels_distribution.png')}")
plt.close()

# Analyze feature statistics
print("\nFeature Statistics:")
feature_means = np.mean(X_train, axis=0)
feature_stds = np.std(X_train, axis=0)

# Organize features by genre for better interpretation
genre_feature_stats = []
for i, genre in enumerate(genre_columns):
    # User preference feature is at index 2*i
    # Movie indicator feature is at index 2*i + 1
    user_pref_idx = 2*i
    movie_ind_idx = 2*i + 1
    
    genre_feature_stats.append({
        'Genre': genre,
        'User_Pref_Mean': feature_means[user_pref_idx],
        'User_Pref_Std': feature_stds[user_pref_idx],
        'Movie_Ind_Mean': feature_means[movie_ind_idx],
        'Movie_Ind_Std': feature_stds[movie_ind_idx]
    })

# Convert to dataframe for easier analysis
feature_stats_df = pd.DataFrame(genre_feature_stats)
print("\nFeature statistics by genre (top 5 genres):")
print(feature_stats_df.sort_values('User_Pref_Mean', ascending=False).head())

# Plot feature distributions for a few genres
plt.figure(figsize=(15, 10))
top_genres = feature_stats_df.sort_values('User_Pref_Mean', ascending=False).head(4)['Genre'].values

for i, genre in enumerate(top_genres):
    genre_idx = genre_columns.index(genre)
    user_pref_idx = 2 * genre_idx
    movie_ind_idx = 2 * genre_idx + 1
    
    plt.subplot(2, 2, i+1)
    sns.histplot(X_train[:, user_pref_idx], label='User Preference', alpha=0.7)
    plt.title(f'{genre} - User Preference Distribution')
    plt.xlabel('Preference Value')
    plt.ylabel('Count')
    
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'feature_distributions.png'))
print(f"Feature distributions saved to {os.path.join(output_path, 'feature_distributions.png')}")
plt.close()

# Save a sample of the training data for reference
sample_indices = np.random.choice(len(X_train), min(5, len(X_train)), replace=False)
sample_data = []

for idx in sample_indices:
    features = X_train[idx]
    rating = y_train[idx]
    
    sample_features = {}
    for i, genre in enumerate(genre_columns):
        user_pref_idx = 2*i
        movie_ind_idx = 2*i + 1
        
        sample_features[f"{genre}_user_pref"] = features[user_pref_idx]
        sample_features[f"{genre}_movie_ind"] = features[movie_ind_idx]
    
    sample_features['rating'] = rating
    sample_data.append(sample_features)

sample_df = pd.DataFrame(sample_data)
print("\nSample of DNN training data (showing first 3 genres for 1 sample):")
print(sample_df.iloc[0][[f"{genre}_user_pref" for genre in genre_columns[:3]] + 
                       [f"{genre}_movie_ind" for genre in genre_columns[:3]] + 
                       ['rating']].to_string())

print("\n" + "="*80)
print("STEP 5: DNN MODEL BUILDING AND TRAINING")
print("="*80)

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
    print("Building and training DNN model...")
    
    # Set memory limit to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
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
    
    print(f"Model training completed. Validation MSE: {val_loss:.4f}, validation MAE: {val_mae:.4f}")
    
    return model, history

# Build and train DNN model
dnn_model, training_history = build_and_train_dnn_model(X_train, X_val, y_train, y_val)

# Save DNN model
dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))
print(f"Saved DNN model to {os.path.join(output_path, 'dnn_model.h5')}")

# Analyze the training results
print("\n" + "-"*50)
print("MODEL ANALYSIS: DNN TRAINING RESULTS")
print("-"*50)

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
print(f"Training history plot saved to {os.path.join(output_path, 'dnn_training_history.png')}")
plt.close()

# Model architecture summary
print("\nDNN Model Architecture:")
dnn_model.summary()

# Analyze convergence
final_train_loss = training_history.history['loss'][-1]
final_val_loss = training_history.history['val_loss'][-1]
final_train_mae = training_history.history['mae'][-1]
final_val_mae = training_history.history['val_mae'][-1]

print(f"\nFinal Training Metrics:")
print(f"- MSE Loss: {final_train_loss:.4f}")
print(f"- MAE: {final_train_mae:.4f}")

print(f"\nFinal Validation Metrics:")
print(f"- MSE Loss: {final_val_loss:.4f}")
print(f"- MAE: {final_val_mae:.4f}")

# Calculate RMSE from MSE
final_train_rmse = np.sqrt(final_train_loss)
final_val_rmse = np.sqrt(final_val_loss)

print(f"\nFinal RMSE:")
print(f"- Training RMSE: {final_train_rmse:.4f}")
print(f"- Validation RMSE: {final_val_rmse:.4f}")
# Save evaluation metrics to CSV
evaluation_metrics = {
    'rmse': final_val_rmse,
    'mae': final_val_mae,
    'num_predictions': len(X_val),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}
metrics_df = pd.DataFrame([evaluation_metrics])
metrics_df.to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)
print(f"Evaluation metrics saved to {os.path.join(output_path, 'dnn_evaluation.csv')}")
# Analyze prediction quality
print("\nPrediction Quality Analysis:")
val_predictions = dnn_model.predict(X_val)
val_errors = val_predictions.flatten() - y_val

# Create error histogram
plt.figure(figsize=(10, 6))
plt.hist(val_errors, bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Validation Prediction Error Distribution')
plt.xlabel('Prediction Error (Predicted - Actual)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_path, 'prediction_error_distribution.png'))
print(f"Prediction error distribution saved to {os.path.join(output_path, 'prediction_error_distribution.png')}")
plt.close()

# Create scatter plot of predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_predictions, alpha=0.3)
plt.plot([0.5, 5.0], [0.5, 5.0], 'r--')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Predicted vs Actual Ratings')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_path, 'predicted_vs_actual.png'))
print(f"Predicted vs actual plot saved to {os.path.join(output_path, 'predicted_vs_actual.png')}")
plt.close()

# Calculate error statistics by rating level
error_by_rating = {}
for rating in sorted(np.unique(np.round(y_val * 2) / 2)):  # Round to nearest 0.5
    mask = (np.round(y_val * 2) / 2 == rating)
    if np.sum(mask) > 0:
        rating_errors = val_errors[mask]
        error_by_rating[rating] = {
            'count': len(rating_errors),
            'mean_error': np.mean(rating_errors),
            'abs_error': np.mean(np.abs(rating_errors)),
            'rmse': np.sqrt(np.mean(rating_errors**2))
        }

print("\nError by rating level:")
error_df = pd.DataFrame.from_dict(error_by_rating, orient='index')
print(error_df)

# Plot error by rating level
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(error_df.index, error_df['mean_error'])
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Mean Error by Rating Level')
plt.xlabel('Actual Rating')
plt.ylabel('Mean Error')

plt.subplot(1, 2, 2)
plt.bar(error_df.index, error_df['rmse'])
plt.title('RMSE by Rating Level')
plt.xlabel('Actual Rating')
plt.ylabel('RMSE')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'error_by_rating.png'))
print(f"Error by rating plot saved to {os.path.join(output_path, 'error_by_rating.png')}")
plt.close()

print("\n" + "="*80)
print("STEP 6: MOVIE RECOMMENDATION GENERATION")
print("="*80)

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

def generate_dnn_recommendations(user_id, dnn_model, user_genre_preferences, movie_genre_features, ratings, n=10):
    """Optimized version with batched predictions"""
    print(f"Generating recommendations for user {user_id}...")
    
    # Skip if user not found in genre preferences
    if user_id not in user_genre_preferences['userId'].values:
        print(f"User {user_id} not found in genre preferences")
        return []
    
    # Get genre columns
    genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
    
    # Get user genre preferences
    user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
    
    # Get movies already rated by the user
    rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    
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
    
    print(f"Generated {len(all_predictions)} predictions for user {user_id}")
    
    # Return top N recommendations
    return all_predictions[:n]

def generate_recommendations_for_all_users(dnn_model, user_genre_preferences, movie_genre_features, ratings, n=10, batch_size=50, max_users=None):
    """
    Generate recommendations for all users using the DNN model with improved batching
    
    Input:
      - dnn_model: Trained DNN model
      - user_genre_preferences: DataFrame with user genre preferences
      - movie_genre_features: DataFrame with movie genre features
      - ratings: DataFrame with ratings
      - n: Number of recommendations to generate per user
      - batch_size: Number of users to process in each batch
      - max_users: Maximum number of users to process (optional)
    
    Output:
      - all_recommendations: Dictionary mapping user IDs to recommendation lists
    """
    print(f"Generating top-{n} DNN recommendations for all users with optimized batching...")
    
    # Get all user IDs
    all_user_ids = user_genre_preferences['userId'].unique()
    
    # Limit to max_users if specified
    if max_users and max_users < len(all_user_ids):
        user_ids = all_user_ids[:max_users]
        print(f"Limiting to {max_users} users out of {len(all_user_ids)} total users")
    else:
        user_ids = all_user_ids
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    # Create a lookup dictionary for user ratings to avoid repeated filtering
    print("Creating user rating lookup dictionary...")
    user_rated_movies = {}
    for _, row in ratings.iterrows():
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
        
        print(f"Processing batch of {len(batch_users)} users ({i+1}-{batch_end} of {total_users})")
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
            # 1. Get all unrated movies
            unrated_movie_ids = set(movie_genre_features['movieId']) - rated_movies
            
            # If too many, limit to a manageable number to improve performance
            max_movies_per_batch = 1000
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
                    print(f"Error making predictions for user {user_id}, batch {j}: {e}")
            
            # Sort predictions by rating and take top n
            predictions.sort(key=lambda x: x[1], reverse=True)
            all_recommendations[user_id] = predictions[:n]
            
            # Log progress for every 10th user or the last one
            if (user_idx + 1) % 10 == 0 or user_idx == len(batch_users) - 1:
                elapsed_batch = time.time() - batch_start_time
                avg_time_per_user = elapsed_batch / (user_idx + 1)
                print(f"  Processed {user_idx + 1}/{len(batch_users)} users in batch, avg time: {avg_time_per_user:.2f}s per user")
        
        # Log batch completion
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / ((batch_end - i) / batch_size)
        progress = batch_end / total_users * 100
        remaining = avg_time_per_batch * ((total_users - batch_end) / batch_size) if batch_end < total_users else 0
        
        print(f"Completed batch {i//batch_size + 1}/{(total_users-1)//batch_size + 1}")
        print(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Force garbage collection
        gc.collect()
    
    print(f"Generated recommendations for {len(all_recommendations)} users")
    return all_recommendations

# Generate DNN recommendations for all users (limiting to a reasonable number for demonstration)
all_user_ids = sorted(user_genre_preferences['userId'].unique())
target_users_count = int(len(all_user_ids) * 0.2)
target_users = all_user_ids[:target_users_count]

print(f"Generating recommendations for first 20% of users ({len(target_users)} out of {len(all_user_ids)} total users)")
print(f"Target user range: {min(target_users)} to {max(target_users)}")

# Create filtered user_genre_preferences with only target users
filtered_user_preferences = user_genre_preferences[user_genre_preferences['userId'].isin(target_users)]
dnn_recommendations = generate_recommendations_for_all_users(
    dnn_model,
    filtered_user_preferences,
    movie_genre_features,
    data['ratings'],
    top_n,
    batch_size=50
)

# Save recommendations
with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
    pickle.dump(dnn_recommendations, f)

# Analyze the recommendations
print("\n" + "-"*50)
print("RECOMMENDATION ANALYSIS: DNN RECOMMENDATIONS")
print("-"*50)

if dnn_recommendations:
    # Create recommendations dataframe for analysis
    rec_list = []
    for user_id, recs in dnn_recommendations.items():
        for rank, (movie_id, rating) in enumerate(recs, 1):
            rec_list.append({
                'userId': user_id,
                'movieId': movie_id,
                'predicted_rating': rating,
                'rank': rank
            })
    
    rec_df = pd.DataFrame(rec_list)
    
    # Save in CSV format
    if not rec_df.empty:
        # Add movie titles if available
        if 'movie_features' in data:
            movie_titles = data['movie_features'][['movieId', 'title']]
            rec_df = pd.merge(rec_df, movie_titles, on='movieId', how='left')
        
        rec_df.to_csv(os.path.join(output_path, 'dnn_recommendations.csv'), index=False)
        print(f"Saved recommendations to {os.path.join(output_path, 'dnn_recommendations.csv')}")
    
    # Basic recommendation statistics
    print(f"\nRecommendation Statistics:")
    print(f"- Users with recommendations: {len(dnn_recommendations)}")
    print(f"- Total recommendation entries: {len(rec_df)}")
    print(f"- Average recommendations per user: {len(rec_df)/len(dnn_recommendations):.2f}")
    
    # Rating distribution
    print(f"\nPredicted Rating Distribution:")
    rating_stats = rec_df['predicted_rating'].describe()
    print(f"- Min: {rating_stats['min']:.2f}")
    print(f"- Max: {rating_stats['max']:.2f}")
    print(f"- Mean: {rating_stats['mean']:.2f}")
    print(f"- Median: {rating_stats['50%']:.2f}")
    print(f"- Std Dev: {rating_stats['std']:.2f}")
    
    # Plot recommendation rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(rec_df['predicted_rating'], bins=20, kde=True)
    plt.title('Distribution of Predicted Ratings in Recommendations')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_path, 'recommendation_rating_distribution.png'))
    print(f"Recommendation rating distribution saved to {os.path.join(output_path, 'recommendation_rating_distribution.png')}")
    plt.close()
    
    # Analyze top recommended movies
    if 'movie_features' in data:
        print("\nTop Recommended Movies:")
        top_movies = rec_df.groupby('movieId').size().reset_index(name='count')
        top_movies = pd.merge(top_movies, data['movie_features'][['movieId', 'title']], on='movieId')
        top_movies = top_movies.sort_values('count', ascending=False).head(10)
        
        for i, (_, row) in enumerate(top_movies.iterrows(), 1):
            print(f"{i}. '{row['title']}' - Recommended to {row['count']} users")
        
        # Get genre distribution of top recommended movies
        top_movie_ids = top_movies['movieId'].values
        top_movie_genres = movie_genre_features[movie_genre_features['movieId'].isin(top_movie_ids)]
        
        genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
        genre_counts = {}
        
        for genre in genre_columns:
            count = top_movie_genres[genre].sum()
            genre_counts[genre] = count
        
        # Plot genre distribution of top recommendations
        if genre_counts:
            plt.figure(figsize=(12, 6))
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            genres, counts = zip(*sorted_genres)
            plt.bar(genres, counts)
            plt.title('Genre Distribution of Top Recommended Movies')
            plt.xlabel('Genre')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'top_recommendations_genre_distribution.png'))
            print(f"Top recommendations genre distribution saved to {os.path.join(output_path, 'top_recommendations_genre_distribution.png')}")
            plt.close()
    
    # Show sample recommendations for a few users
    print("\nSample Recommendations for 3 Users:")
    sample_users = list(dnn_recommendations.keys())[:3]
    
    for user_id in sample_users:
        print(f"\nUser {user_id}:")
        user_recs = dnn_recommendations[user_id][:5]  # Show top 5
        
        for i, (movie_id, rating) in enumerate(user_recs, 1):
            movie_info = f"Movie ID: {movie_id}"
            
            # Try to get movie title and genres if available
            if 'movie_features' in data:
                movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
                if not movie_row.empty:
                    movie_info = movie_row.iloc[0]['title']
            
            print(f"{i}. {movie_info} - Predicted Rating: {rating:.2f}")

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

# Display a sample recommendation for user exploration
print("\nSample recommendation for exploration:")
if dnn_recommendations:
    # Pick a random user
    sample_user_id = np.random.choice(list(dnn_recommendations.keys()))
    
    # Get user's genre preferences
    if sample_user_id in user_genre_preferences['userId'].values:
        user_prefs = user_genre_preferences[user_genre_preferences['userId'] == sample_user_id].iloc[0]
        genre_columns = [col for col in user_genre_preferences.columns if col != 'userId']
        
        print(f"\nUser {sample_user_id} Genre Preferences:")
        # Show top 3 liked and disliked genres
        user_prefs_list = [(genre, user_prefs[genre]) for genre in genre_columns]
        liked_genres = sorted(user_prefs_list, key=lambda x: x[1], reverse=True)[:3]
        disliked_genres = sorted(user_prefs_list, key=lambda x: x[1])[:3]
        
        print(f"- Most liked genres: {', '.join([f'{g} ({v:.2f})' for g, v in liked_genres])}")
        print(f"- Most disliked genres: {', '.join([f'{g} ({v:.2f})' for g, v in disliked_genres])}")
    
    # Show recommendations
    recommend_for_user(sample_user_id, dnn_recommendations, data['movie_features'])

print("\n" + "="*80)
print("SUMMARY: COLLABORATIVE FILTERING WITH DNN")
print("="*80)

# Final summary of model performance and characteristics
print("\nModel Characteristics:")
print(f"- Hidden layer sizes: {dnn_hidden_layers}")
print(f"- Dropout rate: {dnn_dropout_rate}")
print(f"- Learning rate: {dnn_learning_rate}")
print(f"- Batch size: {dnn_batch_size}")

# Display dataset statistics
print("\nDataset Statistics:")
print(f"- Training samples: {len(X_train)}")
print(f"- Validation samples: {len(X_val)}")
print(f"- Feature dimensions: {X_train.shape[1]}")
print(f"- Number of users with genre preferences: {len(user_genre_preferences)}")
print(f"- Number of movies with genre features: {len(movie_genre_features)}")

# Show performance metrics
print("\nValidation Performance Metrics:")
print(f"- RMSE: {final_val_rmse:.4f}")
print(f"- MAE: {final_val_mae:.4f}")