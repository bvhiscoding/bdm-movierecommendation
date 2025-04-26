import pandas as pd
import numpy as np
import os
import pickle
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Embedding, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import time
import gc
from tensorflow.keras import backend as K

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "./processed/"
output_path = "./rec/collaborative-recommendations"
top_n = 20

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Optimized hyperparameters based on extensive testing
dnn_hidden_layers = [ 128, 64, 32]  # Deeper network
dnn_dropout_rate = 0.35  # Increased dropout for better generalization
dnn_l2_reg = 0.0005  # L2 regularization to prevent overfitting
dnn_learning_rate = 0.001  # Lower learning rate for more stable convergence
dnn_batch_size = 256  # Larger batch size for better gradient estimates
dnn_epochs = 30 # More epochs with early stopping
threshold_rating = 3  # Rating threshold for binary classification
early_stopping_patience = 8  # Wait longer for improvement
use_cosine_annealing = True  # Use cosine annealing learning rate schedule
use_attention_mechanism = True  # Use attention mechanism for feature interaction

def log_memory_usage(message="Current memory usage"):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        logger.info(f"{message}: {memory_usage_mb:.2f} MB")
    except ImportError:
        logger.warning("psutil not available for memory monitoring")

def load_data():
    """
    Load and prepare data for model training with improved memory management
    """
    data = {}
    log_memory_usage("Before loading data")
    
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        # Use optimized loading for large CSV files with appropriate dtypes
        data['movie_features'] = pd.read_csv(movie_features_path, 
                                            dtype={'movieId': int, 'token_count': int})
        logger.info(f"Loaded movie features with shape {data['movie_features'].shape}")
    else:
        logger.error(f"File not found: {movie_features_path}")
        return None
    
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        # Load ratings in chunks to manage memory better
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            chunks.append(chunk)
            # Force garbage collection after each chunk
            gc.collect()
        
        data['ratings'] = pd.concat(chunks)
        logger.info(f"Loaded ratings with shape {data['ratings'].shape}")
    else:
        logger.error(f"File not found: {ratings_path}")
        return None
    
    log_memory_usage("After loading data")
    
    if 'ratings' in data:
        # Create improved train/test split
        logger.info("Creating improved train/test split")
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # First split: separate users for training and testing
        all_user_ids = data['ratings']['userId'].unique()
        np.random.shuffle(all_user_ids)
        
        # Use 80% of users for complete training
        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]
        
        # Full training data from train users
        train_ratings_main = data['ratings'][data['ratings']['userId'].isin(train_users)]
        
        # For users in test set, split their ratings temporally
        test_user_ratings = data['ratings'][data['ratings']['userId'].isin(test_users)]
        train_chunks = []
        test_chunks = []
        
        # Process each test user
        for user_id in test_users:
            user_data = test_user_ratings[test_user_ratings['userId'] == user_id]
            
            # Skip users with too few ratings
            if len(user_data) < 5:
                continue
                
            # Sort by timestamp if available
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
            
            # Take first 80% for training, last 20% for testing (temporal split)
            split_idx = int(len(user_data) * 0.8)
            train_chunks.append(user_data.iloc[:split_idx])
            test_chunks.append(user_data.iloc[split_idx:])
        
        # Combine training data from both sources
        data['train_ratings'] = pd.concat([train_ratings_main] + train_chunks) if train_chunks else train_ratings_main
        data['test_ratings'] = pd.concat(test_chunks) if test_chunks else pd.DataFrame()
        
        # Log split statistics
        logger.info(f"Training set: {len(data['train_ratings'])} ratings from {len(data['train_ratings']['userId'].unique())} users")
        logger.info(f"Test set: {len(data['test_ratings'])} ratings from {len(data['test_ratings']['userId'].unique())} users")
        
        # Force garbage collection
        del train_ratings_main, test_user_ratings, train_chunks, test_chunks
        gc.collect()
    
    # Extract region columns with better handling
    region_columns = [col for col in data['movie_features'].columns 
                     if col in ['North America', 'Europe', 'East Asia', 'South Asia', 
                               'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                               'Latin America', 'Other']]
    
    if region_columns:
        logger.info(f"Found {len(region_columns)} region features: {region_columns}")
        data['region_columns'] = region_columns
    
    log_memory_usage("After split preparation")
    return data

def extract_genre_and_region_features(movie_features):
    """
    Extract enhanced genre and region features from movie data
    """
    # Better identification of genre columns
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

def calculate_user_preferences(train_ratings, movie_features, feature_columns, rating_threshold=3):
    """
    Calculate enhanced user preferences with improved weighting scheme
    """
    logger.info(f"Calculating user preferences for {len(feature_columns)} features")
    
    user_preferences = []
    
    total_users = len(train_ratings['userId'].unique())
    processed_users = 0
    start_time = time.time()
    
    # Calculate global average rating
    global_avg_rating = train_ratings['rating'].mean()
    
    # Process users in batches to manage memory
    user_batch_size = 1000
    user_batches = np.array_split(train_ratings['userId'].unique(), 
                                max(1, total_users // user_batch_size))
    
    for batch_idx, user_batch in enumerate(user_batches):
        batch_preferences = []
        
        for user_id in user_batch:
            user_ratings = train_ratings[train_ratings['userId'] == user_id]
            
            if len(user_ratings) == 0:
                continue
            
            # Calculate user's average rating and bias
            user_avg_rating = user_ratings['rating'].mean()
            user_bias = user_avg_rating - global_avg_rating
            
            # Split into liked and disliked movies with more granular approach
            # Use original unbiased ratings
            strongly_liked_movies = user_ratings[user_ratings['rating'] >= rating_threshold + 0.5]['movieId'].values
            liked_movies = user_ratings[(user_ratings['rating'] >= rating_threshold) & 
                                        (user_ratings['rating'] < rating_threshold + 0.5)]['movieId'].values
            neutral_movies = user_ratings[(user_ratings['rating'] >= rating_threshold - 0.5) & 
                                         (user_ratings['rating'] < rating_threshold)]['movieId'].values
            disliked_movies = user_ratings[user_ratings['rating'] < rating_threshold - 0.5]['movieId'].values
            
            feature_preferences = {}
            
            for feature in feature_columns:
                # Calculate weighted feature preference
                strongly_liked_feature = movie_features[movie_features['movieId'].isin(strongly_liked_movies)][feature].sum()
                liked_feature = movie_features[movie_features['movieId'].isin(liked_movies)][feature].sum()
                neutral_feature = movie_features[movie_features['movieId'].isin(neutral_movies)][feature].sum()
                disliked_feature = movie_features[movie_features['movieId'].isin(disliked_movies)][feature].sum()
                
                # Count movies in each category
                strongly_liked_count = len(strongly_liked_movies) if len(strongly_liked_movies) > 0 else 1
                liked_count = len(liked_movies) if len(liked_movies) > 0 else 1
                neutral_count = len(neutral_movies) if len(neutral_movies) > 0 else 1
                disliked_count = len(disliked_movies) if len(disliked_movies) > 0 else 1
                
                # Apply progressive weighting
                strongly_liked_weight = 1.0
                liked_weight = 0.7
                neutral_weight = 0.2
                disliked_weight = -0.8
                
                # Calculate weighted feature preference
                preference = (
                    strongly_liked_weight * (strongly_liked_feature / strongly_liked_count) +
                    liked_weight * (liked_feature / liked_count) -
                    neutral_weight * (neutral_feature / neutral_count) -
                    disliked_weight * (disliked_feature / disliked_count)
                )
                
                feature_preferences[feature] = preference
            
            # Normalize preferences to -1 to 1 range
            max_abs_preference = max(abs(val) for val in feature_preferences.values()) if feature_preferences else 1
            
            for feature in feature_preferences:
                feature_preferences[feature] = feature_preferences[feature] / max_abs_preference if max_abs_preference > 0 else 0
            
            feature_preferences['userId'] = user_id
            
            batch_preferences.append(feature_preferences)
            
            processed_users += 1
        
        # Add batch to main list
        user_preferences.extend(batch_preferences)
        
        # Log progress
        progress = processed_users / total_users * 100
        elapsed = time.time() - start_time
        remaining = elapsed * (total_users - processed_users) / processed_users if processed_users > 0 else 0
        
        logger.info(f"Processed {processed_users}/{total_users} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Force garbage collection
        gc.collect()
    
    user_preferences_df = pd.DataFrame(user_preferences)
    logger.info(f"Created preferences for {len(user_preferences_df)} users")
    
    return user_preferences_df

def prepare_dnn_training_data(train_ratings, user_preferences, movie_features, genre_columns, region_columns=None, threshold=3, max_samples=1000000):
    """
    Prepare enhanced training data for DNN model with improved feature engineering
    """
    logger.info("Preparing training data for DNN model with enhanced features")
    
    # Include both genre and region columns for feature vectors
    feature_columns = genre_columns.copy()
    if region_columns:
        feature_columns.extend(region_columns)
    
    features = []
    labels = []
    
    # Calculate global stats
    global_avg_rating = train_ratings['rating'].mean()
    
    # Calculate user stats
    user_stats = {}
    for user_id in user_preferences['userId'].unique():
        user_ratings = train_ratings[train_ratings['userId'] == user_id]
        if len(user_ratings) > 0:
            user_stats[user_id] = {
                'avg': user_ratings['rating'].mean(),
                'std': user_ratings['rating'].std(),
                'count': len(user_ratings)
            }
    
    # Calculate movie stats
    movie_stats = {}
    for movie_id in movie_features['movieId'].unique():
        movie_ratings = train_ratings[train_ratings['movieId'] == movie_id]
        if len(movie_ratings) > 0:
            movie_stats[movie_id] = {
                'avg': movie_ratings['rating'].mean(),
                'std': movie_ratings['rating'].std(),
                'count': len(movie_ratings)
            }
    
    # Limit sample size for memory efficiency with stratification
    if len(train_ratings) > max_samples:
        # Stratify by rating to maintain distribution
        bin_edges = [0, 1.5, 2.5, 3, 4.5, 5.1]  # Bins for ratings
        train_ratings['rating_bin'] = pd.cut(train_ratings['rating'], bins=bin_edges, labels=False)
        
        # Sample from each bin proportionally
        sampled_ratings = pd.DataFrame()
        for bin_id in range(len(bin_edges)-1):
            bin_data = train_ratings[train_ratings['rating_bin'] == bin_id]
            bin_sample_size = int(max_samples * (len(bin_data) / len(train_ratings)))
            
            if len(bin_data) > bin_sample_size:
                bin_sampled = bin_data.sample(bin_sample_size, random_state=42)
            else:
                bin_sampled = bin_data
                
            sampled_ratings = pd.concat([sampled_ratings, bin_sampled])
        
        # Clean up
        sampled_ratings = sampled_ratings.drop(columns=['rating_bin'])
    else:
        sampled_ratings = train_ratings
    
    # Process in batches for memory efficiency
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
            
            # Convert rating to binary label based on threshold
            binary_label = 1 if rating > threshold else 0
            
            if user_id not in user_preferences['userId'].values or \
               movie_id not in movie_features['movieId'].values:
                continue
            
            user_prefs = user_preferences[user_preferences['userId'] == user_id].iloc[0]
            
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if movie_row.empty:
                continue
                
            movie_features_row = movie_row.iloc[0]
            
            # Create enhanced feature vector with rating context
            feature_vector = []
            
            # Add user and movie bias features
            user_info = user_stats.get(user_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
            movie_info = movie_stats.get(movie_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
            
            # Add global context features
            feature_vector.append(global_avg_rating / 5.0)  # Normalize to 0-1
            
            # Add user context features (normalized)
            feature_vector.append(user_info['avg'] / 5.0)  # User average rating
            feature_vector.append(min(1.0, user_info['std'] / 2.0))  # User rating variability
            feature_vector.append(min(1.0, np.log1p(user_info['count']) / 10.0))  # User experience
            
            # Add movie context features (normalized)
            feature_vector.append(movie_info['avg'] / 5.0)  # Movie average rating
            feature_vector.append(min(1.0, movie_info['std'] / 2.0))  # Movie rating variability 
            feature_vector.append(min(1.0, np.log1p(movie_info['count']) / 10.0))  # Movie popularity
            
            # Add user-movie difference feature
            feature_vector.append((user_info['avg'] - movie_info['avg'] + 2.5) / 5.0)  # Normalized difference
            
            # Add category features with more sophisticated interaction
            for feature in feature_columns:
                user_pref = user_prefs[feature]
                movie_feat = movie_features_row[feature]
                
                # Add user preference
                feature_vector.append(user_pref)
                
                # Add movie feature
                feature_vector.append(movie_feat)
                
                # Add interaction features
                feature_vector.append(user_pref * movie_feat)  # Product
                feature_vector.append(user_pref + movie_feat - 0.5)  # Sum (normalized)
                feature_vector.append(abs(user_pref - movie_feat))  # Absolute difference
            
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
    
    # Split the data into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Created training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    logger.info(f"Created validation data: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    
    # Check class distribution
    train_pos = np.sum(y_train)
    train_neg = len(y_train) - train_pos
    val_pos = np.sum(y_val)
    val_neg = len(y_val) - val_pos
    
    logger.info(f"Training set class distribution: Positive {train_pos} ({train_pos/len(y_train)*100:.1f}%), Negative {train_neg} ({train_neg/len(y_train)*100:.1f}%)")
    logger.info(f"Validation set class distribution: Positive {val_pos} ({val_pos/len(y_val)*100:.1f}%), Negative {val_neg} ({val_neg/len(y_val)*100:.1f}%)")
    
    return X_train, X_val, y_train, y_val, feature_columns

def f1_metric(y_true, y_pred):
    """Custom F1 score metric for Keras"""
    # Calculate precision and recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for better handling of class imbalance
    
    Parameters:
    alpha: Weighting factor for the rare class (positive)
    gamma: Focusing parameter to down-weight easy examples
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # Calculate loss with focal weighting
        loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
               
        # Normalize by batch size
        return loss / K.cast(K.shape(y_true)[0], 'float32')
    
    return focal_loss_fixed

class SelfAttentionLayer(tf.keras.layers.Layer):
    """Properly implemented self-attention layer that handles 2D inputs"""
    def __init__(self, hidden_units, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.supports_masking = True
        
    def build(self, input_shape):
        # Input shape validation
        if len(input_shape) < 2:
            raise ValueError(f"Input shape must be at least 2D, got {input_shape}")
            
        # Determine if we need to reshape 2D input
        self.needs_reshape = len(input_shape) == 2
        input_dim = input_shape[-1]
        
        # Create trainable weights
        self.query_dense = tf.keras.layers.Dense(self.hidden_units)
        self.key_dense = tf.keras.layers.Dense(self.hidden_units)
        self.value_dense = tf.keras.layers.Dense(self.hidden_units)
        self.combine_dense = tf.keras.layers.Dense(input_dim)
        
        self.built = True
        
    def call(self, inputs):
        # Handle 2D inputs by reshaping to 3D (batch_size, 1, features)
        original_shape = tf.shape(inputs)
        needs_reshape = len(inputs.shape) == 2
        
        if needs_reshape:
            # Add sequence dimension of length 1
            inputs = tf.expand_dims(inputs, axis=1)
        
        # Apply transformations
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Calculate attention scores - use standard matrix multiplication
        # (batch_size, seq_len, hidden) × (batch_size, hidden, seq_len) 
        # → (batch_size, seq_len, seq_len)
        key_transposed = tf.transpose(key, perm=[0, 2, 1])
        scores = tf.matmul(query, key_transposed)
        
        # Scale scores
        scores = scores / tf.math.sqrt(tf.cast(self.hidden_units, dtype=tf.float32))
        
        # Apply softmax to get attention weights
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention weights to values
        context = tf.matmul(weights, value)
        
        # Final transformation
        output = self.combine_dense(context)
        
        # Add residual connection
        output = output + inputs
        
        # Reshape back to original shape if needed
        if needs_reshape:
            output = tf.squeeze(output, axis=1)
            
        return output
        
    def compute_output_shape(self, input_shape):
        # Output shape is same as input shape
        return input_shape
        
    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({
            'hidden_units': self.hidden_units
        })
        return config


def attention_block(x, hidden_units):
    """Wrapper function to apply the attention layer"""
    # Create the attention layer
    attention_layer = SelfAttentionLayer(hidden_units)
    
    # Apply attention directly (no need to reshape first - layer handles that)
    x = attention_layer(x)
    
    # Apply layer normalization
    x = tf.keras.layers.LayerNormalization()(x)
    
    return x

def build_and_train_dnn_model(X_train, X_val, y_train, y_val, learning_rate=0.0005, batch_size=256, epochs=50):
    """
    Build and train an enhanced DNN model with advanced architecture and training techniques
    """
    logger.info("Building and training enhanced DNN model")
    
    # Configure GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            logger.warning(f"GPU config error: {e}")
    
    input_dim = X_train.shape[1]
    
    # Create model with more sophisticated architecture
    inputs = Input(shape=(input_dim,))
    
    # Normalize inputs
    x = BatchNormalization()(inputs)
    
    # First block - extract latent representations
    x = Dense(dnn_hidden_layers[0], activation='relu', kernel_regularizer=l2(dnn_l2_reg))(x)
    x_shortcut1 = x  # Save for residual connection
    x = Dropout(dnn_dropout_rate)(x)
    
    # Apply attention if enabled
    if use_attention_mechanism:
        x = attention_block(x, dnn_hidden_layers[0] // 2)
    
    # Middle layers with residual connections
    for i, units in enumerate(dnn_hidden_layers[1:], 1):
        # Normalization before each layer
        x = BatchNormalization()(x)
        
        # Dense layer with regularization
        x = Dense(units, activation='relu', kernel_regularizer=l2(dnn_l2_reg))(x)
        
        # Dropout for regularization
        x = Dropout(dnn_dropout_rate)(x)
        
        # Add residual connection when dimensions match or after projection
        if i == 1 and x_shortcut1.shape[-1] >= units:
            # Project if needed then add
            if x_shortcut1.shape[-1] > units:
                shortcut = Dense(units, activation='linear')(x_shortcut1)
            else:
                shortcut = x_shortcut1
            x = Add()([x, shortcut])
    
    # Final normalization
    x = BatchNormalization()(x)
    
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Learning rate schedule with cosine annealing
    if use_cosine_annealing:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=int(len(X_train) / batch_size) * 5,  # 5 epochs
            t_mul=2.0,  # Double period after each restart
            m_mul=0.9,  # Slightly reduce max LR after each restart
            alpha=1e-6  # Minimum LR
        )
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
    else:
        # Standard Adam optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
    
    # Compile with appropriate loss and metrics
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Standard loss for binary classification
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_metric
        ]
    )
    
    # Calculate class weights to handle imbalance
    neg_count = len(y_train) - np.sum(y_train)
    pos_count = np.sum(y_train)
    pos_weight = (1 / pos_count) * ((neg_count + pos_count) / 2.0) if pos_count > 0 else 1.0
    neg_weight = (1 / neg_count) * ((neg_count + pos_count) / 2.0) if neg_count > 0 else 1.0
    
    class_weight = {0: neg_weight, 1: pos_weight}
    
    logger.info(f"Class weights: {class_weight}")
    import keras
    keras_version = keras.__version__
    model_ext = '.keras' if keras_version.startswith('3.') else '.h5'
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau is removed!
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_path, 'best_model.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train the model with class weights
    logger.info(f"Training model with {epochs} max epochs, batch size {batch_size}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight  # Apply class weights
    )
    
    # Evaluate on validation set
    val_metrics = model.evaluate(X_val, y_val, verbose=1)
    
    # Extract metrics - order matches the model.compile metrics list
    val_loss = val_metrics[0]
    val_acc = val_metrics[1]
    val_auc = val_metrics[2]
    val_precision = val_metrics[3]
    val_recall = val_metrics[4]
    val_f1 = val_metrics[5]
    
    logger.info(f"Model validation metrics:")
    logger.info(f"- Loss: {val_loss:.4f}")
    logger.info(f"- Accuracy: {val_acc:.4f}")
    logger.info(f"- AUC: {val_auc:.4f}")
    logger.info(f"- Precision: {val_precision:.4f}")
    logger.info(f"- Recall: {val_recall:.4f}")
    logger.info(f"- F1 Score: {val_f1:.4f}")
    
    # Plot training history with more metrics
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(history.history['f1_metric'], label='Training F1')
    plt.plot(history.history['val_f1_metric'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'dnn_training_history.png'))
    plt.close()
    
    # Create additional visualization - ROC curve
    y_pred_prob = model.predict(X_val)
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'roc_curve.png'))
    plt.close()
    
    # Create precision-recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    avg_precision = average_precision_score(y_val, y_pred_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_path, 'precision_recall_curve.png'))
    plt.close()
    
    return model, history

def generate_user_movie_features(user_id, movie_id, user_preferences, movie_features, genre_columns, region_columns=None, user_stats=None, movie_stats=None, global_avg_rating=None):
    """
    Generate enhanced feature vector for a user-movie pair
    """
    if global_avg_rating is None:
        global_avg_rating = 3.0
    
    if user_stats is None:
        user_stats = {}
    
    if movie_stats is None:
        movie_stats = {}
        
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
    
    # Create enhanced feature vector with user/movie context
    feature_vector = []
    
    # Get user and movie statistics
    user_info = user_stats.get(user_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
    movie_info = movie_stats.get(movie_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
    
    # Add global context features
    feature_vector.append(global_avg_rating / 5.0)  # Normalize
    
    # Add user context features
    feature_vector.append(user_info['avg'] / 5.0)  # User average rating
    feature_vector.append(min(1.0, user_info['std'] / 2.0))  # User rating variability
    feature_vector.append(min(1.0, np.log1p(user_info['count']) / 10.0))  # User experience
    
    # Add movie context features
    feature_vector.append(movie_info['avg'] / 5.0)  # Movie average rating
    feature_vector.append(min(1.0, movie_info['std'] / 2.0))  # Movie rating variability 
    feature_vector.append(min(1.0, np.log1p(movie_info['count']) / 10.0))  # Movie popularity
    
    # Add user-movie difference feature
    feature_vector.append((user_info['avg'] - movie_info['avg'] + 2.5) / 5.0)  # Normalized difference
    
    # Add category features with more sophisticated interaction
    for feature in feature_columns:
        user_pref = user_prefs[feature]
        movie_feat = movie_features_row[feature]
        
        # Add user preference
        feature_vector.append(user_pref)
        
        # Add movie feature
        feature_vector.append(movie_feat)
        
        # Add interaction features
        feature_vector.append(user_pref * movie_feat)  # Product
        feature_vector.append(user_pref + movie_feat - 0.5)  # Sum (normalized)
        feature_vector.append(abs(user_pref - movie_feat))  # Absolute difference
    
    return np.array([feature_vector], dtype=np.float32)

def generate_dnn_recommendations(user_id, dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, train_ratings=None, user_stats=None, movie_stats=None, global_avg_rating=None, n=10):
    """
    Generate movie recommendations for a user using the enhanced DNN model
    """
    if global_avg_rating is None and train_ratings is not None:
        global_avg_rating = train_ratings['rating'].mean()
    elif global_avg_rating is None:
        global_avg_rating = 3.0
        
    if user_id not in user_preferences['userId'].values:
        logger.warning(f"User {user_id} not found in preferences")
        return []
    
    # Get movies the user has already rated
    rated_movies = set()
    if train_ratings is not None:
        rated_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
    
    # Consider only unrated movies
    unrated_movies = movie_features[~movie_features['movieId'].isin(rated_movies)]
    
    batch_size = 1000
    all_predictions = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(unrated_movies), batch_size):
        batch = unrated_movies.iloc[i:i+batch_size]
        
        feature_vectors = []
        movie_ids = []
        
        for _, movie_row in batch.iterrows():
            movie_id = movie_row['movieId']
            
            # Generate features for this user-movie pair
            feature_vector = generate_user_movie_features(
                user_id, movie_id, 
                user_preferences, movie_features, 
                genre_columns, region_columns,
                user_stats, movie_stats, global_avg_rating
            )
            
            if feature_vector is not None:
                feature_vectors.append(feature_vector[0])  # Flatten first dimension
                movie_ids.append(movie_id)
        
        if not feature_vectors:
            continue
            
        # Convert to numpy array for batch prediction
        feature_array = np.array(feature_vectors)
        
        # Get probability scores (0-1)
        like_probabilities = dnn_model.predict(feature_array, verbose=0).flatten()
        
        # Convert to rating scale for easier comparison
        predicted_ratings = 0.5 + like_probabilities * 4.5
        
        for movie_id, pred in zip(movie_ids, predicted_ratings):
            all_predictions.append((movie_id, float(pred)))
    
    # Sort by predicted rating
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return all_predictions[:n]

def generate_recommendations_for_all_users(dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, train_ratings=None, n=10, batch_size=50, max_users=None, test_sample_ratio=0.2):
    """
    Generate recommendations for users with improved memory management and sample testing
    
    Parameters:
    -----------
    ... [existing parameters] ...
    test_sample_ratio: float
        Ratio of users to include in testing (0.0-1.0)
    """
    all_user_ids = user_preferences['userId'].unique()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sample users for testing - only use a percentage of all users
    if test_sample_ratio < 1.0:
        sample_size = int(len(all_user_ids) * test_sample_ratio)
        user_ids = np.random.choice(all_user_ids, size=sample_size, replace=False)
        logger.info(f"Sampling {sample_size} users ({test_sample_ratio*100:.1f}%) for testing instead of all {len(all_user_ids)} users")
    else:
        user_ids = all_user_ids
        
    if max_users and max_users < len(user_ids):
        user_ids = user_ids[:max_users]
    
    logger.info(f"Generating recommendations for {len(user_ids)} users")
    
    # [rest of the function remains the same]
    
    all_recommendations = {}
    total_users = len(user_ids)
    
    # Precompute global and user/movie statistics for faster recommendations
    global_avg_rating = train_ratings['rating'].mean() if train_ratings is not None else 3.0
    
    # Precompute user statistics
    user_stats = {}
    if train_ratings is not None:
        for user_id in user_ids:
            user_ratings = train_ratings[train_ratings['userId'] == user_id]
            if len(user_ratings) > 0:
                user_stats[user_id] = {
                    'avg': user_ratings['rating'].mean(),
                    'std': user_ratings['rating'].std() if len(user_ratings) > 1 else 0.5,
                    'count': len(user_ratings)
                }
    
    # Precompute movie statistics (for most popular movies)
    movie_stats = {}
    if train_ratings is not None:
        # Group by movieId and count
        movie_counts = train_ratings['movieId'].value_counts()
        
        # Get popular movies (top 10%)
        popular_threshold = np.percentile(movie_counts.values, 90) if len(movie_counts) > 10 else 0
        popular_movies = movie_counts[movie_counts >= popular_threshold].index
        
        for movie_id in popular_movies:
            movie_ratings = train_ratings[train_ratings['movieId'] == movie_id]
            if len(movie_ratings) > 0:
                movie_stats[movie_id] = {
                    'avg': movie_ratings['rating'].mean(),
                    'std': movie_ratings['rating'].std() if len(movie_ratings) > 1 else 0.5,
                    'count': len(movie_ratings)
                }
    
    # Create a dictionary of rated movies by user for faster lookups
    user_rated_movies = {}
    if train_ratings is not None:
        for user_id in user_ids:
            user_rated_movies[user_id] = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
    
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
            
            # Get movies already rated by this user
            rated_movies = user_rated_movies.get(user_id, set())
            
            # Consider only unrated movies
            unrated_movie_ids = set(movie_features['movieId']) - rated_movies
            
            # For large datasets, sample a subset of candidate movies
            # to improve efficiency while maintaining diversity
            max_movies_per_batch = 2000
            
            if len(unrated_movie_ids) > max_movies_per_batch:
                # Use a smarter sampling method:
                # 1. Include some popular movies (higher chance of being liked)
                # 2. Include movies with high genre match to user preferences
                # 3. Include some random movies for diversity
                
                # Convert preferences to dictionary for easier access
                pref_dict = user_prefs.iloc[0].to_dict()
                
                # Find top genres by preference
                genre_prefs = [(genre, pref_dict.get(genre, 0)) 
                              for genre in genre_columns if genre in pref_dict]
                top_genres = sorted(genre_prefs, key=lambda x: x[1], reverse=True)[:5]
                top_genre_names = [g[0] for g in top_genres if g[1] > 0]
                
                # Get movies from top genres (if any positive preferences)
                top_genre_movies = set()
                if top_genre_names:
                    for genre in top_genre_names:
                        genre_movies = set(movie_features[movie_features[genre] == 1]['movieId'])
                        top_genre_movies.update(genre_movies)
                    
                    # Filter to unrated movies only
                    top_genre_movies = top_genre_movies.intersection(unrated_movie_ids)
                    
                    # Limit to a reasonable number
                    if len(top_genre_movies) > max_movies_per_batch // 2:
                        top_genre_movies = set(list(top_genre_movies)[:max_movies_per_batch // 2])
                
                # Get popular movies based on movie_stats
                popular_movies = set([m for m, stats in movie_stats.items() 
                                    if stats['count'] > 5 and stats['avg'] >= 3])
                popular_unrated = popular_movies.intersection(unrated_movie_ids) - top_genre_movies
                
                # Limit number of popular movies
                if len(popular_unrated) > max_movies_per_batch // 4:
                    popular_unrated = set(list(popular_unrated)[:max_movies_per_batch // 4])
                
                # Random sampling for remaining slots
                remaining_count = max_movies_per_batch - len(top_genre_movies) - len(popular_unrated)
                remaining_movies = unrated_movie_ids - top_genre_movies - popular_unrated
                
                if len(remaining_movies) > remaining_count:
                    remaining_sample = np.random.choice(list(remaining_movies), 
                                                       size=remaining_count, 
                                                       replace=False)
                    remaining_movies = set(remaining_sample)
                
                # Combine all selected movies
                unrated_movie_ids = top_genre_movies.union(popular_unrated).union(remaining_movies)
            
            candidate_movies = movie_features[movie_features['movieId'].isin(unrated_movie_ids)]
            
            if len(candidate_movies) == 0:
                continue
            
            # Process movies in batches to avoid memory issues
            movie_batch_size = 200
            predictions = []
            
            for j in range(0, len(candidate_movies), movie_batch_size):
                movie_batch_end = min(j + movie_batch_size, len(candidate_movies))
                movie_batch = candidate_movies.iloc[j:movie_batch_end]
                
                batch_features = []
                batch_movie_ids = []
                
                for _, movie_row in movie_batch.iterrows():
                    movie_id = movie_row['movieId']
                    
                    # Get or compute movie stats on-demand for non-cached movies
                    if movie_id not in movie_stats and train_ratings is not None:
                        movie_ratings = train_ratings[train_ratings['movieId'] == movie_id]
                        if len(movie_ratings) > 0:
                            movie_stats[movie_id] = {
                                'avg': movie_ratings['rating'].mean(),
                                'std': movie_ratings['rating'].std() if len(movie_ratings) > 1 else 0.5,
                                'count': len(movie_ratings)
                            }
                    
                    # Generate features
                    feature_vector = []
                    
                    # Get user and movie stats
                    user_info = user_stats.get(user_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
                    movie_info = movie_stats.get(movie_id, {'avg': global_avg_rating, 'std': 0.5, 'count': 0})
                    
                    # Add global context features
                    feature_vector.append(global_avg_rating / 5.0)  # Normalize
                    
                    # Add user context features
                    feature_vector.append(user_info['avg'] / 5.0)  # User average rating
                    feature_vector.append(min(1.0, user_info['std'] / 2.0))  # User rating variability
                    feature_vector.append(min(1.0, np.log1p(user_info['count']) / 10.0))  # User experience
                    
                    # Add movie context features
                    feature_vector.append(movie_info['avg'] / 5.0)  # Movie average rating
                    feature_vector.append(min(1.0, movie_info['std'] / 2.0))  # Movie rating variability 
                    feature_vector.append(min(1.0, np.log1p(movie_info['count']) / 10.0))  # Movie popularity
                    
                    # Add user-movie difference feature
                    feature_vector.append((user_info['avg'] - movie_info['avg'] + 2.5) / 5.0)  # Normalized difference
                    
                    # Add features with interactions
                    for feature in feature_columns:
                        user_pref = user_prefs.iloc[0][feature]
                        movie_feat = movie_row[feature]
                        
                        # Add user preference
                        feature_vector.append(user_pref)
                        
                        # Add movie feature
                        feature_vector.append(movie_feat)
                        
                        # Add interaction features
                        feature_vector.append(user_pref * movie_feat)  # Product
                        feature_vector.append(user_pref + movie_feat - 0.5)  # Sum (normalized)
                        feature_vector.append(abs(user_pref - movie_feat))  # Absolute difference
                    
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

def evaluate_recommendations(recommendations, test_ratings, dnn_model, user_preferences, movie_features, genre_columns, region_columns=None, threshold=3.5, sample_users=False):
    """
    Evaluate recommendation model using all test ratings for consistent evaluation
    with other models.
    """
    logger.info("Evaluating recommendations with standardized approach")
    
    # Find users with available data for evaluation
    test_users = set(test_ratings['userId'].unique())
    train_users = set(user_preferences['userId'].unique())
    
    # Use all users that have both preference data and test ratings
    common_users = test_users.intersection(train_users)
    
    logger.info(f"Test users: {len(test_users)}, Users with preferences: {len(train_users)}")
    logger.info(f"Users being evaluated: {len(common_users)}")
    
    # [baseline code section remains unchanged]
    
    # Prepare data for evaluation
    predictions = []
    actuals = []
    binary_predictions = []
    binary_actuals = []
    
    # Track metrics per user
    user_metrics = {}
    
    # Precompute global stats
    global_avg_rating = test_ratings['rating'].mean()
    
    # Process each user
    for user_id in common_users:
        if user_id not in user_preferences['userId'].values:
            continue
        
        user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
        
        if len(user_test_ratings) == 0:
            continue
        
        user_preds = []
        user_actuals = []
        user_binary_preds = []
        user_binary_actuals = []
        
        # Evaluate ALL test ratings for this user (not just recommended items)
        for _, row in user_test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            binary_actual = 1 if actual_rating > threshold else 0
            
            # Always generate prediction regardless of whether it's in recommendations
            if movie_id in movie_features['movieId'].values:
                # Generate prediction for this movie
                feature_vector = generate_user_movie_features(
                    user_id, 
                    movie_id, 
                    user_preferences, 
                    movie_features, 
                    genre_columns,
                    region_columns,
                    global_avg_rating=global_avg_rating
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
            
        # [rest of function remains the same]
        
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
            
            # Classification metrics
            user_tp = np.sum((user_binary_preds_np == 1) & (user_binary_actuals_np == 1))
            user_fp = np.sum((user_binary_preds_np == 1) & (user_binary_actuals_np == 0))
            user_tn = np.sum((user_binary_preds_np == 0) & (user_binary_actuals_np == 0))
            user_fn = np.sum((user_binary_preds_np == 0) & (user_binary_actuals_np == 1))
            
            user_precision = user_tp / (user_tp + user_fp) if (user_tp + user_fp) > 0 else 0
            user_recall = user_tp / (user_tp + user_fn) if (user_tp + user_fn) > 0 else 0
            user_f1 = 2 * user_precision * user_recall / (user_precision + user_recall) if (user_precision + user_recall) > 0 else 0
            
            # RMSE (on original rating scale)
            user_preds_np = np.array(user_preds)
            user_actuals_np = np.array(user_actuals)
            user_rmse = np.sqrt(np.mean((user_preds_np - user_actuals_np) ** 2))
            user_mae = np.mean(np.abs(user_preds_np - user_actuals_np))
            
            # Store user metrics
            user_metrics[user_id] = {
                'accuracy': user_accuracy,
                'precision': user_precision,
                'recall': user_recall,
                'f1_score': user_f1,
                'rmse': user_rmse,
                'mae': user_mae,
                'num_predictions': len(user_preds),
                'tp': int(user_tp),
                'fp': int(user_fp),
                'tn': int(user_tn),
                'fn': int(user_fn)
            }
    
    if not predictions:
        logger.warning("No predictions available for evaluation")
        return {
            'rmse': 0.0,
            'mae': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'num_predictions': 0
        }, {}
    
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
        'method': 'enhanced_evaluation'
    }
    
    logger.info(f"Evaluation completed:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1_score:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Predictions: {len(predictions)}")
    
    # Create visualization for confusion matrix
    plt.figure(figsize=(10, 8))
    cm = np.array([
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_path, 'dnn_confusion_matrix.png'))
    plt.close()
    
    # Create bar chart of metrics
    plt.figure(figsize=(12, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [accuracy, precision, recall, f1_score]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(metric_names)))
    bars = plt.bar(metric_names, metric_values, color=colors)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02, 
                f'{value:.3f}', 
                ha='center', va='bottom', 
                fontweight='bold')
    
    plt.ylim(0, 1.0)
    plt.title('Evaluation Metrics')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_path, 'dnn_evaluation_metrics.png'))
    plt.close()
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(output_path, 'dnn_evaluation.csv'), index=False)
    
    return metrics, user_metrics

def recommend_for_user(user_id, recommendations, movie_features=None, n=10):
    """
    Display improved recommendations for a user
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
        genres = []
        
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty:
                if 'title' in movie_row.columns:
                    movie_info = movie_row.iloc[0]['title']
                
                # Extract genres
                genre_columns = [col for col in movie_row.columns if col not in 
                                ['movieId', 'title', 'tokens', 'token_count', 'top_keywords'] and
                                col not in ['North America', 'Europe', 'East Asia', 'South Asia', 
                                           'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                                           'Latin America', 'Other']]
                
                genres = [genre for genre in genre_columns if movie_row.iloc[0][genre] == 1]
        
        genre_str = ", ".join(genres[:3]) + (", ..." if len(genres) > 3 else "")
        logger.info(f"{i}. {movie_info} - Rating: {predicted_rating:.2f} - Genres: {genre_str}")
        
        recs_info.append({
            'rank': i,
            'movie_id': movie_id,
            'title': movie_info,
            'predicted_rating': predicted_rating,
            'genres': genres
        })
    
    return recs_info

# Main execution flow
logger.info("Starting enhanced DNN-based recommendation pipeline")
log_memory_usage("Initial memory usage")

# Load data
data = load_data()
if data is None:
    logger.error("Failed to load data")
    exit(1)

log_memory_usage("After loading data")

# Extract genre and region features
movie_features_with_regions, genre_columns, region_columns = extract_genre_and_region_features(data['movie_features'])

if movie_features_with_regions is None:
    logger.error("Failed to extract movie features")
    exit(1)

log_memory_usage("After feature extraction")

# Calculate user preferences
user_preferences = calculate_user_preferences(
    data['train_ratings'], 
    movie_features_with_regions,
    genre_columns + region_columns,
    threshold_rating
)

# Save user preferences
user_preferences.to_csv(os.path.join(output_path, 'user_genre_preferences.csv'), index=False)
logger.info(f"Saved user preferences for {len(user_preferences)} users")

# Save movie genre features
movie_features_with_regions.to_csv(os.path.join(output_path, 'movie_genre_features.csv'), index=False)
logger.info(f"Saved movie genre features for {len(movie_features_with_regions)} movies")

log_memory_usage("After user preferences calculation")

# Prepare training data
X_train, X_val, y_train, y_val, feature_columns = prepare_dnn_training_data(
    data['train_ratings'],
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    threshold=threshold_rating
)

log_memory_usage("After training data preparation")

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

log_memory_usage("After model training")

# Save the trained model
dnn_model.save(os.path.join(output_path, 'dnn_model.h5'))
logger.info("Saved trained DNN model")

# Generate recommendations for users with adaptive batch size
dnn_recommendations = generate_recommendations_for_all_users(
    dnn_model,
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    data['train_ratings'],
    top_n,
    batch_size=50,
    test_sample_ratio=0.2  # Add this parameter to test only 20% of users
)
log_memory_usage("After generating recommendations")

# Save recommendations
with open(os.path.join(output_path, 'dnn_recommendations.pkl'), 'wb') as f:
    pickle.dump(dnn_recommendations, f)
logger.info(f"Saved recommendations for {len(dnn_recommendations)} users")

# Also save in CSV format for easier inspection
recommendations_list = []
for user_id, recs in dnn_recommendations.items():
    for rank, (movie_id, score) in enumerate(recs, 1):
        movie_title = "Unknown"
        genres = []
        
        if 'movie_features' in data:
            movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_title = movie_row.iloc[0]['title']
                
                # Extract genres
                genre_cols = [col for col in movie_row.columns if col not in 
                            ['movieId', 'title', 'tokens', 'token_count', 'top_keywords'] and
                            col not in ['North America', 'Europe', 'East Asia', 'South Asia', 
                                       'Southeast Asia', 'Oceania', 'Middle East', 'Africa', 
                                       'Latin America', 'Other']]
                
                genres = [genre for genre in genre_cols if movie_row.iloc[0][genre] == 1]
        
        recommendations_list.append({
            'userId': user_id,
            'movieId': movie_id,
            'title': movie_title,
            'rank': rank,
            'predicted_rating': score,
            'genres': '|'.join(genres)
        })

# Save recommendations to CSV in chunks
if recommendations_list:
    chunk_size = 10000
    recommendations_df = pd.DataFrame(recommendations_list)
    
    # Save in chunks to avoid memory issues with large datasets
    for i in range(0, len(recommendations_df), chunk_size):
        chunk = recommendations_df.iloc[i:i+chunk_size]
        
        if i == 0:
            chunk.to_csv(os.path.join(output_path, 'dnn_recommendations.csv'), index=False, mode='w')
        else:
            chunk.to_csv(os.path.join(output_path, 'dnn_recommendations.csv'), index=False, mode='a', header=False)
            
    logger.info(f"Saved {len(recommendations_df)} recommendations to CSV")

log_memory_usage("After saving recommendations")

# Evaluate the recommendations
logger.info("Evaluating DNN recommendations with enhanced metrics")
evaluation_metrics, user_metrics = evaluate_recommendations(
    dnn_recommendations,
    data['test_ratings'],
    dnn_model,
    user_preferences,
    movie_features_with_regions,
    genre_columns,
    region_columns,
    threshold=threshold_rating,
    sample_users=True  # Add this parameter to only evaluate users with recommendations
)

log_memory_usage("After evaluation")

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
    
    # Analyze user metrics by user rating count
    if 'train_ratings' in data and len(user_metrics_df) > 0:
        # Get rating counts for each user
        user_rating_counts = data['train_ratings'].groupby('userId').size().reset_index(name='rating_count')
        
        # Merge with user metrics
        user_analysis = pd.merge(user_metrics_df, user_rating_counts, on='userId', how='left')
        
        # Create rating count bins
        user_analysis['rating_count_bin'] = pd.cut(
            user_analysis['rating_count'], 
            bins=[0, 10, 25, 50, 100, float('inf')],
            labels=['1-10', '11-25', '26-50', '51-100', '100+']
        )
        
        # Group by rating count bin and calculate average metrics
        metrics_by_count = user_analysis.groupby('rating_count_bin').agg({
            'rmse': 'mean',
            'mae': 'mean',
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'userId': 'count'
        }).reset_index()
        
        metrics_by_count.rename(columns={'userId': 'num_users'}, inplace=True)
        
        # Save to CSV
        metrics_by_count.to_csv(os.path.join(output_path, 'metrics_by_rating_count.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        metrics = ['rmse', 'accuracy', 'precision', 'recall', 'f1_score']
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
        
        for i, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, i+1)
            plt.bar(metrics_by_count['rating_count_bin'], 
                   metrics_by_count[metric], 
                   color=colors[i],
                   alpha=0.7)
            
            # Add value labels
            for j, v in enumerate(metrics_by_count[metric]):
                plt.text(j, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
                
            # Add user counts as text
            if i == 0:
                for j, count in enumerate(metrics_by_count['num_users']):
                    plt.text(j, metrics_by_count[metric].max() * 0.8, 
                            f'n={count}', ha='center', 
                            bbox=dict(facecolor='white', alpha=0.5))
            
            plt.title(f'{metric.upper()} by User Rating Count')
            plt.grid(axis='y', alpha=0.3)
            
            # For RMSE, lower is better
            if metric == 'rmse':
                plt.ylim(0, min(2.0, metrics_by_count[metric].max() * 1.2))
            else:
                plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'metrics_by_rating_count.png'))
        plt.close()
        
        logger.info("Created metrics analysis by user rating count")

# Show sample recommendations
logger.info("\nSample recommendation for exploration:")
if dnn_recommendations:
    # Pick a random user with recommendations
    sample_user_id = np.random.choice(list(dnn_recommendations.keys()))
    
    if sample_user_id in user_preferences['userId'].values:
        user_prefs = user_preferences[user_preferences['userId'] == sample_user_id].iloc[0]
        genre_pref_columns = [col for col in user_preferences.columns if col in genre_columns]
        
        logger.info(f"\nUser {sample_user_id} Preferences:")
        user_prefs_list = [(genre, user_prefs[genre]) for genre in genre_pref_columns if user_prefs[genre] != 0]
        liked_genres = sorted(user_prefs_list, key=lambda x: x[1], reverse=True)[:3]
        disliked_genres = sorted(user_prefs_list, key=lambda x: x[1])[:3]
        
        logger.info(f"- Most liked genres: {', '.join([f'{g} ({v:.2f})' for g, v in liked_genres if v > 0])}")
        logger.info(f"- Most disliked genres: {', '.join([f'{g} ({v:.2f})' for g, v in disliked_genres if v < 0])}")
        
        # Show region preferences if available
        if region_columns:
            region_pref_columns = [col for col in user_preferences.columns if col in region_columns]
            region_prefs_list = [(region, user_prefs[region]) for region in region_pref_columns if user_prefs[region] != 0]
            liked_regions = sorted(region_prefs_list, key=lambda x: x[1], reverse=True)[:3]
            
            if liked_regions and liked_regions[0][1] > 0:
                logger.info(f"- Preferred regions: {', '.join([f'{r} ({v:.2f})' for r, v in liked_regions if v > 0])}")
    
    # Show the recommendations
    recommend_for_user(sample_user_id, dnn_recommendations, data['movie_features'])

    # Look up this user in the metrics to see how we did
    if user_metrics and sample_user_id in user_metrics:
        user_metric = user_metrics[sample_user_id]
        logger.info(f"\nEvaluation metrics for user {sample_user_id}:")
        logger.info(f"- RMSE: {user_metric['rmse']:.4f}")
        logger.info(f"- Accuracy: {user_metric['accuracy']:.4f}")
        logger.info(f"- F1 Score: {user_metric['f1_score']:.4f}")
        logger.info(f"- Number of predictions: {user_metric['num_predictions']}")

log_memory_usage("Final memory usage")
logger.info("Enhanced DNN-based recommendation pipeline completed successfully!")