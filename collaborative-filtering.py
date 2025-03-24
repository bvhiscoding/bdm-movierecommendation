import numpy as np
import pandas as pd
import os
import pickle
import logging
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DNNRecommender:
    def __init__(self, data_path="./processed_data", output_path="./recommendations", top_n=10):
        """
        Initialize the DNN-based Recommender system.
        
        Parameters:
        -----------
        data_path : str, default="./processed_data"
            Path to the directory containing processed data
        output_path : str, default="./recommendations"
            Path to save recommendation results
        top_n : int, default=10
            Number of top recommendations to generate
        """
        self.data_path = data_path
        self.output_path = output_path
        self.top_n = top_n
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Initialize data containers
        self.train_df = None
        self.test_df = None
        self.user_stats = None
        self.movie_stats = None
        self.user_genre_prefs = None
        self.movie_genres = None
        
        # DNN model parameters
        self.batch_size = 1024
        self.epochs = 50
        self.learning_rate = 0.001
        self.early_stopping_patience = 3
        self.reduce_lr_patience = 2
        self.weight_decay = 1e-5
        
        # Model containers
        self.model = None
        self.user_scaler = None
        self.movie_scaler = None
    
    def load_data(self):
        """Load processed data required for DNN recommendations"""
        logger.info("Loading processed data...")
        
        # Load training data
        train_path = os.path.join(self.data_path, 'train_ratings.csv')
        if os.path.exists(train_path):
            self.train_df = pd.read_csv(train_path)
            logger.info(f"Loaded {len(self.train_df)} training ratings")
        else:
            logger.error(f"Training data not found at {train_path}")
            return False
        
        # Load test data
        test_path = os.path.join(self.data_path, 'test_ratings.csv')
        if os.path.exists(test_path):
            self.test_df = pd.read_csv(test_path)
            logger.info(f"Loaded {len(self.test_df)} test ratings")
        else:
            logger.error(f"Test data not found at {test_path}")
            return False
        
        # Load user statistics
        user_stats_path = os.path.join(self.data_path, 'user_stats.csv')
        if os.path.exists(user_stats_path):
            self.user_stats = pd.read_csv(user_stats_path)
            logger.info(f"Loaded statistics for {len(self.user_stats)} users")
        else:
            logger.error(f"User statistics not found at {user_stats_path}")
            return False
        
        # Load movie statistics
        movie_stats_path = os.path.join(self.data_path, 'movie_stats.csv')
        if os.path.exists(movie_stats_path):
            self.movie_stats = pd.read_csv(movie_stats_path)
            logger.info(f"Loaded statistics for {len(self.movie_stats)} movies")
        else:
            logger.error(f"Movie statistics not found at {movie_stats_path}")
            return False
        
        # Load user genre preferences
        user_genre_prefs_path = os.path.join(self.data_path, 'user_genre_prefs.csv')
        if os.path.exists(user_genre_prefs_path):
            self.user_genre_prefs = pd.read_csv(user_genre_prefs_path)
            logger.info(f"Loaded genre preferences for {len(self.user_genre_prefs)} users")
        else:
            logger.error(f"User genre preferences not found at {user_genre_prefs_path}")
            return False
        
        # Load movie genres
        movie_genres_path = os.path.join(self.data_path, 'movie_genres.csv')
        if os.path.exists(movie_genres_path):
            self.movie_genres = pd.read_csv(movie_genres_path)
            logger.info(f"Loaded genres for {len(self.movie_genres)} movies")
        else:
            logger.error(f"Movie genres not found at {movie_genres_path}")
            return False
        
        # Check if we have all the necessary data
        if (self.train_df is not None and self.test_df is not None and 
            self.user_stats is not None and self.movie_stats is not None and 
            self.user_genre_prefs is not None and self.movie_genres is not None):
            logger.info("All required data loaded successfully")
            return True
        else:
            logger.error("Failed to load all required data")
            return False
    
    def prepare_features(self):
        """Prepare features for DNN model training"""
        logger.info("Preparing features for DNN model...")
        
        # Extract genre preference columns
        genre_pref_cols = [col for col in self.user_genre_prefs.columns if col.endswith('_pref')]
        
        # Extract genre columns
        genre_cols = [col for col in self.movie_genres.columns if col != 'movieId']
        
        # User features: genre preferences + statistics
        self.user_features = pd.merge(
            self.user_genre_prefs, 
            self.user_stats,
            on='userId',
            how='inner'
        )
        
        # Movie features: genres + statistics
        self.movie_features = pd.merge(
            self.movie_genres,
            self.movie_stats,
            on='movieId',
            how='inner'
        )
        
        # Initialize scalers
        self.user_scaler = StandardScaler()
        self.movie_scaler = StandardScaler()
        
        # Select numeric columns for scaling (exclude IDs and categorical features)
        user_numeric_cols = ['rating_count', 'rating_mean', 'rating_std'] + genre_pref_cols
        movie_numeric_cols = ['rating_count', 'rating_mean', 'rating_std'] + genre_cols
        
        # Fit and transform user features
        self.user_features[user_numeric_cols] = self.user_scaler.fit_transform(
            self.user_features[user_numeric_cols]
        )
        
        # Fit and transform movie features
        self.movie_features[movie_numeric_cols] = self.movie_scaler.fit_transform(
            self.movie_features[movie_numeric_cols]
        )
        
        logger.info(f"Prepared features for {len(self.user_features)} users and {len(self.movie_features)} movies")
        
        # Save the feature dataframes
        self.user_features.to_csv(os.path.join(self.output_path, 'dnn_user_features.csv'), index=False)
        self.movie_features.to_csv(os.path.join(self.output_path, 'dnn_movie_features.csv'), index=False)
        
        # Save the scalers
        with open(os.path.join(self.output_path, 'dnn_user_scaler.pkl'), 'wb') as f:
            pickle.dump(self.user_scaler, f)
        
        with open(os.path.join(self.output_path, 'dnn_movie_scaler.pkl'), 'wb') as f:
            pickle.dump(self.movie_scaler, f)
        
        return True
    
    def _get_features(self, ratings_df, training=True):
        """
        Extract features for the given ratings dataframe
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user-movie ratings
        training : bool, default=True
            Whether this is for training (include ratings) or prediction (exclude ratings)
            
        Returns:
        --------
        tuple
            (user_features, movie_features, ratings) if training=True
            (user_features, movie_features) if training=False
        """
        # Extract unique users and movies from the ratings
        user_ids = ratings_df['userId'].unique()
        movie_ids = ratings_df['movieId'].unique()
        
        # Filter features to include only relevant users and movies
        users_data = self.user_features[self.user_features['userId'].isin(user_ids)]
        movies_data = self.movie_features[self.movie_features['movieId'].isin(movie_ids)]
        
        # Extract user and movie IDs from the features
        user_id_to_idx = {user_id: i for i, user_id in enumerate(users_data['userId'])}
        movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movies_data['movieId'])}
        
        # Number of samples
        n_samples = len(ratings_df)
        
        # Extract user features (excluding userId)
        user_cols = [col for col in users_data.columns if col != 'userId']
        user_features_array = users_data[user_cols].values
        
        # Extract movie features (excluding movieId)
        movie_cols = [col for col in movies_data.columns if col != 'movieId']
        movie_features_array = movies_data[movie_cols].values
        
        # Create arrays to store user and movie indices for each rating
        user_indices = np.zeros(n_samples, dtype=np.int32)
        movie_indices = np.zeros(n_samples, dtype=np.int32)
        
        # Populate the arrays with the indices
        for i, (_, row) in enumerate(ratings_df.iterrows()):
            user_id = row['userId']
            movie_id = row['movieId']
            
            # Skip if user or movie is not in the features (shouldn't happen)
            if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx:
                continue
            
            user_indices[i] = user_id_to_idx[user_id]
            movie_indices[i] = movie_id_to_idx[movie_id]
        
        # Create final feature arrays
        X_user = user_features_array[user_indices]
        X_movie = movie_features_array[movie_indices]
        
        if training:
            # Extract ratings
            y = ratings_df['rating'].values
            # Normalize ratings to [0, 1] range
            y_normalized = (y - 0.5) / 4.5  # Map from [0.5, 5.0] to [0, 1]
            return X_user, X_movie, y_normalized
        else:
            return X_user, X_movie
    
    def build_model(self):
        """Build the DNN model architecture"""
        logger.info("Building DNN model...")
        
        # Get input dimensions
        if not hasattr(self, 'user_features') or not hasattr(self, 'movie_features'):
            logger.error("Features not prepared. Run prepare_features() first.")
            return None
        
        user_dim = len(self.user_features.columns) - 1  # Exclude userId
        movie_dim = len(self.movie_features.columns) - 1  # Exclude movieId
        
        # Model architecture based on the paper
        
        # User branch
        user_input = Input(shape=(user_dim,), name='user_input')
        user_dense1 = Dense(64, activation='relu', kernel_regularizer=l2(self.weight_decay), name='user_dense1')(user_input)
        user_bn1 = BatchNormalization(name='user_bn1')(user_dense1)
        user_dropout1 = Dropout(0.2, name='user_dropout1')(user_bn1)
        user_dense2 = Dense(32, activation='relu', kernel_regularizer=l2(self.weight_decay), name='user_dense2')(user_dropout1)
        user_bn2 = BatchNormalization(name='user_bn2')(user_dense2)
        user_output = Dropout(0.2, name='user_dropout2')(user_bn2)
        
        # Movie branch
        movie_input = Input(shape=(movie_dim,), name='movie_input')
        movie_dense1 = Dense(64, activation='relu', kernel_regularizer=l2(self.weight_decay), name='movie_dense1')(movie_input)
        movie_bn1 = BatchNormalization(name='movie_bn1')(movie_dense1)
        movie_dropout1 = Dropout(0.2, name='movie_dropout1')(movie_bn1)
        movie_dense2 = Dense(32, activation='relu', kernel_regularizer=l2(self.weight_decay), name='movie_dense2')(movie_dropout1)
        movie_bn2 = BatchNormalization(name='movie_bn2')(movie_dense2)
        movie_output = Dropout(0.2, name='movie_dropout2')(movie_bn2)
        
        # Combine user and movie branches
        concat = Concatenate(name='concat')([user_output, movie_output])
        
        # Joint layers
        joint_dense1 = Dense(64, activation='relu', kernel_regularizer=l2(self.weight_decay), name='joint_dense1')(concat)
        joint_bn1 = BatchNormalization(name='joint_bn1')(joint_dense1)
        joint_dropout1 = Dropout(0.2, name='joint_dropout1')(joint_bn1)
        
        joint_dense2 = Dense(32, activation='relu', kernel_regularizer=l2(self.weight_decay), name='joint_dense2')(joint_dropout1)
        joint_bn2 = BatchNormalization(name='joint_bn2')(joint_dense2)
        joint_dropout2 = Dropout(0.2, name='joint_dropout2')(joint_bn2)
        
        joint_dense3 = Dense(16, activation='relu', kernel_regularizer=l2(self.weight_decay), name='joint_dense3')(joint_dropout2)
        joint_bn3 = BatchNormalization(name='joint_bn3')(joint_dense3)
        joint_dropout3 = Dropout(0.2, name='joint_dropout3')(joint_bn3)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(joint_dropout3)
        
        # Create model
        model = Model(inputs=[user_input, movie_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',  # Since we normalized ratings to [0, 1]
            metrics=['mae', 'mse']  # Track both MAE and MSE
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self):
        """Train the DNN model"""
        logger.info("Training DNN model...")
        
        if self.model is None:
            logger.error("Model not built. Run build_model() first.")
            return None
        
        # Prepare training data
        X_user_train, X_movie_train, y_train = self._get_features(self.train_df)
        
        # Prepare validation data (a portion of the training data)
        val_size = int(0.1 * len(self.train_df))
        val_indices = np.random.choice(len(self.train_df), val_size, replace=False)
        train_indices = np.array([i for i in range(len(self.train_df)) if i not in val_indices])
        
        X_user_val = X_user_train[val_indices]
        X_movie_val = X_movie_train[val_indices]
        y_val = y_train[val_indices]
        
        X_user_train = X_user_train[train_indices]
        X_movie_train = X_movie_train[train_indices]
        y_train = y_train[train_indices]
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        start_time = time.time()
        
        history = self.model.fit(
            [X_user_train, X_movie_train],
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([X_user_val, X_movie_val], y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save the model
        model_path = os.path.join(self.output_path, 'dnn_model.h5')
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Plot and save training history
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """Plot and save training history"""
        # Create output directory for plots if it doesn't exist
        plots_dir = os.path.join(self.output_path, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'dnn_training_history.png'))
        plt.close()
        
        # Save history as CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.output_path, 'dnn_training_history.csv'), index=False)
    
    def load_trained_model(self):
        """Load a previously trained model"""
        model_path = os.path.join(self.output_path, 'dnn_model.h5')
        
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"Loaded trained model from {model_path}")
            return True
        else:
            logger.warning(f"No trained model found at {model_path}")
            return False
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        logger.info("Evaluating DNN model...")
        
        if self.model is None:
            loaded = self.load_trained_model()
            if not loaded:
                logger.error("Model not available. Train or load a model first.")
                return None
        
        # Prepare test data
        X_user_test, X_movie_test, y_test = self._get_features(self.test_df)
        
        # Evaluate the model
        evaluation = self.model.evaluate([X_user_test, X_movie_test], y_test, verbose=1)
        
        # Extract metrics
        metrics = {
            'loss': evaluation[0],
            'mae': evaluation[1],
            'mse': evaluation[2],
            'rmse': np.sqrt(evaluation[2])
        }
        
        # Denormalize metrics for easier interpretation
        metrics['mae_original_scale'] = metrics['mae'] * 4.5  # Scale back to original rating range
        metrics['rmse_original_scale'] = metrics['rmse'] * 4.5  # Scale back to original rating range
        
        logger.info(f"Evaluation results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f} (Original scale: {metrics['mae_original_scale']:.4f})")
        logger.info(f"  RMSE: {metrics['rmse']:.4f} (Original scale: {metrics['rmse_original_scale']:.4f})")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_path, 'dnn_evaluation_metrics.csv'), index=False)
        
        return metrics
    
    def predict_ratings(self, user_id, movie_ids=None, top_n=None):
        """
        Predict ratings for a user and a list of movies
        
        Parameters:
        -----------
        user_id : int
            The user ID to predict ratings for
        movie_ids : list, optional
            List of movie IDs to predict ratings for. If None, predict for all movies.
        top_n : int, optional
            Number of top rated movies to return. If None, return all predictions.
            
        Returns:
        --------
        list of tuples
            (movie_id, predicted_rating) pairs sorted by predicted rating in descending order
        """
        if self.model is None:
            loaded = self.load_trained_model()
            if not loaded:
                logger.error("Model not available. Train or load a model first.")
                return None
        
        # Get user features
        if user_id not in self.user_features['userId'].values:
            logger.warning(f"User {user_id} not found in features")
            return None
        
        user_features = self.user_features[self.user_features['userId'] == user_id].iloc[0]
        user_features_array = user_features.drop('userId').values.reshape(1, -1)
        
        # Determine which movies to predict for
        if movie_ids is None:
            # Predict for all movies
            movie_ids = self.movie_features['movieId'].values
        else:
            # Filter to only include movies in our features
            movie_ids = [mid for mid in movie_ids if mid in self.movie_features['movieId'].values]
        
        if not movie_ids:
            logger.warning("No valid movie IDs to predict for")
            return None
        
        # Get already rated movies
        rated_movies = set()
        if hasattr(self, 'train_df'):
            user_ratings = self.train_df[self.train_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'].values)
        
        # Filter out already rated movies
        movie_ids = [mid for mid in movie_ids if mid not in rated_movies]
        
        # Get movie features
        movie_features_list = []
        valid_movie_ids = []
        
        for mid in movie_ids:
            if mid in self.movie_features['movieId'].values:
                movie_row = self.movie_features[self.movie_features['movieId'] == mid].iloc[0]
                movie_features_list.append(movie_row.drop('movieId').values)
                valid_movie_ids.append(mid)
        
        if not valid_movie_ids:
            logger.warning("No valid movies to predict for after filtering")
            return None
        
        movie_features_array = np.array(movie_features_list)
        
        # Predict ratings
        logger.info(f"Predicting ratings for user {user_id} and {len(valid_movie_ids)} movies")
        
        # For large sets of movies, predict in batches
        batch_size = 1000
        num_movies = len(valid_movie_ids)
        predictions = np.zeros(num_movies)
        
        for i in range(0, num_movies, batch_size):
            end_idx = min(i + batch_size, num_movies)
            batch_movies = movie_features_array[i:end_idx]
            
            # Repeat user features for each movie in the batch
            batch_users = np.repeat(user_features_array, len(batch_movies), axis=0)
            
            # Predict
            batch_preds = self.model.predict([batch_users, batch_movies], verbose=0)
            
            # Store predictions
            predictions[i:end_idx] = batch_preds.flatten()
        
        # Denormalize predictions back to original rating scale [0.5, 5.0]
        predictions = 0.5 + predictions * 4.5
        
        # Create list of (movie_id, predicted_rating) pairs
        prediction_pairs = list(zip(valid_movie_ids, predictions))
        
        # Sort by predicted rating (descending)
        prediction_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N if specified
        if top_n is not None:
            prediction_pairs = prediction_pairs[:top_n]
        
        return prediction_pairs
    
    def generate_recommendations(self):
        """Generate recommendations for all users"""
        logger.info(f"Generating top-{self.top_n} recommendations for all users...")
        
        if self.model is None:
            loaded = self.load_trained_model()
            if not loaded:
                logger.error("Model not available. Train or load a model first.")
                return None
        
        # Get all user IDs
        user_ids = self.user_features['userId'].unique()
        
        all_recommendations = {}
        
        for user_id in user_ids:
            recommendations = self.predict_ratings(user_id, top_n=self.top_n)
            if recommendations:
                all_recommendations[user_id] = recommendations
        
        logger.info(f"Generated recommendations for {len(all_recommendations)} users")
        
        # Save recommendations
        with open(os.path.join(self.output_path, 'dnn_recommendations.pkl'), 'wb') as f:
            pickle.dump(all_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in all_recommendations.items():
            for rank, (movie_id, score) in enumerate(recs, 1):
                recommendations_list.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rank': rank,
                    'predicted_rating': score
                })
        
        if recommendations_list:
            recommendations_df = pd.DataFrame(recommendations_list)
            recommendations_df.to_csv(os.path.join(self.output_path, 'dnn_recommendations.csv'), index=False)
            logger.info(f"Saved recommendations to CSV file")
        
        return all_recommendations
    
    def evaluate_recommendations(self):
        """Evaluate recommendations using Hit Rate and ARHR"""
        logger.info("Evaluating recommendations...")
        
        # Generate recommendations if not already done
        if not os.path.exists(os.path.join(self.output_path, 'dnn_recommendations.pkl')):
            logger.info("Recommendations not found. Generating now...")
            self.generate_recommendations()
        
        # Load recommendations
        with open(os.path.join(self.output_path, 'dnn_recommendations.pkl'), 'rb') as f:
            all_recommendations = pickle.load(f)
        
        # Initialize metrics
        hits = 0
        total = 0
        sum_reciprocal_rank = 0
        
        # Get all users in test set
        test_users = self.test_df['userId'].unique()
        
        for user_id in test_users:
            # Skip users without recommendations
            if user_id not in all_recommendations:
                continue
            
            # Get ground truth: movies the user liked in the test set (rating >= 4)
            user_test = self.test_df[self.test_df['userId'] == user_id]
            liked_movies = set(user_test[user_test['rating'] >= 4]['movieId'].values)
            
            if not liked_movies:
                continue
            
            # Get recommendations for this user
            recommendations = all_recommendations[user_id]
            recommended_movies = [movie_id for movie_id, _ in recommendations]
            
            # Check for hits
            user_hits = liked_movies.intersection(set(recommended_movies))
            
            if user_hits:
                hits += 1
                
                # Calculate reciprocal rank (position of first hit)
                for i, (movie_id, _) in enumerate(recommendations, 1):
                    if movie_id in liked_movies:
                        sum_reciprocal_rank += 1.0 / i
                        break
            
            total += 1
        
        # Calculate metrics
        hit_rate = hits / total if total > 0 else 0
        average_reciprocal_hit_rank = sum_reciprocal_rank / total if total > 0 else 0
        
        metrics = {
            'hit_rate': hit_rate,
            'arhr': average_reciprocal_hit_rank,
            'num_users_evaluated': total
        }
        
        logger.info(f"Recommendation evaluation results:")
        logger.info(f"  Hit Rate: {hit_rate:.4f}")
        logger.info(f"  ARHR: {average_reciprocal_hit_rank:.4f}")
        logger.info(f"  Users evaluated: {total}")
        
        # Save metrics
        recommendation_metrics_df = pd.DataFrame([metrics])
        recommendation_metrics_df.to_csv(os.path.join(self.output_path, 'dnn_recommendation_metrics.csv'), index=False)
        
        return metrics
    
    def get_movie_details(self, movie_ids):
        """Get details for a list of movie IDs"""
        movie_details = self.movie_genres[self.movie_genres['movieId'].isin(movie_ids)]
        return movie_details
    
    def run(self):
        """Run the entire recommendation pipeline"""
        # Create a timestamped run ID
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting DNN recommendation pipeline (Run ID: {run_id})")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load required data. Exiting.")
            return None
        
        # Prepare features
        self.prepare_features()
        
        # Check if model exists
        model_exists = self.load_trained_model()
        
        if not model_exists:
            # Build and train model
            self.build_model()
            self.train_model()
        
        # Evaluate model
        evaluation_metrics = self.evaluate_model()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Evaluate recommendations
        recommendation_metrics = self.evaluate_recommendations()
        
        logger.info(f"DNN recommendation pipeline completed (Run ID: {run_id})")
        
        return {
            'run_id': run_id,
            'output_path': self.output_path,
            'evaluation_metrics': evaluation_metrics,
            'recommendation_metrics': recommendation_metrics
        }
    
    def recommend_for_user(self, user_id, n=None):
        """
        Generate and print recommendations for a specific user
        
        Parameters:
        -----------
        user_id : int
            The user ID to generate recommendations for
        n : int, optional
            Number of recommendations to generate (defaults to self.top_n)
        """
        if n is None:
            n = self.top_n
        
        # Predict ratings
        predictions = self.predict_ratings(user_id, top_n=n)
        
        if not predictions:
            print(f"No recommendations found for user {user_id}")
            return None
        
        # Get movie details if available
        movie_ids = [movie_id for movie_id, _ in predictions]
        movie_details = self.get_movie_details(movie_ids)
        
        # Print recommendations
        print(f"\nTop {len(predictions)} recommendations for user {user_id}:")
        
        for i, (movie_id, rating) in enumerate(predictions, 1):
            if movie_details is not None and not movie_details[movie_details['movieId'] == movie_id].empty:
                movie_row = movie_details[movie_details['movieId'] == movie_id].iloc[0]
                movie_name = movie_row['title'] if 'title' in movie_row else f"Movie {movie_id}"
                print(f"{i}. {movie_name} (ID: {movie_id}) - Predicted rating: {rating:.2f}")
            else:
                print(f"{i}. Movie ID: {movie_id} - Predicted rating: {rating:.2f}")
        
        return predictions

if __name__ == "__main__":
    # Create recommender with default paths
    recommender = DNNRecommender(
        data_path="./processed_data", 
        output_path="./recommendations",
        top_n=10
    )
    
    # Run the recommendation pipeline
    result = recommender.run()
    
    if result:
        # Print summary
        print("\nRecommendation generation completed!")
        print(f"Run ID: {result['run_id']}")
        print(f"Output path: {result['output_path']}")
        
        if result['evaluation_metrics']:
            print("\nModel evaluation metrics:")
            print(f"- RMSE: {result['evaluation_metrics']['rmse_original_scale']:.4f}")
            print(f"- MAE: {result['evaluation_metrics']['mae_original_scale']:.4f}")
        
        if result['recommendation_metrics']:
            print("\nRecommendation evaluation metrics:")
            print(f"- Hit Rate: {result['recommendation_metrics']['hit_rate']:.4f}")
            print(f"- Average Reciprocal Hit Rank: {result['recommendation_metrics']['arhr']:.4f}")
            print(f"- Number of users evaluated: {result['recommendation_metrics']['num_users_evaluated']}")
        
        # Example: Generate recommendations for a specific user
        if recommender.user_features is not None:
            sample_user_id = recommender.user_features['userId'].iloc[0]
            print(f"\nExample: Recommendations for user {sample_user_id}")
            recommender.recommend_for_user(sample_user_id, n=5)