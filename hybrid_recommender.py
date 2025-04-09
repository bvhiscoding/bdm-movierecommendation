import numpy as np
import pandas as pd
import os
import torch
import pickle
import logging
import argparse
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device("cuda")
    print(f"CUDA is available: Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available: Using CPU instead")

class HybridRecommender:
    def predict_rating(self, user_id, movie_id):
        """
        Predict a user's rating for a movie using content-based approach
        
        Parameters:
        -----------
        user_id: int
            User ID
        movie_id: int
            Movie ID
            
        Returns:
        --------
        float
            Predicted rating (0.5-5.0 scale)
        """
        # If using user-movie similarities
        if 'user_movie_similarities' in self.data and user_id in self.data['user_movie_similarities']:
            user_sims = self.data['user_movie_similarities'][user_id]
            
            # If movie is in similarities
            if movie_id in user_sims:
                # Convert similarity (0-1) to rating (0.5-5.0)
                sim_score = user_sims[movie_id]
                return 0.5 + 4.5 * sim_score
        
        # Get user's average rating
        if 'train_ratings' in self.data:
            user_ratings = self.data['train_ratings'][self.data['train_ratings']['userId'] == user_id]
            if len(user_ratings) > 0:
                return user_ratings['rating'].mean()
        
        # Default to mid-point if no other information
        return 3.0
        
    def predict_rating_collaborative(self, user_id, movie_id):
        """
        Predict a user's rating for a movie using DNN
        
        Parameters:
        -----------
        user_id: int
            User ID
        movie_id: int
            Movie ID
            
        Returns:
        --------
        float
            Predicted rating (0.5-5.0 scale)
        """
        try:
            # Check if we have the necessary components
            if 'dnn_model' in self.data and 'user_genre_preferences' in self.data and 'movie_genre_features' in self.data:
                # Get user preferences
                user_prefs = self.data['user_genre_preferences'][self.data['user_genre_preferences']['userId'] == user_id]
                if user_prefs.empty:
                    return 3.0  # Default if user not found
                
                # Get movie genres
                movie_genres = self.data['movie_genre_features'][self.data['movie_genre_features']['movieId'] == movie_id]
                if movie_genres.empty:
                    return 3.0  # Default if movie not found
                
                # Create feature vector
                genre_columns = [col for col in self.data['movie_genre_features'].columns if col != 'movieId']
                feature_vector = []
                
                for genre in genre_columns:
                    feature_vector.append(user_prefs.iloc[0][genre])
                    feature_vector.append(movie_genres.iloc[0][genre])
                
                # Reshape for prediction
                feature_vector = np.array([feature_vector])
                
                # Predict movie rating
                predicted_rating = self.data['dnn_model'].predict(feature_vector, verbose=0)[0][0]
                
                # Ensure rating is within bounds
                return max(0.5, min(5.0, predicted_rating))
        except Exception as e:
            print(f"Error in collaborative prediction: {str(e)}")
        
        # Default to mid-point if prediction fails
        return 3.0
    def __init__(self, content_model_path="./content-recommendations", 
                 collab_model_path="./recommendations", 
                 output_path="./hybrid_recommendations", 
                 alpha=0.3):
        """
        Initialize the hybrid recommender with paths to content-based and collaborative filtering models
        
        Parameters:
        -----------
        content_model_path: str
            Path to the directory containing content-based model files
        collab_model_path: str
            Path to the directory containing collaborative filtering model files
        output_path: str
            Path to save hybrid recommendation results
        alpha: float
            Weight for content-based recommendations (1-alpha for collaborative)
        """
        self.content_model_path = content_model_path
        self.collab_model_path = collab_model_path
        self.output_path = output_path
        self.alpha = alpha
        self.data = {}  # Container for all loaded data
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        print("\n" + "="*80)
        print(f"HYBRID MOVIE RECOMMENDATION SYSTEM (alpha={self.alpha:.2f})")
        print("="*80)  
    
    def load_data(self):
        """Load movie data, content-based and collaborative filtering model outputs"""
        print("\nLoading data...")
        start_time = time.time()
        
        # Load movie features
        try:
            movie_features_path = './processed/processed_movie_features.csv'
            self.data['movie_features'] = pd.read_csv(movie_features_path)
            # Convert string representation of tokens and top_keywords back to lists
            if 'tokens' in self.data['movie_features'].columns:
                self.data['movie_features']['tokens'] = self.data['movie_features']['tokens'].apply(
                    lambda x: eval(x) if isinstance(x, str) else []
                )
            if 'top_keywords' in self.data['movie_features'].columns:
                self.data['movie_features']['top_keywords'] = self.data['movie_features']['top_keywords'].apply(
                    lambda x: eval(x) if isinstance(x, str) else []
                )
            print(f"Loaded features for {len(self.data['movie_features'])} movies")
        except Exception as e:
            print(f"Error loading movie features: {str(e)}")
        
        # Load normalized ratings
        try:
            ratings_path = './processed/normalized_ratings.csv'
            self.data['ratings'] = pd.read_csv(ratings_path)
            print(f"Loaded {len(self.data['ratings'])} ratings")
            
            # Create training and testing sets with 80-20 split
            user_groups = self.data['ratings'].groupby('userId')
            train_data = []
            test_data = []
            
            for _, group in user_groups:
                n = len(group)
                split_idx = int(n * 0.8)
                train_data.append(group.iloc[:split_idx])
                test_data.append(group.iloc[split_idx:])
            
            self.data['train_ratings'] = pd.concat(train_data).reset_index(drop=True)
            self.data['test_ratings'] = pd.concat(test_data).reset_index(drop=True)
            
            print(f"Split into {len(self.data['train_ratings'])} training and {len(self.data['test_ratings'])} testing ratings")
        except Exception as e:
            print(f"Error loading ratings data: {str(e)}")
            
        # Load content-based model components
        try:
            # Load content-based recommendations
            content_recs_path = os.path.join(self.content_model_path, 'content_based_recommendations.pkl')
            if os.path.exists(content_recs_path):
                with open(content_recs_path, 'rb') as f:
                    self.data['content_recommendations'] = pickle.load(f)
                print(f"Loaded content-based recommendations for {len(self.data['content_recommendations'])} users")
            
            # Load movie vectors - from non-CUDA version
            movie_vectors_path = os.path.join(self.content_model_path, 'movie_vectors.pkl')
            if os.path.exists(movie_vectors_path):
                with open(movie_vectors_path, 'rb') as f:
                    self.data['movie_vectors'] = pickle.load(f)
                print(f"Loaded content-based vectors for {len(self.data['movie_vectors'])} movies")
            
            # Load user vectors 
            user_vectors_path = os.path.join(self.content_model_path, 'user_vectors.pkl')
            if os.path.exists(user_vectors_path):
                with open(user_vectors_path, 'rb') as f:
                    self.data['user_vectors'] = pickle.load(f)
                print(f"Loaded content-based vectors for {len(self.data['user_vectors'])} users")
            
            # Load user-movie similarities
            similarities_path = os.path.join(self.content_model_path, 'user_movie_similarities.pkl')
            if os.path.exists(similarities_path):
                with open(similarities_path, 'rb') as f:
                    self.data['user_movie_similarities'] = pickle.load(f)
                print(f"Loaded user-movie similarities for {len(self.data['user_movie_similarities'])} users")
            
            # Load content-based evaluation metrics
            try:
                content_eval_path = os.path.join(self.content_model_path, 'content_based_evaluation.csv')
                if os.path.exists(content_eval_path):
                    content_eval_df = pd.read_csv(content_eval_path)
                    if not content_eval_df.empty:
                        self.data['content_evaluation'] = {
                            'rmse': content_eval_df.iloc[0]['rmse'],
                            'mae': content_eval_df.iloc[0]['mae'] if 'mae' in content_eval_df.columns else None,
                            'num_predictions': content_eval_df.iloc[0]['num_predictions'] if 'num_predictions' in content_eval_df.columns else None
                        }
                        print(f"Loaded content-based evaluation metrics: RMSE={self.data['content_evaluation']['rmse']:.4f}")
            except Exception as e:
                print(f"Error loading content-based evaluation metrics: {str(e)}")
                
        except Exception as e:
            print(f"Error loading content-based model components: {str(e)}")

        # Load collaborative filtering model components
        try:
            # Load collaborative filtering recommendations
            collab_recs_path = os.path.join(self.collab_model_path, 'dnn_recommendations.pkl')
            if os.path.exists(collab_recs_path):
                with open(collab_recs_path, 'rb') as f:
                    self.data['collaborative_recommendations'] = pickle.load(f)
                print(f"Loaded collaborative filtering recommendations for {len(self.data['collaborative_recommendations'])} users")
            
            # Load DNN evaluation metrics
            dnn_eval_path = os.path.join(self.collab_model_path, 'dnn_evaluation.csv')
            if os.path.exists(dnn_eval_path):
                dnn_eval_df = pd.read_csv(dnn_eval_path)
                if not dnn_eval_df.empty:
                    self.data['dnn_evaluation'] = {
                        'rmse': dnn_eval_df.iloc[0]['rmse'],
                        'mae': dnn_eval_df.iloc[0]['mae'],
                        'num_predictions': dnn_eval_df.iloc[0]['num_predictions']
                    }
                    print(f"Loaded DNN evaluation metrics: RMSE={self.data['dnn_evaluation']['rmse']:.4f}")
            
            # Load user genre preferences for DNN
            user_prefs_path = os.path.join(self.collab_model_path, 'user_genre_preferences.csv')
            if os.path.exists(user_prefs_path):
                self.data['user_genre_preferences'] = pd.read_csv(user_prefs_path)
                print(f"Loaded user genre preferences for {len(self.data['user_genre_preferences'])} users")
            
            # Load movie genre features for DNN
            movie_genre_path = os.path.join(self.collab_model_path, 'movie_genre_features.csv')
            if os.path.exists(movie_genre_path):
                self.data['movie_genre_features'] = pd.read_csv(movie_genre_path)
                print(f"Loaded movie genre features for {len(self.data['movie_genre_features'])} movies")
            
            # Try to load DNN model if TensorFlow is available
            try:
                import tensorflow as tf
                dnn_model_path = os.path.join(self.collab_model_path, 'dnn_model.h5')
                if os.path.exists(dnn_model_path):
                    self.data['dnn_model'] = tf.keras.models.load_model(dnn_model_path)
                    print(f"Loaded DNN model from {dnn_model_path}")
            except ImportError:
                print("TensorFlow not available - skipping DNN model loading")
                
        except Exception as e:
            print(f"Error loading collaborative filtering model components: {str(e)}")
        
        print(f"Data loading completed in {time.time() - start_time:.2f}s")
        
        # Get common users
        self.common_users = set()
        if 'content_recommendations' in self.data and 'collaborative_recommendations' in self.data:
            self.common_users = set(self.data['content_recommendations'].keys()) & set(self.data['collaborative_recommendations'].keys())
            print(f"Found {len(self.common_users)} users with both content-based and collaborative recommendations")
        
        return self.data
    
    def combine_recommendations(self, top_n=10):
        """
        Combine content-based and collaborative filtering recommendations with weighting
        
        Parameters:
        -----------
        top_n: int
            Number of recommendations to generate per user
            
        Returns:
        --------
        dict
            User ID to list of (movie_id, score) tuples
        """
        print(f"\nCombining recommendations with alpha={self.alpha:.2f}...")
        start_time = time.time()
        
        # Get recommendations from both models
        content_recs = self.data.get('content_recommendations', {})
        collab_recs = self.data.get('collaborative_recommendations', {})
        
        if not content_recs:
            print("Warning: No content-based recommendations available")
        
        if not collab_recs:
            print("Warning: No collaborative filtering recommendations available")
        
        if not content_recs and not collab_recs:
            print("Error: No recommendations available from either model")
            return {}
        
        # Combine recommendations
        combined_recommendations = {}
        
        # Get all users from both recommendation sets
        all_users = set(content_recs.keys()) | set(collab_recs.keys())
        total_users = len(all_users)
        
        for i, user_id in enumerate(all_users):
            # Initialize combined recommendations dictionary for this user
            user_combined_recs = {}
            
            # Add content-based recommendations if available
            if user_id in content_recs:
                for movie_id, score in content_recs[user_id]:
                    # Convert similarity score to rating scale [0.5-5.0]
                    rating = 0.5 + 4.5 * score
                    user_combined_recs[movie_id] = self.alpha * rating
            
            # Add collaborative filtering recommendations if available
            if user_id in collab_recs:
                for movie_id, rating in collab_recs[user_id]:
                    # Collaborative scores are already in rating scale
                    if movie_id in user_combined_recs:
                        user_combined_recs[movie_id] += (1 - self.alpha) * rating
                    else:
                        user_combined_recs[movie_id] = (1 - self.alpha) * rating
            
            # Sort and convert to list of tuples
            sorted_recs = sorted(user_combined_recs.items(), key=lambda x: x[1], reverse=True)
            
            combined_recommendations[user_id] = sorted_recs[:top_n]
            
            # Log progress
            if (i+1) % 1000 == 0 or (i+1) == total_users:
                print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%)")
        
        self.data['combined_recommendations'] = combined_recommendations
        
        print(f"Combined recommendations for {len(combined_recommendations)} users in {time.time() - start_time:.2f}s")
        
        # Save combined recommendations
        with open(os.path.join(self.output_path, 'combined_recommendations.pkl'), 'wb') as f:
            pickle.dump(combined_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in combined_recommendations.items():
            for rank, (movie_id, score) in enumerate(recs, 1):
                movie_title = "Unknown"
                if 'movie_features' in self.data:
                    movie_row = self.data['movie_features'][self.data['movie_features']['movieId'] == movie_id]
                    if not movie_row.empty and 'title' in movie_row.columns:
                        movie_title = movie_row.iloc[0]['title']
                
                recommendations_list.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'title': movie_title,
                    'rank': rank,
                    'score': score
                })
        
        if recommendations_list:
            recommendations_df = pd.DataFrame(recommendations_list)
            recommendations_df.to_csv(os.path.join(self.output_path, 'combined_recommendations.csv'), index=False)
            print(f"Saved combined recommendations to CSV with {len(recommendations_df)} entries")
        
        return combined_recommendations
    
    def evaluate(self):
        """
        Evaluate the hybrid recommendation system using RMSE and MAE
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print("\nEvaluating hybrid recommendation system...")
        start_time = time.time()
        
        # Get combined recommendations and test ratings
        combined_recs = self.data.get('combined_recommendations', {})
        test_ratings = self.data.get('test_ratings')
        
        if not combined_recs:
            print("Error: No combined recommendations available for evaluation")
            return None
        
        if test_ratings is None:
            print("Error: No test ratings available for evaluation")
            return None
        
        # Initialize containers for predictions and actual ratings
        predictions = []
        actuals = []
        
        # Match test ratings with predictions
        for user_id in test_ratings['userId'].unique():
            # Skip users without recommendations
            if user_id not in combined_recs:
                continue
            
            # Get user's test ratings
            user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
            
            # Get user's recommendations (movie_id, score)
            user_recs = dict(combined_recs[user_id])
            
            # Match test ratings with predictions
            for _, row in user_test_ratings.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                # If the movie is in recommendations
                if movie_id in user_recs:
                    predictions.append(user_recs[movie_id])
                    actuals.append(actual_rating)
                # For more complete evaluation, also predict for movies not in top recommendations
                # but using our hybrid prediction approach
                else:
                    # Get content-based and collaborative predictions
                    content_pred = self.predict_rating(user_id, movie_id)
                    collab_pred = self.predict_rating_collaborative(user_id, movie_id)
                    
                    # Combine predictions using alpha
                    hybrid_pred = self.alpha * content_pred + (1 - self.alpha) * collab_pred
                    
                    predictions.append(hybrid_pred)
                    actuals.append(actual_rating)
        
        # Calculate RMSE and MAE if we have predictions
        if predictions:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mae = np.mean(np.abs(predictions - actuals))
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'num_predictions': len(predictions)
            }
            
            print(f"Evaluation completed with {len(predictions)} predictions:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            
            # Save metrics
            pd.DataFrame([metrics]).to_csv(os.path.join(self.output_path, 'evaluation_metrics.csv'), index=False)
            
            # Store metrics in data
            self.data['combined_evaluation_metrics'] = metrics
            
            return metrics
        else:
            print("No predictions available for evaluation")
            return None
    
    def find_optimal_alpha(self, alpha_values=None):
        """
        Find optimal alpha value by evaluating RMSE at different alpha levels
        
        Parameters:
        -----------
        alpha_values: list
            List of alpha values to test
            
        Returns:
        --------
        float
            Optimal alpha value
        """
        if alpha_values is None:
            alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print("\nFinding optimal alpha value...")
        results = []
        
        original_alpha = self.alpha
        
        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha:.1f}")
            self.alpha = alpha
            self.combine_recommendations()
            metrics = self.evaluate()
            
            if metrics:
                results.append({
                    'alpha': alpha,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae']
                })
        
        if results:
            # Convert to DataFrame for easier analysis
            results_df = pd.DataFrame(results)
            
            # Find optimal alpha (minimize RMSE)
            optimal_idx = results_df['rmse'].idxmin()
            optimal_alpha = results_df.loc[optimal_idx, 'alpha']
            optimal_rmse = results_df.loc[optimal_idx, 'rmse']
            
            print(f"\nOptimal alpha = {optimal_alpha:.2f} with RMSE = {optimal_rmse:.4f}")
            
            # Set alpha to optimal value
            self.alpha = optimal_alpha
            
            # Save results
            results_df.to_csv(os.path.join(self.output_path, 'alpha_optimization.csv'), index=False)
            
            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot(results_df['alpha'], results_df['rmse'], 'o-', label='RMSE')
            plt.plot(results_df['alpha'], results_df['mae'], 's-', label='MAE')
            plt.axvline(x=optimal_alpha, color='r', linestyle='--', label=f'Optimal alpha: {optimal_alpha:.2f}')
            plt.xlabel('Alpha (Weight of Content-Based Recommendations)')
            plt.ylabel('Error Metric Value')
            plt.title('Effect of Alpha on Recommendation Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_path, 'alpha_optimization.png'))
            plt.close()
            
            return optimal_alpha
        else:
            # Restore original alpha
            self.alpha = original_alpha
            print("Could not determine optimal alpha due to insufficient data")
            return self.alpha
    
    def get_user_rated_movies(self, user_id):
        """
        Get movies already rated by a user
        
        Parameters:
        -----------
        user_id: int
            User ID
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of user's rated movies with ratings
        """
        if 'train_ratings' not in self.data:
            return pd.DataFrame()
        
        user_ratings = self.data['train_ratings'][self.data['train_ratings']['userId'] == user_id]
        
        if len(user_ratings) > 0 and 'movie_features' in self.data:
            # Join with movie titles
            user_ratings = pd.merge(
                user_ratings,
                self.data['movie_features'][['movieId', 'title']],
                on='movieId',
                how='left'
            )
        
        return user_ratings
    
    def recommend_for_user(self, user_id, n=10):
        """
        Get recommendations for a specific user
        
        Parameters:
        -----------
        user_id: int
            User ID
        n: int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of (movie_id, title, score) tuples
        """
        # Check if user has recommendations
        if 'combined_recommendations' not in self.data or user_id not in self.data['combined_recommendations']:
            return []
        
        # Get recommendations
        recs = self.data['combined_recommendations'][user_id][:n]
        
        # Format recommendations with titles
        formatted_recs = []
        for movie_id, score in recs:
            title = "Unknown"
            if 'movie_features' in self.data:
                movie_row = self.data['movie_features'][self.data['movie_features']['movieId'] == movie_id]
                if not movie_row.empty and 'title' in movie_row.columns:
                    title = movie_row.iloc[0]['title']
            
            formatted_recs.append((movie_id, title, score))
        
        return formatted_recs
    
    def get_movie_details(self, movie_id):
        """
        Get detailed information about a movie
        
        Parameters:
        -----------
        movie_id: int
            Movie ID
            
        Returns:
        --------
        dict
            Movie details
        """
        if 'movie_features' not in self.data:
            return None
        
        movie_row = self.data['movie_features'][self.data['movie_features']['movieId'] == movie_id]
        
        if movie_row.empty:
            return None
        
        # Extract genre columns
        genre_columns = [col for col in movie_row.columns if col not in 
                         ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
        
        # Get genres
        genres = [genre for genre in genre_columns if movie_row[genre].iloc[0] == 1]
        
        # Get keywords
        keywords = []
        if 'top_keywords' in movie_row.columns:
            keywords_raw = movie_row['top_keywords'].iloc[0]
            if isinstance(keywords_raw, str):
                # Parse string representation of list if needed
                try:
                    keywords = eval(keywords_raw)
                except:
                    keywords = []
            else:
                keywords = keywords_raw
        
        # Create details dictionary
        details = {
            'movieId': movie_id,
            'title': movie_row['title'].iloc[0],
            'genres': genres,
            'keywords': keywords
        }
        
        return details
    
    def generate_content_based_recommendations(self):
        """
        Generate content-based recommendations if not already available
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if 'content_recommendations' in self.data and self.data['content_recommendations']:
            print("Content-based recommendations already available, skipping generation")
            return True
            
        # Check if we need to load or generate user-movie similarities
        if 'user_movie_similarities' not in self.data or not self.data['user_movie_similarities']:
            if 'user_vectors' not in self.data or 'movie_vectors' not in self.data:
                print("Error: Required vectors not available for content-based recommendation generation")
                return False
                
            print("\nCalculating user-movie similarities...")
            start_time = time.time()
            
            try:
                user_vectors = self.data['user_vectors']
                movie_vectors = self.data['movie_vectors']
                
                # Initialize user-movie similarities
                user_movie_similarities = {}
                
                # Calculate similarity for each user
                for i, (user_id, user_vector) in enumerate(user_vectors.items()):
                    user_sims = {}
                    
                    for movie_id, movie_vector in movie_vectors.items():
                        # Calculate cosine similarity
                        if len(user_vector) == len(movie_vector):
                            similarity = np.dot(user_vector, movie_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(movie_vector))
                            
                            # Only store if above threshold
                            if similarity > 0.3:  # Minimum similarity threshold
                                user_sims[movie_id] = similarity
                    
                    user_movie_similarities[user_id] = user_sims
                    
                    # Log progress
                    if (i+1) % 100 == 0 or (i+1) == len(user_vectors):
                        print(f"Processed {i+1}/{len(user_vectors)} users ({(i+1)/len(user_vectors)*100:.1f}%)")
                
                self.data['user_movie_similarities'] = user_movie_similarities
                print(f"Calculated user-movie similarities in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                print(f"Error calculating user-movie similarities: {str(e)}")
                return False
        
        # Generate recommendations
        if 'user_movie_similarities' in self.data and self.data['user_movie_similarities']:
            print("\nGenerating content-based recommendations...")
            start_time = time.time()
            
            try:
                user_movie_similarities = self.data['user_movie_similarities']
                
                # Get training ratings to avoid recommending already rated movies
                if 'train_ratings' in self.data:
                    train_ratings = self.data['train_ratings']
                    # Create user to movies mapping for fast lookups
                    user_rated_movies = defaultdict(set)
                    for _, row in train_ratings.iterrows():
                        user_rated_movies[row['userId']].add(row['movieId'])
                else:
                    user_rated_movies = defaultdict(set)
                
                # Generate recommendations for all users
                content_recommendations = {}
                
                # Process users
                total_users = len(user_movie_similarities)
                for i, (user_id, sims) in enumerate(user_movie_similarities.items()):
                    # Get movies already rated by the user
                    rated_movies = user_rated_movies.get(user_id, set())
                    
                    # Filter out already rated movies
                    candidates = [(movie_id, sim) for movie_id, sim in sims.items() 
                                 if movie_id not in rated_movies]
                    
                    # Sort by similarity (descending)
                    recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)[:10]  # Top 10
                    
                    if recommendations:
                        content_recommendations[user_id] = recommendations
                    
                    # Log progress
                    if (i+1) % 100 == 0 or (i+1) == total_users:
                        print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%)")
                
                self.data['content_recommendations'] = content_recommendations
                
                print(f"Generated content-based recommendations for {len(content_recommendations)} users in {time.time() - start_time:.2f}s")
                return True
                
            except Exception as e:
                print(f"Error generating content-based recommendations: {str(e)}")
                return False
        
        return False
    
    def generate_collaborative_recommendations(self):
        """
        Generate collaborative filtering recommendations if not already available
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if 'collaborative_recommendations' in self.data and self.data['collaborative_recommendations']:
            print("Collaborative filtering recommendations already available, skipping generation")
            return True
            
        # Check if we have the necessary components for generating DNN-based recommendations
        if 'dnn_model' not in self.data or 'user_genre_preferences' not in self.data or 'movie_genre_features' not in self.data:
            print("Error: Required components for DNN-based recommendation generation not available")
            return False
        
        print("\nGenerating collaborative filtering recommendations...")
        start_time = time.time()
        
        try:
            # Get required data
            dnn_model = self.data['dnn_model']
            user_genre_preferences = self.data['user_genre_preferences']
            movie_genre_features = self.data['movie_genre_features']
            train_ratings = self.data.get('train_ratings')
            
            # Generate recommendations for all users
            all_users = user_genre_preferences['userId'].unique()
            genre_columns = [col for col in movie_genre_features.columns if col != 'movieId']
            
            collab_recommendations = {}
            
            # Process each user
            for i, user_id in enumerate(all_users):
                # Skip if user not found in genre preferences
                if user_id not in user_genre_preferences['userId'].values:
                    continue
                
                # Get user genre preferences
                user_prefs = user_genre_preferences[user_genre_preferences['userId'] == user_id].iloc[0]
                
                # Get movies already rated by the user
                rated_movies = set()
                if train_ratings is not None:
                    rated_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
                
                # Calculate predictions for all unrated movies
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
                
                # Store top recommendations
                collab_recommendations[user_id] = movie_predictions[:10]  # Top 10 recommendations
                
                # Log progress
                if (i+1) % 100 == 0 or (i+1) == len(all_users):
                    print(f"Processed {i+1}/{len(all_users)} users ({(i+1)/len(all_users)*100:.1f}%)")
            
            self.data['collaborative_recommendations'] = collab_recommendations
            
            print(f"Generated collaborative filtering recommendations for {len(collab_recommendations)} users in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error generating collaborative filtering recommendations: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Hybrid Movie Recommendation System')
    parser.add_argument('--content_path', type=str, default='./content-recommendations', 
                      help='Path to content-based model files')
    parser.add_argument('--collab_path', type=str, default='./collaborative-recommendations',
                      help='Path to collaborative filtering model files')
    parser.add_argument('--output_path', type=str, default='./hybrid_recommendations',
                      help='Path to save hybrid recommendation results')
    parser.add_argument('--alpha', type=float, default=0.3,
                      help='Weight for content-based recommendations (1-alpha for collaborative)')
    parser.add_argument('--optimize_alpha', action='store_true',
                      help='Find optimal alpha value')
    parser.add_argument('--batch_mode', action='store_true',
                      help='Run in batch mode (no interactive prompts)')
    parser.add_argument('--num_recs', type=int, default=10,
                      help='Number of recommendations to generate')
    parser.add_argument('--generate', action='store_true',
                      help='Generate recommendations if they are not already available')
    
    args = parser.parse_args()
    
    # Create and initialize the hybrid recommender
    recommender = HybridRecommender(
        content_model_path=args.content_path,
        collab_model_path=args.collab_path,
        output_path=args.output_path,
        alpha=args.alpha
    )
    
    # Load data
    recommender.load_data()
    
    # Generate recommendations if requested and they don't exist
    if args.generate:
        if 'content_recommendations' not in recommender.data or not recommender.data['content_recommendations']:
            recommender.generate_content_based_recommendations()
        
        if 'collaborative_recommendations' not in recommender.data or not recommender.data['collaborative_recommendations']:
            recommender.generate_collaborative_recommendations()
    
    # Find optimal alpha if requested
    if args.optimize_alpha:
        optimal_alpha = recommender.find_optimal_alpha()
        print(f"Optimal alpha: {optimal_alpha:.2f}")
    
    # Combine recommendations
    recommender.combine_recommendations(top_n=args.num_recs)
    
    # Evaluate
    evaluation_metrics = recommender.evaluate()
    
    # Compare with individual models
    print("\nModel Performance Comparison:")
    headers = ["Model", "RMSE", "MAE", "Predictions"]
    rows = []
    
    # Content-based model metrics
    if 'content_evaluation' in recommender.data:
        rows.append([
            "Content-Based",
            f"{recommender.data['content_evaluation']['rmse']:.4f}",
            f"{recommender.data['content_evaluation'].get('mae', 'N/A')}",
            f"{recommender.data['content_evaluation'].get('num_predictions', 'N/A')}"
        ])
    
    # Collaborative filtering model metrics
    if 'dnn_evaluation' in recommender.data:
        rows.append([
            "Collaborative",
            f"{recommender.data['dnn_evaluation']['rmse']:.4f}",
            f"{recommender.data['dnn_evaluation']['mae']:.4f}",
            f"{recommender.data['dnn_evaluation']['num_predictions']}"
        ])
    
    # Hybrid model metrics
    if evaluation_metrics:
        rows.append([
            f"Hybrid (α={recommender.alpha:.2f})",
            f"{evaluation_metrics['rmse']:.4f}",
            f"{evaluation_metrics['mae']:.4f}",
            f"{evaluation_metrics['num_predictions']}"
        ])
    
    # Print table
    if rows:
        # Calculate column widths
        col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print table header
        print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
        print("| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
        print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
        
        # Print table rows
        for row in rows:
            print("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |")
        
        print("+" + "+".join("-" * (width + 2) for width in col_widths) + "+")
    
    if args.batch_mode:
        print("\nHybrid Recommendation System completed successfully!")
        return
    
    # Interactive mode - prompt for user IDs
    while True:
        try:
            user_input = input("\nEnter user id to recommend (blank to stop): ")
            
            if not user_input.strip():
                print("\nExiting recommendation system. Goodbye!")
                break
            
            try:
                user_id = int(user_input)
            except ValueError:
                print("Please enter a valid numeric user ID.")
                continue
            
            print(f"\nGenerating recommendations for User ID: {user_id}")
            
            # Get user's rated movies
            rated_movies = recommender.get_user_rated_movies(user_id)
            
            if len(rated_movies) > 0:
                print(f"\nUser {user_id} has rated {len(rated_movies)} movies")
                print("\nSample of user's highest rated movies:")
                top_rated = rated_movies.sort_values('rating', ascending=False).head(5)
                for _, row in top_rated.iterrows():
                    print(f"  '{row['title']}' - Rating: {row['rating']:.1f}")
            else:
                print(f"\nUser {user_id} has no rated movies in the training set")
            
            # Get recommendations
            recommendations = recommender.recommend_for_user(user_id, n=args.num_recs)
            
            if recommendations:
                print(f"\nTop {len(recommendations)} recommendations for User {user_id}:")
                for i, (movie_id, title, score) in enumerate(recommendations, 1):
                    print(f"{i}. '{title}' - Predicted Rating: {score:.2f}")
                    
                    # Get movie details
                    details = recommender.get_movie_details(movie_id)
                    if details and details['genres']:
                        print(f"   Genres: {', '.join(details['genres'])}")
                    if details and details['keywords']:
                        print(f"   Keywords: {', '.join(details['keywords'][:5])}")
            else:
                print(f"\nNo recommendations found for User {user_id}")
        
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
    
    print("\nHybrid Recommendation System completed successfully!")

if __name__ == "__main__":
    main()