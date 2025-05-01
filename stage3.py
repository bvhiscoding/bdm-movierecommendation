import numpy as np
import pandas as pd
import os
import pickle
import logging
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, content_model_path="./rec/content-recommendations", 
                 collab_model_path="./rec/collaborative-recommendations", 
                 output_path="./rec/hybrid_recommendations", 
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
        self.optimal_alphas = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        print("\n" + "="*80)
        print(f"OPTIMIZED HYBRID MOVIE RECOMMENDATION SYSTEM (alpha={self.alpha:.2f})")
        print("="*80)
    
    def load_data(self):
        """
        Optimized data loading - only load the necessary evaluation files and recommendations
        """
        print("\nLoading essential data...")
        start_time = time.time()
        
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
        
        # Load DNN evaluation metrics
        try:
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
        except Exception as e:
            print(f"Error loading DNN evaluation metrics: {str(e)}")
        
        # Load content-based recommendations
        try:
            content_recs_path = os.path.join(self.content_model_path, 'content_based_recommendations.pkl')
            if os.path.exists(content_recs_path):
                with open(content_recs_path, 'rb') as f:
                    self.data['content_recommendations'] = pickle.load(f)
                print(f"Loaded content-based recommendations for {len(self.data['content_recommendations'])} users")
        except Exception as e:
            print(f"Error loading content-based recommendations: {str(e)}")
        
        # Load collaborative filtering recommendations
        try:
            collab_recs_path = os.path.join(self.collab_model_path, 'dnn_recommendations.pkl')
            if os.path.exists(collab_recs_path):
                with open(collab_recs_path, 'rb') as f:
                    self.data['collaborative_recommendations'] = pickle.load(f)
                print(f"Loaded collaborative filtering recommendations for {len(self.data['collaborative_recommendations'])} users")
        except Exception as e:
            print(f"Error loading collaborative filtering recommendations: {str(e)}")
                
        # Load minimal user rating data needed just for calculating adaptive alpha
        try:
            # We only need userId and rating count, so we'll parse this from the CSV directly
            ratings_path = './processed/normalized_ratings.csv'
            if os.path.exists(ratings_path):
                # Read only the userId column to calculate rating counts
                ratings_df = pd.read_csv(ratings_path, usecols=['userId'])
                user_rating_counts = ratings_df['userId'].value_counts().reset_index()
                user_rating_counts.columns = ['userId', 'rating_count']
                self.data['user_rating_counts'] = user_rating_counts
                print(f"Loaded rating counts for {len(user_rating_counts)} users")
        except Exception as e:
            print(f"Error loading user rating counts: {str(e)}")
            
        # Optionally load minimal movie metadata (just for displaying recommendations)
        try:
            movie_features_path = './processed/processed_movie_features.csv'
            if os.path.exists(movie_features_path):
                # Read only the essential columns
                self.data['movie_features'] = pd.read_csv(
                    movie_features_path, 
                    usecols=['movieId', 'title']  # Only load the columns we need
                )
                print(f"Loaded minimal movie metadata for {len(self.data['movie_features'])} movies")
        except Exception as e:
            print(f"Error loading movie metadata: {str(e)}")
        
        # Try to load previously learned optimal alphas if they exist
        try:
            optimal_alphas_path = os.path.join(self.output_path, 'optimal_alphas.json')
            if os.path.exists(optimal_alphas_path):
                with open(optimal_alphas_path, 'r') as f:
                    self.optimal_alphas = json.load(f)
                print(f"Loaded learned optimal alpha values for {len(self.optimal_alphas)} rating bins")
        except Exception as e:
            print(f"No learned alpha values found: {str(e)}")
        
        print(f"Data loading completed in {time.time() - start_time:.2f}s")
        
        # Get common users for both recommendation systems
        self.common_users = set()
        if 'content_recommendations' in self.data and 'collaborative_recommendations' in self.data:
            self.common_users = set(self.data['content_recommendations'].keys()) & set(self.data['collaborative_recommendations'].keys())
            print(f"Found {len(self.common_users)} users with both content-based and collaborative recommendations")
        
        return self.data

    def get_adaptive_alpha(self, user_id):
        """
        Get optimized alpha value based on user's rating count and learned optimal values
        
        Parameters:
        -----------
        user_id: int
            User ID
            
        Returns:
        --------
        float
            Optimized alpha value
        """
        # Get user's rating count
        rating_count = 0
        if 'user_rating_counts' in self.data:
            user_data = self.data['user_rating_counts'][self.data['user_rating_counts']['userId'] == user_id]
            if not user_data.empty:
                rating_count = user_data.iloc[0]['rating_count']
        
        # Check if we have learned optimal alpha values
        if self.optimal_alphas:
            # Find appropriate bin for this user
            for bin_name, alpha in self.optimal_alphas.items():
                if '-' in bin_name:
                    # Range bin (e.g., "10-24")
                    lower, upper = map(int, bin_name.split('-'))
                    if lower <= rating_count <= upper:
                        return alpha
                elif '+' in bin_name:
                    # Last bin (e.g., "300+")
                    threshold = int(bin_name.replace('+', ''))
                    if rating_count >= threshold:
                        return alpha
        
        # Fallback to default values if no learned values are available
        if rating_count <= 10:
            return 0.01
        elif rating_count <= 25:
            return 0.05
        elif rating_count <= 50:
            return 0.15
        elif rating_count <= 100:
            return 0.25
        elif rating_count <= 150:
            return 0.35
        elif rating_count <= 200:
            return 0.45
        elif rating_count <= 300:
            return 0.55
        else:  # > 300
            return 0.65
    
    def normalize_prediction(self, prediction):
        """
        Normalize a prediction to the 0-1 range
        
        Parameters:
        -----------
        prediction: float
            Prediction value in the 0.5-5.0 range
            
        Returns:
        --------
        float
            Normalized prediction in the 0-1 range
        """
        # Normalize from rating scale [0.5, 5.0] to [0, 1]
        return (prediction - 0.5) / 4.5
    
    def denormalize_prediction(self, normalized_prediction):
        """
        Convert a normalized prediction back to the 0.5-5.0 range
        
        Parameters:
        -----------
        normalized_prediction: float
            Normalized prediction in the 0-1 range
            
        Returns:
        --------
        float
            Prediction value in the 0.5-5.0 range
        """
        # Convert from [0, 1] back to rating scale [0.5, 5.0]
        return 0.5 + 4.5 * normalized_prediction
    
    def _prepare_validation_data(self):
        """
        Prepare validation data for alpha optimization
        
        Returns:
        --------
        dict
            Validation data for evaluating different alpha values
        """
        # We need content and collaborative predictions for the same user-item pairs
        validation_data = {}
        
        # Get common users with both types of recommendations
        common_users = self.common_users
        if not common_users:
            print("Error: No common users found with both types of recommendations")
            return None
        
        # Sample users for validation
        sample_size = min(1000, len(common_users))
        validation_users = np.random.choice(list(common_users), sample_size, replace=False)
        
        # For each user, find items with both content and collaborative predictions
        for user_id in validation_users:
            user_validation = {}
            
            # Get content recommendations
            content_recs = {movie_id: score for movie_id, score 
                           in self.data['content_recommendations'].get(user_id, [])}
            
            # Get collaborative recommendations
            collab_recs = {movie_id: rating for movie_id, rating 
                          in self.data['collaborative_recommendations'].get(user_id, [])}
            
            # Find common movies
            common_movies = set(content_recs.keys()) & set(collab_recs.keys())
            
            if not common_movies:
                continue
            
            # Add to validation data
            for movie_id in common_movies:
                user_validation[movie_id] = {
                    'content_score': content_recs[movie_id],
                    'collab_score': self.normalize_prediction(collab_recs[movie_id])
                }
            
            if user_validation:
                validation_data[user_id] = user_validation
        
        print(f"Prepared validation data for {len(validation_data)} users")
        return validation_data

    def _evaluate_alpha_for_users(self, alpha, user_ids, validation_data):
        """
        Evaluate RMSE for a specific alpha value on a set of users
        
        Parameters:
        -----------
        alpha: float
            Alpha value to evaluate
        user_ids: list
            List of user IDs to evaluate
        validation_data: dict
            Validation data
            
        Returns:
        --------
        float
            RMSE for the given alpha value
        """
        squared_errors = []
        
        for user_id in user_ids:
            if user_id not in validation_data:
                continue
            
            user_validation = validation_data[user_id]
            
            for movie_id, data in user_validation.items():
                # Combine predictions using alpha
                combined_score = alpha * data['content_score'] + (1 - alpha) * data['collab_score']
                
                # Convert to rating scale
                final_rating = self.denormalize_prediction(combined_score)
                
                # Compare with ground truth (use collaborative rating as ground truth)
                true_rating = self.denormalize_prediction(data['collab_score'])
                
                # Calculate squared error
                squared_error = (final_rating - true_rating) ** 2
                squared_errors.append(squared_error)
        
        if not squared_errors:
            return float('inf')
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        return rmse
    
    def learn_optimal_alpha_values(self, rating_bins=[0, 10, 25, 50, 100, 150, 200, 300], 
                                 alpha_values=None):
        """
        Learn optimal alpha values for different rating count bins using grid search
        
        Parameters:
        -----------
        rating_bins: list
            Rating count thresholds for binning users
        alpha_values: array
            Possible alpha values to explore
            
        Returns:
        --------
        dict
            Mapping of rating bins to optimal alpha values
        """
        if alpha_values is None:
            alpha_values = np.arange(0.0, 1.05, 0.05)
            
        print(f"\nLearning optimal alpha values with grid search for {len(rating_bins)} bins...")
        
        # Get user rating counts and test data
        if 'user_rating_counts' not in self.data:
            print("Error: User rating counts not available")
            return None
        
        # Create user bins based on rating counts
        user_bins = {}
        for i in range(len(rating_bins)):
            if i == len(rating_bins) - 1:
                bin_name = f"{rating_bins[i]}+"
            else:
                bin_name = f"{rating_bins[i]}-{rating_bins[i+1]-1}"
            user_bins[bin_name] = []
        
        # Assign users to bins
        for _, row in self.data['user_rating_counts'].iterrows():
            user_id = row['userId']
            rating_count = row['rating_count']
            
            # Find appropriate bin
            for i in range(len(rating_bins)):
                if i == len(rating_bins) - 1:
                    if rating_count >= rating_bins[i]:
                        bin_name = f"{rating_bins[i]}+"
                        user_bins[bin_name].append(user_id)
                        break
                else:
                    if rating_bins[i] <= rating_count < rating_bins[i+1]:
                        bin_name = f"{rating_bins[i]}-{rating_bins[i+1]-1}"
                        user_bins[bin_name].append(user_id)
                        break
        
        # Print user distribution in bins
        print("\nUser distribution in rating bins:")
        for bin_name, users in user_bins.items():
            print(f"  {bin_name} ratings: {len(users)} users")
        
        # Prepare validation data
        validation_data = self._prepare_validation_data()
        if validation_data is None:
            print("Error: Could not prepare validation data")
            return None
        
        # For each bin, find optimal alpha value
        optimal_alphas = {}
        
        for bin_name, user_ids in user_bins.items():
            print(f"\nFinding optimal alpha for bin {bin_name}...")
            
            if not user_ids:
                print(f"  No users in bin {bin_name}, skipping")
                continue
            
            # Filter users to those with validation data
            valid_users = [uid for uid in user_ids if uid in validation_data]
            
            if not valid_users:
                print(f"  No users with validation data in bin {bin_name}, skipping")
                continue
            
            # Select a sample of users for faster evaluation if bin is large
            sample_size = min(500, len(valid_users))
            sample_users = np.random.choice(valid_users, sample_size, replace=False)
            
            best_rmse = float('inf')
            best_alpha = 0.0
            
            for alpha in alpha_values:
                # Evaluate RMSE for this alpha value
                rmse = self._evaluate_alpha_for_users(alpha, sample_users, validation_data)
                
                print(f"  Alpha {alpha:.2f}: RMSE {rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha
            
            optimal_alphas[bin_name] = best_alpha
            print(f"  Optimal alpha for bin {bin_name}: {best_alpha:.2f} (RMSE: {best_rmse:.4f})")
        
        # Save optimal alpha values
        self.optimal_alphas = optimal_alphas
        
        # Save to file
        with open(os.path.join(self.output_path, 'optimal_alphas.json'), 'w') as f:
            json.dump(optimal_alphas, f, indent=2)
        
        print(f"\nOptimal alpha values saved to {os.path.join(self.output_path, 'optimal_alphas.json')}")
        
        return optimal_alphas
    
    def combine_recommendations(self, top_n=10, use_adaptive_alpha=True):
        """
        Combine content-based and collaborative filtering recommendations with optimized weighting
        
        Parameters:
        -----------
        top_n: int
            Number of recommendations to generate per user
        use_adaptive_alpha: bool
            Whether to use adaptive alpha based on user rating count
            
        Returns:
        --------
        dict
            User ID to list of (movie_id, score) tuples
        """
        print(f"\nCombining recommendations with {'adaptive' if use_adaptive_alpha else 'fixed'} alpha values...")
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
        alpha_stats = {'values': [], 'count_categories': {}}
        
        # Get all users from both recommendation sets
        all_users = set(content_recs.keys()) | set(collab_recs.keys())
        total_users = len(all_users)
        
        for i, user_id in enumerate(all_users):
            # Get appropriate alpha value for this user
            if use_adaptive_alpha:
                alpha = self.get_adaptive_alpha(user_id)
                # Track alpha statistics
                alpha_stats['values'].append(alpha)
                
                # Get user's rating count
                rating_count = 0
                if 'user_rating_counts' in self.data:
                    user_data = self.data['user_rating_counts'][self.data['user_rating_counts']['userId'] == user_id]
                    if not user_data.empty:
                        rating_count = user_data.iloc[0]['rating_count']
                
                # Categorize for statistics
                count_category = "<=10" if rating_count <= 10 else "11-25" if rating_count <= 25 else "26-50" if rating_count <= 50 else "51-100" if rating_count <= 100 else "101-150" if rating_count <= 150 else "151-200" if rating_count <= 200 else "201-300" if rating_count <= 300 else ">300"
                if count_category in alpha_stats['count_categories']:
                    alpha_stats['count_categories'][count_category]['count'] += 1
                    alpha_stats['count_categories'][count_category]['alpha_sum'] += alpha
                else:
                    alpha_stats['count_categories'][count_category] = {'count': 1, 'alpha_sum': alpha}
            else:
                alpha = self.alpha
            
            # Initialize combined recommendations dictionary for this user
            user_combined_recs = {}
            
            # Add content-based recommendations if available
            if user_id in content_recs:
                for movie_id, score in content_recs[user_id]:
                    # Scores from content-based are already normalized (0-1), just store them
                    user_combined_recs[movie_id] = {'content_score': score, 'content_available': True}
            
            # Add collaborative filtering recommendations if available
            if user_id in collab_recs:
                for movie_id, rating in collab_recs[user_id]:
                    # Normalize the collaborative rating to 0-1 scale
                    collab_score = self.normalize_prediction(rating)
                    
                    if movie_id in user_combined_recs:
                        user_combined_recs[movie_id]['collab_score'] = collab_score
                        user_combined_recs[movie_id]['collab_available'] = True
                    else:
                        user_combined_recs[movie_id] = {
                            'collab_score': collab_score, 
                            'collab_available': True,
                            'content_available': False
                        }
            
            # Calculate final scores with proper normalization
            final_recommendations = []
            for movie_id, data in user_combined_recs.items():
                # Check which models provided predictions
                content_available = data.get('content_available', False)
                collab_available = data.get('collab_available', False)
                
                if content_available and collab_available:
                    # We have both predictions, use the weighted average
                    content_score = data['content_score']
                    collab_score = data['collab_score']
                    combined_score = alpha * content_score + (1 - alpha) * collab_score
                elif content_available:
                    # Only content-based prediction available
                    combined_score = data['content_score']
                elif collab_available:
                    # Only collaborative prediction available
                    combined_score = data['collab_score']
                
                # Convert back to rating scale for storage
                final_rating = self.denormalize_prediction(combined_score)
                final_recommendations.append((movie_id, final_rating))
            
            # Sort by final score and limit to top_n
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
            combined_recommendations[user_id] = final_recommendations[:top_n]
            
            # Log progress
            if (i+1) % 1000 == 0 or (i+1) == total_users:
                print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%)")
        
        self.data['combined_recommendations'] = combined_recommendations
        
        # Print alpha statistics if using adaptive alpha
        if use_adaptive_alpha and alpha_stats['values']:
            print("\nAdaptive Alpha Statistics:")
            print(f"Average alpha: {np.mean(alpha_stats['values']):.4f}")
            print(f"Min alpha: {min(alpha_stats['values']):.4f}, Max alpha: {max(alpha_stats['values']):.4f}")
            print("\nAlpha by user rating count:")
            
            # Sort categories by rating count
            def sort_key(category):
                if category.startswith("<="):
                    return (0, int(category[2:]))
                elif category.startswith(">"):
                    return (999, int(category[1:]))
                else:
                    # Format like "11-25"
                    lower = int(category.split("-")[0])
                    return (lower, lower)
                
            for category, stats in sorted(alpha_stats['count_categories'].items(), key=lambda x: sort_key(x[0])):
                avg_alpha = stats['alpha_sum'] / stats['count']
                print(f"  {category} ratings: {stats['count']} users, avg alpha = {avg_alpha:.4f}")
            
            # Save alpha statistics to a file
            with open(os.path.join(self.output_path, 'alpha_stats.txt'), 'w') as f:
                f.write(f"Adaptive Alpha Statistics:\n")
                f.write(f"Average alpha: {np.mean(alpha_stats['values']):.4f}\n")
                f.write(f"Min alpha: {min(alpha_stats['values']):.4f}, Max alpha: {max(alpha_stats['values']):.4f}\n\n")
                f.write("Alpha by user rating count:\n")
                for category, stats in sorted(alpha_stats['count_categories'].items(), key=lambda x: sort_key(x[0])):
                    avg_alpha = stats['alpha_sum'] / stats['count']
                    f.write(f"  {category} ratings: {stats['count']} users, avg alpha = {avg_alpha:.4f}\n")
            
            # Create a visualization of alpha distribution
            plt.figure(figsize=(12, 6))
            plt.hist(alpha_stats['values'], bins=20, alpha=0.7)
            plt.title('Distribution of Alpha Values')
            plt.xlabel('Alpha Value')
            plt.ylabel('Number of Users')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_path, 'alpha_distribution.png'))
            plt.close()
        
        print(f"Combined recommendations for {len(combined_recommendations)} users in {time.time() - start_time:.2f}s")
        
        # Save combined recommendations
        with open(os.path.join(self.output_path, 'combined_recommendations.pkl'), 'wb') as f:
            pickle.dump(combined_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in combined_recommendations.items():
            # Get user's alpha
            if use_adaptive_alpha:
                user_alpha = self.get_adaptive_alpha(user_id)
            else:
                user_alpha = self.alpha
                
            # Get user's rating count
            rating_count = 0
            if 'user_rating_counts' in self.data:
                user_data = self.data['user_rating_counts'][self.data['user_rating_counts']['userId'] == user_id]
                if not user_data.empty:
                    rating_count = user_data.iloc[0]['rating_count']
            
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
                    'score': score,
                    'alpha': user_alpha,
                    'rating_count': rating_count
                })
        
        if recommendations_list:
            recommendations_df = pd.DataFrame(recommendations_list)
            recommendations_df.to_csv(os.path.join(self.output_path, 'combined_recommendations.csv'), index=False)
            print(f"Saved combined recommendations to CSV with {len(recommendations_df)} entries")
        
        return combined_recommendations
    
    def evaluate(self, use_adaptive_alpha=True):
        """
        Evaluate the hybrid recommendation system using the pre-computed metrics
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print(f"\nEvaluating hybrid recommendation system using pre-computed metrics...")
        
        # Check if we have the pre-computed metrics
        if 'content_evaluation' not in self.data or 'dnn_evaluation' not in self.data:
            print("Cannot evaluate: Missing pre-computed evaluation metrics")
            return None
        
        content_rmse = self.data['content_evaluation']['rmse']
        dnn_rmse = self.data['dnn_evaluation']['rmse']
        
        content_mae = self.data['content_evaluation'].get('mae', 0)
        dnn_mae = self.data['dnn_evaluation'].get('mae', 0)
        
        # Calculate combined metrics based on alpha distribution
        if use_adaptive_alpha and 'user_rating_counts' in self.data:
            # Get user distribution in different rating count bins
            rating_counts = {}
            for _, row in self.data['user_rating_counts'].iterrows():
                rating_count = row['rating_count']
                if rating_count <= 10:
                    bin_name = "<=10"
                elif rating_count <= 25:
                    bin_name = "11-25"
                elif rating_count <= 50:
                    bin_name = "26-50"
                elif rating_count <= 100:
                    bin_name = "51-100"
                elif rating_count <= 150:
                    bin_name = "101-150"
                elif rating_count <= 200:
                    bin_name = "151-200"
                elif rating_count <= 300:
                    bin_name = "201-300"
                else:  # > 300
                    bin_name = ">300"
                
                if bin_name in rating_counts:
                    rating_counts[bin_name] += 1
                else:
                    rating_counts[bin_name] = 1
            
            total_users = sum(rating_counts.values())
            
            # Calculate weighted RMSE and MAE
            weighted_rmse = 0
            weighted_mae = 0
            
            # Function to get alpha for a bin
            def get_bin_alpha(bin_name):
                if self.optimal_alphas and bin_name in self.optimal_alphas:
                    return self.optimal_alphas[bin_name]
                
                # Fallback to default values
                if bin_name == "<=10":
                    return 0.01
                elif bin_name == "11-25":
                    return 0.05
                elif bin_name == "26-50":
                    return 0.15
                elif bin_name == "51-100":
                    return 0.25
                elif bin_name == "101-150":
                    return 0.35
                elif bin_name == "151-200":
                    return 0.45
                elif bin_name == "201-300":
                    return 0.55
                else:  # ">300"
                    return 0.65
            
            for bin_name, count in rating_counts.items():
                weight = count / total_users
                alpha = get_bin_alpha(bin_name)
                
                # Calculate weighted metrics
                bin_rmse = alpha * content_rmse + (1 - alpha) * dnn_rmse
                bin_mae = alpha * content_mae + (1 - alpha) * dnn_mae
                
                weighted_rmse += weight * bin_rmse
                weighted_mae += weight * bin_mae
                
                print(f"  Bin {bin_name}: {count} users, alpha = {alpha:.4f}, RMSE = {bin_rmse:.4f}")
            
            # Store hybrid evaluation metrics
            hybrid_metrics = {
                'rmse': weighted_rmse,
                'mae': weighted_mae,
                'num_predictions': self.data['content_evaluation'].get('num_predictions', 0),
                'use_adaptive_alpha': True
            }
        else:
            # Use fixed alpha
            hybrid_rmse = self.alpha * content_rmse + (1 - self.alpha) * dnn_rmse
            hybrid_mae = self.alpha * content_mae + (1 - self.alpha) * dnn_mae
            
            hybrid_metrics = {
                'rmse': hybrid_rmse,
                'mae': hybrid_mae,
                'num_predictions': self.data['content_evaluation'].get('num_predictions', 0),
                'use_adaptive_alpha': False
            }
        
        print(f"\nHybrid model evaluation (with {'adaptive' if use_adaptive_alpha else 'fixed'} alpha):")
        print(f"RMSE: {hybrid_metrics['rmse']:.4f}")
        print(f"MAE: {hybrid_metrics['mae']:.4f}")
        
        # Save metrics
        pd.DataFrame([hybrid_metrics]).to_csv(os.path.join(self.output_path, 'evaluation_metrics.csv'), index=False)
        
        return hybrid_metrics
    
    def update_alpha_values(self, new_ratings, learning_rate=0.01):
        """
        Update alpha values based on new user ratings
        
        Parameters:
        -----------
        new_ratings: DataFrame
            New ratings data with userId, movieId, rating columns
        learning_rate: float
            Learning rate for updates
            
        Returns:
        --------
        dict
            Updated alpha values
        """
        print("Updating alpha values with new ratings data...")
        
        if not self.optimal_alphas:
            print("No optimal alpha values to update")
            return None
        
        # Group new ratings by user
        user_ratings = new_ratings.groupby('userId')
        
        for user_id, ratings in user_ratings:
            # Get current alpha for this user
            current_alpha = self.get_adaptive_alpha(user_id)
            
            # Calculate error for content-based and collaborative predictions
            content_errors = []
            collab_errors = []
            
            for _, row in ratings.iterrows():
                movie_id = row['movieId']
                true_rating = row['rating']
                
                # Get content-based prediction if available
                content_pred = None
                if user_id in self.data.get('content_recommendations', {}) and movie_id in [m for m, _ in self.data['content_recommendations'][user_id]]:
                    for m, s in self.data['content_recommendations'][user_id]:
                        if m == movie_id:
                            content_pred = self.denormalize_prediction(s)
                            break
                
                # Get collaborative prediction if available
                collab_pred = None
                if user_id in self.data.get('collaborative_recommendations', {}) and movie_id in [m for m, _ in self.data['collaborative_recommendations'][user_id]]:
                    for m, r in self.data['collaborative_recommendations'][user_id]:
                        if m == movie_id:
                            collab_pred = r
                            break
                
                # Calculate errors if both predictions are available
                if content_pred is not None and collab_pred is not None:
                    content_error = (content_pred - true_rating) ** 2
                    collab_error = (collab_pred - true_rating) ** 2
                    
                    content_errors.append(content_error)
                    collab_errors.append(collab_error)
            
            # Update alpha if we have errors
            if content_errors and collab_errors:
                avg_content_error = np.mean(content_errors)
                avg_collab_error = np.mean(collab_errors)
                
                # Adjust alpha based on relative performance
                # If content error is higher, decrease alpha; if collab error is higher, increase alpha
                error_diff = avg_content_error - avg_collab_error
                delta_alpha = learning_rate * error_diff
                
                new_alpha = current_alpha - delta_alpha
                new_alpha = max(0.0, min(1.0, new_alpha))  # Clip to [0, 1]
                
                # Find appropriate bin for this user
                rating_count = 0
                if 'user_rating_counts' in self.data:
                    user_data = self.data['user_rating_counts'][self.data['user_rating_counts']['userId'] == user_id]
                    if not user_data.empty:
                        rating_count = user_data.iloc[0]['rating_count']
                
                for bin_name in self.optimal_alphas.keys():
                    if '-' in bin_name:
                        lower, upper = map(int, bin_name.split('-'))
                        if lower <= rating_count <= upper:
                            # Weighted update of bin alpha
                            self.optimal_alphas[bin_name] = 0.9 * self.optimal_alphas[bin_name] + 0.1 * new_alpha
                            break
                    elif '+' in bin_name:
                        threshold = int(bin_name.replace('+', ''))
                        if rating_count >= threshold:
                            self.optimal_alphas[bin_name] = 0.9 * self.optimal_alphas[bin_name] + 0.1 * new_alpha
                            break
        
        # Save updated alpha values
        with open(os.path.join(self.output_path, 'optimal_alphas.json'), 'w') as f:
            json.dump(self.optimal_alphas, f, indent=2)
        
        print(f"Updated alpha values saved to {os.path.join(self.output_path, 'optimal_alphas.json')}")
        
        return self.optimal_alphas
    
    def recommend_for_user(self, user_id, n=10, use_adaptive_alpha=True):
        """
        Get recommendations for a specific user
        
        Parameters:
        -----------
        user_id: int
            User ID
        n: int
            Number of recommendations to return
        use_adaptive_alpha: bool
            Whether to use adaptive alpha based on user rating count
            
        Returns:
        --------
        list
            List of (movie_id, title, score) tuples
        """
        # Check if user has recommendations
        if 'combined_recommendations' not in self.data or user_id not in self.data['combined_recommendations']:
            print(f"No pre-computed recommendations found for user {user_id}")
            
            # Look for individual model recommendations
            content_recs = []
            if 'content_recommendations' in self.data and user_id in self.data['content_recommendations']:
                content_recs = self.data['content_recommendations'][user_id]
            
            collab_recs = []
            if 'collaborative_recommendations' in self.data and user_id in self.data['collaborative_recommendations']:
                collab_recs = self.data['collaborative_recommendations'][user_id]
            
            if not content_recs and not collab_recs:
                print(f"No recommendations available for user {user_id}")
                return []
            
            # Get alpha for this user
            alpha = self.get_adaptive_alpha(user_id) if use_adaptive_alpha else self.alpha
            
            # Combine available recommendations
            combined_recs = {}
            
            # Add content-based recs
            for movie_id, score in content_recs:
                combined_recs[movie_id] = {"content_score": score, "has_content": True}
            
            # Add collaborative recs
            for movie_id, rating in collab_recs:
                collab_score = self.normalize_prediction(rating)
                if movie_id in combined_recs:
                    combined_recs[movie_id]["collab_score"] = collab_score
                    combined_recs[movie_id]["has_collab"] = True
                else:
                    combined_recs[movie_id] = {"collab_score": collab_score, "has_collab": True}
            
            # Calculate final scores
            recommendations = []
            for movie_id, data in combined_recs.items():
                if data.get("has_content", False) and data.get("has_collab", False):
                    combined_score = alpha * data["content_score"] + (1 - alpha) * data["collab_score"]
                elif data.get("has_content", False):
                    combined_score = data["content_score"]
                elif data.get("has_collab", False):
                    combined_score = data["collab_score"]
                else:
                    continue
                
                final_rating = self.denormalize_prediction(combined_score)
                recommendations.append((movie_id, final_rating))
            
            # Sort and get top-n
            recommendations.sort(key=lambda x: x[1], reverse=True)
            recommendations = recommendations[:n]
        else:
            # Use pre-computed recommendations
            recommendations = self.data['combined_recommendations'][user_id][:n]
        
        # Format recommendations with titles
        formatted_recs = []
        for movie_id, score in recommendations:
            title = "Unknown"
            if 'movie_features' in self.data:
                movie_row = self.data['movie_features'][self.data['movie_features']['movieId'] == movie_id]
                if not movie_row.empty and 'title' in movie_row.columns:
                    title = movie_row.iloc[0]['title']
            
            formatted_recs.append((movie_id, title, score))
        
        return formatted_recs

def main():
    # Configuration section
    content_path = "./rec/content-recommendations"
    collab_path = "./rec/collaborative-recommendations"
    output_path = "./rec/hybrid_recommendations"
    alpha = 0.3
    learn_alpha = True  # Enable alpha learning
    adaptive_alpha = True
    num_recs = 10
    
    # Custom rating bins for learning
    rating_bins = [0, 10, 25, 50, 100, 150, 200, 300]
    
    # Create and initialize the hybrid recommender
    recommender = HybridRecommender(
        content_model_path=content_path,
        collab_model_path=collab_path,
        output_path=output_path,
        alpha=alpha
    )

    # Load data
    recommender.load_data()

    # Learn optimal alpha values if requested
    if learn_alpha:
        recommender.learn_optimal_alpha_values(rating_bins=rating_bins)

    # Combine recommendations with learned alphas
    recommender.combine_recommendations(top_n=num_recs, use_adaptive_alpha=adaptive_alpha)

    # Evaluate 
    evaluation_metrics = recommender.evaluate(use_adaptive_alpha=adaptive_alpha)

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
        alpha_desc = "Adaptive (Learned)" if adaptive_alpha else f"α={recommender.alpha:.2f}"
        rows.append([
            f"Hybrid ({alpha_desc})",
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

    # Show sample recommendations
    if 'combined_recommendations' in recommender.data:
        # Get a sample user ID
        sample_user_id = next(iter(recommender.data['combined_recommendations'].keys()))
        
        print(f"\nSample recommendations for user {sample_user_id}:")
        recs = recommender.recommend_for_user(sample_user_id, n=5)
        
        for i, (movie_id, title, score) in enumerate(recs, 1):
            print(f"{i}. {title} (ID: {movie_id}) - Score: {score:.2f}")
    
    # Create visualizations of results
    if 'combined_recommendations' in recommender.data:
        # Create visualization of recommendation scores
        scores = []
        for user_id, recs in recommender.data['combined_recommendations'].items():
            for _, score in recs:
                scores.append(score)
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7)
        plt.title('Distribution of Recommendation Scores')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'recommendation_scores.png'))
        plt.close()
        
        print(f"Visualization saved to {os.path.join(output_path, 'recommendation_scores.png')}")
    
    print("\nHybrid Recommendation System with Learning completed successfully!")

if __name__ == "__main__":
    main()