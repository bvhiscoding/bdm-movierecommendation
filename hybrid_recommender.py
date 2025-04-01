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
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

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
    def __init__(self, content_model_path="./cuda_optimized_recommendations", collab_model_path="./recommendations", 
                 output_path="./hybrid_recommendations", alpha=0.3):
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
            self.data['movie_features'] = pd.read_csv('./processed/processed_movie_features.csv')
            print(f"Loaded features for {len(self.data['movie_features'])} movies")
        except Exception as e:
            print(f"Error loading movie features: {str(e)}")
            
        # Load content-based recommendations
        try:
            with open(os.path.join(self.content_model_path, 'content_based_recommendations.pkl'), 'rb') as f:
                self.data['content_recommendations'] = pickle.load(f)
            print(f"Loaded content-based recommendations for {len(self.data['content_recommendations'])} users")
        except Exception as e:
            print(f"Error loading content-based recommendations: {str(e)}")
            self.data['content_recommendations'] = {}
            
        # Load collaborative filtering recommendations
        try:
            with open(os.path.join(self.collab_model_path, 'dnn_recommendations.pkl'), 'rb') as f:
                self.data['collaborative_recommendations'] = pickle.load(f)
            print(f"Loaded collaborative filtering recommendations for {len(self.data['collaborative_recommendations'])} users")
        except Exception as e:
            print(f"Error loading collaborative filtering recommendations: {str(e)}")
            self.data['collaborative_recommendations'] = {}
            
        # Load rating data for evaluation
        try:
            self.data['ratings'] = pd.read_csv('./processed/normalized_ratings.csv')
            
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
            
            print(f"Loaded {len(self.data['ratings'])} ratings")
            print(f"Split into {len(self.data['train_ratings'])} training and {len(self.data['test_ratings'])} testing ratings")
        except Exception as e:
            print(f"Error loading ratings data: {str(e)}")
            
        print(f"Data loading completed in {time.time() - start_time:.2f}s")
        
        # Get common users
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
        
        # Get recommendations
        content_recs = self.data['content_recommendations']
        collab_recs = self.data['collaborative_recommendations']
        
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
        combined_recs = self.data['combined_recommendations']
        test_ratings = self.data['test_ratings']
        
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

def main():
    parser = argparse.ArgumentParser(description='Hybrid Movie Recommendation System')
    parser.add_argument('--content_path', type=str, default='./cuda_optimized_recommendations', 
                      help='Path to content-based model files')
    parser.add_argument('--collab_path', type=str, default='./recommendations',
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
    
    # Find optimal alpha if requested
    if args.optimize_alpha:
        optimal_alpha = recommender.find_optimal_alpha()
        print(f"Optimal alpha: {optimal_alpha:.2f}")
    
    # Combine recommendations
    recommender.combine_recommendations(top_n=args.num_recs)
    
    # Evaluate
    recommender.evaluate()
    
    if args.batch_mode:
        print("\nHybrid Recommendation System completed successfully!")
        return
    
    # Interactive mode - prompt for user IDs
    while True:
        try:
            user_input = input("\nEnter user id to recommend: \n(blank to stop) ")
            
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