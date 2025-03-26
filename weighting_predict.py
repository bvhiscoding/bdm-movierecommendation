import numpy as np
import pandas as pd
import os
import pickle
import logging
from datetime import datetime
import heapq
from collections import defaultdict

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, data_path="./processed_data", recommendation_path="./recommendations", output_path="./hybrid_recommendations", content_weight=0.5, top_n=10):
        """
        Initialize the Hybrid Recommender system.
        
        Parameters:
        -----------
        data_path : str, default="./processed_data"
            Path to the directory containing processed data
        recommendation_path : str, default="./recommendations"
            Path to the directory containing CB and CF recommendations
        output_path : str, default="./hybrid_recommendations"
            Path to save hybrid recommendation results
        content_weight : float, default=0.5
            Weight for content-based recommendations (0.0 to 1.0)
            Collaborative filtering weight will be (1 - content_weight)
        top_n : int, default=10
            Number of top recommendations to generate
        """
        self.data_path = data_path
        self.recommendation_path = recommendation_path
        self.output_path = output_path
        self.content_weight = content_weight
        self.cf_weight = 1.0 - content_weight
        self.top_n = top_n
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Initialize data containers
        self.content_based_recs = None
        self.collaborative_recs = None
        self.movie_metadata = None
        self.user_ratings = None
    
    def load_recommendations(self):
        """Load pre-computed recommendations from both models"""
        logger.info("Loading recommendations from content-based and collaborative filtering models...")
        
        # Load content-based recommendations
        content_based_path = os.path.join(self.recommendation_path, 'content_based_recommendations.pkl')
        if os.path.exists(content_based_path):
            with open(content_based_path, 'rb') as f:
                self.content_based_recs = pickle.load(f)
            logger.info(f"Loaded content-based recommendations for {len(self.content_based_recs)} users")
        else:
            logger.error(f"Content-based recommendations not found at {content_based_path}")
        
        # Load collaborative filtering recommendations
        collaborative_path = os.path.join(self.recommendation_path, 'dnn_recommendations.pkl')
        if os.path.exists(collaborative_path):
            with open(collaborative_path, 'rb') as f:
                self.collaborative_recs = pickle.load(f)
            logger.info(f"Loaded collaborative filtering recommendations for {len(self.collaborative_recs)} users")
        else:
            logger.error(f"Collaborative filtering recommendations not found at {collaborative_path}")
        
        # Load movie metadata for recommendation display
        movie_metadata_path = os.path.join(self.data_path, 'movie_genres.csv')
        if os.path.exists(movie_metadata_path):
            self.movie_metadata = pd.read_csv(movie_metadata_path)
            logger.info(f"Loaded metadata for {len(self.movie_metadata)} movies")
        
        # Load user ratings for evaluation
        train_ratings_path = os.path.join(self.data_path, 'train_ratings.csv')
        if os.path.exists(train_ratings_path):
            self.user_ratings = pd.read_csv(train_ratings_path)
            logger.info(f"Loaded {len(self.user_ratings)} user ratings for filtering")
        
        # Validate data loading
        if self.content_based_recs is not None and self.collaborative_recs is not None:
            # Find common users in both recommendation sets
            content_users = set(self.content_based_recs.keys())
            collab_users = set(self.collaborative_recs.keys())
            common_users = content_users.intersection(collab_users)
            logger.info(f"Found {len(common_users)} users with recommendations from both models")
            return True
        else:
            logger.error("Failed to load recommendations from both models")
            return False
    
    def _normalize_scores(self, recommendations):
        """
        Normalize recommendation scores to range [0, 1]
        
        Parameters:
        -----------
        recommendations : list of tuples
            (item_id, score) pairs
            
        Returns:
        --------
        list of tuples
            (item_id, normalized_score) pairs
        """
        if not recommendations:
            return []
        
        # Extract scores
        scores = [score for _, score in recommendations]
        min_score = min(scores)
        max_score = max(scores)
        
        # Check if all scores are the same
        if max_score == min_score:
            # Return original recommendations with normalized scores of 1.0
            return [(item_id, 1.0) for item_id, _ in recommendations]
        
        # Normalize to [0, 1]
        normalized_recs = [
            (item_id, (score - min_score) / (max_score - min_score))
            for item_id, score in recommendations
        ]
        
        return normalized_recs
    
    def _get_hybrid_recommendations(self, user_id, already_rated=None):
        """
        Generate hybrid recommendations for a specific user
        
        Parameters:
        -----------
        user_id : int
            The user ID to generate recommendations for
        already_rated : set, optional
            Set of item IDs already rated by the user (to be excluded)
            
        Returns:
        --------
        list of tuples
            (movie_id, hybrid_score) pairs sorted by score in descending order
        """
        # Initialize dictionaries to store scores
        content_scores = {}
        cf_scores = {}
        
        # Get content-based recommendations if available
        if user_id in self.content_based_recs:
            content_recs = self.content_based_recs[user_id]
            normalized_content_recs = self._normalize_scores(content_recs)
            
            # Store in dictionary for easy lookup
            for item_id, score in normalized_content_recs:
                content_scores[item_id] = score
        
        # Get collaborative filtering recommendations if available
        if user_id in self.collaborative_recs:
            cf_recs = self.collaborative_recs[user_id]
            normalized_cf_recs = self._normalize_scores(cf_recs)
            
            # Store in dictionary for easy lookup
            for item_id, score in normalized_cf_recs:
                cf_scores[item_id] = score
        
        # If no recommendations from either model, return empty list
        if not content_scores and not cf_scores:
            logger.warning(f"No recommendations available for user {user_id} from either model")
            return []
        
        # Combine unique item IDs from both recommendation sets
        all_items = set(content_scores.keys()).union(set(cf_scores.keys()))
        
        # Filter out already rated items if provided
        if already_rated:
            all_items = all_items - already_rated
        
        # Calculate hybrid scores
        hybrid_scores = []
        
        for item_id in all_items:
            # Get scores from each model (default to 0 if item not in recommendations)
            content_score = content_scores.get(item_id, 0.0)
            cf_score = cf_scores.get(item_id, 0.0)
            
            # Calculate weighted hybrid score
            hybrid_score = (self.content_weight * content_score) + (self.cf_weight * cf_score)
            
            hybrid_scores.append((item_id, hybrid_score))
        
        # Sort by hybrid score (descending)
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_scores
    
    def generate_recommendations(self):
        """Generate hybrid recommendations for all users"""
        logger.info(f"Generating hybrid recommendations with content weight = {self.content_weight:.2f}")
        
        # First, load recommendations if not already loaded
        if self.content_based_recs is None or self.collaborative_recs is None:
            if not self.load_recommendations():
                logger.error("Failed to load recommendations. Exiting.")
                return None
        
        # Find users with recommendations from at least one model
        all_users = set(self.content_based_recs.keys()).union(set(self.collaborative_recs.keys()))
        logger.info(f"Generating hybrid recommendations for {len(all_users)} users")
        
        hybrid_recommendations = {}
        
        for user_id in all_users:
            # Get already rated items for this user
            already_rated = set()
            if self.user_ratings is not None:
                user_ratings = self.user_ratings[self.user_ratings['userId'] == user_id]
                already_rated = set(user_ratings['movieId'].values)
            
            # Generate hybrid recommendations
            user_recs = self._get_hybrid_recommendations(user_id, already_rated)
            
            # Take top-N
            hybrid_recommendations[user_id] = user_recs[:self.top_n]
        
        # Save the recommendations
        with open(os.path.join(self.output_path, f'hybrid_recommendations_cw{self.content_weight:.1f}.pkl'), 'wb') as f:
            pickle.dump(hybrid_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in hybrid_recommendations.items():
            for rank, (movie_id, score) in enumerate(recs, 1):
                # Get movie title if available
                title = "Unknown"
                if self.movie_metadata is not None:
                    movie_row = self.movie_metadata[self.movie_metadata['movieId'] == movie_id]
                    if not movie_row.empty and 'title' in movie_row.columns:
                        title = movie_row['title'].iloc[0]
                
                recommendations_list.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'title': title,
                    'rank': rank,
                    'hybrid_score': score,
                    'content_weight': self.content_weight
                })
        
        if recommendations_list:
            recommendations_df = pd.DataFrame(recommendations_list)
            csv_path = os.path.join(self.output_path, f'hybrid_recommendations_cw{self.content_weight:.1f}.csv')
            recommendations_df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(recommendations_list)} hybrid recommendations to {csv_path}")
        
        return hybrid_recommendations
    
    def evaluate_recommendations(self, test_df=None):
        """
        Evaluate the hybrid recommendations
        
        Parameters:
        -----------
        test_df : pandas.DataFrame, optional
            Test ratings dataframe. If None, will load from default location.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating hybrid recommendations (content weight = {self.content_weight:.2f})...")
        
        # Load test ratings if not provided
        if test_df is None:
            test_path = os.path.join(self.data_path, 'test_ratings.csv')
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                logger.info(f"Loaded {len(test_df)} test ratings")
            else:
                logger.error(f"Test ratings not found at {test_path}")
                return None
        
        # Load or generate hybrid recommendations
        hybrid_rec_path = os.path.join(self.output_path, f'hybrid_recommendations_cw{self.content_weight:.1f}.pkl')
        
        if os.path.exists(hybrid_rec_path):
            with open(hybrid_rec_path, 'rb') as f:
                hybrid_recommendations = pickle.load(f)
            logger.info(f"Loaded existing hybrid recommendations from {hybrid_rec_path}")
        else:
            logger.info("Generating hybrid recommendations...")
            hybrid_recommendations = self.generate_recommendations()
        
        if not hybrid_recommendations:
            logger.error("No hybrid recommendations available for evaluation")
            return None
        
        # Initialize metrics
        hits = 0
        total = 0
        sum_reciprocal_rank = 0
        
        # Get all users in test set
        test_users = test_df['userId'].unique()
        
        for user_id in test_users:
            # Skip users without recommendations
            if user_id not in hybrid_recommendations:
                continue
            
            # Get ground truth: movies the user liked in the test set (rating >= 4)
            user_test = test_df[test_df['userId'] == user_id]
            liked_movies = set(user_test[user_test['rating'] >= 4]['movieId'].values)
            
            if not liked_movies:
                continue
            
            # Get recommendations for this user
            recommendations = hybrid_recommendations[user_id]
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
            'content_weight': self.content_weight,
            'cf_weight': self.cf_weight,
            'hit_rate': hit_rate,
            'arhr': average_reciprocal_hit_rank,
            'num_users_evaluated': total
        }
        
        logger.info(f"Evaluation results: Hit Rate = {hit_rate:.4f}, ARHR = {average_reciprocal_hit_rank:.4f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(self.output_path, f'hybrid_evaluation_cw{self.content_weight:.1f}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        return metrics
    
    def find_optimal_weights(self, weight_range=None):
        """
        Find the optimal weight combination by evaluating multiple weight settings
        
        Parameters:
        -----------
        weight_range : list, optional
            List of content weights to try. If None, defaults to [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            
        Returns:
        --------
        dict
            Dictionary with optimal weights and their performance metrics
        """
        if weight_range is None:
            weight_range = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        
        logger.info(f"Finding optimal weights from {len(weight_range)} options: {weight_range}")
        
        # Load test data once
        test_path = os.path.join(self.data_path, 'test_ratings.csv')
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded {len(test_df)} test ratings")
        else:
            logger.error(f"Test ratings not found at {test_path}")
            return None
        
        # Store metrics for each weight setting
        all_metrics = []
        
        # Test each weight combination
        for weight in weight_range:
            # Update weight
            self.content_weight = weight
            self.cf_weight = 1.0 - weight
            
            # Generate and evaluate recommendations
            hybrid_recommendations = self.generate_recommendations()
            metrics = self.evaluate_recommendations(test_df)
            
            if metrics:
                all_metrics.append(metrics)
        
        # Combine all metrics into a dataframe
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Save combined metrics
            combined_path = os.path.join(self.output_path, 'hybrid_weights_comparison.csv')
            metrics_df.to_csv(combined_path, index=False)
            
            # Find optimal weights based on ARHR (could also use hit_rate)
            optimal_row = metrics_df.loc[metrics_df['arhr'].idxmax()]
            optimal = {
                'content_weight': optimal_row['content_weight'],
                'cf_weight': optimal_row['cf_weight'],
                'hit_rate': optimal_row['hit_rate'],
                'arhr': optimal_row['arhr']
            }
            
            logger.info(f"Optimal weight configuration found:")
            logger.info(f"Content weight: {optimal['content_weight']:.2f}, CF weight: {optimal['cf_weight']:.2f}")
            logger.info(f"Hit Rate: {optimal['hit_rate']:.4f}, ARHR: {optimal['arhr']:.4f}")
            
            return optimal
        
        return None
    
    def recommend_for_user(self, user_id, n=None):
        """
        Generate and print hybrid recommendations for a specific user
        
        Parameters:
        -----------
        user_id : int
            The user ID to generate recommendations for
        n : int, optional
            Number of recommendations to generate (defaults to self.top_n)
        """
        if n is None:
            n = self.top_n
        
        # Get already rated items for this user
        already_rated = set()
        if self.user_ratings is not None:
            user_ratings = self.user_ratings[self.user_ratings['userId'] == user_id]
            already_rated = set(user_ratings['movieId'].values)
        
        # Generate hybrid recommendations
        recommendations = self._get_hybrid_recommendations(user_id, already_rated)
        
        if not recommendations:
            print(f"No recommendations found for user {user_id}")
            return None
        
        # Take top N
        recommendations = recommendations[:n]
        
        # Get movie details if available
        movie_ids = [movie_id for movie_id, _ in recommendations]
        
        # Print recommendations
        print(f"\nTop {len(recommendations)} hybrid recommendations for user {user_id}:")
        print(f"Content weight: {self.content_weight:.2f}, Collaborative weight: {self.cf_weight:.2f}")
        print("-" * 80)
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            title = "Unknown"
            if self.movie_metadata is not None:
                movie_row = self.movie_metadata[self.movie_metadata['movieId'] == movie_id]
                if not movie_row.empty and 'title' in movie_row.columns:
                    title = movie_row['title'].iloc[0]
            
            # Get individual model scores
            content_score = 0.0
            if user_id in self.content_based_recs:
                for mid, scr in self.content_based_recs[user_id]:
                    if mid == movie_id:
                        content_score = scr
                        break
            
            cf_score = 0.0
            if user_id in self.collaborative_recs:
                for mid, scr in self.collaborative_recs[user_id]:
                    if mid == movie_id:
                        cf_score = scr
                        break
            
            print(f"{i}. {title} (ID: {movie_id})")
            print(f"   Hybrid score: {score:.4f}")
            print(f"   Content score: {content_score:.4f}, CF score: {cf_score:.4f}")
        
        return recommendations
    
    def run(self):
        """Run the hybrid recommendation pipeline"""
        # Create a timestamped run ID
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting hybrid recommendation pipeline (Run ID: {run_id})")
        
        # Load recommendations
        if not self.load_recommendations():
            logger.error("Failed to load recommendations from models. Exiting.")
            return None
        
        # Find optimal weights
        optimal = self.find_optimal_weights()
        
        if optimal:
            # Set to optimal weights
            self.content_weight = optimal['content_weight']
            self.cf_weight = optimal['cf_weight']
            
            # Generate final recommendations with optimal weights
            logger.info(f"Generating final recommendations with optimal weights")
            self.generate_recommendations()
        else:
            logger.warning("Could not determine optimal weights, using default")
            self.generate_recommendations()
        
        logger.info(f"Hybrid recommendation pipeline completed (Run ID: {run_id})")
        
        return {
            'run_id': run_id,
            'output_path': self.output_path,
            'content_weight': self.content_weight,
            'cf_weight': self.cf_weight,
            'optimal_weights': optimal
        }

if __name__ == "__main__":
    # Create hybrid recommender with default paths
    recommender = HybridRecommender(
        data_path="./processed_data", 
        recommendation_path="./recommendations", 
        output_path="./hybrid_recommendations",
        content_weight=0.5,  # Equal weighting as default
        top_n=10
    )
    
    # Run the recommendation pipeline
    result = recommender.run()
    
    if result:
        # Print summary
        print("\nHybrid recommendation generation completed!")
        print(f"Run ID: {result['run_id']}")
        print(f"Output path: {result['output_path']}")
        
        if result['optimal_weights']:
            print("\nOptimal weight configuration:")
            print(f"Content weight: {result['optimal_weights']['content_weight']:.2f}")
            print(f"Collaborative weight: {result['optimal_weights']['cf_weight']:.2f}")
            print(f"Hit Rate: {result['optimal_weights']['hit_rate']:.4f}")
            print(f"ARHR: {result['optimal_weights']['arhr']:.4f}")
        
        # Example: Generate recommendations for a specific user
        sample_user_id = 1
        print(f"\nExample: Hybrid recommendations for user {sample_user_id}")
        recommender.recommend_for_user(sample_user_id, n=5)