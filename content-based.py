import numpy as np
import pandas as pd
import os
import pickle
import logging
import heapq
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import time

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, data_path="./processed_data", output_path="./recommendations", top_n=10):
        """
        Initialize the Content-Based Recommender system.
        
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
        self.movie_vectors = None
        self.user_vectors = None
        self.movie_id_to_idx = None
        self.user_id_to_idx = None
        self.movie_metadata = None
        self.user_ratings = None
        self.test_ratings = None
        
        # Model parameters
        self.similarity_threshold = 0.3  # Minimum similarity to consider
    
    def load_data(self):
        """Load processed data required for content-based recommendations"""
        logger.info("Loading processed data...")
        
        # Load movie vectors
        movie_vectors_path = os.path.join(self.data_path, 'movie_vectors.pkl')
        if os.path.exists(movie_vectors_path):
            with open(movie_vectors_path, 'rb') as f:
                self.movie_vectors = pickle.load(f)
            logger.info(f"Loaded feature vectors for {len(self.movie_vectors)} movies")
        else:
            logger.error(f"Movie vectors not found at {movie_vectors_path}")
        
        # Load user vectors
        user_vectors_path = os.path.join(self.data_path, 'user_vectors.pkl')
        if os.path.exists(user_vectors_path):
            with open(user_vectors_path, 'rb') as f:
                self.user_vectors = pickle.load(f)
            logger.info(f"Loaded feature vectors for {len(self.user_vectors)} users")
        else:
            logger.error(f"User vectors not found at {user_vectors_path}")
        
        # Load ID mappings
        movie_id_to_idx_path = os.path.join(self.data_path, 'movie_id_to_idx.pkl')
        if os.path.exists(movie_id_to_idx_path):
            with open(movie_id_to_idx_path, 'rb') as f:
                self.movie_id_to_idx = pickle.load(f)
        
        user_id_to_idx_path = os.path.join(self.data_path, 'user_id_to_idx.pkl')
        if os.path.exists(user_id_to_idx_path):
            with open(user_id_to_idx_path, 'rb') as f:
                self.user_id_to_idx = pickle.load(f)
        
        # Load movie metadata
        movie_genres_path = os.path.join(self.data_path, 'movie_genres.csv')
        if os.path.exists(movie_genres_path):
            self.movie_metadata = pd.read_csv(movie_genres_path)
            logger.info(f"Loaded metadata for {len(self.movie_metadata)} movies")
        
        # Load movie stats (if available)
        movie_stats_path = os.path.join(self.data_path, 'movie_stats.csv')
        if os.path.exists(movie_stats_path):
            self.movie_stats = pd.read_csv(movie_stats_path)
            logger.info(f"Loaded statistics for {len(self.movie_stats)} movies")

        # Load training and test ratings
        train_ratings_path = os.path.join(self.data_path, 'train_ratings.csv')
        if os.path.exists(train_ratings_path):
            self.user_ratings = pd.read_csv(train_ratings_path)
            logger.info(f"Loaded {len(self.user_ratings)} training ratings")
        
        test_ratings_path = os.path.join(self.data_path, 'test_ratings.csv')
        if os.path.exists(test_ratings_path):
            self.test_ratings = pd.read_csv(test_ratings_path)
            logger.info(f"Loaded {len(self.test_ratings)} test ratings")
        
        # Validate data
        if (self.movie_vectors is not None and self.user_vectors is not None and 
            self.movie_id_to_idx is not None and self.user_id_to_idx is not None and 
            self.user_ratings is not None):
            logger.info("All required data loaded successfully")
            return True
        else:
            logger.error("Failed to load all required data")
            return False
    
    def calculate_similarity(self):
        """Calculate similarity between users and movies"""
        logger.info("Calculating user-movie similarity...")
        
        if self.movie_vectors is None or self.user_vectors is None:
            logger.error("Movie or user vectors not loaded")
            return False
        
        # Convert dictionaries to arrays for vectorized computation
        movie_ids = list(self.movie_vectors.keys())
        user_ids = list(self.user_vectors.keys())
        
        movie_vectors_array = np.array([self.movie_vectors[mid] for mid in movie_ids])
        user_vectors_array = np.array([self.user_vectors[uid] for uid in user_ids])
        
        logger.info(f"Computing similarity for {len(user_ids)} users and {len(movie_ids)} movies")
        
        # For large datasets, compute similarity in batches
        batch_size = 1000  # Adjust based on memory constraints
        num_users = len(user_ids)
        
        # Store similarities in a dictionary of dictionaries
        # {user_id: {movie_id: similarity_score}}
        self.user_movie_similarities = {}
        
        start_time = time.time()
        
        for i in range(0, num_users, batch_size):
            batch_end = min(i + batch_size, num_users)
            batch_users = user_ids[i:batch_end]
            batch_vectors = user_vectors_array[i:batch_end]
            
            # Compute cosine similarity for this batch
            # Shape: (batch_size, num_movies)
            batch_similarities = cosine_similarity(batch_vectors, movie_vectors_array)
            
            # Store the similarities
            for idx, user_id in enumerate(batch_users):
                user_sims = {}
                for movie_idx, movie_id in enumerate(movie_ids):
                    similarity = batch_similarities[idx, movie_idx]
                    
                    # Only store if above threshold
                    if similarity > self.similarity_threshold:
                        user_sims[movie_id] = similarity
                
                self.user_movie_similarities[user_id] = user_sims
            
            if (i + batch_size) % (batch_size * 10) == 0 or batch_end == num_users:
                elapsed = time.time() - start_time
                logger.info(f"Processed {batch_end}/{num_users} users ({batch_end/num_users*100:.1f}%) in {elapsed:.1f}s")
        
        logger.info(f"Similarity calculation completed in {time.time() - start_time:.1f}s")
        
        # Save the similarities for future use
        with open(os.path.join(self.output_path, 'user_movie_similarities.pkl'), 'wb') as f:
            pickle.dump(self.user_movie_similarities, f)
        
        logger.info(f"Saved user-movie similarities to {self.output_path}/user_movie_similarities.pkl")
        return True
    
    def load_similarities(self):
        """Load pre-computed similarities if available"""
        similarity_path = os.path.join(self.output_path, 'user_movie_similarities.pkl')
        if os.path.exists(similarity_path):
            with open(similarity_path, 'rb') as f:
                self.user_movie_similarities = pickle.load(f)
            logger.info(f"Loaded pre-computed similarities for {len(self.user_movie_similarities)} users")
            return True
        return False
    
    def _get_user_rated_movies(self, user_id):
        """Get the set of movies already rated by a user"""
        if self.user_ratings is None:
            return set()
        
        user_data = self.user_ratings[self.user_ratings['userId'] == user_id]
        return set(user_data['movieId'].values)
    
    def get_top_n_recommendations(self, user_id, n=None):
        """
        Generate top-N recommendations for a specific user
        
        Parameters:
        -----------
        user_id : int
            The user ID to generate recommendations for
        n : int, optional
            Number of recommendations to generate (defaults to self.top_n)
            
        Returns:
        --------
        list of tuples
            (movie_id, similarity_score) pairs sorted by similarity in descending order
        """
        if n is None:
            n = self.top_n
        
        if self.user_movie_similarities is None:
            logger.error("Similarities not calculated yet")
            return []
        
        if user_id not in self.user_movie_similarities:
            logger.warning(f"User {user_id} not found in similarity matrix")
            return []
        
        # Get movies already rated by the user
        rated_movies = self._get_user_rated_movies(user_id)
        
        # Get user's similarities
        user_sims = self.user_movie_similarities[user_id]
        
        # Filter out already rated movies and sort by similarity
        candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                     if movie_id not in rated_movies]
        
        # Sort by similarity (descending)
        recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # Return top N
        return recommendations[:n]
    
    def generate_recommendations(self):
        """Generate recommendations for all users"""
        logger.info(f"Generating top-{self.top_n} recommendations for all users...")
        
        if not hasattr(self, 'user_movie_similarities') or self.user_movie_similarities is None:
            logger.info("Similarities not found, calculating now...")
            if not self.load_similarities():
                self.calculate_similarity()
        
        # Get all user IDs
        user_ids = list(self.user_vectors.keys())
        
        all_recommendations = {}
        
        for user_id in user_ids:
            recommendations = self.get_top_n_recommendations(user_id)
            all_recommendations[user_id] = recommendations
        
        logger.info(f"Generated recommendations for {len(all_recommendations)} users")
        
        # Save recommendations
        with open(os.path.join(self.output_path, 'content_based_recommendations.pkl'), 'wb') as f:
            pickle.dump(all_recommendations, f)
        
        # Also save in a more readable CSV format
        recommendations_list = []
        
        for user_id, recs in all_recommendations.items():
            for rank, (movie_id, score) in enumerate(recs, 1):
                recommendations_list.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rank': rank,
                    'similarity_score': score
                })
        
        if recommendations_list:
            recommendations_df = pd.DataFrame(recommendations_list)
            recommendations_df.to_csv(os.path.join(self.output_path, 'content_based_recommendations.csv'), index=False)
            logger.info(f"Saved recommendations to CSV file")
        
        return all_recommendations
    
    def evaluate_recommendations(self):
        """Evaluate the recommendations against test data"""
        logger.info("Evaluating recommendations...")
        
        if self.test_ratings is None:
            logger.error("Test ratings not available for evaluation")
            return None
        
        if not hasattr(self, 'user_movie_similarities') or self.user_movie_similarities is None:
            logger.info("Similarities not found, loading or calculating now...")
            if not self.load_similarities():
                self.calculate_similarity()
        
        # Initialize metrics
        hits = 0
        total = 0
        sum_reciprocal_rank = 0
        
        # Get all users in test set
        test_users = self.test_ratings['userId'].unique()
        
        for user_id in test_users:
            # Skip users without similarity data
            if user_id not in self.user_movie_similarities:
                continue
            
            # Get ground truth: movies the user liked in the test set (rating >= 4)
            user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
            liked_movies = set(user_test[user_test['rating'] >= 4]['movieId'].values)
            
            if not liked_movies:
                continue
            
            # Get recommendations for this user
            recommendations = self.get_top_n_recommendations(user_id, n=self.top_n)
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
        
        logger.info(f"Evaluation results: Hit Rate = {hit_rate:.4f}, ARHR = {average_reciprocal_hit_rank:.4f}")
        
        # Save metrics
        evaluation_results = pd.DataFrame([metrics])
        evaluation_results.to_csv(os.path.join(self.output_path, 'content_based_evaluation.csv'), index=False)
        
        return metrics
    
    def get_movie_details(self, movie_ids):
        """Get details for a list of movie IDs"""
        if self.movie_metadata is None:
            logger.warning("Movie metadata not available")
            return None
        
        movie_details = self.movie_metadata[self.movie_metadata['movieId'].isin(movie_ids)]
        return movie_details
    
    def run(self):
        """Run the entire recommendation pipeline"""
        # Create a timestamped run ID
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting content-based recommendation pipeline (Run ID: {run_id})")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load required data. Exiting.")
            return None
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Evaluate recommendations
        metrics = self.evaluate_recommendations()
        
        logger.info(f"Content-based recommendation pipeline completed (Run ID: {run_id})")
        
        return {
            'run_id': run_id,
            'output_path': self.output_path,
            'recommendations': recommendations,
            'metrics': metrics
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
        
        # Ensure data is loaded
        if not hasattr(self, 'user_movie_similarities') or self.user_movie_similarities is None:
            if not self.load_similarities():
                if not self.load_data():
                    logger.error("Failed to load required data. Exiting.")
                    return None
                self.calculate_similarity()
        
        # Get recommendations
        recommendations = self.get_top_n_recommendations(11, 10)
        
        if not recommendations:
            print(f"No recommendations found for user {user_id}")
            return None
        
        # Get movie details if available
        movie_ids = [movie_id for movie_id, _ in recommendations]
        movie_details = self.get_movie_details(movie_ids)
        
        # Print recommendations
        print(f"\nTop {len(recommendations)} recommendations for user {user_id}:")
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            if movie_details is not None and not movie_details[movie_details['movieId'] == movie_id].empty:
                movie_row = movie_details[movie_details['movieId'] == movie_id].iloc[0]
                movie_name = movie_row['title'] if 'title' in movie_row else f"Movie {movie_id}"
                print(f"{i}. {movie_name} (ID: {movie_id}) - Similarity: {score:.4f}")
            else:
                print(f"{i}. Movie ID: {movie_id} - Similarity: {score:.4f}")
        
        return recommendations

if __name__ == "__main__":
    # Create recommender with default paths
    recommender = ContentBasedRecommender(
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
        
        if result['metrics']:
            print("\nEvaluation metrics:")
            print(f"- Hit Rate: {result['metrics']['hit_rate']:.4f}")
            print(f"- Average Reciprocal Hit Rank: {result['metrics']['arhr']:.4f}")
            print(f"- Number of users evaluated: {result['metrics']['num_users_evaluated']}")
        
        # Example: Generate recommendations for a specific user
        if recommender.user_vectors:
            sample_user_id = next(iter(recommender.user_vectors.keys()))
            print(f"\nExample: Recommendations for user {sample_user_id}")
            recommender.recommend_for_user(sample_user_id, n=5)