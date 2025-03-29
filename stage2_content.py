import numpy as np
import pandas as pd
import os
import pickle
import logging
import heapq
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ast  # To safely evaluate string representations of lists

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set paths
input_path = "./"  # Current directory where stage1.py saved the files
output_path = "./recommendations"
top_n = 10

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize NLTK tools
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Model parameters
similarity_threshold = 0.3  # Minimum similarity to consider
word2vec_dim = 100  # Dimensionality of Word2Vec embeddings

def load_data():
    """Load processed data from stage1.py"""
    logger.info("Loading processed data from stage1.py...")
    
    # Data containers
    data = {}
    
    # Load movie features
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
        # Convert string representation of tokens and top_keywords back to lists
        data['movie_features']['tokens'] = data['movie_features']['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        data['movie_features']['top_keywords'] = data['movie_features']['top_keywords'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        logger.info(f"Loaded features for {len(data['movie_features'])} movies")
    else:
        logger.error(f"Movie features not found at {movie_features_path}")
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        data['ratings'] = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(data['ratings'])} normalized ratings")
    else:
        logger.error(f"Normalized ratings not found at {ratings_path}")
    
    # Create training and testing sets with 80-20 split
    if 'ratings' in data:
        # Sort by timestamp if available to ensure reproducibility
        if 'timestamp' in data['ratings'].columns:
            data['ratings'] = data['ratings'].sort_values('timestamp')
        
        # Group by user to ensure each user has both training and testing data
        user_groups = data['ratings'].groupby('userId')
        train_data = []
        test_data = []
        
        for _, group in user_groups:
            n = len(group)
            split_idx = int(n * 0.8)
            train_data.append(group.iloc[:split_idx])
            test_data.append(group.iloc[split_idx:])
        
        data['train_ratings'] = pd.concat(train_data).reset_index(drop=True)
        data['test_ratings'] = pd.concat(test_data).reset_index(drop=True)
        
        logger.info(f"Split ratings into {len(data['train_ratings'])} training and {len(data['test_ratings'])} testing samples")
    
    return data

# Load the data
data = load_data()

# Build corpus word counts from movie features
if 'movie_features' in data:
    corpus_word_counts = Counter()
    
    for tokens in data['movie_features']['tokens']:
        corpus_word_counts.update(tokens)
    
    data['corpus_word_counts'] = corpus_word_counts
    
    # Save corpus word counts
    with open(os.path.join(output_path, 'corpus_word_counts.pkl'), 'wb') as f:
        pickle.dump(corpus_word_counts, f)
    
    logger.info(f"Built vocabulary with {len(corpus_word_counts)} unique words")

def calculate_log_likelihood(movie_features, corpus_word_counts):
    """Calculate Log-Likelihood values for words in each movie"""
    logger.info("Calculating Log-Likelihood values...")
    
    # Calculate total corpus size
    total_corpus_size = sum(corpus_word_counts.values())
    
    # Initialize container for movie features
    movie_ll_values = {}
    
    # Process each movie document
    for _, row in movie_features.iterrows():
        movie_id = row['movieId']
        tokens = row['tokens']
        
        if not tokens:
            continue
        
        # Count word occurrences in this movie
        movie_word_counts = Counter(tokens)
        movie_size = sum(movie_word_counts.values())
        
        # Calculate Log-Likelihood for each word
        movie_ll_values[movie_id] = {}
        
        for word, count in movie_word_counts.items():
            # Observed frequencies
            a = count  # Occurrences in this movie
            b = corpus_word_counts[word] - count  # Occurrences in other movies
            c = movie_size  # Total words in this movie
            d = total_corpus_size - movie_size  # Total words in other movies
            
            # Expected counts
            e1 = c * (a + b) / (c + d)
            e2 = d * (a + b) / (c + d)
            
            # Log-Likelihood calculation
            ll = 0
            if a > 0 and e1 > 0:
                ll += a * math.log(a / e1)
            if b > 0 and e2 > 0:
                ll += b * math.log(b / e2)
            
            ll = 2 * ll
            movie_ll_values[movie_id][word] = ll
    
    return movie_ll_values

# Calculate Log-Likelihood if movie features are available
if 'movie_features' in data and 'corpus_word_counts' in data:
    movie_ll_values = calculate_log_likelihood(data['movie_features'], data['corpus_word_counts'])
    data['movie_ll_values'] = movie_ll_values
    
    # Save Log-Likelihood values
    with open(os.path.join(output_path, 'movie_ll_values.pkl'), 'wb') as f:
        pickle.dump(movie_ll_values, f)
    
    logger.info(f"Calculated Log-Likelihood values for {len(movie_ll_values)} movies")

def train_word2vec(movie_features, vector_size=100):
    """Train Word2Vec model on movie tokens"""
    logger.info(f"Training Word2Vec model with {vector_size} dimensions...")
    
    # Extract token lists from movie features
    tokenized_corpus = list(movie_features['tokens'])
    
    # Train Word2Vec model using CBOW approach
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        epochs=15,
        sg=0  # CBOW model
    )
    
    return word2vec_model

# Train Word2Vec if movie features are available
if 'movie_features' in data:
    word2vec_model = train_word2vec(data['movie_features'], word2vec_dim)
    data['word2vec_model'] = word2vec_model
    
    # Save Word2Vec model
    word2vec_path = os.path.join(output_path, 'word2vec_model')
    word2vec_model.save(word2vec_path)
    
    logger.info(f"Trained Word2Vec model with {len(word2vec_model.wv)} words")

def generate_movie_vectors(movie_ll_values, word2vec_model):
    """Generate movie feature vectors using Log-Likelihood and Word2Vec"""
    logger.info("Generating movie feature vectors...")
    
    movie_vectors = {}
    
    for movie_id, ll_values in movie_ll_values.items():
        # Sort words by LL value and select top 200
        top_words = sorted(ll_values.items(), key=lambda x: x[1], reverse=True)[:200]
        
        if not top_words:
            continue
        
        # Combine Word2Vec vectors weighted by Log-Likelihood values
        weighted_vectors = []
        ll_sum = 0
        
        for word, ll_value in top_words:
            if ll_value <= 0:
                continue
            
            if word in word2vec_model.wv:
                weighted_vectors.append(word2vec_model.wv[word] * ll_value)
                ll_sum += ll_value
        
        if weighted_vectors and ll_sum > 0:
            # Calculate the weighted average vector
            movie_vector = np.sum(weighted_vectors, axis=0) / ll_sum
            
            # Normalize to unit length
            norm = np.linalg.norm(movie_vector)
            if norm > 0:
                movie_vector = movie_vector / norm
            
            movie_vectors[movie_id] = movie_vector
    
    return movie_vectors

# Generate movie vectors if Word2Vec and LL values are available
if 'word2vec_model' in data and 'movie_ll_values' in data:
    movie_vectors = generate_movie_vectors(data['movie_ll_values'], data['word2vec_model'])
    data['movie_vectors'] = movie_vectors
    
    # Save movie vectors
    with open(os.path.join(output_path, 'movie_vectors.pkl'), 'wb') as f:
        pickle.dump(movie_vectors, f)
    
    # Create movie ID to index mapping
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_vectors.keys())}
    data['movie_id_to_idx'] = movie_id_to_idx
    
    # Save the mapping
    with open(os.path.join(output_path, 'movie_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(movie_id_to_idx, f)
    
    logger.info(f"Generated feature vectors for {len(movie_vectors)} movies")

def generate_user_vectors(movie_vectors, train_ratings):
    """Generate user feature vectors based on rated movies and their content"""
    logger.info("Generating user feature vectors...")
    
    user_vectors = {}
    
    # Process each user
    for user_id in train_ratings['userId'].unique():
        # Get user ratings
        user_data = train_ratings[train_ratings['userId'] == user_id]
        
        if len(user_data) == 0:
            continue
        
        weighted_vectors = []
        weight_sum = 0
        
        for _, rating_row in user_data.iterrows():
            movie_id = rating_row['movieId']
            
            # Use the normalized rating if available, otherwise use original rating
            if 'normalized_rating' in rating_row:
                rating = rating_row['normalized_rating']
                # Convert from [0,1] to [-0.5,0.5] to match the paper's approach
                weight = rating - 0.5
            else:
                rating = rating_row['rating']
                # Center rating at 3.0 as described in the papers
                weight = rating - 3.0
            
            # Skip if movie vector is not available
            if movie_id not in movie_vectors:
                continue
            
            if weight != 0:
                weighted_vectors.append(movie_vectors[movie_id] * weight)
                weight_sum += abs(weight)
        
        if weighted_vectors and weight_sum > 0:
            # Calculate the weighted average vector
            user_vector = np.sum(weighted_vectors, axis=0) / weight_sum
            
            # Normalize to unit length
            norm = np.linalg.norm(user_vector)
            if norm > 0:
                user_vector = user_vector / norm
            
            user_vectors[user_id] = user_vector
    
    return user_vectors

# Generate user vectors if movie vectors and training ratings are available
if 'movie_vectors' in data and 'train_ratings' in data:
    user_vectors = generate_user_vectors(data['movie_vectors'], data['train_ratings'])
    data['user_vectors'] = user_vectors
    
    # Save user vectors
    with open(os.path.join(output_path, 'user_vectors.pkl'), 'wb') as f:
        pickle.dump(user_vectors, f)
    
    # Create user ID to index mapping
    user_id_to_idx = {user_id: i for i, user_id in enumerate(user_vectors.keys())}
    data['user_id_to_idx'] = user_id_to_idx
    
    # Save the mapping
    with open(os.path.join(output_path, 'user_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(user_id_to_idx, f)
    
    logger.info(f"Generated feature vectors for {len(user_vectors)} users")

def calculate_user_movie_similarity(user_vectors, movie_vectors, threshold=0.3):
    """Calculate similarity between users and movies"""
    logger.info("Calculating user-movie similarity...")
    
    # Store similarities in a dictionary of dictionaries
    # {user_id: {movie_id: similarity_score}}
    user_movie_similarities = {}
    
    # Calculate similarity for each user
    total_users = len(user_vectors)
    for i, (user_id, user_vector) in enumerate(user_vectors.items()):
        user_sims = {}
        
        for movie_id, movie_vector in movie_vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(user_vector, movie_vector)
            
            # Only store if above threshold
            if similarity > threshold:
                user_sims[movie_id] = similarity
        
        user_movie_similarities[user_id] = user_sims
        
        # Log progress
        if (i+1) % 100 == 0 or (i+1) == total_users:
            logger.info(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%)")
    
    return user_movie_similarities

# Calculate similarities if user and movie vectors are available
if 'user_vectors' in data and 'movie_vectors' in data:
    user_movie_similarities = calculate_user_movie_similarity(
        data['user_vectors'], 
        data['movie_vectors'], 
        threshold=similarity_threshold
    )
    data['user_movie_similarities'] = user_movie_similarities
    
    # Save the similarities
    with open(os.path.join(output_path, 'user_movie_similarities.pkl'), 'wb') as f:
        pickle.dump(user_movie_similarities, f)
    
    logger.info(f"Calculated similarities for {len(user_movie_similarities)} users")

def get_user_rated_movies(user_id, train_ratings):
    """Get the set of movies already rated by a user"""
    if train_ratings is None:
        return set()
    
    user_data = train_ratings[train_ratings['userId'] == user_id]
    return set(user_data['movieId'].values)

def get_top_n_recommendations(user_id, user_movie_similarities, train_ratings, n=10):
    """
    Generate top-N recommendations for a specific user
    
    Parameters:
    -----------
    user_id : int
        The user ID to generate recommendations for
    user_movie_similarities : dict
        Dictionary of user-movie similarities
    train_ratings : pd.DataFrame
        DataFrame of user ratings
    n : int, optional
        Number of recommendations to generate
        
    Returns:
    --------
    list of tuples
        (movie_id, similarity_score) pairs sorted by similarity in descending order
    """
    if user_id not in user_movie_similarities:
        logger.warning(f"User {user_id} not found in similarity matrix")
        return []
    
    # Get movies already rated by the user
    rated_movies = get_user_rated_movies(user_id, train_ratings)
    
    # Get user's similarities
    user_sims = user_movie_similarities[user_id]
    
    # Filter out already rated movies and sort by similarity
    candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                 if movie_id not in rated_movies]
    
    # Sort by similarity (descending)
    recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    # Return top N
    return recommendations[:n]

def generate_recommendations_for_all_users(user_movie_similarities, train_ratings, n=10):
    """Generate recommendations for all users"""
    logger.info(f"Generating top-{n} recommendations for all users...")
    
    # Get all user IDs
    user_ids = list(user_movie_similarities.keys())
    
    all_recommendations = {}
    
    for user_id in user_ids:
        recommendations = get_top_n_recommendations(user_id, user_movie_similarities, train_ratings, n)
        if recommendations:
            all_recommendations[user_id] = recommendations
    
    return all_recommendations

# Generate recommendations if similarities are available
if 'user_movie_similarities' in data and 'train_ratings' in data:
    all_recommendations = generate_recommendations_for_all_users(
        data['user_movie_similarities'], 
        data['train_ratings'], 
        n=top_n
    )
    data['all_recommendations'] = all_recommendations
    
    # Save recommendations
    with open(os.path.join(output_path, 'content_based_recommendations.pkl'), 'wb') as f:
        pickle.dump(all_recommendations, f)
    
    # Also save in a more readable CSV format
    recommendations_list = []
    
    for user_id, recs in all_recommendations.items():
        for rank, (movie_id, score) in enumerate(recs, 1):
            movie_title = "Unknown"
            if 'movie_features' in data:
                movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
                if not movie_row.empty and 'title' in movie_row.columns:
                    movie_title = movie_row.iloc[0]['title']
                    
            recommendations_list.append({
                'userId': user_id,
                'movieId': movie_id,
                'title': movie_title,
                'rank': rank,
                'similarity_score': score
            })
    
    if recommendations_list:
        recommendations_df = pd.DataFrame(recommendations_list)
        recommendations_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False)
    
    logger.info(f"Generated recommendations for {len(all_recommendations)} users")

def evaluate_recommendations(user_movie_similarities, train_ratings, test_ratings, top_n=10):
    """Evaluate the recommendations against test data"""
    logger.info("Evaluating recommendations...")
    
    # Initialize metrics
    hits = 0
    total = 0
    sum_reciprocal_rank = 0
    
    # Get all users in test set
    test_users = test_ratings['userId'].unique()
    
    for user_id in test_users:
        # Skip users without similarity data
        if user_id not in user_movie_similarities:
            continue
        
        # Get ground truth: movies the user liked in the test set (rating >= 4)
        user_test = test_ratings[test_ratings['userId'] == user_id]
        liked_movies = set(user_test[user_test['rating'] >= 4]['movieId'].values)
        
        if not liked_movies:
            continue
        
        # Get recommendations for this user
        recommendations = get_top_n_recommendations(
            user_id, user_movie_similarities, train_ratings, n=top_n
        )
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
    
    return metrics

# Evaluate recommendations if test ratings are available
if 'user_movie_similarities' in data and 'train_ratings' in data and 'test_ratings' in data:
    evaluation_metrics = evaluate_recommendations(
        data['user_movie_similarities'],
        data['train_ratings'],
        data['test_ratings'],
        top_n=top_n
    )
    data['evaluation_metrics'] = evaluation_metrics
    
    # Save metrics
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'content_based_evaluation.csv'), index=False)
    
    # Display evaluation metrics
    print("\nEvaluation Results:")
    print(f"Hit Rate: {evaluation_metrics['hit_rate']:.4f}")
    print(f"Average Reciprocal Hit Rank: {evaluation_metrics['arhr']:.4f}")
    print(f"Number of users evaluated: {evaluation_metrics['num_users_evaluated']}")

def recommend_for_user(user_id, user_movie_similarities, train_ratings, movie_features=None, n=10):
    """Generate and print recommendations for a specific user"""
    # Get recommendations
    recommendations = get_top_n_recommendations(
        user_id, user_movie_similarities, train_ratings, n
    )
    
    if not recommendations:
        print(f"No recommendations found for user {user_id}")
        return None
    
    # Print recommendations
    print(f"\nTop {len(recommendations)} recommendations for user {user_id}:")
    
    for i, (movie_id, score) in enumerate(recommendations, 1):
        movie_info = f"Movie ID: {movie_id}"
        
        # Try to get movie title if available
        if movie_features is not None:
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty and 'title' in movie_row.columns:
                movie_info = movie_row.iloc[0]['title']
        
        print(f"{i}. {movie_info} - Similarity: {score:.4f}")
    
    return recommendations

# Demonstrate recommendations for a specific user
if 'user_movie_similarities' in data and 'train_ratings' in data:
    # Find a user with recommendations
    if 'all_recommendations' in data and data['all_recommendations']:
        sample_user_id = next(iter(data['all_recommendations'].keys()))
        
        print(f"\nGenerating sample recommendations for User ID {sample_user_id}:")
        
        # Get user ratings for context
        if 'train_ratings' in data:
            user_ratings = data['train_ratings'][data['train_ratings']['userId'] == sample_user_id]
            
            print(f"\nThis user has rated {len(user_ratings)} movies.")
            if len(user_ratings) > 0:
                print("Sample of their highest-rated movies:")
                
                # Get top 5 highest-rated movies
                top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
                
                for _, row in top_rated.iterrows():
                    movie_info = f"Movie ID: {row['movieId']}"
                    if 'movie_features' in data:
                        movie_row = data['movie_features'][data['movie_features']['movieId'] == row['movieId']]
                        if not movie_row.empty and 'title' in movie_row.columns:
                            movie_info = movie_row.iloc[0]['title']
                    print(f"  {movie_info} - Rating: {row['rating']}")
        
        # Generate recommendations
        movie_features = data.get('movie_features')
        recommend_for_user(
            sample_user_id, 
            data['user_movie_similarities'], 
            data['train_ratings'], 
            movie_features,
            n=10
        )

# Print final summary
print("\nSummary of Content-Based Filtering with Log-Likelihood and Word2Vec:")
print("=" * 80)

# Data information
print("\nData Information:")
if 'movie_features' in data:
    print(f"- Processed {len(data['movie_features'])} movie feature records")
if 'corpus_word_counts' in data:
    print(f"- Vocabulary size: {len(data['corpus_word_counts'])} unique words")
if 'movie_vectors' in data:
    print(f"- Generated feature vectors for {len(data['movie_vectors'])} movies")
if 'user_vectors' in data:
    print(f"- Generated feature vectors for {len(data['user_vectors'])} users")
if 'user_movie_similarities' in data:
    avg_similar_movies = sum(len(sims) for sims in data['user_movie_similarities'].values()) / len(data['user_movie_similarities'])
    print(f"- Average similar movies per user: {avg_similar_movies:.2f}")
if 'all_recommendations' in data:
    avg_recommendations = sum(len(recs) for recs in data['all_recommendations'].values()) / len(data['all_recommendations'])
    print(f"- Average recommendations per user: {avg_recommendations:.2f}")

# Evaluation metrics
if 'evaluation_metrics' in data:
    print("\nPerformance Metrics:")
    print(f"- Hit Rate: {data['evaluation_metrics']['hit_rate']:.4f}")
    print(f"- Average Reciprocal Hit Rank: {data['evaluation_metrics']['arhr']:.4f}")
    print(f"- Users evaluated: {data['evaluation_metrics']['num_users_evaluated']}")

# Model advantages
print("\nAdvantages of this approach:")
print("- Log-Likelihood identifies more meaningful words compared to TF-IDF")
print("- Word2Vec captures semantic relationships between words")
print("- Handles new movies effectively (cold start for items)")
print("- Generates personalized recommendations based on content preferences")
print("- Doesn't require item-item similarity calculations")

# Saved files
print("\nSaved Files:")
for file in os.listdir(output_path):
    print(f"- {file}")

print("\nContent-Based Filtering Model Successfully Implemented!")