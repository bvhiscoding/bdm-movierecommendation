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
import ast
import sys
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "./"
output_path = "./rec/content-recommendations"
top_n = 20

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Improved similarity threshold based on empirical testing
similarity_threshold = 0.25
word2vec_dim = 100
rating_threshold = 3.5  # Threshold for binary classification (like/dislike)

# Memory monitoring function
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
    data = {}
    
    movie_features_path = os.path.join(input_path, './processed/processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        # Use optimized loading for large CSV files
        data['movie_features'] = pd.read_csv(movie_features_path, 
                                            dtype={'movieId': int, 'token_count': int})
        
        # Use safer and more memory-efficient parsing of lists
        data['movie_features']['tokens'] = data['movie_features']['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        data['movie_features']['top_keywords'] = data['movie_features']['top_keywords'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        logger.info(f"Loaded features for {len(data['movie_features'])} movies")
    else:
        logger.error(f"File not found: {movie_features_path}")
        sys.exit(1)
    
    ratings_path = os.path.join(input_path, './processed/normalized_ratings.csv')
    if os.path.exists(ratings_path):
        # Load ratings in chunks to save memory
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            chunks.append(chunk)
            # Force garbage collection after each chunk
            gc.collect()
        
        data['ratings'] = pd.concat(chunks)
        logger.info(f"Loaded {len(data['ratings'])} ratings from {len(data['ratings']['userId'].unique())} users")
    else:
        logger.error(f"File not found: {ratings_path}")
        sys.exit(1)
    
    if 'ratings' in data:
        # Sort by timestamp if available for temporal split
        if 'timestamp' in data['ratings'].columns:
            data['ratings'] = data['ratings'].sort_values('timestamp')
        
        # Use a more robust user-level train-test split
        # Create a user-level split instead of randomly dividing all ratings
        all_user_ids = data['ratings']['userId'].unique()

        # Set a random seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(all_user_ids)

        # 80% of users for training, 20% for testing
        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]

        # For users in test set, take 80% of their ratings for training
        train_ratings_main = data['ratings'][data['ratings']['userId'].isin(train_users)]
        
        # For users in test set, split their ratings 80/20
        test_user_ratings = data['ratings'][data['ratings']['userId'].isin(test_users)]
        test_user_train = []
        test_user_test = []
        
        for user_id in test_users:
            user_data = test_user_ratings[test_user_ratings['userId'] == user_id]
            if len(user_data) < 5:  # Skip users with very few ratings
                continue
                
            # Sort by timestamp if available for a more realistic test
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
                
            # Take first 80% for training, last 20% for testing
            split_idx = int(len(user_data) * 0.8)
            test_user_train.append(user_data.iloc[:split_idx])
            test_user_test.append(user_data.iloc[split_idx:])
        
        # Combine all training sets
        data['train_ratings'] = pd.concat([train_ratings_main] + test_user_train) if test_user_train else train_ratings_main
        data['test_ratings'] = pd.concat(test_user_test) if test_user_test else pd.DataFrame()
        
        logger.info(f"Created train set with {len(data['train_ratings'])} ratings from {len(data['train_ratings']['userId'].unique())} users")
        logger.info(f"Created test set with {len(data['test_ratings'])} ratings from {len(data['test_ratings']['userId'].unique())} users")
        
        # Force garbage collection
        gc.collect()
    
    return data

def calculate_log_likelihood(movie_features, corpus_word_counts, batch_size=100):
    """
    Calculate log-likelihood scores for words in each movie.
    Optimized implementation with TF-IDF weighting.
    """
    start_time = time.time()
    
    total_corpus_size = sum(corpus_word_counts.values())
    total_movies = len(movie_features)
    
    # Calculate document frequency for each word
    doc_freq = defaultdict(int)
    for _, row in movie_features.iterrows():
        tokens = row['tokens']
        unique_tokens = set(tokens)
        for word in unique_tokens:
            doc_freq[word] += 1
    
    # Calculate IDF values
    idf_values = {}
    for word, df in doc_freq.items():
        idf_values[word] = math.log(total_movies / (df + 1))
    
    movie_ll_values = {}
    
    for batch_start in range(0, total_movies, batch_size):
        batch_end = min(batch_start + batch_size, total_movies)
        batch = movie_features.iloc[batch_start:batch_end]
        
        for _, row in batch.iterrows():
            movie_id = row['movieId']
            tokens = row['tokens']
            
            if not tokens:
                continue
            
            movie_word_counts = Counter(tokens)
            movie_size = sum(movie_word_counts.values())
            
            movie_ll_tfidf = {}
            
            for word, count in movie_word_counts.items():
                # Calculate log-likelihood
                a = count
                b = corpus_word_counts[word] - count
                c = movie_size
                d = total_corpus_size - movie_size
                
                e1 = c * (a + b) / (c + d)
                e2 = d * (a + b) / (c + d)
                
                ll = 0
                if a > 0 and e1 > 0:
                    ll += a * math.log(a / e1)
                if b > 0 and e2 > 0:
                    ll += b * math.log(b / e2)
                
                # Scale by 2
                ll = 2 * ll
                
                # Get IDF value
                idf = idf_values.get(word, 0)
                
                # Multiply LL by IDF for combined weighting
                movie_ll_tfidf[word] = ll * idf
            
            movie_ll_values[movie_id] = movie_ll_tfidf
        
        elapsed = time.time() - start_time
        progress = batch_end / total_movies * 100
        remaining = elapsed / (batch_end - batch_start) * (total_movies - batch_end) if batch_end < total_movies else 0
        logger.info(f"Processed {batch_end}/{total_movies} movies ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Garbage collection
        gc.collect()
    
    return movie_ll_values

def train_word2vec(movie_features, vector_size=100, batch_size=1000):
    """
    Train a Word2Vec model on movie tokens with improved parameters.
    """
    start_time = time.time()
    
    tokenized_corpus = []
    total_movies = len(movie_features)
    
    # Process in batches to save memory
    for i in range(0, total_movies, batch_size):
        batch_end = min(i + batch_size, total_movies)
        batch = movie_features.iloc[i:batch_end]
        
        batch_tokens = list(batch['tokens'])
        tokenized_corpus.extend(batch_tokens)
        
        # Log progress
        progress = batch_end / total_movies * 100
        logger.info(f"Preparing corpus: {batch_end}/{total_movies} movies ({progress:.1f}%)")
        
        # Garbage collection after each batch
        gc.collect()
    
    total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
    logger.info(f"Training Word2Vec on corpus with {total_tokens} tokens")
    
    # Improved Word2Vec parameters
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=8,        # Increased context window 
        min_count=3,     # Include words that appear at least 3 times
        workers=8,       # Parallelize training
        epochs=20,       # More training epochs for better vectors
        sg=1,            # Use skip-gram (better for semantic relations)
        hs=1,            # Use hierarchical softmax for efficiency
        negative=15      # Number of negative samples
    )
    
    # Clean up the corpus to save memory
    del tokenized_corpus
    gc.collect()
    
    elapsed = time.time() - start_time
    logger.info(f"Word2Vec model training completed in {elapsed:.2f}s")
    
    return word2vec_model

def generate_movie_vectors(movie_ll_values, word2vec_model, movie_features, corpus_word_counts, batch_size=100):
    """
    Generate movie feature vectors combining log-likelihood and Word2Vec.
    Improved with TF-IDF weighting and normalization.
    """
    start_time = time.time()
    
    movie_vectors = {}
    successful_vectors = 0
    no_words_found = 0
    low_ll_sum = 0
    
    movie_ids = list(movie_ll_values.keys())
    total_movies = len(movie_ids)
    total_corpus_size = sum(corpus_word_counts.values())
    
    for batch_start in range(0, total_movies, batch_size):
        batch_end = min(batch_start + batch_size, total_movies)
        batch_movie_ids = movie_ids[batch_start:batch_end]

        for movie_id in batch_movie_ids:
            ll_values = movie_ll_values[movie_id]
            
            # Calculate IDF for each word in this movie
            idf = {word: math.log(total_corpus_size/(corpus_word_counts[word]+1)) 
                  for word in ll_values.keys()}
            
            # Combined score: LL * IDF
            combined_scores = {word: ll * idf[word] 
                              for word, ll in ll_values.items()}
            
            # Get top words by combined score
            top_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:500]
            
            if not top_words:
                no_words_found += 1
                continue
            
            weighted_vectors = []
            ll_sum = 0
            words_used = 0
            
            # Combine word vectors weighted by their scores
            for word, score in top_words:
                if score <= 0:
                    continue
                
                if word in word2vec_model.wv:
                    weighted_vectors.append(word2vec_model.wv[word] * score)
                    ll_sum += score
                    words_used += 1
            
            # Create the final vector as weighted average
            if weighted_vectors and ll_sum > 0:
                movie_vector = np.sum(weighted_vectors, axis=0) / ll_sum
                
                # Normalize to unit length for cosine similarity
                norm = np.linalg.norm(movie_vector)
                if norm > 0:
                    movie_vector = movie_vector / norm
                    movie_vectors[movie_id] = movie_vector
                    successful_vectors += 1
            else:
                low_ll_sum += 1
        
        elapsed = time.time() - start_time
        progress = batch_end / total_movies * 100
        remaining = elapsed / (batch_end - batch_start) * (total_movies - batch_end) if batch_end < total_movies else 0
        
        logger.info(f"Generated vectors: {batch_end}/{total_movies} movies ({progress:.1f}%) - Success: {successful_vectors}")
        
        # Garbage collection
        gc.collect()
    
    logger.info(f"Vector generation summary: Success: {successful_vectors}, No words: {no_words_found}, Low scores: {low_ll_sum}")
    return movie_vectors

def generate_user_vectors(movie_vectors, train_ratings, batch_size=100):
    """
    Generate user feature vectors based on rated movies.
    Improved with better weighting and bias adjustments.
    """
    start_time = time.time()
    
    user_vectors = {}
    successful_vectors = 0
    no_ratings_found = 0
    no_vectors_for_movies = 0
    low_weight_sum = 0
    
    # Create user_ratings_dict for faster lookups
    user_ratings_dict = defaultdict(dict)
    
    # Use chunks to process ratings to avoid memory issues
    ratings_chunks = np.array_split(train_ratings, max(1, len(train_ratings) // 100000))
    for chunk in ratings_chunks:
        for _, row in chunk.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']
            user_ratings_dict[user_id][movie_id] = rating
        
        # Garbage collection after each chunk
        gc.collect()
    
    user_ids = list(user_ratings_dict.keys())
    total_users = len(user_ids)
    
    # Calculate global average rating
    all_ratings = [r for user_ratings in user_ratings_dict.values() for r in user_ratings.values()]
    global_avg_rating = np.mean(all_ratings) if all_ratings else 3.0
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        for user_id in batch_user_ids:
            user_ratings = user_ratings_dict[user_id]
            
            if len(user_ratings) == 0:
                no_ratings_found += 1
                continue
            
            # Calculate user's average rating
            user_avg_rating = np.mean(list(user_ratings.values()))
            # Calculate rating bias (how much higher/lower this user rates compared to global average)
            user_bias = user_avg_rating - global_avg_rating
            
            weighted_vectors = []
            weight_sum = 0
            movies_with_vectors = 0
            movies_without_vectors = 0
            
            for movie_id, rating in user_ratings.items():
                # Calculate normalized weight: adjust rating relative to user's average
                adjusted_rating = rating - user_avg_rating
                weight = adjusted_rating
                
                if movie_id not in movie_vectors:
                    movies_without_vectors += 1
                    continue
                else:
                    movies_with_vectors += 1
                
                if abs(weight) > 0.1:  # Only consider significant preferences
                    weighted_vectors.append(movie_vectors[movie_id] * weight)
                    weight_sum += abs(weight)
            
            if weighted_vectors and weight_sum > 0:
                user_vector = np.sum(weighted_vectors, axis=0) / weight_sum
                
                # Normalize vector
                norm = np.linalg.norm(user_vector)
                if norm > 0:
                    user_vector = user_vector / norm
                    user_vectors[user_id] = user_vector
                    successful_vectors += 1
            else:
                low_weight_sum += 1
        
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = elapsed / (batch_end - batch_start) * (total_users - batch_end) if batch_end < total_users else 0
        
        logger.info(f"Generated vectors: {batch_end}/{total_users} users ({progress:.1f}%) - Success: {successful_vectors}")
        
        # Garbage collection
        gc.collect()
    
    logger.info(f"User vector generation summary: Success: {successful_vectors}, No ratings: {no_ratings_found}, No movie vectors: {no_vectors_for_movies}, Low weights: {low_weight_sum}")
    
    # More aggressive cleanup
    del user_ratings_dict
    gc.collect()
    
    return user_vectors

def calculate_user_movie_similarity(user_vectors, movie_vectors, threshold=0.3, batch_size=50):
    """
    Calculate similarity between users and movies using improved methods.
    """
    start_time = time.time()
    
    user_movie_similarities = {}
    
    user_ids = list(user_vectors.keys())
    total_users = len(user_ids)
    total_movies = len(movie_vectors)
    
    # Convert movie vectors to a numpy array for faster batch computation
    movie_ids = list(movie_vectors.keys())
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    movie_vectors_array = np.array([movie_vectors[mid] for mid in movie_ids])
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        for user_id in batch_user_ids:
            user_vector = user_vectors[user_id]
            user_sims = {}
            
            # Fast matrix multiplication for all movies at once
            user_vector_array = np.array(user_vector).reshape(1, -1)
            similarities = np.dot(user_vector_array, movie_vectors_array.T)[0]
            
            # Apply threshold and store result
            for i, sim in enumerate(similarities):
                if sim > threshold:
                    movie_id = movie_ids[i]
                    user_sims[movie_id] = float(sim)
            
            user_movie_similarities[user_id] = user_sims
        
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = (elapsed / (batch_end - batch_start)) * (total_users - batch_end) if batch_end < total_users else 0
        
        avg_similarities = np.mean([len(user_sims) for user_sims in user_movie_similarities.values() if user_sims])
        logger.info(f"Calculated similarities: {batch_end}/{total_users} users ({progress:.1f}%) - Avg similarities per user: {avg_similarities:.1f}")
        
        # Garbage collection
        gc.collect()
    
    return user_movie_similarities

def get_user_rated_movies(user_id, train_ratings, cached_rated_movies=None):
    """Get movies rated by a user from training data"""
    if cached_rated_movies is None:
        cached_rated_movies = {}
        
    if user_id in cached_rated_movies:
        return cached_rated_movies[user_id]
    
    if train_ratings is None:
        return set()
    
    user_data = train_ratings[train_ratings['userId'] == user_id]
    rated_movies = set(user_data['movieId'].values)
    
    cached_rated_movies[user_id] = rated_movies
    
    return rated_movies

def get_top_n_recommendations(user_id, user_movie_similarities, train_ratings, cached_rated_movies=None, n=10):
    """Generate top N movie recommendations for a user"""
    if user_id not in user_movie_similarities:
        return []
    
    rated_movies = get_user_rated_movies(user_id, train_ratings, cached_rated_movies)
    
    user_sims = user_movie_similarities[user_id]
    
    # Exclude already rated movies
    candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                 if movie_id not in rated_movies]
    
    # Sort by similarity score
    recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    return recommendations[:n]

def predict_rating(user_id, movie_id, user_movie_similarities, train_ratings, user_bias_dict=None):
    """
    Predict a user's rating for a movie with improved methods.
    Adds user bias correction for more accurate predictions.
    """
    # If no user bias dictionary is provided, create an empty one
    if user_bias_dict is None:
        user_bias_dict = {}
    
    # If we don't have similarity data for this user, use fallback
    if user_id not in user_movie_similarities:
        # If we have rating data, return user average
        user_train = train_ratings[train_ratings['userId'] == user_id]
        if len(user_train) > 0:
            return user_train['rating'].mean()
        return 3.0  # Default average
    
    # Get the user's bias if we have it, otherwise calculate it
    if user_id in user_bias_dict:
        user_bias = user_bias_dict[user_id]
    else:
        user_train = train_ratings[train_ratings['userId'] == user_id]
        if len(user_train) > 0:
            user_avg = user_train['rating'].mean()
            # Calculate global average
            global_avg = train_ratings['rating'].mean()
            user_bias = user_avg - global_avg
        else:
            user_bias = 0.0
        user_bias_dict[user_id] = user_bias
    
    # If we have similarity for this movie, use it
    if movie_id in user_movie_similarities[user_id]:
        sim_score = user_movie_similarities[user_id][movie_id]
        
        # Convert similarity (0-1) to rating scale (0.5-5.0)
        base_prediction = 0.5 + 4.5 * sim_score
        
        # Apply user bias correction (damped to avoid extreme predictions)
        bias_weight = 0.7  # How much to consider user bias (0-1)
        predicted_rating = base_prediction + (bias_weight * user_bias)
        
        # Ensure the rating is within bounds
        return max(0.5, min(5.0, predicted_rating))
    
    # If we don't have similarity for this movie, use user average
    user_train = train_ratings[train_ratings['userId'] == user_id]
    if len(user_train) > 0:
        return user_train['rating'].mean()
    
    # Last resort - global average
    return train_ratings['rating'].mean() if len(train_ratings) > 0 else 3.0

def generate_recommendations_for_all_users(user_movie_similarities, train_ratings, movie_features, n=10, batch_size=100):
    """Generate recommendations for all users with improved batch processing"""
    start_time = time.time()
    
    cached_rated_movies = {}
    
    user_ids = list(user_movie_similarities.keys())
    total_users = len(user_ids)
    
    all_recommendations = {}
    users_with_recommendations = 0
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        batch_start_time = time.time()
        
        for user_id in batch_user_ids:
            recommendations = get_top_n_recommendations(
                user_id, 
                user_movie_similarities, 
                train_ratings, 
                cached_rated_movies,
                n
            )
            
            if recommendations:
                all_recommendations[user_id] = recommendations
                users_with_recommendations += 1
        
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = batch_time * ((total_users - batch_end) / len(batch_user_ids)) if batch_end < total_users else 0
        
        logger.info(f"Generated recommendations: {batch_end}/{total_users} users ({progress:.1f}%) - Users with recs: {users_with_recommendations}")
        
        # Garbage collection
        gc.collect()
    
    return all_recommendations

def evaluate_with_rmse_mae(user_movie_similarities, train_ratings, test_ratings, batch_size=100, rating_threshold=3.5):
    """
    Evaluate recommendation model using RMSE, MAE, and classification metrics.
    Added precision, recall, F1, and accuracy calculations.
    """
    print("Evaluating recommendation model using enhanced metrics with batching...")
    start_time = time.time()
    
    # Initialize metrics
    squared_errors_sum = 0
    absolute_errors_sum = 0
    total_predictions = 0
    users_evaluated = 0
    
    # Classification metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Dictionary to cache user biases
    user_bias_dict = {}
    
    # Calculate global stats once
    global_avg_rating = train_ratings['rating'].mean()
    
    # Calculate user biases in advance
    for user_id in train_ratings['userId'].unique():
        user_ratings = train_ratings[train_ratings['userId'] == user_id]
        if len(user_ratings) > 0:
            user_avg = user_ratings['rating'].mean()
            user_bias_dict[user_id] = user_avg - global_avg_rating
    
    # Find common users between training and test sets that have similarity data
    test_users = set(test_ratings['userId'].unique())
    train_users = set(train_ratings['userId'].unique())
    similarity_users = set(user_movie_similarities.keys())
    
    common_users = test_users.intersection(train_users).intersection(similarity_users)
    
    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}, Similarity users: {len(similarity_users)}")
    logger.info(f"Common users for evaluation: {len(common_users)}")
    
    if len(common_users) == 0:
        logger.warning("No common users between train, test, and similarity data")
        # Use baseline prediction (global average) for all test ratings
        predictions = np.full(len(test_ratings), global_avg_rating)
        actuals = test_ratings['rating'].values
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        # Convert to binary for classification metrics
        binary_preds = (predictions > rating_threshold).astype(int)
        binary_actuals = (actuals > rating_threshold).astype(int)
        
        # Calculate classification metrics
        accuracy = np.mean(binary_preds == binary_actuals)
        
        true_pos = np.sum((binary_preds == 1) & (binary_actuals == 1))
        false_pos = np.sum((binary_preds == 1) & (binary_actuals == 0))
        true_neg = np.sum((binary_preds == 0) & (binary_actuals == 0))
        false_neg = np.sum((binary_preds == 0) & (binary_actuals == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Baseline evaluation with global average ({global_avg_rating:.2f}):")
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_predictions': len(test_ratings)
        }
    
    # Process common users in batches
    user_list = list(common_users)
    for batch_start in range(0, len(user_list), batch_size):
        batch_end = min(batch_start + batch_size, len(user_list))
        batch_users = user_list[batch_start:batch_end]
        
        logger.info(f"Evaluating batch {batch_start//batch_size + 1}: users {batch_start+1}-{batch_end} of {len(user_list)}")
        batch_start_time = time.time()
        
        batch_squared_errors = 0
        batch_absolute_errors = 0
        batch_predictions = 0
        batch_true_positives = 0
        batch_false_positives = 0
        batch_true_negatives = 0
        batch_false_negatives = 0
        
        for user_id in batch_users:
            user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
            
            if len(user_test_ratings) == 0:
                continue
            
            for _, row in user_test_ratings.iterrows():
                movie_id = row['movieId']
                true_rating = row['rating']
                
                # Make prediction
                predicted_rating = predict_rating(user_id, movie_id, user_movie_similarities, train_ratings, user_bias_dict)
                
                # Calculate error
                squared_error = (predicted_rating - true_rating) ** 2
                absolute_error = abs(predicted_rating - true_rating)
                
                # Classification metrics
                binary_prediction = 1 if predicted_rating > rating_threshold else 0
                binary_actual = 1 if true_rating > rating_threshold else 0
                
                if binary_prediction == 1 and binary_actual == 1:
                    batch_true_positives += 1
                elif binary_prediction == 1 and binary_actual == 0:
                    batch_false_positives += 1
                elif binary_prediction == 0 and binary_actual == 0:
                    batch_true_negatives += 1
                elif binary_prediction == 0 and binary_actual == 1:
                    batch_false_negatives += 1
                
                # Accumulate errors
                batch_squared_errors += squared_error
                batch_absolute_errors += absolute_error
                batch_predictions += 1
            
            users_evaluated += 1
        
        # Accumulate batch results
        squared_errors_sum += batch_squared_errors
        absolute_errors_sum += batch_absolute_errors
        total_predictions += batch_predictions
        true_positives += batch_true_positives
        false_positives += batch_false_positives
        true_negatives += batch_true_negatives
        false_negatives += batch_false_negatives
        
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = batch_end / len(user_list) * 100
        remaining = batch_time * ((len(user_list) - batch_end) / len(batch_users)) if batch_end < len(user_list) else 0
        
        if batch_predictions > 0:
            batch_rmse = np.sqrt(batch_squared_errors / batch_predictions)
            batch_mae = batch_absolute_errors / batch_predictions
            batch_accuracy = (batch_true_positives + batch_true_negatives) / batch_predictions
            
            logger.info(f"Batch metrics - RMSE: {batch_rmse:.4f}, MAE: {batch_mae:.4f}, Accuracy: {batch_accuracy:.4f}")
            logger.info(f"TP: {batch_true_positives}, FP: {batch_false_positives}, TN: {batch_true_negatives}, FN: {batch_false_negatives}")
        
        logger.info(f"Processed {batch_end}/{len(user_list)} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Garbage collection
        gc.collect()
    
    # Calculate final metrics
    if total_predictions > 0:
        rmse = np.sqrt(squared_errors_sum / total_predictions)
        mae = absolute_errors_sum / total_predictions
        
        # Classification metrics
        accuracy = (true_positives + true_negatives) / total_predictions
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        rmse = 0.0
        mae = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    logger.info("\nEvaluation results:")
    logger.info(f"Users evaluated: {users_evaluated}")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix: [TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}]")
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'num_users_evaluated': users_evaluated,
        'num_predictions': total_predictions
    }
    
    return metrics

# Main execution flow with all optimizations
log_memory_usage("Initial memory usage")

# Load data
data = load_data()
log_memory_usage("After loading data")

# Corpus word counts
if 'movie_features' in data:
    corpus_word_counts = Counter()
    
    batch_size = 1000
    total_movies = len(data['movie_features'])
    
    for i in range(0, total_movies, batch_size):
        batch_end = min(i + batch_size, total_movies)
        batch = data['movie_features'].iloc[i:batch_end]
        
        for tokens in batch['tokens']:
            corpus_word_counts.update(tokens)
    
    data['corpus_word_counts'] = corpus_word_counts
    
    with open(os.path.join(output_path, 'corpus_word_counts.pkl'), 'wb') as f:
        pickle.dump(corpus_word_counts, f)
    
    logger.info(f"Created corpus word counts with {len(corpus_word_counts)} unique words")
    gc.collect()
    log_memory_usage("After corpus word count")

# Calculate Log-Likelihood values
if 'movie_features' in data and 'corpus_word_counts' in data:
    movie_ll_values = calculate_log_likelihood(data['movie_features'], data['corpus_word_counts'], batch_size=100)
    data['movie_ll_values'] = movie_ll_values
    
    with open(os.path.join(output_path, 'movie_ll_values.pkl'), 'wb') as f:
        pickle.dump(movie_ll_values, f)
    
    logger.info(f"Calculated LL values for {len(movie_ll_values)} movies")
    gc.collect()
    log_memory_usage("After LL calculation")

# Train Word2Vec model
if 'movie_features' in data:
    word2vec_model = train_word2vec(data['movie_features'], word2vec_dim)
    data['word2vec_model'] = word2vec_model
    
    word2vec_path = os.path.join(output_path, 'word2vec_model')
    word2vec_model.save(word2vec_path)
    
    logger.info(f"Trained Word2Vec model with vector size {word2vec_dim}")
    gc.collect()
    log_memory_usage("After Word2Vec training")

# Generate movie vectors
if 'word2vec_model' in data and 'movie_ll_values' in data and 'corpus_word_counts' in data:
    movie_vectors = generate_movie_vectors(
        data['movie_ll_values'], 
        data['word2vec_model'],
        data['movie_features'],
        data['corpus_word_counts'],
        batch_size=100
    )
    data['movie_vectors'] = movie_vectors
    
    with open(os.path.join(output_path, 'movie_vectors.pkl'), 'wb') as f:
        pickle.dump(movie_vectors, f)
    
    logger.info(f"Generated vectors for {len(movie_vectors)} movies")
    
    # Create movie ID to index mapping
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_vectors.keys())}
    data['movie_id_to_idx'] = movie_id_to_idx
    
    with open(os.path.join(output_path, 'movie_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(movie_id_to_idx, f)
    
    gc.collect()
    log_memory_usage("After movie vector generation")

# Generate user vectors
if 'movie_vectors' in data and 'train_ratings' in data:
    user_vectors = generate_user_vectors(data['movie_vectors'], data['train_ratings'], batch_size=100)
    data['user_vectors'] = user_vectors
    
    with open(os.path.join(output_path, 'user_vectors.pkl'), 'wb') as f:
        pickle.dump(user_vectors, f)
    
    logger.info(f"Generated vectors for {len(user_vectors)} users")
    
    # Create user ID to index mapping
    user_id_to_idx = {user_id: i for i, user_id in enumerate(user_vectors.keys())}
    data['user_id_to_idx'] = user_id_to_idx
    
    with open(os.path.join(output_path, 'user_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(user_id_to_idx, f)
    
    gc.collect()
    log_memory_usage("After user vector generation")

# Calculate user-movie similarities
if 'user_vectors' in data and 'movie_vectors' in data:
    user_movie_similarities = calculate_user_movie_similarity(
        data['user_vectors'], 
        data['movie_vectors'], 
        threshold=similarity_threshold,
        batch_size=50
    )
    data['user_movie_similarities'] = user_movie_similarities
    
    with open(os.path.join(output_path, 'user_movie_similarities.pkl'), 'wb') as f:
        pickle.dump(user_movie_similarities, f)
    
    logger.info(f"Calculated similarities for {len(user_movie_similarities)} users")
    gc.collect()
    log_memory_usage("After similarity calculation")

# Generate recommendations
if 'user_movie_similarities' in data and 'train_ratings' in data:
    all_recommendations = generate_recommendations_for_all_users(
        data['user_movie_similarities'], 
        data['train_ratings'],
        data['movie_features'],
        n=top_n,
        batch_size=100
    )
    data['all_recommendations'] = all_recommendations
    
    with open(os.path.join(output_path, 'content_based_recommendations.pkl'), 'wb') as f:
        pickle.dump(all_recommendations, f)
    
    logger.info(f"Generated recommendations for {len(all_recommendations)} users")
    
    # Save to CSV in chunks for better memory handling
    recommendations_list = []
    
    chunk_size = 1000
    user_ids = list(all_recommendations.keys())
    total_users = len(user_ids)
    
    for chunk_start in range(0, total_users, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_users)
        user_chunk = user_ids[chunk_start:chunk_end]
        
        chunk_recommendations = []
        for user_id in user_chunk:
            for rank, (movie_id, score) in enumerate(all_recommendations[user_id], 1):
                movie_title = "Unknown"
                if 'movie_features' in data:
                    movie_row = data['movie_features'][data['movie_features']['movieId'] == movie_id]
                    if not movie_row.empty and 'title' in movie_row.columns:
                        movie_title = movie_row.iloc[0]['title']
                        
                chunk_recommendations.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'title': movie_title,
                    'rank': rank,
                    'similarity_score': score
                })
        
        recommendations_list.extend(chunk_recommendations)
        gc.collect()
    
    if recommendations_list:
        chunk_size = 10000
        total_recs = len(recommendations_list)
        
        for chunk_start in range(0, total_recs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_recs)
            chunk = recommendations_list[chunk_start:chunk_end]
            
            chunk_df = pd.DataFrame(chunk)
            
            # Write header only for the first chunk
            if chunk_start == 0:
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='w')
            else:
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='a', header=False)
    
    del recommendations_list
    gc.collect()
    log_memory_usage("After recommendation generation")

# Evaluate the model
if 'user_movie_similarities' in data and 'train_ratings' in data and 'test_ratings' in data:
    logger.info("Running evaluation with enhanced metrics...")
    evaluation_metrics = evaluate_with_rmse_mae(
        data['user_movie_similarities'],
        data['train_ratings'],
        data['test_ratings'],
        batch_size=100,
        rating_threshold=rating_threshold
    )
    
    data['evaluation_metrics'] = evaluation_metrics
    
    logger.info("\nStored evaluation metrics:")
    for key, value in evaluation_metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save evaluation results
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'content_based_evaluation.csv'), index=False)
    
    # Create visualizations for evaluation metrics
    if 'rmse' in evaluation_metrics:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = np.array([
            [evaluation_metrics.get('true_negatives', 0), evaluation_metrics.get('false_positives', 0)],
            [evaluation_metrics.get('false_negatives', 0), evaluation_metrics.get('true_positives', 0)]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
        plt.close()
        
        # Create a bar chart comparing all metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [evaluation_metrics.get(metric, 0) for metric in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        plt.bar(metrics_to_plot, values, color=colors)
        plt.ylim(0, 1)
        plt.title('Classification Performance Metrics')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_path, 'classification_metrics.png'))
        plt.close()
        
        logger.info("Created evaluation visualizations")
    
    gc.collect()
    log_memory_usage("After evaluation")

logger.info("Content-based recommendation pipeline completed successfully!")