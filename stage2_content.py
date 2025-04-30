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
import sys
import gc  # Add garbage collector for memory management

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("CONTENT-BASED MOVIE RECOMMENDATION SYSTEM WITH LOG-LIKELIHOOD AND WORD2VEC")
print("="*80)

# Set paths
input_path = "./"  # Current directory where stage1.py saved the files
output_path = "./rec/content-recommendations"
top_n =50

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# # Initialize NLTK tools
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# Model parameters/
similarity_threshold = 0.8  # Minimum similarity to consider
word2vec_dim = 200  # Dimensionality of Word2Vec embeddings

print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

def load_data():
    """Load processed data from stage1.py"""
    print("Loading processed data from stage1.py...")
    
    # Data containers
    data = {}
    
    # Load movie features
    movie_features_path = os.path.join(input_path, './processed/processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        # First, load without the tokens column to save memory
        data['movie_features'] = pd.read_csv(movie_features_path)
        # Convert string representation of tokens and top_keywords back to lists
        data['movie_features']['tokens'] = data['movie_features']['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        data['movie_features']['top_keywords'] = data['movie_features']['top_keywords'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        print(f"Loaded features for {len(data['movie_features'])} movies")
        print("\nSample of movie features data:")
        print(data['movie_features'][['movieId', 'title', 'top_keywords']].head(3))
        
        # Print token statistics
        token_lengths = [len(tokens) for tokens in data['movie_features']['tokens']]
        print(f"\nAverage token count per movie: {np.mean(token_lengths):.2f}")
        print(f"Min token count: {min(token_lengths)}, Max token count: {max(token_lengths)}")
    else:
        print(f"Error: Movie features not found at {movie_features_path}")
        sys.exit(1)
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, './processed/normalized_ratings.csv')
    if os.path.exists(ratings_path):
        # Read in chunks to save memory
        chunk_size = 100000  # Adjust based on dataset size
        chunks = []
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            chunks.append(chunk)
        data['ratings'] = pd.concat(chunks)
        print(f"\nLoaded {len(data['ratings'])} normalized ratings")
        print("\nSample of normalized ratings data:")
        print(data['ratings'].head(3))
        
        # Print rating statistics
        print(f"\nNumber of unique users: {data['ratings']['userId'].nunique()}")
        print(f"Number of unique movies: {data['ratings']['movieId'].nunique()}")
        print(f"Rating sparsity: {(1 - len(data['ratings']) / (data['ratings']['userId'].nunique() * data['ratings']['movieId'].nunique())) * 100:.4f}%")
    else:
        print(f"Error: Normalized ratings not found at {ratings_path}")
        sys.exit(1)
    
    # Create training and testing sets with 80-20 split
    # Create training and testing sets with 80-20 split
    if 'ratings' in data:
        print("Splitting ratings into training and testing sets...")
        
        # Import train_test_split if not already imported
        from sklearn.model_selection import train_test_split
        
        # Split ratings into training and testing sets
        train_ratings, test_ratings = train_test_split(
            data['ratings'], 
            test_size=0.2
        )
        
        # Store the split data
        data['train_ratings'] = train_ratings
        data['test_ratings'] = test_ratings
        
        print(f"\nSplit ratings into {len(data['train_ratings'])} training and {len(data['test_ratings'])} testing samples")
        print(f"Training set covers {data['train_ratings']['userId'].nunique()} users and {data['train_ratings']['movieId'].nunique()} movies")
        print(f"Testing set covers {data['test_ratings']['userId'].nunique()} users and {data['test_ratings']['movieId'].nunique()} movies")
    
    return data

# Load the data
data = load_data()

print("\n" + "="*80)
print("STEP 2: CORPUS ANALYSIS")
print("="*80)

# Build corpus word counts from movie features
if 'movie_features' in data:
    print("Building vocabulary and word frequency counts...")
    
    corpus_word_counts = Counter()
    
    # Process in batches to avoid memory spikes
    batch_size = 1000
    total_movies = len(data['movie_features'])
    
    for i in range(0, total_movies, batch_size):
        batch_end = min(i + batch_size, total_movies)
        batch = data['movie_features'].iloc[i:batch_end]
        
        for tokens in batch['tokens']:
            corpus_word_counts.update(tokens)
        
        # Log progress
        print(f"Processed {batch_end}/{total_movies} movies ({batch_end/total_movies*100:.1f}%)")
    
    data['corpus_word_counts'] = corpus_word_counts
    
    # Save corpus word counts
    with open(os.path.join(output_path, 'corpus_word_counts.pkl'), 'wb') as f:
        pickle.dump(corpus_word_counts, f)
    
    print(f"Built vocabulary with {len(corpus_word_counts)} unique words")
    print(f"Total words in corpus: {sum(corpus_word_counts.values())}")
    
    # Display top 20 most common words
    print("\nTop 20 most common words in the corpus:")
    for word, count in corpus_word_counts.most_common(20):
        print(f"'{word}': {count}")

    # Clear memory
    gc.collect()

print("\n" + "="*80)
print("STEP 3: LOG-LIKELIHOOD CALCULATION")
print("="*80)

def calculate_log_likelihood(movie_features, corpus_word_counts, batch_size=100):
    """Calculate Log-Likelihood values for words in each movie in batches"""
    print("Calculating Log-Likelihood values for all movies in batches...")
    start_time = time.time()
    
    # Calculate total corpus size
    total_corpus_size = sum(corpus_word_counts.values())
    print(f"Total corpus size: {total_corpus_size} words")
    
    # Initialize container for movie features
    movie_ll_values = {}
    
    # Process each movie document in batches
    total_movies = len(movie_features)
    
    for batch_start in range(0, total_movies, batch_size):
        batch_end = min(batch_start + batch_size, total_movies)
        print(f"Processing batch {batch_start//batch_size + 1}: movies {batch_start+1}-{batch_end} of {total_movies}")
        
        # Get batch of movies
        batch = movie_features.iloc[batch_start:batch_end]
        
        for _, row in batch.iterrows():
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
                
                # Expected counts based on corpus distribution
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
        
        # Log progress and elapsed time
        elapsed = time.time() - start_time
        progress = batch_end / total_movies * 100
        remaining = elapsed / (batch_end - batch_start) * (total_movies - batch_end) if batch_end < total_movies else 0
        print(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Show sample LL values for a movie
    if movie_ll_values:
        sample_movie_id = next(iter(movie_ll_values.keys()))
        sample_movie_title = movie_features[movie_features['movieId'] == sample_movie_id]['title'].values[0]
        print(f"\nSample Log-Likelihood values for movie '{sample_movie_title}' (ID: {sample_movie_id}):")
        
        # Get top 10 words by LL value
        top_ll_words = sorted(movie_ll_values[sample_movie_id].items(), key=lambda x: x[1], reverse=True)[:10]
        for word, ll_value in top_ll_words:
            print(f"Word: '{word}', LL Value: {ll_value:.2f}")
    
    return movie_ll_values

# Calculate Log-Likelihood if movie features are available
if 'movie_features' in data and 'corpus_word_counts' in data:
    movie_ll_values = calculate_log_likelihood(data['movie_features'], data['corpus_word_counts'], batch_size=100)
    data['movie_ll_values'] = movie_ll_values
    
    # Save Log-Likelihood values
    with open(os.path.join(output_path, 'movie_ll_values.pkl'), 'wb') as f:
        pickle.dump(movie_ll_values, f)
    
    print(f"Calculated Log-Likelihood values for {len(movie_ll_values)} movies")
    
    # Calculate average number of words with high LL values
    high_ll_counts = []
    for movie_id, ll_dict in movie_ll_values.items():
        high_ll_words = [word for word, value in ll_dict.items() if value > 10]  # Threshold of 10
        high_ll_counts.append(len(high_ll_words))
    
    print(f"Average number of words with LL > 10 per movie: {np.mean(high_ll_counts):.2f}")
    print(f"Min: {min(high_ll_counts)}, Max: {max(high_ll_counts)}")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("STEP 4: WORD2VEC MODEL TRAINING")
print("="*80)

def train_word2vec(movie_features, vector_size=100, batch_size=1000):
    """Train Word2Vec model on movie tokens with memory optimization"""
    print(f"Training Word2Vec model with {vector_size} dimensions...")
    start_time = time.time()
    
    # Extract token lists from movie features in batches
    tokenized_corpus = []
    total_movies = len(movie_features)
    
    for i in range(0, total_movies, batch_size):
        batch_end = min(i + batch_size, total_movies)
        batch = movie_features.iloc[i:batch_end]
        
        batch_tokens = list(batch['tokens'])
        tokenized_corpus.extend(batch_tokens)
        
        # Log progress
        print(f"Loaded tokens from {batch_end}/{total_movies} movies ({batch_end/total_movies*100:.1f}%)")
    
    # Print corpus statistics
    total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
    print(f"Training corpus size: {total_tokens} tokens from {len(tokenized_corpus)} documents")
    
    # Train Word2Vec model using CBOW approach with memory optimization
    print("Starting Word2Vec training (this may take a few minutes)...")
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=10,
        min_count=3,
        workers=8,
        epochs=25,  # Reduced from 50 to save memory
        sg=1,  # CBOW model
        hs=1,
        negative=20
    )
    
    # Free memory - no longer need the full corpus
    del tokenized_corpus
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"Word2Vec training completed in {elapsed:.2f} seconds")
    
    # Print model statistics
    vocab_size = len(word2vec_model.wv)
    print(f"Word2Vec model vocabulary size: {vocab_size} words")
    
    # Show some example vectors for common words
    print("\nExample word vectors from the trained model:")
    common_words = [word for word, _ in data['corpus_word_counts'].most_common(10)]
    for word in common_words:
        if word in word2vec_model.wv:
            # Show just the first 5 dimensions of the vector
            print(f"'{word}': {word2vec_model.wv[word][:5]}...")
    
    # Show some word similarities
    if len(word2vec_model.wv) > 0:
        print("\nExample word similarities:")
        try:
            # Try some movie-related terms
            for word in ['action', 'love', 'hero', 'villain']:
                if word in word2vec_model.wv:
                    similar_words = word2vec_model.wv.most_similar(word, topn=5)
                    print(f"Words similar to '{word}': {similar_words}")
        except Exception as e:
            print(f"Could not compute word similarities: {str(e)}")
    
    return word2vec_model

# Train Word2Vec if movie features are available
if 'movie_features' in data:
    word2vec_model = train_word2vec(data['movie_features'], word2vec_dim)
    data['word2vec_model'] = word2vec_model
    
    # Save Word2Vec model
    word2vec_path = os.path.join(output_path, 'word2vec_model')
    word2vec_model.save(word2vec_path)
    
    print(f"Trained and saved Word2Vec model with {len(word2vec_model.wv)} words")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("STEP 5: MOVIE VECTOR GENERATION")
print("="*80)

def generate_movie_vectors(movie_ll_values, word2vec_model, movie_features, batch_size=100):
    """Generate movie feature vectors using Log-Likelihood and Word2Vec in batches"""
    print("Generating movie feature vectors using Log-Likelihood + Word2Vec in batches...")
    start_time = time.time()
    
    movie_vectors = {}
    successful_vectors = 0
    no_words_found = 0
    low_ll_sum = 0
    
    # Get movie IDs from LL values
    movie_ids = list(movie_ll_values.keys())
    total_movies = len(movie_ids)
    
    # Process movies in batches
    for batch_start in range(0, total_movies, batch_size):
        batch_end = min(batch_start + batch_size, total_movies)
        batch_movie_ids = movie_ids[batch_start:batch_end]
        total_corpus_size = sum(corpus_word_counts.values())

        print(f"Processing batch {batch_start//batch_size + 1}: movies {batch_start+1}-{batch_end} of {total_movies}")
        
        for movie_id in batch_movie_ids:
            # Sort words by LL value and select top 200
            ll_values = movie_ll_values[movie_id]
    # Tính IDF cho từng từ
            idf = {word: math.log(total_corpus_size/(corpus_word_counts[word]+1)) 
                for word in ll_values.keys()}
            
            # Kết hợp LL và IDF
            combined_scores = {word: ll * idf[word] 
                            for word, ll in ll_values.items()}
            
            # Chọn top words dựa trên combined scores
            top_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:500]            
            if not top_words:
                no_words_found += 1
                continue
            
            # Combine Word2Vec vectors weighted by Log-Likelihood values
            weighted_vectors = []
            ll_sum = 0
            words_used = 0
            
            for word, ll_value in top_words:
                if ll_value <= 0:
                    continue
                
                if word in word2vec_model.wv:
                    weighted_vectors.append(word2vec_model.wv[word] * ll_value)
                    ll_sum += ll_value
                    words_used += 1
            
            if weighted_vectors and ll_sum > 0:
                # Calculate the weighted average vector
                movie_vector = np.sum(weighted_vectors, axis=0) / ll_sum
                
                # Normalize to unit length
                norm = np.linalg.norm(movie_vector)
                if norm > 0:
                    movie_vector = movie_vector / norm
                    movie_vectors[movie_id] = movie_vector
                    successful_vectors += 1
            else:
                low_ll_sum += 1
        
        # Log progress and elapsed time
        elapsed = time.time() - start_time
        progress = batch_end / total_movies * 100
        remaining = elapsed / (batch_end - batch_start) * (total_movies - batch_end) if batch_end < total_movies else 0
        print(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        print(f"Successfully created vectors: {successful_vectors}/{batch_end}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    print(f"\nVector generation complete:")
    print(f"Successfully created vectors: {successful_vectors}/{total_movies} ({successful_vectors/total_movies*100:.1f}%)")
    print(f"Movies with no words found: {no_words_found}")
    print(f"Movies with too low LL sum: {low_ll_sum}")
    
    # Display sample movie vectors
    if movie_vectors:
        print("\nSample movie vectors:")
        for movie_id in list(movie_vectors.keys())[:3]:
            movie_title = movie_features[movie_features['movieId'] == movie_id]['title'].values[0]
            vector = movie_vectors[movie_id]
            print(f"Movie: '{movie_title}' (ID: {movie_id})")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector norm: {np.linalg.norm(vector):.4f}")
            print(f"First 5 dimensions: {vector[:5]}")
            print("---")
    
    return movie_vectors

# Generate movie vectors if Word2Vec and LL values are available
if 'word2vec_model' in data and 'movie_ll_values' in data:
    movie_vectors = generate_movie_vectors(
        data['movie_ll_values'], 
        data['word2vec_model'],
        data['movie_features'],
        batch_size=100
    )
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
    
    print(f"Generated and saved feature vectors for {len(movie_vectors)} movies")
    
    # Calculate and display vector statistics
    vector_norms = [np.linalg.norm(v) for v in movie_vectors.values()]
    print(f"\nVector statistics:")
    print(f"Average vector norm: {np.mean(vector_norms):.4f}")
    print(f"Vector dimensionality: {word2vec_dim}")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("STEP 6: USER VECTOR GENERATION")
print("="*80)

def generate_user_vectors(movie_vectors, train_ratings, batch_size=100):
    """Generate user feature vectors based on rated movies and their content in batches"""
    print("Generating user feature vectors based on movie ratings in batches...")
    start_time = time.time()
    
    user_vectors = {}
    successful_vectors = 0
    no_ratings_found = 0
    no_vectors_for_movies = 0
    low_weight_sum = 0
    
    # Create a rating cache for quick lookups
    # This can be memory intensive for large datasets, but speeds up processing
    user_ratings_dict = {}
    print("Creating user ratings lookup dictionary...")
    
    # Process in chunks to avoid memory issues
    for _, row in train_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        if user_id not in user_ratings_dict:
            user_ratings_dict[user_id] = {}
        
        user_ratings_dict[user_id][movie_id] = rating
    
    # Process each user
    user_ids = list(user_ratings_dict.keys())
    total_users = len(user_ids)
    print(f"Processing {total_users} users in batches...")
    
    # Process users in batches
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: users {batch_start+1}-{batch_end} of {total_users}")
        
        for user_id in batch_user_ids:
            # Get user ratings
            user_ratings = user_ratings_dict[user_id]
            
            if len(user_ratings) == 0:
                no_ratings_found += 1
                continue
            
            weighted_vectors = []
            weight_sum = 0
            movies_with_vectors = 0
            movies_without_vectors = 0
            
            for movie_id, rating in user_ratings.items():
                # Center rating at 3.0 as described in the papers
                weight = rating - 3.0
                
                # Skip if movie vector is not available
                if movie_id not in movie_vectors:
                    movies_without_vectors += 1
                    continue
                else:
                    movies_with_vectors += 1
                
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
                    successful_vectors += 1
            else:
                low_weight_sum += 1
        
        # Log progress
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = elapsed / (batch_end - batch_start) * (total_users - batch_end) if batch_end < total_users else 0
        print(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        print(f"Successfully created vectors: {successful_vectors}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    print(f"\nUser vector generation complete:")
    print(f"Successfully created vectors: {successful_vectors}/{total_users} ({successful_vectors/total_users*100:.1f}%)")
    print(f"Users with no ratings: {no_ratings_found}")
    print(f"Users with no vectorized movies: {no_vectors_for_movies}")
    print(f"Users with too low weight sum: {low_weight_sum}")
    
    # Free memory
    del user_ratings_dict
    gc.collect()
    
    # Display sample user vectors
    if user_vectors:
        print("\nSample user vectors:")
        for user_id in list(user_vectors.keys())[:3]:
            vector = user_vectors[user_id]
            user_rating_count = len([r for r in train_ratings[train_ratings['userId'] == user_id]])
            print(f"User ID: {user_id}")
            print(f"Number of ratings: {user_rating_count}")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector norm: {np.linalg.norm(vector):.4f}")
            print(f"First 5 dimensions: {vector[:5]}")
            print("---")
    
    return user_vectors

# Generate user vectors if movie vectors and training ratings are available
if 'movie_vectors' in data and 'train_ratings' in data:
    user_vectors = generate_user_vectors(data['movie_vectors'], data['train_ratings'], batch_size=100)
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
    
    print(f"Generated and saved feature vectors for {len(user_vectors)} users")
    
    # Calculate and display vector statistics
    vector_norms = [np.linalg.norm(v) for v in user_vectors.values()]
    print(f"\nVector statistics:")
    print(f"Average vector norm: {np.mean(vector_norms):.4f}")
    print(f"Vector dimensionality: {word2vec_dim}")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("STEP 7: USER-MOVIE SIMILARITY CALCULATION")
print("="*80)

def calculate_user_movie_similarity(user_vectors, movie_vectors, threshold=0.8, batch_size=50):
    """Calculate similarity between users and movies in batches"""
    print(f"Calculating user-movie similarity with threshold {threshold} in batches...")
    start_time = time.time()
    
    # Store similarities in a dictionary of dictionaries
    # {user_id: {movie_id: similarity_score}}
    user_movie_similarities = {}
    
    # Get all user IDs
    user_ids = list(user_vectors.keys())
    total_users = len(user_ids)
    total_movies = len(movie_vectors)
    
    # Process users in batches
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: users {batch_start+1}-{batch_end} of {total_users}")
        
        for user_id in batch_user_ids:
            user_vector = user_vectors[user_id]
            user_sims = {}
            user_similarities = 0
            user_above_threshold = 0
            
            # Calculate similarity for all movies at once (vectorized)
            # Convert both user and movie vectors to arrays for faster computation
            user_vector_array = np.array(user_vector).reshape(1, -1)
            
            # Process movies in chunks to avoid memory issues
            movie_ids = list(movie_vectors.keys())
            movie_chunk_size = 1000  # Adjust based on memory availability
            
            for movie_chunk_start in range(0, len(movie_ids), movie_chunk_size):
                movie_chunk_end = min(movie_chunk_start + movie_chunk_size, len(movie_ids))
                chunk_movie_ids = movie_ids[movie_chunk_start:movie_chunk_end]
                
                # Create array of movie vectors for this chunk
                movie_vectors_array = np.array([movie_vectors[mid] for mid in chunk_movie_ids])
                
                # Calculate cosine similarity in a vectorized way
                similarities = np.dot(user_vector_array, movie_vectors_array.T)[0]
                
                # Filter by threshold and store
                for i, sim in enumerate(similarities):
                    if sim > threshold:
                        movie_id = chunk_movie_ids[i]
                        user_sims[movie_id] = float(sim)  # Convert to native Python float
                        user_above_threshold += 1
                    user_similarities += 1
            
            user_movie_similarities[user_id] = user_sims
            
            # Log progress for this user
            if len(batch_user_ids) <= 10 or (user_id == batch_user_ids[-1]):
                print(f"User {user_id}: {user_above_threshold}/{total_movies} movies above threshold ({user_above_threshold/total_movies*100:.2f}%)")
        
        # Log progress for this batch
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = (elapsed / (batch_end - batch_start)) * (total_users - batch_end) if batch_end < total_users else 0
        print(f"Processed {batch_end}/{total_users} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Force garbage collection after each batch
        gc.collect()
    
    avg_above_threshold = sum(len(sims) for sims in user_movie_similarities.values()) / len(user_movie_similarities) if user_movie_similarities else 0
    
    print(f"\nSimilarity calculation complete:")
    print(f"Total users processed: {len(user_movie_similarities)}")
    print(f"Total movies per user: {total_movies}")
    print(f"Average movies above threshold per user: {avg_above_threshold:.2f}")
    
    # Display sample user similarities
    if user_movie_similarities:
        print("\nSample user-movie similarities:")
        for user_id in list(user_movie_similarities.keys())[:3]:
            sims = user_movie_similarities[user_id]
            print(f"User ID: {user_id}")
            print(f"Number of movies above threshold: {len(sims)}")
            if sims:
                top_movies = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 most similar movies:")
                for movie_id, sim in top_movies:
                    movie_title = data['movie_features'][data['movie_features']['movieId'] == movie_id]['title'].values[0] if 'movie_features' in data else f"Movie {movie_id}"
                    print(f"  '{movie_title}' (ID: {movie_id}): {sim:.4f}")
            print("---")
    
    return user_movie_similarities
# Calculate similarities if user and movie vectors are available
if 'user_vectors' in data and 'movie_vectors' in data:
    user_movie_similarities = calculate_user_movie_similarity(
        data['user_vectors'], 
        data['movie_vectors'], 
        threshold=similarity_threshold,
        batch_size=50
    )
    data['user_movie_similarities'] = user_movie_similarities
    
    # Save the similarities
    with open(os.path.join(output_path, 'user_movie_similarities.pkl'), 'wb') as f:
        pickle.dump(user_movie_similarities, f)
    
    print(f"Calculated and saved similarities for {len(user_movie_similarities)} users")
    
    # Calculate and display similarity statistics
    similarity_counts = [len(sims) for sims in user_movie_similarities.values()]
    print(f"\nSimilarity statistics:")
    print(f"Average number of similar movies per user: {np.mean(similarity_counts):.2f}")
    print(f"Min: {min(similarity_counts)}, Max: {max(similarity_counts)}")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("STEP 8: RECOMMENDATION GENERATION")
print("="*80)

def get_user_rated_movies(user_id, train_ratings, cached_rated_movies=None):
    """Get the set of movies already rated by a user with caching for efficiency"""
    # Initialize cache if not provided
    if cached_rated_movies is None:
        cached_rated_movies = {}
        
    # Return from cache if available
    if user_id in cached_rated_movies:
        return cached_rated_movies[user_id]
    
    # Get from ratings dataframe
    if train_ratings is None:
        return set()
    
    user_data = train_ratings[train_ratings['userId'] == user_id]
    rated_movies = set(user_data['movieId'].values)
    
    # Cache for future use
    cached_rated_movies[user_id] = rated_movies
    
    return rated_movies

def get_top_n_recommendations(user_id, user_movie_similarities, train_ratings, cached_rated_movies=None, n=10):
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
    cached_rated_movies : dict, optional
        Cache of user rated movies for efficiency
    n : int, optional
        Number of recommendations to generate
        
    Returns:
    --------
    list of tuples
        (movie_id, similarity_score) pairs sorted by similarity in descending order
    """
    if user_id not in user_movie_similarities:
        return []
    
    # Get movies already rated by the user (using cache)
    rated_movies = get_user_rated_movies(user_id, train_ratings, cached_rated_movies)
    
    # Get user's similarities
    user_sims = user_movie_similarities[user_id]
    
    # Filter out already rated movies and sort by similarity
    candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                 if movie_id not in rated_movies]
    
    # Sort by similarity (descending)
    recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    # Return top N
    return recommendations[:n]

def predict_rating(user_id, movie_id, user_movie_similarities, train_ratings):
    """
    Predict a user's rating for a movie
    
    Parameters:
    -----------
    user_id : int
        The user ID
    movie_id : int
        The movie ID
    user_movie_similarities : dict
        Dictionary of user-movie similarities
    train_ratings : pd.DataFrame
        DataFrame of user ratings
        
    Returns:
    --------
    float
        Predicted rating (0.5-5.0 scale)
    """
    # If user not in similarity matrix, return average rating
    if user_id not in user_movie_similarities:
        return 3.0
    
    # Get user's average rating from training data
    user_train = train_ratings[train_ratings['userId'] == user_id]
    user_avg_rating = user_train['rating'].mean() if len(user_train) > 0 else 3.0
    
    # If movie not in similarity matrix, return user's average rating
    if movie_id not in user_movie_similarities[user_id]:
        return user_avg_rating
    
    # Convert similarity score to rating prediction
    # Similarity is in range [0,1], convert to rating range [0.5,5]
    sim_score = user_movie_similarities[user_id][movie_id]
    predicted_rating = 0.5 + 4.5 * sim_score
    
    return predicted_rating

def generate_recommendations_for_all_users(user_movie_similarities, train_ratings, movie_features, n=10, batch_size=100):
    """Generate recommendations for all users with memory efficiency in mind"""
    print(f"Generating top-{n} recommendations for all users in batches...")
    start_time = time.time()
    
    # Create a shared cache for rated movies
    cached_rated_movies = {}
    
    # Get all user IDs
    user_ids = list(user_movie_similarities.keys())
    total_users = len(user_ids)
    
    all_recommendations = {}
    users_with_recommendations = 0
    
    # Process users in batches
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: users {batch_start+1}-{batch_end} of {total_users}")
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
        
        # Log progress after each batch
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = batch_time * ((total_users - batch_end) / len(batch_user_ids)) if batch_end < total_users else 0
        
        print(f"Processed {batch_end}/{total_users} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        print(f"Users with recommendations so far: {users_with_recommendations}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Calculate statistics
    if all_recommendations:
        total_recommendations = sum(len(recs) for recs in all_recommendations.values())
        avg_recommendations = total_recommendations / users_with_recommendations if users_with_recommendations > 0 else 0
        
        print(f"\nRecommendation generation complete:")
        print(f"Users with recommendations: {users_with_recommendations}/{total_users} ({users_with_recommendations/total_users*100:.1f}%)")
        print(f"Total recommendations generated: {total_recommendations}")
        print(f"Average recommendations per user: {avg_recommendations:.2f}")
    
    # Display sample recommendations for a few users
    if all_recommendations:
        print("\nSample recommendations for 3 users:")
        for user_id in list(all_recommendations.keys())[:3]:
            print(f"User ID: {user_id}")
            print("Top 5 recommended movies:")
            
            for rank, (movie_id, score) in enumerate(all_recommendations[user_id][:5], 1):
                movie_title = "Unknown"
                if movie_features is not None:
                    movie_row = movie_features[movie_features['movieId'] == movie_id]
                    if not movie_row.empty and 'title' in movie_row.columns:
                        movie_title = movie_row.iloc[0]['title']
                print(f"  {rank}. '{movie_title}' (ID: {movie_id}): {score:.4f}")
            print("---")
    
    return all_recommendations

# Generate recommendations if similarities are available
if 'user_movie_similarities' in data and 'train_ratings' in data:
    all_recommendations = generate_recommendations_for_all_users(
        data['user_movie_similarities'], 
        data['train_ratings'],
        data['movie_features'],
        n=top_n,
        batch_size=100
    )
    data['all_recommendations'] = all_recommendations
    
    # Save recommendations
    with open(os.path.join(output_path, 'content_based_recommendations.pkl'), 'wb') as f:
        pickle.dump(all_recommendations, f)
    
    # Also save in a more readable CSV format
    recommendations_list = []
    
    # Process in chunks to avoid memory issues
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
        print(f"Processed recommendation chunk {chunk_start//chunk_size + 1}: users {chunk_start+1}-{chunk_end} of {total_users}")
        gc.collect()
    
    if recommendations_list:
        # Write CSV in chunks to avoid memory issues
        chunk_size = 10000
        total_recs = len(recommendations_list)
        
        for chunk_start in range(0, total_recs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_recs)
            chunk = recommendations_list[chunk_start:chunk_end]
            
            chunk_df = pd.DataFrame(chunk)
            
            # For first chunk, write with header
            if chunk_start == 0:
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='w')
            else:
                # For subsequent chunks, append without header
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='a', header=False)
            
            print(f"Saved recommendation chunk {chunk_start//chunk_size + 1}: recommendations {chunk_start+1}-{chunk_end} of {total_recs}")
            
        print(f"Saved recommendations to CSV file with {total_recs} entries")
    
    # Free memory
    del recommendations_list
    gc.collect()

print("\n" + "="*80)
print("STEP 9: MODEL EVALUATION")
print("="*80)

# Thay đổi hàm evaluate_with_rmse_mae như sau:
def evaluate_with_rmse_mae(user_movie_similarities, train_ratings, test_ratings, batch_size=100):
    """
    Evaluate the recommendations using RMSE and MAE with batching for memory efficiency
    
    Parameters:
    -----------
    user_movie_similarities : dict
        Dictionary of user-movie similarities
    train_ratings : pd.DataFrame
        DataFrame of training ratings
    test_ratings : pd.DataFrame
        DataFrame of test ratings
    batch_size : int
        Size of user batches to process
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("Evaluating recommendation model using RMSE and MAE with batching...")
    start_time = time.time()
    
    # Lấy common users giữa test_ratings và training users
    test_users = set(test_ratings['userId'].unique())
    train_users = set(train_ratings['userId'].unique())
    users_to_evaluate = test_users.intersection(train_users)
    
    print(f"Users in test set: {len(test_users)}")
    print(f"Users in training set: {len(train_users)}")
    print(f"Users to evaluate (intersection): {len(users_to_evaluate)}")
    
    # Kiểm tra nếu chúng ta đang sử dụng user-based split
    if len(users_to_evaluate) == 0:
        print("Using user-based split - no common users between train and test.")
        print("Evaluating using average rating for all predictions instead.")
        
        # Lấy average rating từ training set
        avg_rating = train_ratings['rating'].mean()
        
        # Đếm số lượng dự đoán
        total_predictions = len(test_ratings)
        
        # Tính MSE và MAE bằng cách dùng average rating làm dự đoán
        squared_errors_sum = ((test_ratings['rating'] - avg_rating) ** 2).sum()
        absolute_errors_sum = (abs(test_ratings['rating'] - avg_rating)).sum()
        
        # Tính RMSE và MAE
        overall_rmse = np.sqrt(squared_errors_sum / total_predictions)
        overall_mae = absolute_errors_sum / total_predictions
        
        print("\nEvaluation results using average rating baseline:")
        print(f"Total predictions: {total_predictions}")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        
        # Tạo metrics dictionary
        metrics = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'num_predictions': total_predictions,
            'evaluation_method': 'average_rating_baseline'
        }
        
        return metrics
    
    # Nếu có common users, tiếp tục với cách đánh giá gốc
    # (Phần code gốc của bạn bắt đầu từ đây, giữ nguyên)
    users_with_similarity = set(user_movie_similarities.keys())
    users_in_test_with_similarity = users_to_evaluate.intersection(users_with_similarity)
    
    print(f"Users in test set with similarity data: {len(users_in_test_with_similarity)}/{len(users_to_evaluate)} ({len(users_in_test_with_similarity)/len(users_to_evaluate)*100:.1f}%)")
    
    # Track metrics in chunks instead of storing all predictions
    squared_errors_sum = 0
    absolute_errors_sum = 0
    total_predictions = 0
    users_evaluated = 0
    
    # Process users in batches
    user_list = list(users_in_test_with_similarity)
    for batch_start in range(0, len(user_list), batch_size):
        batch_end = min(batch_start + batch_size, len(user_list))
        batch_users = user_list[batch_start:batch_end]
        
        print(f"Evaluating batch {batch_start//batch_size + 1}: users {batch_start+1}-{batch_end} of {len(user_list)}")
        batch_start_time = time.time()
        
        batch_squared_errors = 0
        batch_absolute_errors = 0
        batch_predictions = 0
        
        for user_id in batch_users:
            # Get user test ratings
            user_test = test_ratings[test_ratings['userId'] == user_id]
            
            if len(user_test) == 0:
                continue
            
            # Get user's average rating from training data
            user_train = train_ratings[train_ratings['userId'] == user_id]
            user_avg_rating = user_train['rating'].mean() if len(user_train) > 0 else 3.0
            
            # Predict ratings for test items
            for _, row in user_test.iterrows():
                movie_id = row['movieId']
                true_rating = row['rating']
                
                # Get similarity-based prediction
                if movie_id in user_movie_similarities.get(user_id, {}):
                    # Convert similarity score to rating prediction
                    sim_score = user_movie_similarities[user_id][movie_id]
                    predicted_rating = 0.5 + 4.5 * sim_score
                else:
                    # Use user's average rating as fallback
                    predicted_rating = user_avg_rating
                
                # Calculate error
                squared_error = (predicted_rating - true_rating) ** 2
                absolute_error = abs(predicted_rating - true_rating)
                
                batch_squared_errors += squared_error
                batch_absolute_errors += absolute_error
                batch_predictions += 1
            
            users_evaluated += 1
        
        # Accumulate batch metrics
        squared_errors_sum += batch_squared_errors
        absolute_errors_sum += batch_absolute_errors
        total_predictions += batch_predictions
        
        # Log progress
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = batch_end / len(user_list) * 100
        remaining = batch_time * ((len(user_list) - batch_end) / len(batch_users)) if batch_end < len(user_list) else 0
        
        # Periodically calculate and log intermediate metrics
        if batch_predictions > 0:
            batch_rmse = np.sqrt(batch_squared_errors / batch_predictions)
            batch_mae = batch_absolute_errors / batch_predictions
            print(f"Batch metrics - RMSE: {batch_rmse:.4f}, MAE: {batch_mae:.4f}, Predictions: {batch_predictions}")
        
        print(f"Processed {batch_end}/{len(user_list)} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Calculate overall RMSE and MAE
    if total_predictions > 0:
        overall_rmse = np.sqrt(squared_errors_sum / total_predictions)
        overall_mae = absolute_errors_sum / total_predictions
    else:
        overall_rmse = 0.0
        overall_mae = 0.0
    
    print("\nEvaluation results:")
    print(f"Users evaluated: {users_evaluated}")
    print(f"Total predictions: {total_predictions}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    
    # Create metrics dictionary
    metrics = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'num_users_evaluated': users_evaluated,
        'num_predictions': total_predictions
    }
    
    return metrics
# Evaluate recommendations if test ratings are available
if 'user_movie_similarities' in data and 'train_ratings' in data and 'test_ratings' in data:
    # Call the new evaluation function
    print("Running evaluation with RMSE and MAE metrics...")
    evaluation_metrics = evaluate_with_rmse_mae(
        data['user_movie_similarities'],
        data['train_ratings'],
        data['test_ratings'],
        batch_size=100
    )
    
    # Store the metrics in the data dictionary
    data['evaluation_metrics'] = evaluation_metrics
    
    # Print the metrics to confirm they're stored correctly
    print("\nStored evaluation metrics:")
    for key, value in evaluation_metrics.items():
        print(f"  {key}: {value}")
    
    # Save metrics to CSV
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'content_based_evaluation.csv'), index=False)
    
    # Also save user metrics
    if 'user_metrics' in data:
        user_metrics_df = pd.DataFrame.from_dict(data['user_metrics'], orient='index')
        user_metrics_df.reset_index(inplace=True)
        user_metrics_df.rename(columns={'index': 'userId'}, inplace=True)
        user_metrics_df.to_csv(os.path.join(output_path, 'user_metrics.csv'), index=False)
    
    print(f"Saved evaluation metrics to CSV files")
    
    # Free memory
    gc.collect()

print("\n" + "="*80)
print("SUMMARY OF CONTENT-BASED RECOMMENDATION SYSTEM")
print("="*80)

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

# Safely display evaluation metrics without assuming specific keys
print("\nPerformance Metrics:")
if 'evaluation_metrics' in data:
    # Safely check for each expected metric
    if 'rmse' in data['evaluation_metrics']:
        print(f"- RMSE: {data['evaluation_metrics']['rmse']:.4f}")
    if 'mae' in data['evaluation_metrics']:
        print(f"- MAE: {data['evaluation_metrics']['mae']:.4f}")
    if 'num_users_evaluated' in data['evaluation_metrics']:
        print(f"- Users evaluated: {data['evaluation_metrics']['num_users_evaluated']}")
    if 'num_predictions' in data['evaluation_metrics']:
        print(f"- Total predictions: {data['evaluation_metrics']['num_predictions']}")
else:
    print("- No evaluation metrics available")

# Model advantages
print("\nAdvantages of this approach:")
print("- Log-Likelihood identifies more meaningful words compared to TF-IDF")
print("- Word2Vec captures semantic relationships between words")
print("- Handles new movies effectively (cold start for items)")
print("- Generates personalized recommendations based on content preferences")
print("- Doesn't require item-item similarity calculations")
print("- Memory-optimized batch processing prevents RAM overflow during long runs")

# Saved files
print("\nSaved Files:")
for file in os.listdir(output_path):
    file_path = os.path.join(output_path, file)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"- {file} ({file_size:.2f} MB)")

# Memory usage information
import psutil 
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
print(f"\nFinal memory usage: {memory_usage:.2f} MB")

print("\nContent-Based Filtering Model Successfully Implemented!")