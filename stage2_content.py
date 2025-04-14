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

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("CONTENT-BASED MOVIE RECOMMENDATION SYSTEM WITH LOG-LIKELIHOOD AND WORD2VEC")
print("="*80)

# Set paths
input_path = "./"  # Current directory where stage1.py saved the files
output_path = "./content-recommendations"
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
        data['ratings'] = pd.read_csv(ratings_path)
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
    for tokens in data['movie_features']['tokens']:
        corpus_word_counts.update(tokens)
    
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

print("\n" + "="*80)
print("STEP 3: LOG-LIKELIHOOD CALCULATION")
print("="*80)

def calculate_log_likelihood(movie_features, corpus_word_counts):
    """Calculate Log-Likelihood values for words in each movie"""
    print("Calculating Log-Likelihood values for all movies...")
    start_time = time.time()
    
    # Calculate total corpus size
    total_corpus_size = sum(corpus_word_counts.values())
    print(f"Total corpus size: {total_corpus_size} words")
    
    # Initialize container for movie features
    movie_ll_values = {}
    
    # Process each movie document
    total_movies = len(movie_features)
    for i, (_, row) in enumerate(movie_features.iterrows()):
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
        
        # Log progress
        if (i+1) % 1000 == 0 or (i+1) == total_movies:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_movies} movies ({(i+1)/total_movies*100:.1f}%) - Elapsed: {elapsed:.2f}s")
    
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
# def calculate_log_likelihood(movie_features, corpus_word_counts):
    """Enhanced calculation of Log-Likelihood values for words in each movie with smoothing"""
    print("Calculating enhanced Log-Likelihood values for all movies...")
    start_time = time.time()
    
    # Calculate total corpus size
    total_corpus_size = sum(corpus_word_counts.values())
    print(f"Total corpus size: {total_corpus_size} words")
    
    # Initialize container for movie features
    movie_ll_values = {}
    
    # Smoothing parameter to handle zero frequencies
    alpha = 0.01
    
    # Process each movie document
    total_movies = len(movie_features)
    for i, (_, row) in enumerate(movie_features.iterrows()):
        movie_id = row['movieId']
        tokens = row['tokens']
        
        if not tokens:
            continue
        
        # Count word occurrences in this movie
        movie_word_counts = Counter(tokens)
        movie_size = sum(movie_word_counts.values())
        
        # Calculate Log-Likelihood for each word
        movie_ll_values[movie_id] = {}
        
        # Track max LL value for normalization
        max_ll = 0
        
        for word, count in movie_word_counts.items():
            try:
                # Observed frequencies with smoothing
                a = count + alpha  # Occurrences in this movie
                b = (corpus_word_counts[word] - count) + alpha  # Occurrences in other movies
                c = movie_size + (alpha * len(movie_word_counts))  # Total words in this movie
                d = (total_corpus_size - movie_size) + (alpha * len(corpus_word_counts))  # Total words in other movies
                
                # Expected counts based on corpus distribution
                e1 = c * (a + b) / (c + d)
                e2 = d * (a + b) / (c + d)
                
                # Log-Likelihood calculation
                ll_parts = []
                if a > 0 and e1 > 0:
                    ll_parts.append(a * math.log(a / e1))
                if b > 0 and e2 > 0:
                    ll_parts.append(b * math.log(b / e2))
                
                ll = 2 * sum(ll_parts)
                
                # Apply importance scaling: penalize extremely common words
                word_rarity = 1.0 - (corpus_word_counts[word] / total_corpus_size)
                ll = ll * (0.5 + 0.5 * word_rarity)
                
                movie_ll_values[movie_id][word] = ll
                max_ll = max(max_ll, ll)
            except Exception as e:
                # Skip words with calculation errors
                print(f"Error calculating LL for word '{word}' in movie {movie_id}: {e}")
                continue
        
        # Normalize LL values by the maximum value to ensure consistency across movies
        if max_ll > 0:
            for word in movie_ll_values[movie_id]:
                movie_ll_values[movie_id][word] = movie_ll_values[movie_id][word] / max_ll
        
        # Log progress
        if (i+1) % 1000 == 0 or (i+1) == total_movies:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_movies} movies ({(i+1)/total_movies*100:.1f}%) - Elapsed: {elapsed:.2f}s")
    
    return movie_ll_values
# Calculate Log-Likelihood if movie features are available
if 'movie_features' in data and 'corpus_word_counts' in data:
    movie_ll_values = calculate_log_likelihood(data['movie_features'], data['corpus_word_counts'])
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

print("\n" + "="*80)
print("STEP 4: WORD2VEC MODEL TRAINING")
print("="*80)

def train_word2vec(movie_features, vector_size=100):
    """Train Word2Vec model on movie tokens"""
    print(f"Training Word2Vec model with {vector_size} dimensions...")
    start_time = time.time()
    
    # Extract token lists from movie features
    tokenized_corpus = list(movie_features['tokens'])
    
    # Print corpus statistics
    total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
    print(f"Training corpus size: {total_tokens} tokens from {len(tokenized_corpus)} documents")
    
    # Train Word2Vec model using CBOW approach
    print("Starting Word2Vec training (this may take a few minutes)...")
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        epochs=50,
        sg=0  # CBOW model
    )
    
    elapsed = time.time() - start_time
    print(f"Word2Vec training completed in {elapsed:.2f} seconds")
    
    # Print model statistics
    vocab_size = len(word2vec_model.wv)
    print(f"Word2Vec model vocabulary size: {vocab_size} words")
    print(f"Words not included in model due to min_count filter: {len(corpus_word_counts) - vocab_size}")
    
    # Show some example vectors for common words
    print("\nExample word vectors from the trained model:")
    common_words = [word for word, _ in corpus_word_counts.most_common(10)]
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

print("\n" + "="*80)
print("STEP 5: MOVIE VECTOR GENERATION")
print("="*80)

def generate_movie_vectors(movie_ll_values, word2vec_model, movie_features):
    """Generate movie feature vectors using Log-Likelihood and Word2Vec"""
    print("Generating movie feature vectors using Log-Likelihood + Word2Vec...")
    start_time = time.time()
    
    movie_vectors = {}
    successful_vectors = 0
    no_words_found = 0
    low_ll_sum = 0
    
    total_movies = len(movie_ll_values)
    print(f"Processing {total_movies} movies...")
    
    for i, (movie_id, ll_values) in enumerate(movie_ll_values.items()):
        # Sort words by LL value and select top 200
        top_words = sorted(ll_values.items(), key=lambda x: x[1], reverse=True)[:200]
        
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
        
        # Log progress
        if (i+1) % 1000 == 0 or (i+1) == total_movies:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_movies} movies ({(i+1)/total_movies*100:.1f}%) - Elapsed: {elapsed:.2f}s")
            print(f"Successfully created vectors: {successful_vectors}")
    
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
        data['movie_features']
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

print("\n" + "="*80)
print("STEP 6: USER VECTOR GENERATION")
print("="*80)

def generate_user_vectors(movie_vectors, train_ratings):
    """Generate user feature vectors based on rated movies and their content"""
    print("Generating user feature vectors based on movie ratings...")
    start_time = time.time()
    
    user_vectors = {}
    successful_vectors = 0
    no_ratings_found = 0
    no_vectors_for_movies = 0
    low_weight_sum = 0
    
    # Process each user
    user_ids = train_ratings['userId'].unique()
    total_users = len(user_ids)
    print(f"Processing {total_users} users...")
    
    for i, user_id in enumerate(user_ids):
        # Get user ratings
        user_data = train_ratings[train_ratings['userId'] == user_id]
        
        if len(user_data) == 0:
            no_ratings_found += 1
            continue
        
        weighted_vectors = []
        weight_sum = 0
        movies_with_vectors = 0
        movies_without_vectors = 0
        
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
        if (i+1) % 1000 == 0 or (i+1) == total_users:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%) - Elapsed: {elapsed:.2f}s")
            print(f"Successfully created vectors: {successful_vectors}")
    
    print(f"\nUser vector generation complete:")
    print(f"Successfully created vectors: {successful_vectors}/{total_users} ({successful_vectors/total_users*100:.1f}%)")
    print(f"Users with no ratings: {no_ratings_found}")
    print(f"Users with no vectorized movies: {no_vectors_for_movies}")
    print(f"Users with too low weight sum: {low_weight_sum}")
    
    # Display sample user vectors
    if user_vectors:
        print("\nSample user vectors:")
        for user_id in list(user_vectors.keys())[:3]:
            vector = user_vectors[user_id]
            user_data = train_ratings[train_ratings['userId'] == user_id]
            print(f"User ID: {user_id}")
            print(f"Number of ratings: {len(user_data)}")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector norm: {np.linalg.norm(vector):.4f}")
            print(f"First 5 dimensions: {vector[:5]}")
            print("---")
    
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
    
    print(f"Generated and saved feature vectors for {len(user_vectors)} users")
    
    # Calculate and display vector statistics
    vector_norms = [np.linalg.norm(v) for v in user_vectors.values()]
    print(f"\nVector statistics:")
    print(f"Average vector norm: {np.mean(vector_norms):.4f}")
    print(f"Vector dimensionality: {word2vec_dim}")

print("\n" + "="*80)
print("STEP 7: USER-MOVIE SIMILARITY CALCULATION")
print("="*80)

def calculate_user_movie_similarity(user_vectors, movie_vectors, threshold=0.3):
    """Calculate similarity between users and movies"""
    print(f"Calculating user-movie similarity with threshold {threshold}...")
    start_time = time.time()
    
    # Store similarities in a dictionary of dictionaries
    # {user_id: {movie_id: similarity_score}}
    user_movie_similarities = {}
    
    # Calculate similarity for each user
    total_users = len(user_vectors)
    total_similarities = 0
    similarities_above_threshold = 0
    
    for i, (user_id, user_vector) in enumerate(user_vectors.items()):
        user_sims = {}
        user_similarities = 0
        user_above_threshold = 0
        
        for movie_id, movie_vector in movie_vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(user_vector, movie_vector)
            user_similarities += 1
            
            # Only store if above threshold
            if similarity > threshold:
                user_sims[movie_id] = similarity
                user_above_threshold += 1
        
        user_movie_similarities[user_id] = user_sims
        total_similarities += user_similarities
        similarities_above_threshold += user_above_threshold
        
        # Log progress
        if (i+1) % 100 == 0 or (i+1) == total_users:
            elapsed = time.time() - start_time
            remaining = (elapsed / (i+1)) * (total_users - (i+1)) if i < total_users - 1 else 0
            print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
            print(f"User {user_id}: {user_above_threshold}/{len(movie_vectors)} movies above threshold ({user_above_threshold/len(movie_vectors)*100:.2f}%)")
    
    avg_above_threshold = similarities_above_threshold / total_users if total_users > 0 else 0
    threshold_percentage = similarities_above_threshold / total_similarities * 100 if total_similarities > 0 else 0
    
    print(f"\nSimilarity calculation complete:")
    print(f"Total users processed: {total_users}")
    print(f"Total movies per user: {len(movie_vectors)}")
    print(f"Total similarity calculations: {total_similarities}")
    print(f"Similarities above threshold: {similarities_above_threshold} ({threshold_percentage:.2f}%)")
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
                    movie_title = data['movie_features'][data['movie_features']['movieId'] == movie_id]['title'].values[0]
                    print(f"  '{movie_title}' (ID: {movie_id}): {sim:.4f}")
            print("---")
    
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
    
    print(f"Calculated and saved similarities for {len(user_movie_similarities)} users")
    
    # Calculate and display similarity statistics
    similarity_counts = [len(sims) for sims in user_movie_similarities.values()]
    print(f"\nSimilarity statistics:")
    print(f"Average number of similar movies per user: {np.mean(similarity_counts):.2f}")
    print(f"Min: {min(similarity_counts)}, Max: {max(similarity_counts)}")

print("\n" + "="*80)
print("STEP 8: RECOMMENDATION GENERATION")
print("="*80)

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

def generate_recommendations_for_all_users(user_movie_similarities, train_ratings, movie_features, n=10):
    """Generate recommendations for all users"""
    print(f"Generating top-{n} recommendations for all users...")
    start_time = time.time()
    
    # Get all user IDs
    user_ids = list(user_movie_similarities.keys())
    total_users = len(user_ids)
    
    all_recommendations = {}
    users_with_recommendations = 0
    total_recommendations = 0
    
    print(f"Processing {total_users} users...")
    
    for i, user_id in enumerate(user_ids):
        recommendations = get_top_n_recommendations(user_id, user_movie_similarities, train_ratings, n)
        
        if recommendations:
            all_recommendations[user_id] = recommendations
            users_with_recommendations += 1
            total_recommendations += len(recommendations)
        
        # Log progress
        if (i+1) % 1000 == 0 or (i+1) == total_users:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_users} users ({(i+1)/total_users*100:.1f}%) - Elapsed: {elapsed:.2f}s")
            print(f"Users with recommendations: {users_with_recommendations}")
    
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
        print(f"Saved recommendations to CSV file with {len(recommendations_df)} entries")

# This section replaces the evaluation section in stage2_content.py

print("\n" + "="*80)
print("STEP 9: MODEL EVALUATION")
print("="*80)

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

def evaluate_with_rmse_mae(user_movie_similarities, train_ratings, test_ratings):
    """
    Evaluate the recommendations using RMSE and MAE
    
    Parameters:
    -----------
    user_movie_similarities : dict
        Dictionary of user-movie similarities
    train_ratings : pd.DataFrame
        DataFrame of training ratings
    test_ratings : pd.DataFrame
        DataFrame of test ratings
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    dict
        Dictionary of per-user metrics
    """
    print("Evaluating recommendation model using RMSE and MAE...")
    start_time = time.time()
    
    # Initialize metrics
    all_predictions = []
    all_true_ratings = []
    
    # Get all users in test set
    test_users = test_ratings['userId'].unique()
    total_test_users = len(test_users)
    
    print(f"Evaluating predictions for {total_test_users} users in the test set...")
    
    # Users with similarity data
    users_with_similarity = set(user_movie_similarities.keys())
    users_in_test_with_similarity = set(test_users).intersection(users_with_similarity)
    
    print(f"Users in test set with similarity data: {len(users_in_test_with_similarity)}/{total_test_users} ({len(users_in_test_with_similarity)/total_test_users*100:.1f}%)")
    
    # Track individual user metrics
    user_metrics = {}
    users_evaluated = 0
    
    for i, user_id in enumerate(test_users):
        # Skip users without similarity data
        if user_id not in user_movie_similarities:
            continue
        
        # Get user test ratings
        user_test = test_ratings[test_ratings['userId'] == user_id]
        
        if len(user_test) == 0:
            continue
        
        # Predict ratings for test items
        user_predictions = []
        user_true_ratings = []
        
        for _, row in user_test.iterrows():
            movie_id = row['movieId']
            true_rating = row['rating']
            
            # Use the predict_rating function
            predicted_rating = predict_rating(user_id, movie_id, user_movie_similarities, train_ratings)
            
            user_predictions.append(predicted_rating)
            user_true_ratings.append(true_rating)
        
        # Calculate user RMSE and MAE
        user_rmse = np.sqrt(np.mean(np.square(np.array(user_predictions) - np.array(user_true_ratings))))
        user_mae = np.mean(np.abs(np.array(user_predictions) - np.array(user_true_ratings)))
        
        # Store individual user metrics
        user_metrics[user_id] = {
            'rmse': user_rmse,
            'mae': user_mae,
            'num_predictions': len(user_predictions)
        }
        
        # Accumulate all predictions for overall metrics
        all_predictions.extend(user_predictions)
        all_true_ratings.extend(user_true_ratings)
        users_evaluated += 1
        
        # Log progress
        if (i+1) % 100 == 0 or (i+1) == total_test_users:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_test_users} users ({(i+1)/total_test_users*100:.1f}%) - Elapsed: {elapsed:.2f}s")
    
    # Calculate overall RMSE and MAE
    if len(all_predictions) > 0:
        overall_rmse = np.sqrt(np.mean(np.square(np.array(all_predictions) - np.array(all_true_ratings))))
        overall_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_true_ratings)))
    else:
        overall_rmse = 0.0
        overall_mae = 0.0
    
    print("\nEvaluation results:")
    print(f"Users evaluated: {users_evaluated}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    
    # Analyze metrics by user rating count
    if 'train_ratings' in data:
        print("\nMetrics by user rating count:")
        
        # Count ratings for each user
        user_rating_counts = train_ratings.groupby('userId').size()
        
        # Define groups
        groups = [
            (0, 10, "0-10 ratings"),
            (10, 50, "10-50 ratings"),
            (50, 100, "50-100 ratings"),
            (100, float('inf'), "100+ ratings")
        ]
        
        for min_count, max_count, label in groups:
            group_users = [
                user_id for user_id in user_metrics
                if min_count <= user_rating_counts.get(user_id, 0) < max_count
            ]
            
            if group_users:
                group_rmse = np.mean([user_metrics[user_id]['rmse'] for user_id in group_users])
                group_mae = np.mean([user_metrics[user_id]['mae'] for user_id in group_users])
                group_predictions = sum(user_metrics[user_id]['num_predictions'] for user_id in group_users)
                
                print(f"{label}: {len(group_users)} users, {group_predictions} predictions, RMSE={group_rmse:.4f}, MAE={group_mae:.4f}")
    
    # Create metrics dictionary with RMSE and MAE (no HR or ARHR)
    metrics = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'num_users_evaluated': users_evaluated,
        'num_predictions': len(all_predictions)
    }
    
    return metrics, user_metrics

# Evaluate recommendations if test ratings are available
if 'user_movie_similarities' in data and 'train_ratings' in data and 'test_ratings' in data:
    # Call the new evaluation function
    print("Running evaluation with RMSE and MAE metrics...")
    evaluation_metrics, user_metrics = evaluate_with_rmse_mae(
        data['user_movie_similarities'],
        data['train_ratings'],
        data['test_ratings']
    )
    
    # Store the metrics in the data dictionary
    data['evaluation_metrics'] = evaluation_metrics
    data['user_metrics'] = user_metrics
    
    # Print the metrics to confirm they're stored correctly
    print("\nStored evaluation metrics:")
    for key, value in evaluation_metrics.items():
        print(f"  {key}: {value}")
    
    # Save metrics to CSV
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'content_based_evaluation.csv'), index=False)
    
    # Also save user metrics
    user_metrics_df = pd.DataFrame.from_dict(user_metrics, orient='index')
    user_metrics_df.reset_index(inplace=True)
    user_metrics_df.rename(columns={'index': 'userId'}, inplace=True)
    user_metrics_df.to_csv(os.path.join(output_path, 'user_metrics.csv'), index=False)
    
    print(f"Saved evaluation metrics to CSV files")

# This section replaces the summary section at the end of stage2_content.py

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

# Saved files
print("\nSaved Files:")
for file in os.listdir(output_path):
    file_path = os.path.join(output_path, file)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"- {file} ({file_size:.2f} MB)")

print("\nContent-Based Filtering Model Successfully Implemented!")