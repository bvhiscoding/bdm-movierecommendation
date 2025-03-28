# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from collections import Counter
import seaborn as sns
import os
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# GPU acceleration libraries
import torch
import cupy as cp  # CUDA-accelerated NumPy alternative
from cuml import preprocessing  # GPU-accelerated ML preprocessing
from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cudf import DataFrame as cuDataFrame  # GPU-accelerated pandas
from cuml.metrics.pairwise_distance import pairwise_distances

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Set device to GPU
    device = torch.device("cuda:0")
    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Total Memory: {total_memory:.2f} GB")
else:
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

print("Libraries imported successfully!")

# Step 1: Movie Text Feature Extraction with GPU Acceleration
print("\n" + "="*80)
print("STEP 1: MOVIE TEXT FEATURE EXTRACTION (GPU-Accelerated)")
print("="*80)

start_time = time.time()

# Helper function to load data with timing
def load_data_with_timing(file_path, message):
    start = time.time()
    data = pd.read_csv(file_path)
    end = time.time()
    print(f"{message}: {len(data)} records in {end-start:.2f} seconds")
    return data

# Load MovieLens movie data
movies_df = load_data_with_timing('data/20M/demo_movie.csv', "Loaded movies")

# Load TMDB data
tmdb_df = load_data_with_timing('data/20M/demo_tmdb.csv', "Loaded TMDB data")

# Load links data
links_df = load_data_with_timing('data/20M/demo_link.csv', "Loaded movie links")

# Load tags data
tags_df = load_data_with_timing('data/20M/demo_tag.csv', "Loaded movie tags")

# Transfer to GPU memory for faster merging operations if CUDA is available
if CUDA_AVAILABLE:
    try:
        # Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
        cu_movies_df = cuDataFrame.from_pandas(movies_df)
        cu_links_df = cuDataFrame.from_pandas(links_df)
        cu_tmdb_df = cuDataFrame.from_pandas(tmdb_df)
        cu_tags_df = cuDataFrame.from_pandas(tags_df)
        
        # Merge DataFrames on GPU
        merge_start = time.time()
        cu_movie_data = cu_movies_df.merge(cu_links_df, on='movieId', how='left')
        cu_movie_data = cu_movie_data.merge(cu_tmdb_df, left_on='tmdbId', right_on='id', how='left')
        
        # Aggregate tags on GPU
        cu_tags_by_movie = cu_tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.fillna('')))
        cu_tags_by_movie = cu_tags_by_movie.reset_index()
        
        # Merge tags
        cu_movie_data = cu_movie_data.merge(cu_tags_by_movie, on='movieId', how='left')
        
        # Convert back to pandas for further processing
        movie_data = cu_movie_data.to_pandas()
        merge_end = time.time()
        print(f"GPU-accelerated merging completed in {merge_end-merge_start:.2f} seconds")
    except Exception as e:
        print(f"GPU acceleration failed: {e}. Falling back to CPU.")
        # Fallback to CPU processing
        movie_data = pd.merge(movies_df, links_df, on='movieId', how='left')
        movie_data = pd.merge(movie_data, tmdb_df, left_on='tmdbId', right_on='id', how='left')
        tags_by_movie = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.fillna(''))).reset_index()
        movie_data = pd.merge(movie_data, tags_by_movie, on='movieId', how='left')
else:
    # CPU processing
    merge_start = time.time()
    movie_data = pd.merge(movies_df, links_df, on='movieId', how='left')
    movie_data = pd.merge(movie_data, tmdb_df, left_on='tmdbId', right_on='id', how='left')
    tags_by_movie = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.fillna(''))).reset_index()
    movie_data = pd.merge(movie_data, tags_by_movie, on='movieId', how='left')
    merge_end = time.time()
    print(f"CPU merging completed in {merge_end-merge_start:.2f} seconds")

# Create text corpus for each movie
movie_data['text_corpus'] = ""
movie_data['text_corpus'] += movie_data['title'].fillna("")
movie_data['text_corpus'] += " " + movie_data['overview'].fillna("")
movie_data['text_corpus'] += " " + movie_data['cast'].fillna("")
movie_data['text_corpus'] += " " + movie_data['director'].fillna("")
movie_data['text_corpus'] += " " + movie_data['tag'].fillna("")

end_time = time.time()
print(f"Total time for Step 1: {end_time-start_time:.2f} seconds")

# Step 2: Text Preprocessing with GPU Acceleration
print("\n" + "="*80)
print("STEP 2: TEXT PREPROCESSING (GPU-Accelerated)")
print("="*80)

start_time = time.time()

# Define helper functions for text preprocessing
def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag"""
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def preprocess_text(text, stop_words, lemmatizer):
    """Tokenize, remove stopwords, and lemmatize text"""
    if not isinstance(text, str) or text == "":
        return []
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize text
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    try:
        # Try lemmatizing tokens with POS tagging
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                             for word, tag in tagged_tokens]
    except LookupError:
        # Fallback to simple lemmatization without POS tagging
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized_tokens

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean all text first
print("Cleaning text corpus...")
movie_data['cleaned_text'] = movie_data['text_corpus'].apply(clean_text)

# Parallel text processing using ThreadPoolExecutor
print("Tokenizing and lemmatizing text corpus in parallel...")
preprocess_func = partial(preprocess_text, stop_words=stop_words, lemmatizer=lemmatizer)

# Process in batches to avoid excessive memory usage
num_cores = os.cpu_count()
batch_size = min(1000, len(movie_data))
all_tokens = []

with ThreadPoolExecutor(max_workers=num_cores) as executor:
    for i in range(0, len(movie_data), batch_size):
        batch = movie_data['cleaned_text'].iloc[i:i+batch_size]
        batch_tokens = list(executor.map(preprocess_func, batch))
        all_tokens.extend(batch_tokens)
        print(f"Processed batch {i//batch_size + 1}/{(len(movie_data)-1)//batch_size + 1}")

movie_data['tokens'] = all_tokens

# Count corpus words using GPU if available
if CUDA_AVAILABLE:
    try:
        # Use CUDA-accelerated CountVectorizer for word frequency analysis
        print("Calculating word frequencies using GPU...")
        count_start = time.time()
        
        # Create a single document for CountVectorizer
        all_text = " ".join([" ".join(tokens) for tokens in movie_data['tokens']])
        
        # Use GPU-accelerated CountVectorizer
        vectorizer = CountVectorizer(lowercase=False)  # Already lowercase from cleaning
        X = vectorizer.fit_transform([all_text])
        
        # Get vocabulary and counts
        vocab = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        
        # Create word count dictionary
        corpus_word_counts = {word: count for word, count in zip(vocab, counts)}
        
        count_end = time.time()
        print(f"GPU word frequency calculation completed in {count_end-count_start:.2f} seconds")
    except Exception as e:
        print(f"GPU word frequency calculation failed: {e}. Falling back to CPU.")
        # Fallback to CPU
        all_words = []
        for tokens in movie_data['tokens']:
            all_words.extend(tokens)
        corpus_word_counts = Counter(all_words)
else:
    # CPU processing
    count_start = time.time()
    all_words = []
    for tokens in movie_data['tokens']:
        all_words.extend(tokens)
    corpus_word_counts = Counter(all_words)
    count_end = time.time()
    print(f"CPU word frequency calculation completed in {count_end-count_start:.2f} seconds")

# Calculate document frequency
doc_freq = {}
for tokens in movie_data['tokens']:
    for word in set(tokens):  # Count each word only once per document
        doc_freq[word] = doc_freq.get(word, 0) + 1

end_time = time.time()
print(f"Total time for Step 2: {end_time-start_time:.2f} seconds")

# Step 3: Data Normalization with GPU Acceleration
print("\n" + "="*80)
print("STEP 3: DATA NORMALIZATION (GPU-Accelerated)")
print("="*80)

start_time = time.time()

# Load user ratings data
ratings_df = load_data_with_timing('data/20M/demo_rating.csv', "Loaded ratings")

# Calculate rating statistics by user
if CUDA_AVAILABLE:
    try:
        # Move to GPU
        cu_ratings_df = cuDataFrame.from_pandas(ratings_df)
        
        # Calculate statistics on GPU
        stats_start = time.time()
        cu_user_stats = cu_ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        
        # Rename columns
        cu_user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
        
        # Fill NA values in std with 0 (for users with only one rating)
        cu_user_stats['rating_std'] = cu_user_stats['rating_std'].fillna(0)
        
        # Convert back to pandas
        user_stats = cu_user_stats.to_pandas()
        stats_end = time.time()
        print(f"GPU statistics calculation completed in {stats_end-stats_start:.2f} seconds")
    except Exception as e:
        print(f"GPU statistics calculation failed: {e}. Falling back to CPU.")
        # CPU fallback
        user_stats = ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
else:
    # CPU processing
    stats_start = time.time()
    user_stats = ratings_df.groupby('userId').agg({
        'rating': ['count', 'mean', 'std']
    }).reset_index()
    user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
    user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
    stats_end = time.time()
    print(f"CPU statistics calculation completed in {stats_end-stats_start:.2f} seconds")

# Normalize ratings using Z-score and min-max scaling with GPU acceleration if available
def normalize_ratings_cpu(ratings, user_stats):
    normalized_ratings = ratings.copy()
    normalized_ratings['normalized_rating'] = 0.0  # Initialize
    
    for user_id in normalized_ratings['userId'].unique():
        # Get user's data
        user_indices = normalized_ratings['userId'] == user_id
        user_data = user_stats[user_stats['userId'] == user_id]
        
        if user_data.empty:
            # Skip if user not found in stats
            normalized_ratings.loc[user_indices, 'normalized_rating'] = 0.5  # Default mid-value
            continue
            
        mean_rating = user_data.iloc[0]['rating_mean']
        std_rating = user_data.iloc[0]['rating_std']
        
        if std_rating > 0:
            # Z-score normalization followed by scaling to [0,1]
            z_scores = (normalized_ratings.loc[user_indices, 'rating'] - mean_rating) / std_rating
            normalized_ratings.loc[user_indices, 'normalized_rating'] = (z_scores + 3) / 6
            normalized_ratings.loc[user_indices, 'normalized_rating'] = normalized_ratings.loc[user_indices, 'normalized_rating'].clip(0, 1)
        else:
            # Min-max scaling if std is 0
            normalized_ratings.loc[user_indices, 'normalized_rating'] = (normalized_ratings.loc[user_indices, 'rating'] - 0.5) / 4.5
    
    return normalized_ratings

def normalize_ratings_gpu(ratings, user_stats):
    # Transfer to GPU
    cu_ratings = cuDataFrame.from_pandas(ratings)
    cu_stats = cuDataFrame.from_pandas(user_stats)
    
    # Initialize normalized ratings
    cu_ratings['normalized_rating'] = 0.0
    
    # Process each user
    for user_id in cu_ratings['userId'].unique().to_array():
        # Get user's data
        user_indices = cu_ratings['userId'] == user_id
        user_data = cu_stats[cu_stats['userId'] == user_id]
        
        if len(user_data) == 0:
            # Skip if user not found in stats
            cu_ratings.loc[user_indices, 'normalized_rating'] = 0.5  # Default mid-value
            continue
            
        mean_rating = user_data['rating_mean'].iloc[0]
        std_rating = user_data['rating_std'].iloc[0]
        
        if std_rating > 0:
            # Z-score normalization followed by scaling to [0,1]
            z_scores = (cu_ratings.loc[user_indices, 'rating'] - mean_rating) / std_rating
            cu_ratings.loc[user_indices, 'normalized_rating'] = (z_scores + 3) / 6
            
            # Clip values to [0,1]
            mask_below = cu_ratings.loc[user_indices, 'normalized_rating'] < 0
            mask_above = cu_ratings.loc[user_indices, 'normalized_rating'] > 1
            
            if mask_below.any():
                cu_ratings.loc[user_indices & mask_below, 'normalized_rating'] = 0
            if mask_above.any():
                cu_ratings.loc[user_indices & mask_above, 'normalized_rating'] = 1
        else:
            # Min-max scaling if std is 0
            cu_ratings.loc[user_indices, 'normalized_rating'] = (cu_ratings.loc[user_indices, 'rating'] - 0.5) / 4.5
    
    # Transfer back to CPU
    return cu_ratings.to_pandas()

# Apply normalization with appropriate method
norm_start = time.time()
if CUDA_AVAILABLE:
    try:
        normalized_ratings = normalize_ratings_gpu(ratings_df, user_stats)
        print(f"GPU-accelerated normalization completed in {time.time()-norm_start:.2f} seconds")
    except Exception as e:
        print(f"GPU normalization failed: {e}. Falling back to CPU.")
        normalized_ratings = normalize_ratings_cpu(ratings_df, user_stats)
        print(f"CPU normalization completed in {time.time()-norm_start:.2f} seconds")
else:
    normalized_ratings = normalize_ratings_cpu(ratings_df, user_stats)
    print(f"CPU normalization completed in {time.time()-norm_start:.2f} seconds")

end_time = time.time()
print(f"Total time for Step 3: {end_time-start_time:.2f} seconds")

# Step 4: Genre Encoding with GPU Acceleration
print("\n" + "="*80)
print("STEP 4: GENRE ENCODING (GPU-Accelerated)")
print("="*80)

start_time = time.time()

# Count total unique genres
all_genres = set()
for genres in movies_df['genres'].str.split('|'):
    if isinstance(genres, list):
        all_genres.update(genres)

print(f"Found {len(all_genres)} unique genres: {sorted(all_genres)}")

# One-hot encode genres
if CUDA_AVAILABLE:
    try:
        # GPU accelerated one-hot encoding
        # First convert to appropriate format for GPU processing
        genre_data = []
        for _, movie in movies_df.iterrows():
            movie_id = movie['movieId']
            genres = movie['genres'].split('|') if isinstance(movie['genres'], str) else []
            
            for genre in genres:
                genre_data.append({'movieId': movie_id, 'genre': genre})
        
        # Convert to DataFrame and then to cuDF
        genre_df = pd.DataFrame(genre_data)
        cu_genre_df = cuDataFrame.from_pandas(genre_df)
        
        # Create GPU-accelerated one-hot encoding
        encode_start = time.time()
        cu_one_hot = cu_genre_df.pivot_table(
            index='movieId',
            columns='genre',
            values='genre',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        # Convert back to pandas
        genre_one_hot = cu_one_hot.to_pandas()
        
        # Flatten column names
        genre_one_hot.columns.name = None
        
        encode_end = time.time()
        print(f"GPU one-hot encoding completed in {encode_end-encode_start:.2f} seconds")
    except Exception as e:
        print(f"GPU one-hot encoding failed: {e}. Falling back to CPU.")
        # CPU fallback
        genre_data = []
        for _, movie in movies_df.iterrows():
            movie_id = movie['movieId']
            genres = movie['genres'].split('|') if isinstance(movie['genres'], str) else []
            
            for genre in genres:
                genre_data.append({'movieId': movie_id, 'genre': genre})
        
        genre_df = pd.DataFrame(genre_data)
        genre_one_hot = pd.pivot_table(
            genre_df, 
            index='movieId', 
            columns='genre', 
            aggfunc=lambda x: 1, 
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        genre_one_hot.columns.name = None
else:
    # CPU processing
    encode_start = time.time()
    genre_data = []
    for _, movie in movies_df.iterrows():
        movie_id = movie['movieId']
        genres = movie['genres'].split('|') if isinstance(movie['genres'], str) else []
        
        for genre in genres:
            genre_data.append({'movieId': movie_id, 'genre': genre})
    
    genre_df = pd.DataFrame(genre_data)
    genre_one_hot = pd.pivot_table(
        genre_df, 
        index='movieId', 
        columns='genre', 
        aggfunc=lambda x: 1, 
        fill_value=0
    ).reset_index()
    
    # Flatten column names
    genre_one_hot.columns.name = None
    encode_end = time.time()
    print(f"CPU one-hot encoding completed in {encode_end-encode_start:.2f} seconds")

# Merge with original movie data
merge_start = time.time()
if CUDA_AVAILABLE:
    try:
        # GPU-accelerated merge
        cu_movies_sample = cuDataFrame.from_pandas(movies_df[['movieId', 'title']])
        cu_genre_one_hot = cuDataFrame.from_pandas(genre_one_hot)
        
        cu_movie_genres = cu_movies_sample.merge(
            cu_genre_one_hot,
            on='movieId',
            how='left'
        )
        
        # Fill NaN values with 0
        for genre in all_genres:
            if genre in cu_movie_genres.columns:
                cu_movie_genres[genre] = cu_movie_genres[genre].fillna(0).astype('int32')
        
        # Convert back to pandas
        movie_genres = cu_movie_genres.to_pandas()
        merge_end = time.time()
        print(f"GPU genre merge completed in {merge_end-merge_start:.2f} seconds")
    except Exception as e:
        print(f"GPU genre merge failed: {e}. Falling back to CPU.")
        # CPU fallback
        movie_genres = pd.merge(
            movies_df[['movieId', 'title']], 
            genre_one_hot, 
            on='movieId', 
            how='left'
        )
        
        # Fill NaN values with 0
        for genre in all_genres:
            if genre in movie_genres.columns:
                movie_genres[genre] = movie_genres[genre].fillna(0).astype(int)
else:
    # CPU processing
    movie_genres = pd.merge(
        movies_df[['movieId', 'title']], 
        genre_one_hot, 
        on='movieId', 
        how='left'
    )
    
    # Fill NaN values with 0
    for genre in all_genres:
        if genre in movie_genres.columns:
            movie_genres[genre] = movie_genres[genre].fillna(0).astype(int)
    merge_end = time.time()
    print(f"CPU genre merge completed in {merge_end-merge_start:.2f} seconds")

end_time = time.time()
print(f"Total time for Step 4: {end_time-start_time:.2f} seconds")

# Final Output: Combined Movie Features
print("\n" + "="*80)
print("FINAL OUTPUT: COMBINED MOVIE FEATURES (GPU-Accelerated)")
print("="*80)

start_time = time.time()

# Define helper function to get top keywords
def get_top_keywords(tokens, n=5):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return []
    
    word_counts = Counter(tokens)
    return [word for word, _ in word_counts.most_common(n)]

# Combine all features into one DataFrame
if CUDA_AVAILABLE:
    try:
        # GPU-accelerated merge for final features
        cu_movie_genres = cuDataFrame.from_pandas(movie_genres)
        
        # We need to convert 'tokens' to a string representation for cuDF
        preprocessed_df = movie_data[['movieId', 'tokens']]
        preprocessed_df['tokens_str'] = preprocessed_df['tokens'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        cu_preprocessed = cuDataFrame.from_pandas(preprocessed_df[['movieId', 'tokens_str']])
        
        # Merge on GPU
        cu_movie_features = cu_movie_genres.merge(
            cu_preprocessed,
            on='movieId',
            how='left'
        )
        
        # Convert back to pandas for final processing
        movie_features_temp = cu_movie_features.to_pandas()
        
        # Convert token strings back to lists
        movie_features_temp['tokens'] = movie_features_temp['tokens_str'].apply(
            lambda x: x.split(',') if x else []
        )
        movie_features_temp.drop('tokens_str', axis=1, inplace=True)
        
        # Calculate token counts and keywords
        movie_features_temp['token_count'] = movie_features_temp['tokens'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        movie_features_temp['top_keywords'] = movie_features_temp['tokens'].apply(get_top_keywords)
        
        # Final movie features
        movie_features = movie_features_temp
        print(f"GPU-accelerated feature combination completed in {time.time()-start_time:.2f} seconds")
    except Exception as e:
        print(f"GPU feature combination failed: {e}. Falling back to CPU.")
        # CPU fallback
        movie_features = pd.merge(
            movie_genres,
            movie_data[['movieId', 'tokens']],
            on='movieId',
            how='left'
        )
        
        # Add token count and keywords
        movie_features['token_count'] = movie_features['tokens'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        movie_features['top_keywords'] = movie_features['tokens'].apply(get_top_keywords)
else:
    # CPU processing
    movie_features = pd.merge(
        movie_genres,
        movie_data[['movieId', 'tokens']],
        on='movieId',
        how='left'
    )
    
    # Add token count and keywords
    movie_features['token_count'] = movie_features['tokens'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    movie_features['top_keywords'] = movie_features['tokens'].apply(get_top_keywords)

# Create display version without tokens for better readability
display_features = movie_features.drop(columns=['tokens'])

# Save the processed data to disk
save_start = time.time()
movie_features.to_csv('processed_movie_features.csv', index=False)
normalized_ratings.to_csv('normalized_ratings.csv', index=False)
save_end = time.time()
print(f"Data saved to disk in {save_end-save_start:.2f} seconds")

end_time = time.time()
print(f"Total time for Final Output: {end_time-start_time:.2f} seconds")

# Summary and GPU statistics
print("\n" + "="*80)
print("SUMMARY OF GPU-ACCELERATED STAGE 1 DATA PROCESSING")
print("="*80)
print(f"1. Extracted text features for {len(movie_data)} movies")
print(f"2. Preprocessed text resulting in a vocabulary of {len(corpus_word_counts)} unique words")
print(f"3. Normalized {len(normalized_ratings)} ratings from {len(user_stats)} users")
print(f"4. Created one-hot encodings for {len(all_genres)} genres")
print(f"5. Final dataset contains {len(movie_features)} movies with complete feature sets")

if CUDA_AVAILABLE:
    # Get GPU memory usage
    allocated_memory = torch.cuda.memory_allocated() / 1e9
    reserved_memory = torch.cuda.memory_reserved() / 1e9
    print(f"\nGPU Memory Stats:")
    print(f"- Allocated: {allocated_memory:.2f} GB")
    print(f"- Reserved: {reserved_memory:.2f} GB")
    print(f"- Total Available: {total_memory:.2f} GB")
    print(f"- Utilization: {(allocated_memory/total_memory)*100:.2f}%")

print("="*80)