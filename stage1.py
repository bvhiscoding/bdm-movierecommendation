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

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("Libraries imported successfully!")

# Step 1: Movie Text Feature Extraction
print("\n" + "="*80)
print("STEP 1: MOVIE TEXT FEATURE EXTRACTION")
print("="*80)
# Load MovieLens movie data
movies_df = pd.read_csv('data/20M/movie.csv')
print(f"Loaded {len(movies_df)} movies from MovieLens dataset")
print(movies_df.head(3))

# Load TMDB data (containing movie overviews, cast, director)
tmdb_df = pd.read_csv('data/20M/tmdb.csv')
print(f"\nLoaded {len(tmdb_df)} movies from TMDB dataset")
print(tmdb_df.head(3)[['id', 'tmdb_title', 'overview']])

# Load links data to connect MovieLens IDs with TMDB IDs
links_df = pd.read_csv('data/20M/link.csv')
print(f"\nLoaded {len(links_df)} movie links")
print(links_df.head(3))

# Load tags data for additional text information
tags_df = pd.read_csv('data/20M/tag.csv')
print(f"\nLoaded {len(tags_df)} movie tags")
print(tags_df.head(3))
# Merge movie data with TMDB data via links_df
movie_data = pd.merge(movies_df, links_df, on='movieId', how='left')
movie_data = pd.merge(movie_data, tmdb_df, left_on='tmdbId', right_on='id', how='left')
# Create text corpus for each movie
movie_data['text_corpus'] = ""

# Add title to corpus
movie_data['text_corpus'] += movie_data['title'].fillna("")

# Add TMDB overview to corpus
movie_data['text_corpus'] += " " + movie_data['overview'].fillna("")

# Add TMDB cast to corpus
movie_data['text_corpus'] += " " + movie_data['cast'].fillna("")

# Add TMDB director to corpus
movie_data['text_corpus'] += " " + movie_data['director'].fillna("")

# Aggregate tags by movieId
tags_by_movie = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.fillna(''))).reset_index()

# Merge tags with movies
movie_data = pd.merge(movie_data, tags_by_movie, on='movieId', how='left')

# Add tags to corpus
movie_data['text_corpus'] += " " + movie_data['tag'].fillna("")

# Display a sample text corpus
print("\nSample movie text corpus:")
sample_movie = movie_data.iloc[0]
print(f"Movie: {sample_movie['title']}")
print(f"Text corpus: {sample_movie['text_corpus'][:300]}...")

# Output: Movie text corpus data
print(f"\nCreated text corpus for {len(movie_data)} movies")
movie_corpus_df = movie_data[['movieId', 'title', 'text_corpus']]
print(movie_corpus_df.head(3))

# Step 2: Text Preprocessing
print("\n" + "="*80)
print("STEP 2: TEXT PREPROCESSING")
print("="*80)

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

# Apply preprocessing to movie text corpus
print("Cleaning and tokenizing text corpus...")
movie_data['cleaned_text'] = movie_data['text_corpus'].apply(clean_text)
movie_data['tokens'] = movie_data['cleaned_text'].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))

# Display sample of preprocessed text
print("\nSample of preprocessed text:")
sample_idx = 0
print(f"Movie: {movie_data.iloc[sample_idx]['title']}")
print(f"Original text: {movie_data.iloc[sample_idx]['text_corpus'][:100]}...")
print(f"Cleaned text: {movie_data.iloc[sample_idx]['cleaned_text'][:100]}...")
print(f"Tokens: {movie_data.iloc[sample_idx]['tokens'][:20]}...")

# Count corpus words
all_words = []
for tokens in movie_data['tokens']:
    all_words.extend(tokens)

corpus_word_counts = Counter(all_words)
print(f"\nVocabulary size: {len(corpus_word_counts)} unique words")
print(f"Top 20 most common words: {corpus_word_counts.most_common(20)}")

# Calculate document frequency (number of documents containing each word)
doc_freq = {}
for tokens in movie_data['tokens']:
    for word in set(tokens):  # Count each word only once per document
        doc_freq[word] = doc_freq.get(word, 0) + 1

print(f"\nDocument frequency of top words:")
for word, count in sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"'{word}' appears in {count} documents")

# Output: Preprocessed text data
preprocessed_df = movie_data[['movieId', 'title', 'tokens']]
print("\nPreprocessed movie text data:")
print(preprocessed_df.head(3))

# Plot token length distribution
token_lengths = [len(tokens) for tokens in movie_data['tokens']]
plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Token Count per Movie')
plt.xlabel('Number of Tokens')
plt.ylabel('Number of Movies')
plt.grid(True, alpha=0.3)
plt.savefig('token_distribution.png')
print("\nToken distribution plot saved as 'token_distribution.png'")
plt.close()

# Step 3: Data Normalization
print("\n" + "="*80)
print("STEP 3: DATA NORMALIZATION")
print("="*80)

# Load user ratings data
ratings_df = pd.read_csv('data/20M/rating.csv')
print(f"Loaded {len(ratings_df)} ratings from {len(ratings_df['userId'].unique())} users")
print(ratings_df.head())

# Calculate rating statistics by user
user_stats = ratings_df.groupby('userId').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()
user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']

# Fill NA values in std with 0 (for users with only one rating)
user_stats['rating_std'] = user_stats['rating_std'].fillna(0)

print("\nUser rating statistics:")
print(user_stats.head())

# Plot user rating distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(user_stats['rating_mean'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mean Ratings per User')
plt.xlabel('Mean Rating')
plt.ylabel('Number of Users')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(user_stats['rating_std'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Rating Standard Deviation per User')
plt.xlabel('Rating Standard Deviation')
plt.ylabel('Number of Users')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(user_stats['rating_count'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Rating Count per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('user_rating_stats.png')
print("\nUser rating statistics plot saved as 'user_rating_stats.png'")
plt.close()

# Normalize ratings using z-score and min-max scaling
def normalize_ratings(ratings, user_stats):
    normalized_ratings = ratings.copy()
    
    for index, row in normalized_ratings.iterrows():
        user_id = row['userId']
        user_data = user_stats[user_stats['userId'] == user_id]
        
        if user_data.empty:
            # Skip if user not found in stats
            normalized_ratings.at[index, 'normalized_rating'] = 0.5  # Default mid-value
            continue
            
        user_data = user_data.iloc[0]
        mean_rating = user_data['rating_mean']
        std_rating = user_data['rating_std']
        
        if std_rating > 0:
            # Z-score normalization
            normalized_rating = (row['rating'] - mean_rating) / std_rating
            # Scale to [0, 1]
            normalized_rating = (normalized_rating + 3) / 6  # Assuming range is [-3, 3] after z-score
            normalized_rating = min(max(normalized_rating, 0), 1)  # Clip to [0, 1]
        else:
            # If std is 0, just use min-max scaling from [0.5, 5] to [0, 1]
            normalized_rating = (row['rating'] - 0.5) / 4.5
        
        normalized_ratings.at[index, 'normalized_rating'] = normalized_rating
    
    return normalized_ratings

# Apply normalization
print("\nNormalizing ratings...")
normalized_ratings = normalize_ratings(ratings_df, user_stats)

print("\nOriginal vs. Normalized ratings:")
print(normalized_ratings[['userId', 'movieId', 'rating', 'normalized_rating']].head(10))

# Plot original vs normalized ratings
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(ratings_df['rating'], bins=9, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Original Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(normalized_ratings['normalized_rating'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Distribution of Normalized Ratings')
plt.xlabel('Normalized Rating')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rating_normalization.png')
print("\nRating normalization plot saved as 'rating_normalization.png'")
plt.close()

# Output: Normalized ratings data
print("\nNormalized ratings data:")
print(normalized_ratings.head())

# Step 4: Genre Encoding (Binary Representation)
print("\n" + "="*80)
print("STEP 4: GENRE ENCODING")
print("="*80)

# Extract genres from movies dataframe
print("\nExample of raw genres format:")
print(movies_df[['movieId', 'title', 'genres']].head())

# Count total unique genres
all_genres = set()
for genres in movies_df['genres'].str.split('|'):
    if isinstance(genres, list):
        all_genres.update(genres)

print(f"\nFound {len(all_genres)} unique genres: {sorted(all_genres)}")

# One-hot encode genres
# First, create a DataFrame with movieId and genre columns
genre_data = []
for _, movie in movies_df.iterrows():
    movie_id = movie['movieId']
    genres = movie['genres'].split('|') if isinstance(movie['genres'], str) else []
    
    for genre in genres:
        genre_data.append({'movieId': movie_id, 'genre': genre})

# Convert to DataFrame
genre_df = pd.DataFrame(genre_data)

# Create pivot table for one-hot encoding
genre_one_hot = pd.pivot_table(
    genre_df, 
    index='movieId', 
    columns='genre', 
    aggfunc=lambda x: 1, 
    fill_value=0
).reset_index()

# Flatten the column names
genre_one_hot.columns.name = None

print("\nOne-hot encoded genres (sample):")
print(genre_one_hot.head())

# Merge with original movie data
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

print("\nMovies with genre encodings (sample):")
print(movie_genres.head())

# Plot genre distribution
genre_counts = {}
for genre in all_genres:
    if genre in movie_genres.columns:
        genre_counts[genre] = movie_genres[genre].sum()

# Sort genres by count
sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(14, 7))
plt.bar([x[0] for x in sorted_genres], [x[1] for x in sorted_genres], color='skyblue', edgecolor='black')
plt.title('Distribution of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('genre_distribution.png')
print("\nGenre distribution plot saved as 'genre_distribution.png'")
plt.close()

# Output: Genre-encoded data
print("\nFinal genre-encoded data:")
print(movie_genres.head())

# Final Output: Combined Movie Features
print("\n" + "="*80)
print("FINAL OUTPUT: COMBINED MOVIE FEATURES")
print("="*80)

# Combine all features into one DataFrame
movie_features = pd.merge(
    movie_genres,  # Contains movieId, title, and genre encodings
    preprocessed_df[['movieId', 'tokens']],  # Contains preprocessed text tokens
    on='movieId',
    how='left'
)

# Add a column for text corpus length (token count)
movie_features['token_count'] = movie_features['tokens'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Add a column for the top 5 keywords for each movie (based on frequency)
def get_top_keywords(tokens, n=5):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return []
    
    word_counts = Counter(tokens)
    return [word for word, _ in word_counts.most_common(n)]

movie_features['top_keywords'] = movie_features['tokens'].apply(get_top_keywords)

# Drop the tokens column to make the DataFrame more readable for display
display_features = movie_features.drop(columns=['tokens'])

print("\nFinal movie features (sample):")
print(display_features.head())

# Save the processed data for later use
movie_features.to_csv('processed_movie_features.csv', index=False)
normalized_ratings.to_csv('normalized_ratings.csv', index=False)

print("\nProcessed data saved to 'processed_movie_features.csv' and 'normalized_ratings.csv'")

# Summary of the data processing pipeline
print("\n" + "="*80)
print("SUMMARY OF STAGE 1 DATA PROCESSING")
print("="*80)
print(f"1. Extracted text features for {len(movie_data)} movies")
print(f"2. Preprocessed text resulting in a vocabulary of {len(corpus_word_counts)} unique words")
print(f"3. Normalized {len(normalized_ratings)} ratings from {len(user_stats)} users")
print(f"4. Created one-hot encodings for {len(all_genres)} genres")
print(f"5. Final dataset contains {len(movie_features)} movies with complete feature sets")
print("="*80)