import pandas as pd
import numpy as np
import os
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import math
import logging
from datetime import datetime
import pickle
from scipy import sparse

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)  # Add this line if missing

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path="./data/20m/", output_path="./processed_data", random_state=42):
        """
        Initialize the DataProcessor.
        
        Parameters:
        -----------
        data_path : str, default="./data"
            Path to the directory containing MovieLens dataset files
        output_path : str, default="./processed_data"
            Path to save processed data
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_path = output_path
        self.random_state = random_state
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize data containers
        self.ratings_df = None
        self.movies_df = None
        self.links_df = None
        self.tags_df = None
        self.tmdb_df = None
        
        # Initialize processed data
        self.movie_features = {}
        self.user_features = {}
        self.word2vec_model = None
        self.corpus_word_counts = None
        self.user_genre_preferences = None
        
    def load_data(self):
        """Load MovieLens dataset files"""
        logger.info("Loading MovieLens dataset...")
        
        # Load rating data
        ratings_path = os.path.join(self.data_path, "demo_rating.csv")
        if os.path.exists(ratings_path):
            self.ratings_df = pd.read_csv(ratings_path)
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
        else:
            logger.error(f"File not found: {ratings_path}")
            
        # Load movie data
        movies_path = os.path.join(self.data_path, "demo_movie.csv")
        if os.path.exists(movies_path):
            self.movies_df = pd.read_csv(movies_path)
            logger.info(f"Loaded {len(self.movies_df)} movies")
        else:
            logger.error(f"File not found: {movies_path}")
            
        # Load links data
        links_path = os.path.join(self.data_path, "demo_link.csv")
        if os.path.exists(links_path):
            self.links_df = pd.read_csv(links_path)
            logger.info(f"Loaded {len(self.links_df)} links")
        else:
            logger.error(f"File not found: {links_path}")
            
        # Load tags data (if available)
        tags_path = os.path.join(self.data_path, "demo_tag.csv")
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
            logger.info(f"Loaded {len(self.tags_df)} tags")
        
        # Load TMDB data (if available)
        tmdb_path = os.path.join(self.data_path, "demo_tmdb.csv")
        if os.path.exists(tmdb_path):
            self.tmdb_df = pd.read_csv(tmdb_path)
            logger.info(f"Loaded {len(self.tmdb_df)} TMDB records")
        
        # Perform simple validation
        if self.movies_df is not None and self.ratings_df is not None:
            movie_ids_in_movies = set(self.movies_df['movieId'])
            movie_ids_in_ratings = set(self.ratings_df['movieId'])
            common_ids = movie_ids_in_movies.intersection(movie_ids_in_ratings)
            logger.info(f"Found {len(common_ids)} movies with both metadata and ratings")
    
    def clean_data(self):
        """Clean and prepare the loaded data"""
        logger.info("Cleaning data...")
        
        # 1. Clean ratings data
        if self.ratings_df is not None:
            # Convert timestamp to datetime - check if already in datetime format
            if 'timestamp' in self.ratings_df.columns:
                # Try to determine if timestamps are unix seconds or already formatted
                first_timestamp = str(self.ratings_df['timestamp'].iloc[0])
                if first_timestamp.isdigit():
                    # If numeric, treat as Unix timestamp
                    self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
                else:
                    # If already formatted as a date string
                    self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'])
        # 2. Clean movie data
        if self.movies_df is not None:
            # Extract year from title if it's in format "Movie Title (YYYY)"
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$')
            
            # Convert genres from pipe-separated string to list
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
            
            # One-hot encode genres
            genres_dummies = self.movies_df['genres'].str.get_dummies('|')
            self.movies_df = pd.concat([self.movies_df, genres_dummies], axis=1)
            
            logger.info(f"Processed {len(self.movies_df)} movies")
        
        # 3. Clean TMDB data
        if self.tmdb_df is not None:
            # Clean HTML tags and special characters from overview
            if 'overview' in self.tmdb_df.columns:
                self.tmdb_df['overview'] = self.tmdb_df['overview'].fillna('')
                self.tmdb_df['overview'] = self.tmdb_df['overview'].apply(self._clean_text)
            
            logger.info(f"Processed {len(self.tmdb_df)} TMDB records")
        
        # 4. Merge movie data with TMDB data if available
        if self.movies_df is not None and self.tmdb_df is not None and self.links_df is not None:
            # Merge based on tmdbId from links_df
            merged_df = pd.merge(self.movies_df, self.links_df, on='movieId', how='left')
            self.movies_df = pd.merge(merged_df, self.tmdb_df, left_on='tmdbId', right_on='id', how='left')
            logger.info(f"Merged movie data with TMDB data: {len(self.movies_df)} records")
    
    def _get_wordnet_pos(self, tag):
        """Map POS tag to WordNet POS tag"""
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)
    
    def _clean_text(self, text):
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
    
    def _preprocess_text(self, text):
        """Tokenize, remove stopwords, and lemmatize text"""
        if not isinstance(text, str) or text == "":
            return []
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        
        try:
            # Lemmatize tokens with POS tagging
            tagged_tokens = pos_tag(tokens)
            lemmatized_tokens = [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag)) 
                                for word, tag in tagged_tokens]
        except LookupError:
            # Fallback to simple lemmatization without POS tagging
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return lemmatized_tokens
    
    def preprocess_data(self):
        """Preprocess the data for feature extraction"""
        logger.info("Preprocessing data...")
        
        # 1. Preprocess movie text data
        if self.movies_df is not None:
            # Create a corpus for each movie from available text fields
            self.movies_df['text_corpus'] = ""
            
            # Add title to corpus
            self.movies_df['text_corpus'] += self.movies_df['title'].fillna("").apply(self._clean_text)
            
            # Add TMDB overview to corpus if available
            if 'overview' in self.movies_df.columns:
                self.movies_df['text_corpus'] += " " + self.movies_df['overview'].fillna("").apply(self._clean_text)
            
            # Add TMDB cast to corpus if available
            if 'cast' in self.movies_df.columns:
                self.movies_df['text_corpus'] += " " + self.movies_df['cast'].fillna("").apply(self._clean_text)
            
            # Add TMDB director to corpus if available
            if 'director' in self.movies_df.columns:
                self.movies_df['text_corpus'] += " " + self.movies_df['director'].fillna("").apply(self._clean_text)
            
            # Add tags to corpus if available
            if self.tags_df is not None:
                # Fill NaN values with empty strings before aggregation
                self.tags_df['tag'] = self.tags_df['tag'].fillna('').astype(str)
                
                # Aggregate tags by movieId
                tags_by_movie = self.tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
                tags_by_movie['tag'] = tags_by_movie['tag'].apply(self._clean_text)
                
                # Merge tags with movies
                self.movies_df = pd.merge(self.movies_df, tags_by_movie, on='movieId', how='left')
                
                # Add tags to corpus
                self.movies_df['text_corpus'] += " " + self.movies_df['tag'].fillna("")
            
            # Tokenize and lemmatize the corpus
            logger.info("Tokenizing and lemmatizing movie text corpus...")
            self.movies_df['tokens'] = self.movies_df['text_corpus'].apply(self._preprocess_text)
            
            # Count corpus words
            all_words = []
            for tokens in self.movies_df['tokens']:
                all_words.extend(tokens)
            
            self.corpus_word_counts = Counter(all_words)
            logger.info(f"Built vocabulary with {len(self.corpus_word_counts)} unique words")
            
            # Calculate document frequency (number of documents containing each word)
            self.doc_freq = {}
            for tokens in self.movies_df['tokens']:
                for word in set(tokens):  # Count each word only once per document
                    self.doc_freq[word] = self.doc_freq.get(word, 0) + 1
            
            # Save the tokenized corpus for Word2Vec training
            tokenized_corpus = self.movies_df['tokens'].tolist()
            with open(os.path.join(self.output_path, 'tokenized_corpus.pkl'), 'wb') as f:
                pickle.dump(tokenized_corpus, f)
            
            logger.info(f"Preprocessed text for {len(self.movies_df)} movies")
        
        # 2. Preprocess user ratings
        if self.ratings_df is not None:
            # Calculate rating statistics by user
            user_stats = self.ratings_df.groupby('userId').agg({
                'rating': ['count', 'mean', 'std']
            }).reset_index()
            user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
            
            # Fill NA values in std with 0 (for users with only one rating)
            user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
            
            # Calculate rating statistics by movie
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['count', 'mean', 'std']
            }).reset_index()
            movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std']
            
            # Fill NA values in std with 0 (for movies with only one rating)
            movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
            
            # Merge statistics with original dataframes
            self.user_stats = user_stats
            self.movie_stats = movie_stats
            
            logger.info(f"Calculated rating statistics for {len(user_stats)} users and {len(movie_stats)} movies")
            
            # Create train-test split for evaluation
            self.train_df, self.test_df = train_test_split(
                self.ratings_df, test_size=0.2, random_state=self.random_state
            )
            
            logger.info(f"Created train-test split: {len(self.train_df)} training, {len(self.test_df)} testing")
        
        # 3. Calculate user preferences for genres
        if self.ratings_df is not None and self.movies_df is not None:
    # Merge ratings with movie genre data
            common_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                            'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
                            'Sci-Fi', 'TV Movie', 'Thriller', 'War', 'Western']

            # Then filter columns that are either in common_genres or have 'genre' in the name 
            genre_columns = [col for col in self.movies_df.columns 
                            if col in common_genres or 
                            'genre' in col.lower() or 
                            any(genre.lower() in col.lower() for genre in common_genres)]

            # Explicitly exclude non-genre columns that might get caught in the filter
            non_genre_columns = ['movieId', 'title', 'genres', 'text_corpus', 'tokens', 'year', 
                                'genres_list', 'tmdb_title', 'overview', 'cast', 'director', 
                                'tag', 'tmdbId', 'id']
            genre_columns = [col for col in genre_columns if col not in non_genre_columns]

            logger.info(f"Identified {len(genre_columns)} genre columns: {genre_columns}")
            
            # Keep only genre columns (assuming they are one-hot encoded)
            genre_columns = [col for col in genre_columns if col in self.movies_df.columns]
            
            if genre_columns:
                # Create a new DataFrame with only the columns we need
                movies_genres_df = self.movies_df[['movieId'] + genre_columns].copy()
                
                # Handle non-numeric columns and convert to safe numeric values
                for genre in genre_columns:
                    try:
                        # Convert to numeric if not already
                        if not pd.api.types.is_numeric_dtype(movies_genres_df[genre]):
                            movies_genres_df[genre] = pd.to_numeric(movies_genres_df[genre], errors='coerce')
                        
                        # Replace all non-finite values with 0 and convert to int
                        movies_genres_df[genre] = movies_genres_df[genre].replace([np.inf, -np.inf], np.nan).fillna(0)
                        movies_genres_df[genre] = movies_genres_df[genre].astype('Int64')  # Use Int64 pandas nullable integer type
                    except Exception as e:
                        logger.warning(f"Error processing genre column {genre}: {str(e)}")
                        # Remove problematic column
                        movies_genres_df.drop(columns=[genre], inplace=True)
                        genre_columns.remove(genre)
                
                # Merge with ratings
                ratings_with_genres = pd.merge(
                    self.ratings_df,
                    movies_genres_df,
                    on='movieId'
                )
                
                # Define liked and disliked movies
                ratings_with_genres['liked'] = ratings_with_genres['rating'] >= 3.0
                
                # Calculate user preferences for each genre
                user_genre_prefs = []
                
                for user_id in ratings_with_genres['userId'].unique():
                    user_data = ratings_with_genres[ratings_with_genres['userId'] == user_id]
                    
                    if len(user_data) == 0:
                        continue
                    
                    liked_data = user_data[user_data['liked']]
                    disliked_data = user_data[~user_data['liked']]
                    
                    user_prefs = {'userId': user_id}
                    
                    for genre in genre_columns:
                        try:
                            # Count liked and disliked movies in this genre
                            num_liked = liked_data[genre].sum()
                            num_disliked = disliked_data[genre].sum()
                            total = num_liked + num_disliked
                            
                            # Calculate preference score: (liked - disliked) / total
                            if total > 0:
                                preference = (num_liked - num_disliked) / total
                            else:
                                preference = 0
                            
                            # Normalize to [-1, 1]
                            user_prefs[f'{genre}_pref'] = preference
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error processing genre {genre} for user {user_id}: {str(e)}")
                            user_prefs[f'{genre}_pref'] = 0
                    
                    user_genre_prefs.append(user_prefs)
                
                # Create DataFrame from user preferences
                self.user_genre_preferences = pd.DataFrame(user_genre_prefs)
                
                logger.info(f"Calculated genre preferences for {len(self.user_genre_preferences)} users")
    def split_and_save_ratings(self):
        """Split ratings into train and test sets and save them"""
        if self.ratings_df is not None and len(self.ratings_df) > 0:
            logger.info("Splitting ratings data into train and test sets...")
            
            # Create train-test split
            self.train_df, self.test_df = train_test_split(
                self.ratings_df, test_size=0.2, random_state=self.random_state
            )
            
            # Save to files
            train_path = os.path.join(self.output_path, 'train_ratings.csv')
            test_path = os.path.join(self.output_path, 'test_ratings.csv')
            
            self.train_df.to_csv(train_path, index=False)
            self.test_df.to_csv(test_path, index=False)
            
            logger.info(f"Created and saved train-test split: {len(self.train_df)} training, {len(self.test_df)} testing")
            logger.info(f"Train data saved to: {train_path}")
            logger.info(f"Test data saved to: {test_path}")
            
            return True
        else:
            logger.error("Cannot split ratings: No ratings data available")
            return False
    def extract_features(self):
        """Extract features for content-based and collaborative filtering"""
        logger.info("Extracting features...")
        
        # 1. Train Word2Vec model on tokenized corpus
        logger.info("Training Word2Vec model...")
        tokenized_corpus_path = os.path.join(self.output_path, 'tokenized_corpus.pkl')
        
        if os.path.exists(tokenized_corpus_path):
            with open(tokenized_corpus_path, 'rb') as f:
                tokenized_corpus = pickle.load(f)
            
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(
                sentences=tokenized_corpus,
                vector_size=300,  # 300 dimensions as specified in the paper
                window=10,        # Context window size
                min_count=5,      # Ignore words with fewer occurrences
                workers=4,        # Number of threads
                epochs=30,        # Number of training epochs
                sg=1              # Skip-gram model as specified in the paper
            )
            
            # Save Word2Vec model
            word2vec_path = os.path.join(self.output_path, 'word2vec_model')
            self.word2vec_model.save(word2vec_path)
            logger.info(f"Trained Word2Vec model with {len(self.word2vec_model.wv)} words")
        else:
            logger.error(f"Tokenized corpus not found at {tokenized_corpus_path}")
        
        # 2. Calculate Log-Likelihood values for words in each movie
        if hasattr(self, 'movies_df') and 'tokens' in self.movies_df.columns:
            logger.info("Calculating Log-Likelihood values...")
            
            # Calculate total corpus size
            total_corpus_size = sum(self.corpus_word_counts.values())
            
            # Initialize container for movie features
            movie_features = {}
            
            # Process each movie
            for idx, row in self.movies_df.iterrows():
                movie_id = row['movieId']
                tokens = row['tokens']
                
                if not tokens:
                    continue
                
                # Count word occurrences in this movie
                movie_word_counts = Counter(tokens)
                movie_size = sum(movie_word_counts.values())
                
                # Calculate Log-Likelihood for each word
                movie_ll_values = {}
                
                for word, count in movie_word_counts.items():
                    # Observed and expected frequencies
                    a = count  # Occurrences in this movie
                    b = self.corpus_word_counts[word] - count  # Occurrences in other movies
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
                    movie_ll_values[word] = ll
                
                # Store the Log-Likelihood values for this movie
                movie_features[movie_id] = {'ll_values': movie_ll_values, 'tokens': tokens}
            
            self.movie_features = movie_features
            logger.info(f"Calculated Log-Likelihood values for {len(movie_features)} movies")
            
            # Save Log-Likelihood values
            with open(os.path.join(self.output_path, 'movie_ll_values.pkl'), 'wb') as f:
                pickle.dump(movie_features, f)
        
        # 3. Generate movie feature vectors (combining Log-Likelihood and Word2Vec)
        if hasattr(self, 'word2vec_model') and hasattr(self, 'movie_features'):
            logger.info("Generating movie feature vectors...")
            
            movie_vectors = {}
            
            for movie_id, features in self.movie_features.items():
                ll_values = features['ll_values']
                tokens = features['tokens']
                
                # Sort words by LL value and select top 200 as specified in the paper
                top_words = sorted(ll_values.items(), key=lambda x: x[1], reverse=True)[:200]
                
                if not top_words:
                    continue
                
                # Combine Word2Vec vectors weighted by Log-Likelihood values
                weighted_vectors = []
                ll_sum = 0
                
                for word, ll_value in top_words:
                    if ll_value <= 0:
                        continue
                    
                    if word in self.word2vec_model.wv:
                        weighted_vectors.append(self.word2vec_model.wv[word] * ll_value)
                        ll_sum += ll_value
                
                if weighted_vectors and ll_sum > 0:
                    # Calculate the weighted average vector
                    movie_vector = np.sum(weighted_vectors, axis=0) / ll_sum
                    
                    # Normalize to unit length
                    norm = np.linalg.norm(movie_vector)
                    if norm > 0:
                        movie_vector = movie_vector / norm
                    
                    movie_vectors[movie_id] = movie_vector
            
            # Save movie vectors
            with open(os.path.join(self.output_path, 'movie_vectors.pkl'), 'wb') as f:
                pickle.dump(movie_vectors, f)
            
            logger.info(f"Generated feature vectors for {len(movie_vectors)} movies")
            
            # Movie ID to index mapping
            movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_vectors.keys())}
            
            # Save the mapping
            with open(os.path.join(self.output_path, 'movie_id_to_idx.pkl'), 'wb') as f:
                pickle.dump(movie_id_to_idx, f)
        
        # 4. Generate user feature vectors using the same approach as the paper
        if hasattr(self, 'train_df') and hasattr(self, 'word2vec_model') and hasattr(self, 'movie_features'):
            logger.info("Generating user feature vectors...")
            
            user_vectors = {}
            
            # Load movie vectors
            with open(os.path.join(self.output_path, 'movie_vectors.pkl'), 'rb') as f:
                movie_vectors = pickle.load(f)
            
            # Process each user
            for user_id in self.train_df['userId'].unique():
                # Get user ratings
                user_ratings = self.train_df[self.train_df['userId'] == user_id]
                
                if len(user_ratings) == 0:
                    continue
                
                weighted_vectors = []
                weight_sum = 0
                
                for _, rating_row in user_ratings.iterrows():
                    movie_id = rating_row['movieId']
                    rating = rating_row['rating']
                    
                    # Skip if movie vector is not available
                    if movie_id not in movie_vectors:
                        continue
                    
                    # Weight is the rating centered at 3.0
                    weight = rating - 3.0
                    
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
            
            # Save user vectors
            with open(os.path.join(self.output_path, 'user_vectors.pkl'), 'wb') as f:
                pickle.dump(user_vectors, f)
            
            logger.info(f"Generated feature vectors for {len(user_vectors)} users")
            
            # User ID to index mapping
            user_id_to_idx = {user_id: i for i, user_id in enumerate(user_vectors.keys())}
            
            # Save the mapping
            with open(os.path.join(self.output_path, 'user_id_to_idx.pkl'), 'wb') as f:
                pickle.dump(user_id_to_idx, f)
        
        # 5. Process data for the DNN model
        # 5. Process data for the DNN model
        if hasattr(self, 'user_genre_preferences') and self.user_genre_preferences is not None:
            logger.info("Preparing data for DNN model...")
            
            # Prepare genre data for DNN model
            genres_df = self.movies_df.drop(columns=['title', 'genres', 'text_corpus', 'tokens', 'year', 'genres_list'], errors='ignore')
            
            # Keep only genre columns (one-hot encoded)
            genre_columns = [col for col in genres_df.columns if col not in ['movieId']]
            
            # Ensure we have genre data
            if genre_columns:
                # Save genres data
                genres_df.to_csv(os.path.join(self.output_path, 'movie_genres.csv'), index=False)
                
                # Save user genre preferences
                self.user_genre_preferences.to_csv(os.path.join(self.output_path, 'user_genre_prefs.csv'), index=False)
                
                # Save user and movie stats directly
                if hasattr(self, 'user_stats') and self.user_stats is not None:
                    self.user_stats.to_csv(os.path.join(self.output_path, 'user_stats.csv'), index=False)
                    logger.info(f"Saved statistics for {len(self.user_stats)} users to user_stats.csv")
                    
                if hasattr(self, 'movie_stats') and self.movie_stats is not None:
                    self.movie_stats.to_csv(os.path.join(self.output_path, 'movie_stats.csv'), index=False)
                    logger.info(f"Saved statistics for {len(self.movie_stats)} movies to movie_stats.csv")
                
                # Save train and test sets
                if hasattr(self, 'train_df') and hasattr(self, 'test_df'):
                    self.train_df.to_csv(os.path.join(self.output_path, 'train_ratings.csv'), index=False)
                    self.test_df.to_csv(os.path.join(self.output_path, 'test_ratings.csv'), index=False)
                
                logger.info(f"Prepared data for DNN model")
        else:
            logger.warning("User genre preferences not available. Skipping DNN data preparation.")
    
    def process(self):
        """Process the data pipeline"""
        # Create a timestamped run ID
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting data processing pipeline (Run ID: {run_id})")
        
        # Load data
        self.load_data()
        
        # Clean data
        self.clean_data()
        
        # Explicitly split and save ratings before continuing
        self.split_and_save_ratings()
        
        # Continue with remaining processing
        self.preprocess_data()
        self.extract_features()
    
        
        logger.info(f"Data processing pipeline completed (Run ID: {run_id})")
        
        # Save the processor state
        with open(os.path.join(self.output_path, f'processor_state_{run_id}.pkl'), 'wb') as f:
            pickle.dump({
                'corpus_word_counts': self.corpus_word_counts,
                'doc_freq': self.doc_freq,
                'movie_stats': self.movie_stats,
                'user_stats': self.user_stats
            }, f)
        
        logger.info(f"Saved processor state to {self.output_path}/processor_state_{run_id}.pkl")
        
        return {
            'run_id': run_id,
            'output_path': self.output_path,
            'stats': {
                'num_movies': len(self.movies_df) if hasattr(self, 'movies_df') else 0,
                'num_users': len(self.user_stats) if hasattr(self, 'user_stats') else 0,
                'num_ratings': len(self.ratings_df) if hasattr(self, 'ratings_df') else 0,
                'num_movie_vectors': len(self.movie_features) if hasattr(self, 'movie_features') else 0
            }
        }

if __name__ == "__main__":
    # Create processor with default paths
    processor = DataProcessor(data_path="./data/20m", output_path="./processed_data")
    
    # Run the processing pipeline
    result = processor.process()
    
    # Print summary
    print("\nProcessing completed!")
    print(f"Run ID: {result['run_id']}")
    print(f"Output path: {result['output_path']}")
    print("\nStatistics:")
    for key, value in result['stats'].items():
        print(f"- {key}: {value}")