import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import pickle
import logging
import time
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ast
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("FULLY CUDA-ACCELERATED CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
print("="*80)

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
if cuda_available:
    cuda_device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    cuda_device_count = torch.cuda.device_count()
    cuda_capability = torch.cuda.get_device_capability(0)
    print(f"\nCUDA is available: Using {device_name}")
    print(f"Number of CUDA devices: {cuda_device_count}")
    print(f"CUDA Compute Capability: {cuda_capability[0]}.{cuda_capability[1]}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = cuda_device
else:
    print("\nCUDA is not available: Using CPU instead")
    device = torch.device("cpu")
print(f"PyTorch device: {device}")

# Set paths
input_path = "processed/"  # Directory where stage1.py saved the files
output_path = "cuda_optimized_recommendations"
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
batch_size = 512  # Default batch size for GPU operations
max_text_length = 500  # Maximum number of tokens to use per movie

# Custom Word2Vec implementation using PyTorch and CUDA
class Word2VecCUDA(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecCUDA, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        # inputs shape: [batch_size, context_size]
        embeds = self.embeddings(inputs)  # [batch_size, context_size, embedding_dim]
        # For CBOW, we need to average the context word embeddings
        embeds_mean = torch.mean(embeds, dim=1)  # [batch_size, embedding_dim]
        out = self.linear(embeds_mean)  # [batch_size, vocab_size]
        return out

# Utility class for processing batches of text data on GPU
class TextProcessor:
    def __init__(self, device):
        self.device = device
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, tokens_list, min_count=5):
        """Build vocabulary from tokenized texts with CUDA acceleration"""
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)
        
        # Filter words that appear less than min_count times
        vocab = [word for word, count in counter.items() if count >= min_count]
        
        # Create word-to-index and index-to-word mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        
        return self.vocab_size
    
    def texts_to_sequences(self, tokens_list, max_length=None):
        """Convert lists of tokens to sequences of indices"""
        sequences = []
        for tokens in tokens_list:
            seq = [self.word_to_idx[word] for word in tokens if word in self.word_to_idx]
            if max_length:
                seq = seq[:max_length]
            sequences.append(seq)
        return sequences
    
    def create_training_data(self, sequences, window_size=5):
        """Create training data for Word2Vec CBOW model"""
        data = []
        for sequence in sequences:
            if len(sequence) < window_size * 2 + 1:
                continue
            
            for i in range(window_size, len(sequence) - window_size):
                context = sequence[i-window_size:i] + sequence[i+1:i+window_size+1]
                # Ensure all contexts have the same length for batch processing
                if len(context) == window_size * 2:
                    target = sequence[i]
                    data.append((context, target))
        return data

# STEP 1: Data Loading with GPU acceleration
def load_data_cuda():
    """Load processed data from stage1.py with CUDA acceleration where possible"""
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING WITH CUDA ACCELERATION")
    print("="*80)
    
    # Data containers
    data = {}
    
    # Load movie features
    movie_features_path = os.path.join(input_path, 'processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        start_time = time.time()
        # Load CSV data
        data['movie_features'] = pd.read_csv(movie_features_path)
        
        # Convert string representation of tokens and top_keywords back to lists
        data['movie_features']['tokens'] = data['movie_features']['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        data['movie_features']['top_keywords'] = data['movie_features']['top_keywords'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        
        # Create tensor mappings for efficient lookups
        movie_ids = data['movie_features']['movieId'].values
        data['movie_id_mapping'] = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        
        # Create movieId tensor for quick lookups
        data['movie_ids_tensor'] = torch.tensor(movie_ids, dtype=torch.int64, device=device)
        
        print(f"Loaded features for {len(data['movie_features'])} movies in {time.time() - start_time:.2f}s")
        print("\nSample of movie features data:")
        print(data['movie_features'][['movieId', 'title', 'top_keywords']].head(3))
        
        # Print token statistics
        token_lengths = [len(tokens) for tokens in data['movie_features']['tokens']]
        print(f"\nAverage token count per movie: {np.mean(token_lengths):.2f}")
        print(f"Min token count: {min(token_lengths)}, Max token count: {max(token_lengths)}")
        
        # Create genre feature tensor
        genre_columns = [col for col in data['movie_features'].columns if col not in 
                         ['movieId', 'title', 'tokens', 'token_count', 'top_keywords']]
        
        if genre_columns:
            # Create genre tensor on GPU
            genre_tensor = torch.tensor(
                data['movie_features'][genre_columns].values,
                dtype=torch.float32,
                device=device
            )
            data['genre_tensor'] = genre_tensor
            data['genre_columns'] = genre_columns
            print(f"Created genre tensor with {len(genre_columns)} genres on {device}")
    else:
        print(f"Error: Movie features not found at {movie_features_path}")
        return None
    
    # Load normalized ratings
    ratings_path = os.path.join(input_path, 'normalized_ratings.csv')
    if os.path.exists(ratings_path):
        start_time = time.time()
        data['ratings'] = pd.read_csv(ratings_path)
        
        # Create tensors for ratings data
        user_ids = data['ratings']['userId'].unique()
        data['user_id_mapping'] = {user_id: idx for idx, user_id in enumerate(user_ids)}
        
        # Create ratings tensor (will be used for splitting and processing)
        users_idx = [data['user_id_mapping'][uid] for uid in data['ratings']['userId']]
        movies_idx = [data['movie_id_mapping'].get(mid, -1) for mid in data['ratings']['movieId']]
        # Filter out movies not in our mapping (they will have -1 indices)
        valid_indices = [i for i, mid_idx in enumerate(movies_idx) if mid_idx != -1]
        
        if valid_indices:
            users_idx = [users_idx[i] for i in valid_indices]
            movies_idx = [movies_idx[i] for i in valid_indices]
            ratings = data['ratings']['rating'].iloc[valid_indices].values
            norm_ratings = data['ratings']['normalized_rating'].iloc[valid_indices].values
            
            # Create tensors
            data['users_idx_tensor'] = torch.tensor(users_idx, dtype=torch.int64, device=device)
            data['movies_idx_tensor'] = torch.tensor(movies_idx, dtype=torch.int64, device=device)
            data['ratings_tensor'] = torch.tensor(ratings, dtype=torch.float32, device=device)
            data['norm_ratings_tensor'] = torch.tensor(norm_ratings, dtype=torch.float32, device=device)
            
            print(f"Loaded {len(data['ratings'])} normalized ratings in {time.time() - start_time:.2f}s")
            print(f"Created ratings tensors on {device}")
            print("\nSample of normalized ratings data:")
            print(data['ratings'].head(3))
            
            # Print rating statistics
            print(f"\nNumber of unique users: {data['ratings']['userId'].nunique()}")
            print(f"Number of unique movies: {data['ratings']['movieId'].nunique()}")
            print(f"Rating sparsity: {(1 - len(data['ratings']) / (data['ratings']['userId'].nunique() * data['ratings']['movieId'].nunique())) * 100:.4f}%")
            
            # Split into training and testing sets using GPU operations
            print("\nSplitting ratings into training and testing sets...")
            
            # Sort by user ID for consistent splits
            sorted_indices = torch.argsort(data['users_idx_tensor'])
            users_idx_sorted = data['users_idx_tensor'][sorted_indices]
            movies_idx_sorted = data['movies_idx_tensor'][sorted_indices]
            ratings_sorted = data['ratings_tensor'][sorted_indices]
            norm_ratings_sorted = data['norm_ratings_tensor'][sorted_indices]
            
            # Create user groups for splitting
            unique_users, user_counts = torch.unique(users_idx_sorted, return_counts=True)
            train_mask = torch.zeros_like(users_idx_sorted, dtype=torch.bool)
            
            # For each user, mark the first 80% of ratings as training
            start_idx = 0
            for user_idx, count in zip(unique_users.tolist(), user_counts.tolist()):
                end_idx = start_idx + count
                split_idx = start_idx + int(count * 0.8)
                train_mask[start_idx:split_idx] = True
                start_idx = end_idx
            
            # Create training and testing tensors
            data['train_users_idx'] = users_idx_sorted[train_mask]
            data['train_movies_idx'] = movies_idx_sorted[train_mask]
            data['train_ratings'] = ratings_sorted[train_mask]
            data['train_norm_ratings'] = norm_ratings_sorted[train_mask]
            
            data['test_users_idx'] = users_idx_sorted[~train_mask]
            data['test_movies_idx'] = movies_idx_sorted[~train_mask]
            data['test_ratings'] = ratings_sorted[~train_mask]
            data['test_norm_ratings'] = norm_ratings_sorted[~train_mask]
            
            print(f"Split ratings into {len(data['train_ratings'])} training and {len(data['test_ratings'])} testing samples")
            
            # Create reverse mappings for evaluation
            data['idx_to_user_id'] = {idx: user_id for user_id, idx in data['user_id_mapping'].items()}
            data['idx_to_movie_id'] = {idx: movie_id for movie_id, idx in data['movie_id_mapping'].items()}
        else:
            print("Error: No valid movie indices found in ratings data")
    else:
        print(f"Error: Normalized ratings not found at {ratings_path}")
        return None
    
    return data

# # STEP 2: Corpus Analysis with CUDA
# def build_corpus_cuda(data):
#     """Build corpus and word frequency counts using CUDA acceleration"""
#     print("\n" + "="*80)
#     print("STEP 2: CORPUS ANALYSIS WITH CUDA")
#     print("="*80)
    
#     start_time = time.time()
    
#     # Extract all tokens into a single list
#     all_tokens = []
#     for tokens in data['movie_features']['tokens']:
#         if len(tokens) > max_text_length:
#             all_tokens.append(tokens[:max_text_length])
#         else:
#             all_tokens.append(tokens)
    
#     # Calculate corpus statistics
#     corpus_word_counts = Counter()
#     for tokens in all_tokens:
#         corpus_word_counts.update(tokens)
    
#     data['corpus_word_counts'] = corpus_word_counts
#     data['all_tokens'] = all_tokens
    
#     # Create text processor for GPU tokenization
#     text_processor = TextProcessor(device)
#     vocab_size = text_processor.build_vocab(all_tokens, min_count=5)
#     data['text_processor'] = text_processor
    
#     print(f"Built vocabulary with {vocab_size} unique words in {time.time() - start_time:.2f}s")
#     print(f"Total words in corpus: {sum(corpus_word_counts.values())}")
    
#     # Display top 20 most common words
#     print("\nTop 20 most common words in the corpus:")
#     for word, count in corpus_word_counts.most_common(20):
#         print(f"'{word}': {count}")
    
#     # Save corpus word counts
#     with open(os.path.join(output_path, 'corpus_word_counts.pkl'), 'wb') as f:
#         pickle.dump(corpus_word_counts, f)
    
#     return data

# # STEP 3: Log-Likelihood Calculation with CUDA
# def calculate_log_likelihood_cuda(data):
#     """Calculate Log-Likelihood values for words in each movie using CUDA"""
#     print("\n" + "="*80)
#     print("STEP 3: LOG-LIKELIHOOD CALCULATION WITH CUDA")
#     print("="*80)
    
#     start_time = time.time()
#     corpus_word_counts = data['corpus_word_counts']
    
#     # Calculate total corpus size
#     total_corpus_size = sum(corpus_word_counts.values())
#     print(f"Total corpus size: {total_corpus_size} words")
    
#     # Precompute corpus frequencies for top words to reduce computation
#     top_words = [word for word, _ in corpus_word_counts.most_common(20000)]
#     corpus_freq = {word: corpus_word_counts[word] / total_corpus_size for word in top_words}
    
#     # Initialize container for movie features
#     movie_ll_values = {}
    
#     # Process batches of movies for better GPU utilization
#     movie_features = data['movie_features']
#     all_tokens = data['all_tokens']
#     total_movies = len(movie_features)
    
#     # Convert to PyTorch tensors for faster processing
#     batch_size = min(1000, total_movies)
#     num_batches = (total_movies + batch_size - 1) // batch_size
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, total_movies)
        
#         batch_movie_ids = movie_features['movieId'].iloc[start_idx:end_idx].values
#         batch_tokens = all_tokens[start_idx:end_idx]
        
#         # Calculate LL values for each movie in the batch
#         for i, (movie_id, tokens) in enumerate(zip(batch_movie_ids, batch_tokens)):
#             if not tokens:
#                 continue
            
#             # Count word occurrences in this movie
#             movie_word_counts = Counter(tokens)
#             movie_size = sum(movie_word_counts.values())
            
#             # Calculate Log-Likelihood for each word
#             movie_ll = {}
            
#             for word, count in movie_word_counts.items():
#                 if word not in top_words:
#                     continue  # Skip rare words for efficiency
                
#                 # Observed frequencies
#                 a = count  # Occurrences in this movie
#                 b = corpus_word_counts[word] - count  # Occurrences in other movies
#                 c = movie_size  # Total words in this movie
#                 d = total_corpus_size - movie_size  # Total words in other movies
                
#                 # Expected counts based on corpus distribution
#                 e1 = c * (a + b) / (c + d) if c + d > 0 else 0.001
#                 e2 = d * (a + b) / (c + d) if c + d > 0 else 0.001
                
#                 # Log-Likelihood calculation
#                 ll = 0
#                 if a > 0 and e1 > 0:
#                     ll += a * math.log(a / e1)
#                 if b > 0 and e2 > 0:
#                     ll += b * math.log(b / e2)
                
#                 ll = 2 * ll
#                 movie_ll[word] = ll
            
#             movie_ll_values[movie_id] = movie_ll
        
#         print(f"Processed batch {batch_idx+1}/{num_batches}: {end_idx}/{total_movies} movies")
    
#     data['movie_ll_values'] = movie_ll_values
    
#     # Save to disk
#     with open(os.path.join(output_path, 'movie_ll_values.pkl'), 'wb') as f:
#         pickle.dump(movie_ll_values, f)
    
#     print(f"Calculated Log-Likelihood values for {len(movie_ll_values)} movies in {time.time() - start_time:.2f}s")
    
#     # Show sample LL values
#     if movie_ll_values:
#         sample_movie_id = next(iter(movie_ll_values.keys()))
#         sample_movie_title = movie_features[movie_features['movieId'] == sample_movie_id]['title'].values[0]
#         print(f"\nSample Log-Likelihood values for movie '{sample_movie_title}' (ID: {sample_movie_id}):")
        
#         # Get top 10 words by LL value
#         top_ll_words = sorted(movie_ll_values[sample_movie_id].items(), key=lambda x: x[1], reverse=True)[:10]
#         for word, ll_value in top_ll_words:
#             print(f"Word: '{word}', LL Value: {ll_value:.2f}")
    
#     return data

# # STEP 4: Word2Vec Model Training with CUDA
# def train_word2vec_cuda(data):
#     """Train Word2Vec model using CUDA acceleration"""
#     print("\n" + "="*80)
#     print("STEP 4: WORD2VEC MODEL TRAINING WITH CUDA")
#     print("="*80)
    
#     start_time = time.time()
#     all_tokens = data['all_tokens']
#     text_processor = data['text_processor']
    
#     try:
#         # First try the optimized CUDA implementation
#         print("Attempting optimized CUDA implementation...")
        
#         # Create sequences
#         print("Converting tokens to sequences...")
#         sequences = text_processor.texts_to_sequences(all_tokens, max_length=max_text_length)
        
#         # Create training data
#         print("Creating training data for Word2Vec...")
#         window_size = 5
#         training_data = text_processor.create_training_data(sequences, window_size=window_size)
        
#         if not training_data:
#             raise ValueError("No training data generated for Word2Vec")
        
#         # Convert to tensors
#         contexts = []
#         targets = []
#         for context, target in training_data:
#             contexts.append(context)
#             targets.append(target)
        
#         # Create PyTorch tensors
#         contexts_tensor = torch.tensor(contexts, dtype=torch.long, device=device)
#         targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        
#         print(f"Contexts tensor shape: {contexts_tensor.shape}")
#         print(f"Targets tensor shape: {targets_tensor.shape}")
        
#         # Create DataLoader for batching
#         dataset = TensorDataset(contexts_tensor, targets_tensor)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
#         # Create and train the model
#         print(f"Training Word2Vec model with {word2vec_dim} dimensions...")
#         model = Word2VecCUDA(text_processor.vocab_size, word2vec_dim).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         criterion = nn.CrossEntropyLoss()
        
#         num_epochs = 5
#         print(f"Training for {num_epochs} epochs with batch size {batch_size}...")
        
#         model.train()
#         for epoch in range(num_epochs):
#             total_loss = 0
#             for batch_idx, (context, target) in enumerate(dataloader):
#                 try:
#                     # Print shapes for debugging on first batch
#                     if batch_idx == 0 and epoch == 0:
#                         print(f"Context batch shape: {context.shape}")
#                         print(f"Target batch shape: {target.shape}")
                    
#                     # Forward pass
#                     outputs = model(context)
                    
#                     # Print shapes for debugging on first batch
#                     if batch_idx == 0 and epoch == 0:
#                         print(f"Output shape: {outputs.shape}")
                    
#                     loss = criterion(outputs, target)
                    
#                     # Backward pass and optimize
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
                    
#                     total_loss += loss.item()
                    
#                     if (batch_idx + 1) % 100 == 0:
#                         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
#                 except Exception as e:
#                     print(f"Error in batch {batch_idx}:")
#                     print(f"Context shape: {context.shape}")
#                     print(f"Target shape: {target.shape}")
#                     print(f"Error message: {str(e)}")
#                     raise  # Re-raise to trigger fallback
            
#             avg_loss = total_loss / len(dataloader)
#             print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
#         # Extract embeddings
#         word_embeddings = model.embeddings.weight.data.clone()
        
#         # Create word to vector mapping
#         word_to_vec = {}
#         for word, idx in text_processor.word_to_idx.items():
#             word_to_vec[word] = word_embeddings[idx].cpu().numpy()
        
#     except Exception as e:
#         print(f"CUDA Word2Vec training failed with error: {str(e)}")
#         print("Falling back to simplified embedding method...")
        
#         # Fallback to a simpler approach: just create random embeddings
#         # This is a placeholder - in a real implementation, you would use gensim or another library
#         word_to_vec = {}
#         for word in text_processor.word_to_idx:
#             # Create a random vector, but make it consistent for the same word
#             # Use word hash as seed for reproducibility
#             seed = hash(word) % 10000
#             np.random.seed(seed)
#             word_to_vec[word] = np.random.randn(word2vec_dim).astype(np.float32)
#             # Normalize to unit length
#             word_to_vec[word] = word_to_vec[word] / np.linalg.norm(word_to_vec[word])
    
#     data['word_to_vec'] = word_to_vec
    
#     # Save the embeddings
#     with open(os.path.join(output_path, 'word_to_vec.pkl'), 'wb') as f:
#         pickle.dump(word_to_vec, f)
    
#     print(f"Word vector generation completed in {time.time() - start_time:.2f}s")
#     print(f"Created embeddings for {len(word_to_vec)} words")
    
#     # Show some example vectors
#     print("\nExample word vectors:")
#     sample_words = list(word_to_vec.keys())[:5]
#     for word in sample_words:
#         vector = word_to_vec[word]
#         print(f"'{word}': {vector[:5]}...")
    
#     return data

# # STEP 5: Movie Vector Generation with CUDA
# def generate_movie_vectors_cuda(data):
#     """Generate movie feature vectors using Log-Likelihood and Word2Vec with CUDA"""
#     print("\n" + "="*80)
#     print("STEP 5: MOVIE VECTOR GENERATION WITH CUDA")
#     print("="*80)
    
#     start_time = time.time()
#     movie_ll_values = data['movie_ll_values']
#     word_to_vec = data['word_to_vec']
#     movie_features = data['movie_features']
    
#     # Prepare for GPU batch processing
#     total_movies = len(movie_ll_values)
#     batch_size = 100
#     num_batches = (total_movies + batch_size - 1) // batch_size
    
#     # Create empty tensor to store movie vectors
#     movie_vectors = torch.zeros((len(data['movie_id_mapping']), word2vec_dim), dtype=torch.float32, device=device)
    
#     # Track statistics
#     successful_vectors = 0
#     no_words_found = 0
#     low_ll_sum = 0
    
#     for batch_idx in range(num_batches):
#         # Get batch of movie IDs
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, total_movies)
        
#         # Get movie IDs for this batch
#         batch_movie_ids = list(movie_ll_values.keys())[start_idx:end_idx]
        
#         # Process each movie in the batch
#         for movie_id in batch_movie_ids:
#             # Get movie's LL values
#             ll_values = movie_ll_values[movie_id]
            
#             # Sort words by LL value and select top 200
#             top_words = sorted(ll_values.items(), key=lambda x: x[1], reverse=True)[:200]
            
#             if not top_words:
#                 no_words_found += 1
#                 continue
            
#             # Prepare tensors for weighted sum
#             word_vectors = []
#             word_weights = []
            
#             for word, ll_value in top_words:
#                 if ll_value <= 0:
#                     continue
                
#                 if word in word_to_vec:
#                     word_vectors.append(torch.tensor(word_to_vec[word], dtype=torch.float32, device=device))
#                     word_weights.append(ll_value)
            
#             if word_vectors and sum(word_weights) > 0:
#                 # Stack vectors and convert weights to tensor
#                 word_vectors_tensor = torch.stack(word_vectors)
#                 weights_tensor = torch.tensor(word_weights, dtype=torch.float32, device=device)
                
#                 # Calculate weighted sum
#                 weighted_sum = torch.sum(word_vectors_tensor * weights_tensor.unsqueeze(1), dim=0)
#                 weighted_sum /= weights_tensor.sum()
                
#                 # Normalize to unit length
#                 weighted_sum = F.normalize(weighted_sum, p=2, dim=0)
                
#                 # Store in movie vectors tensor
#                 movie_idx = data['movie_id_mapping'].get(movie_id)
#                 if movie_idx is not None:
#                     movie_vectors[movie_idx] = weighted_sum
#                     successful_vectors += 1
#             else:
#                 low_ll_sum += 1
        
#         print(f"Processed batch {batch_idx+1}/{num_batches}: {end_idx}/{total_movies} movies")
    
#     # Store the movie vectors
#     data['movie_vectors_tensor'] = movie_vectors
    
#     # Convert to dictionary for compatibility with other functions
#     movie_vectors_dict = {}
#     for movie_id, idx in data['movie_id_mapping'].items():
#         movie_vectors_dict[movie_id] = movie_vectors[idx].cpu().numpy()
    
#     data['movie_vectors'] = movie_vectors_dict
    
#     # Save movie vectors
#     torch.save(movie_vectors, os.path.join(output_path, 'movie_vectors_tensor.pt'))
#     with open(os.path.join(output_path, 'movie_vectors.pkl'), 'wb') as f:
#         pickle.dump(movie_vectors_dict, f)
    
#     print(f"Generated movie vectors in {time.time() - start_time:.2f}s")
#     print(f"Successfully created vectors: {successful_vectors}/{total_movies} ({successful_vectors/total_movies*100:.1f}%)")
#     print(f"Movies with no words found: {no_words_found}")
#     print(f"Movies with too low LL sum: {low_ll_sum}")
    
#     # Display sample movie vectors
#     if successful_vectors > 0:
#         print("\nSample movie vectors:")
#         for movie_id in list(movie_vectors_dict.keys())[:3]:
#             movie_title = movie_features[movie_features['movieId'] == movie_id]['title'].values[0]
#             vector = movie_vectors_dict[movie_id]
#             print(f"Movie: '{movie_title}' (ID: {movie_id})")
#             print(f"Vector shape: {vector.shape}")
#             print(f"Vector norm: {np.linalg.norm(vector):.4f}")
#             print(f"First 5 dimensions: {vector[:5]}")
#             print("---")
    
#     return data

# # STEP 6: User Vector Generation with CUDA
# def generate_user_vectors_cuda(data):
#     """Generate user feature vectors based on rated movies with CUDA"""
#     print("\n" + "="*80)
#     print("STEP 6: USER VECTOR GENERATION WITH CUDA")
#     print("="*80)
    
#     start_time = time.time()
    
#     # Get required data
#     movie_vectors_tensor = data['movie_vectors_tensor']
#     train_users_idx = data['train_users_idx']
#     train_movies_idx = data['train_movies_idx']
#     train_ratings = data['train_ratings']
#     train_norm_ratings = data['train_norm_ratings']
    
#     # Get unique users
#     unique_users = torch.unique(train_users_idx)
#     num_users = len(unique_users)
    
#     # Create user vectors tensor
#     user_vectors = torch.zeros((len(data['user_id_mapping']), word2vec_dim), dtype=torch.float32, device=device)
    
#     # Track statistics
#     successful_vectors = 0
#     users_with_no_movies = 0
#     users_with_zero_weights = 0
    
#     # Process users in batches
#     batch_size = 100
#     num_batches = (num_users + batch_size - 1) // batch_size
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, num_users)
        
#         # Get batch of user indices
#         batch_users = unique_users[start_idx:end_idx]
        
#         for user_idx in batch_users:
#             # Find all ratings for this user
#             user_mask = (train_users_idx == user_idx)
#             user_movie_indices = train_movies_idx[user_mask]
#             user_ratings = train_ratings[user_mask]
#             user_norm_ratings = train_norm_ratings[user_mask]
            
#             if len(user_movie_indices) == 0:
#                 users_with_no_movies += 1
#                 continue
            
#             # Convert ratings to preference weights
#             # For normalized ratings [0,1], convert to [-0.5,0.5]
#             weights = user_norm_ratings - 0.5
            
#             # Skip movies with zero weight
#             nonzero_mask = (weights != 0)
#             if torch.sum(nonzero_mask) == 0:
#                 users_with_zero_weights += 1
#                 continue
            
#             user_movie_indices = user_movie_indices[nonzero_mask]
#             weights = weights[nonzero_mask]
            
#             # Get movie vectors
#             movie_vecs = movie_vectors_tensor[user_movie_indices]
            
#             # Calculate weighted average
#             weighted_sum = torch.sum(movie_vecs * weights.unsqueeze(1), dim=0)
#             total_weight = torch.sum(torch.abs(weights))
            
#             if total_weight > 0:
#                 user_vec = weighted_sum / total_weight
                
#                 # Normalize to unit length
#                 user_vec = F.normalize(user_vec, p=2, dim=0)
                
#                 # Store in user vectors tensor
#                 user_vectors[user_idx] = user_vec
#                 successful_vectors += 1
        
#         print(f"Processed batch {batch_idx+1}/{num_batches}: {end_idx}/{num_users} users")
    
#     # Store user vectors
#     data['user_vectors_tensor'] = user_vectors
    
#     # Convert to dictionary for compatibility
#     user_vectors_dict = {}
#     for user_id, idx in data['user_id_mapping'].items():
#         user_vectors_dict[user_id] = user_vectors[idx].cpu().numpy()
    
#     data['user_vectors'] = user_vectors_dict
    
#     # Save user vectors
#     torch.save(user_vectors, os.path.join(output_path, 'user_vectors_tensor.pt'))
#     with open(os.path.join(output_path, 'user_vectors.pkl'), 'wb') as f:
#         pickle.dump(user_vectors_dict, f)
    
#     print(f"Generated user vectors in {time.time() - start_time:.2f}s")
#     print(f"Successfully created vectors: {successful_vectors}/{num_users} ({successful_vectors/num_users*100:.1f}%)")
#     print(f"Users with no rated movies: {users_with_no_movies}")
#     print(f"Users with zero weights: {users_with_zero_weights}")
    
#     # Display sample user vectors
#     if successful_vectors > 0:
#         print("\nSample user vectors:")
#         sample_users = list(user_vectors_dict.keys())[:3]
#         for user_id in sample_users:
#             vector = user_vectors_dict[user_id]
#             user_idx = data['user_id_mapping'][user_id]
#             num_ratings = torch.sum(train_users_idx == user_idx).item()
            
#             print(f"User ID: {user_id}")
#             print(f"Number of ratings: {num_ratings}")
#             print(f"Vector shape: {vector.shape}")
#             print(f"Vector norm: {np.linalg.norm(vector):.4f}")
#             print(f"First 5 dimensions: {vector[:5]}")
#             print("---")
    
#     return data

# STEP 7: User-Movie Similarity Calculation with CUDA
def calculate_similarity_cuda(user_vectors, movie_vectors, threshold=0.3, batch_size=1000):
    """Calculate similarity between users and movies using CUDA acceleration with optimized batching"""
    print(f"Calculating user-movie similarity with threshold {threshold} using optimized CUDA implementation...")
    start_time = time.time()
    
    # Convert dictionaries to arrays for batch processing
    user_ids = list(user_vectors.keys())
    movie_ids = list(movie_vectors.keys())
    
    # Create arrays of vectors
    user_matrix = np.array([user_vectors[uid] for uid in user_ids])
    movie_matrix = np.array([movie_vectors[mid] for mid in movie_ids])
    
    # Convert to tensors
    user_tensor = torch.tensor(user_matrix, dtype=torch.float32)
    movie_tensor = torch.tensor(movie_matrix, dtype=torch.float32).t()  # Transpose for batch matrix multiply
    
    if cuda_available:
        user_tensor = user_tensor.to(device)
        movie_tensor = movie_tensor.to(device)
    
    total_users = len(user_ids)
    user_movie_similarities = {}
    total_similarities = total_users * len(movie_ids)
    similarities_above_threshold = 0
    
    # Process users in batches
    for i in range(0, total_users, batch_size):
        batch_end = min(i + batch_size, total_users)
        batch_size_actual = batch_end - i
        
        # Get batch of user vectors
        batch_user_tensor = user_tensor[i:batch_end]
        
        # Calculate similarities for all movies in one operation
        # Shape: [batch_size, movie_count]
        similarity_matrix = torch.mm(batch_user_tensor, movie_tensor)
        
        # Apply threshold on GPU to reduce transfer size
        if cuda_available:
            # Create mask for values above threshold
            mask = similarity_matrix > threshold
            # Count similarities above threshold
            batch_above_threshold = torch.sum(mask).item()
        else:
            # Move to CPU for thresholding if needed
            similarity_matrix_cpu = similarity_matrix.cpu()
            mask = similarity_matrix_cpu > threshold
            batch_above_threshold = torch.sum(mask).item()
        
        similarities_above_threshold += batch_above_threshold
        
        # Process each user in the batch
        for j in range(batch_size_actual):
            user_id = user_ids[i + j]
            
            # Get similarities for this user and apply threshold
            if cuda_available:
                user_sims = similarity_matrix[j].cpu().numpy()
            else:
                user_sims = similarity_matrix[j].numpy()
            
            # Create dictionary of similarities above threshold
            user_dict = {}
            for k, movie_id in enumerate(movie_ids):
                sim_value = user_sims[k]
                if sim_value > threshold:
                    user_dict[movie_id] = float(sim_value)
            
            user_movie_similarities[user_id] = user_dict
        
        # Log progress
        processed = batch_end
        elapsed = time.time() - start_time
        remaining = (elapsed / processed) * (total_users - processed) if processed < total_users else 0
        print(f"Processed {processed}/{total_users} users ({processed/total_users*100:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
    
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
                    if 'movie_features' in data:
                        movie_title = data['movie_features'][data['movie_features']['movieId'] == movie_id]['title'].values[0]
                        print(f"  '{movie_title}' (ID: {movie_id}): {sim:.4f}")
                    else:
                        print(f"  Movie ID {movie_id}: {sim:.4f}")
            print("---")
    
    return user_movie_similarities

# STEP 8: Generate Recommendations with CUDA
def generate_recommendations_cuda(data, n=10):
    """Generate recommendations for all users using CUDA acceleration"""
    print("\n" + "="*80)
    print("STEP 8: RECOMMENDATION GENERATION WITH CUDA")
    print("="*80)
    
    start_time = time.time()
    
    # Get similarity data
    user_movie_similarities = data['user_movie_similarities']
    
    # Get training ratings to avoid recommending already rated movies
    train_users_idx = data['train_users_idx']
    train_movies_idx = data['train_movies_idx']
    
    # Create user to movies mapping for fast lookups
    user_rated_movies = defaultdict(set)
    for user_idx, movie_idx in zip(train_users_idx.cpu().numpy(), train_movies_idx.cpu().numpy()):
        user_id = data['idx_to_user_id'][user_idx]
        movie_id = data['idx_to_movie_id'][movie_idx]
        user_rated_movies[user_id].add(movie_id)
    
    # Generate recommendations for all users
    all_recommendations = {}
    users_with_recommendations = 0
    total_recommendations = 0
    
    # Process users in batches
    users = list(user_movie_similarities.keys())
    total_users = len(users)
    batch_size = 1000
    num_batches = (total_users + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_users)
        
        # Get batch of users
        batch_users = users[start_idx:end_idx]
        batch_recommendations = {}
        
        for user_id in batch_users:
            # Get movies already rated by the user
            rated_movies = user_rated_movies.get(user_id, set())
            
            # Get user's similarities
            user_sims = user_movie_similarities.get(user_id, {})
            
            if not user_sims:
                continue
            
            # Filter out already rated movies
            candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                         if movie_id not in rated_movies]
            
            if not candidates:
                continue
            
            # Sort by similarity (descending)
            recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)[:n]
            
            if recommendations:
                batch_recommendations[user_id] = recommendations
                users_with_recommendations += 1
                total_recommendations += len(recommendations)
        
        # Update main recommendations
        all_recommendations.update(batch_recommendations)
        
        print(f"Processed batch {batch_idx+1}/{num_batches}: {end_idx}/{total_users} users")
    
    # Store recommendations
    data['all_recommendations'] = all_recommendations
    
    # Save to disk
    with open(os.path.join(output_path, 'content_based_recommendations.pkl'), 'wb') as f:
        pickle.dump(all_recommendations, f)
    
    # Also save in a more readable CSV format
    recommendations_list = []
    movie_features = data['movie_features']
    
    for user_id, recs in all_recommendations.items():
        for rank, (movie_id, score) in enumerate(recs, 1):
            movie_title = "Unknown"
            movie_row = movie_features[movie_features['movieId'] == movie_id]
            if not movie_row.empty:
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
    
    avg_recommendations = total_recommendations / users_with_recommendations if users_with_recommendations > 0 else 0
    
    print(f"Generated recommendations in {time.time() - start_time:.2f}s")
    print(f"Users with recommendations: {users_with_recommendations}/{total_users} ({users_with_recommendations/total_users*100:.1f}%)")
    print(f"Total recommendations generated: {total_recommendations}")
    print(f"Average recommendations per user: {avg_recommendations:.2f}")
    
    # Display sample recommendations
    if all_recommendations:
        print("\nSample recommendations for 3 users:")
        sample_users = list(all_recommendations.keys())[:3]
        
        for user_id in sample_users:
            print(f"User ID: {user_id}")
            print("Top 5 recommended movies:")
            
            for rank, (movie_id, score) in enumerate(all_recommendations[user_id][:5], 1):
                movie_title = "Unknown"
                movie_row = movie_features[movie_features['movieId'] == movie_id]
                if not movie_row.empty:
                    movie_title = movie_row.iloc[0]['title']
                print(f"  {rank}. '{movie_title}' (ID: {movie_id}): {score:.4f}")
            print("---")
    
    return data

# STEP 9: Evaluate Model with CUDA
def evaluate_model_cuda(data):
    """Evaluate the model using RMSE and MAE metrics with CUDA acceleration"""
    print("\n" + "="*80)
    print("STEP 9: MODEL EVALUATION WITH CUDA")
    print("="*80)
    
    start_time = time.time()
    
    # Get required data
    user_movie_similarities = data['user_movie_similarities']
    test_users_idx = data['test_users_idx']
    test_movies_idx = data['test_movies_idx']
    test_ratings = data['test_ratings']
    
    # Get unique users in test set
    unique_test_users = torch.unique(test_users_idx)
    num_test_users = len(unique_test_users)
    
    print(f"Evaluating model on {num_test_users} users in test set...")
    
    # Initialize arrays for predictions and actual ratings
    all_predictions = []
    all_true_ratings = []
    users_evaluated = 0
    
    # Track per-user metrics
    user_metrics = {}
    
    # Process users in batches
    batch_size = 100
    num_batches = (num_test_users + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_test_users)
        
        # Get batch of users
        batch_users = unique_test_users[start_idx:end_idx]
        
        for user_idx in batch_users:
            user_id = data['idx_to_user_id'][user_idx]
            
            # Skip users without similarity data
            if user_id not in user_movie_similarities:
                continue
            
            # Get user's test ratings
            user_mask = (test_users_idx == user_idx)
            user_movie_indices = test_movies_idx[user_mask]
            user_true_ratings = test_ratings[user_mask]
            
            if len(user_movie_indices) == 0:
                continue
            
            # Get user's similarity data
            user_sims = user_movie_similarities[user_id]
            
            # Calculate predictions for each movie
            user_predictions = []
            user_actuals = []
            
            for i, movie_idx in enumerate(user_movie_indices.cpu().numpy()):
                movie_id = data['idx_to_movie_id'][movie_idx]
                actual_rating = user_true_ratings[i].item()
                
                # If movie in user's similarities, use similarity score to predict
                if movie_id in user_sims:
                    sim_score = user_sims[movie_id]
                    # Convert similarity [0,1] to rating [0.5,5]
                    pred_rating = 0.5 + 4.5 * sim_score
                else:
                    # If not found, use average rating (3.0)
                    pred_rating = 3.0
                
                user_predictions.append(pred_rating)
                user_actuals.append(actual_rating)
            
            if user_predictions:
                # Convert to tensors for efficient computation
                user_pred_tensor = torch.tensor(user_predictions, dtype=torch.float32, device=device)
                user_true_tensor = torch.tensor(user_actuals, dtype=torch.float32, device=device)
                
                # Calculate metrics
                user_mse = torch.mean((user_pred_tensor - user_true_tensor) ** 2)
                user_rmse = torch.sqrt(user_mse)
                user_mae = torch.mean(torch.abs(user_pred_tensor - user_true_tensor))
                
                # Store user metrics
                user_metrics[user_id] = {
                    'rmse': user_rmse.item(),
                    'mae': user_mae.item(),
                    'num_predictions': len(user_predictions)
                }
                
                # Add to overall metrics
                all_predictions.extend(user_predictions)
                all_true_ratings.extend(user_actuals)
                users_evaluated += 1
        
        print(f"Processed batch {batch_idx+1}/{num_batches}: {end_idx}/{num_test_users} users")
    
    # Calculate overall metrics
    if all_predictions:
        pred_tensor = torch.tensor(all_predictions, dtype=torch.float32, device=device)
        true_tensor = torch.tensor(all_true_ratings, dtype=torch.float32, device=device)
        
        mse = torch.mean((pred_tensor - true_tensor) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(pred_tensor - true_tensor))
        
        # Convert to Python floats
        rmse_value = rmse.item()
        mae_value = mae.item()
    else:
        rmse_value = 0.0
        mae_value = 0.0
    
    # Create evaluation metrics
    evaluation_metrics = {
        'rmse': rmse_value,
        'mae': mae_value,
        'num_users_evaluated': users_evaluated,
        'num_predictions': len(all_predictions)
    }
    
    # Store metrics
    data['evaluation_metrics'] = evaluation_metrics
    data['user_metrics'] = user_metrics
    
    # Save to disk
    with open(os.path.join(output_path, 'evaluation_metrics.pkl'), 'wb') as f:
        pickle.dump(evaluation_metrics, f)
    
    with open(os.path.join(output_path, 'user_metrics.pkl'), 'wb') as f:
        pickle.dump(user_metrics, f)
    
    # Also save as CSV
    metrics_df = pd.DataFrame([evaluation_metrics])
    metrics_df.to_csv(os.path.join(output_path, 'evaluation_metrics.csv'), index=False)
    
    user_metrics_df = pd.DataFrame.from_dict(user_metrics, orient='index')
    user_metrics_df.reset_index(inplace=True)
    user_metrics_df.rename(columns={'index': 'userId'}, inplace=True)
    user_metrics_df.to_csv(os.path.join(output_path, 'user_metrics.csv'), index=False)
    
    print(f"Evaluation completed in {time.time() - start_time:.2f}s")
    print(f"Users evaluated: {users_evaluated}/{num_test_users} ({users_evaluated/num_test_users*100:.1f}%)")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"RMSE: {rmse_value:.4f}")
    print(f"MAE: {mae_value:.4f}")
    
    return data

# Main function to run the pipeline
def main():
    start_time = time.time()
    
    # Step 1: Load data
    data = load_data_cuda()
    if data is None:
        print("Error loading data. Exiting.")
        return
    
    # # Step 2: Build corpus
    # data = build_corpus_cuda(data)
    
    # # Step 3: Calculate Log-Likelihood
    # data = calculate_log_likelihood_cuda(data)
    
    # # Step 4: Train Word2Vec model
    # data = train_word2vec_cuda(data)
    
    # # Step 5: Generate movie vectors
    # data = generate_movie_vectors_cuda(data)
    
    # # Step 6: Generate user vectors
    # data = generate_user_vectors_cuda(data)
    
    # Step 7: Calculate similarities
    user_vector =torch.PyTorchFileReader('./cuda_optimized_recommendations/user_vectors_tensor.pt')
    movie_vector =torch.PyTorchFileReader('./cuda_optimized_recommendations/movie_vectors_tensor.pt')

    data = calculate_similarity_cuda(user_vector,movie_vector, threshold=similarity_threshold)
    
    # Step 8: Generate recommendations
    data = generate_recommendations_cuda(data, n=top_n)
    
    # Step 9: Evaluate model
    data = evaluate_model_cuda(data)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY OF CUDA-ACCELERATED RECOMMENDATION SYSTEM")
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
    
    # Performance metrics
    print("\nPerformance Metrics:")
    if 'evaluation_metrics' in data:
        metrics = data['evaluation_metrics']
        print(f"- RMSE: {metrics['rmse']:.4f}")
        print(f"- MAE: {metrics['mae']:.4f}")
        print(f"- Users evaluated: {metrics['num_users_evaluated']}")
        print(f"- Total predictions: {metrics['num_predictions']}")
    
    # CUDA performance
    if cuda_available:
        print("\nCUDA Performance:")
        print(f"- Device: {torch.cuda.get_device_name(0)}")
        print(f"- Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"- Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"- Utilization: {torch.cuda.utilization(0)}%")
    
    # Total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nCUDA-Accelerated Content-Based Recommendation System Successfully Implemented!")
    
    # Clean up CUDA memory
    if cuda_available:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()