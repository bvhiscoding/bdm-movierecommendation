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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "./"
output_path = "./rec/content-recommendations"
top_n = 20

if not os.path.exists(output_path):
    os.makedirs(output_path)

similarity_threshold = 0.3
word2vec_dim = 100

def load_data():
    data = {}
    
    movie_features_path = os.path.join(input_path, './processed/processed_movie_features.csv')
    if os.path.exists(movie_features_path):
        data['movie_features'] = pd.read_csv(movie_features_path)
        data['movie_features']['tokens'] = data['movie_features']['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
        data['movie_features']['top_keywords'] = data['movie_features']['top_keywords'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        sys.exit(1)
    
    ratings_path = os.path.join(input_path, './processed/normalized_ratings.csv')
    if os.path.exists(ratings_path):
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            chunks.append(chunk)
        data['ratings'] = pd.concat(chunks)
    else:
        sys.exit(1)
    
    if 'ratings' in data:
        if 'timestamp' in data['ratings'].columns:
            data['ratings'] = data['ratings'].sort_values('timestamp')
        
        user_groups = data['ratings'].groupby('userId')
        train_chunks = []
        test_chunks = []
        
        user_count = 0
        total_users = len(user_groups)
        
        for user_id, group in user_groups:
            n = len(group)
            split_idx = int(n * 0.8)
            train_chunks.append(group.iloc[:split_idx])
            test_chunks.append(group.iloc[split_idx:])
            
            user_count += 1
            if len(train_chunks) >= 1000 or user_count == total_users:
                gc.collect()
        
        all_user_ids = data['ratings']['userId'].unique()

        np.random.seed(42)
        np.random.shuffle(all_user_ids)

        split_idx = int(len(all_user_ids) * 0.8)
        train_users = all_user_ids[:split_idx]
        test_users = all_user_ids[split_idx:]

        data['train_ratings'] = data['ratings'][data['ratings']['userId'].isin(train_users)]
        data['test_ratings'] = data['ratings'][data['ratings']['userId'].isin(test_users)]
        
        del train_chunks, test_chunks
        gc.collect()
    
    return data

def calculate_log_likelihood(movie_features, corpus_word_counts, batch_size=100):
    start_time = time.time()
    
    total_corpus_size = sum(corpus_word_counts.values())
    
    movie_ll_values = {}
    
    total_movies = len(movie_features)
    
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
            
            movie_ll_values[movie_id] = {}
            
            for word, count in movie_word_counts.items():
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
                
                ll = 2 * ll
                movie_ll_values[movie_id][word] = ll
        
        elapsed = time.time() - start_time
        progress = batch_end / total_movies * 100
        remaining = elapsed / (batch_end - batch_start) * (total_movies - batch_end) if batch_end < total_movies else 0
        
        gc.collect()
    
    return movie_ll_values

def train_word2vec(movie_features, vector_size=100, batch_size=1000):
    start_time = time.time()
    
    tokenized_corpus = []
    total_movies = len(movie_features)
    
    for i in range(0, total_movies, batch_size):
        batch_end = min(i + batch_size, total_movies)
        batch = movie_features.iloc[i:batch_end]
        
        batch_tokens = list(batch['tokens'])
        tokenized_corpus.extend(batch_tokens)
    
    total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
    
    word2vec_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=10,
        min_count=3,
        workers=8,
        epochs=25,
        sg=1,
        hs=1,
        negative=20
    )
    
    del tokenized_corpus
    gc.collect()
    
    elapsed = time.time() - start_time
    
    return word2vec_model

def generate_movie_vectors(movie_ll_values, word2vec_model, movie_features, batch_size=100):
    start_time = time.time()
    
    movie_vectors = {}
    successful_vectors = 0
    no_words_found = 0
    low_ll_sum = 0
    
    movie_ids = list(movie_ll_values.keys())
    total_movies = len(movie_ids)
    
    for batch_start in range(0, total_movies, batch_size):
        batch_end = min(batch_start + batch_size, total_movies)
        batch_movie_ids = movie_ids[batch_start:batch_end]
        total_corpus_size = sum(corpus_word_counts.values())

        for movie_id in batch_movie_ids:
            ll_values = movie_ll_values[movie_id]
            idf = {word: math.log(total_corpus_size/(corpus_word_counts[word]+1)) 
                for word in ll_values.keys()}
            
            combined_scores = {word: ll * idf[word] 
                            for word, ll in ll_values.items()}
            
            top_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:500]            
            if not top_words:
                no_words_found += 1
                continue
            
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
                movie_vector = np.sum(weighted_vectors, axis=0) / ll_sum
                
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
        
        gc.collect()
    
    return movie_vectors

def generate_user_vectors(movie_vectors, train_ratings, batch_size=100):
    start_time = time.time()
    
    user_vectors = {}
    successful_vectors = 0
    no_ratings_found = 0
    no_vectors_for_movies = 0
    low_weight_sum = 0
    
    user_ratings_dict = {}
    
    for _, row in train_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        if user_id not in user_ratings_dict:
            user_ratings_dict[user_id] = {}
        
        user_ratings_dict[user_id][movie_id] = rating
    
    user_ids = list(user_ratings_dict.keys())
    total_users = len(user_ids)
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        for user_id in batch_user_ids:
            user_ratings = user_ratings_dict[user_id]
            
            if len(user_ratings) == 0:
                no_ratings_found += 1
                continue
            
            weighted_vectors = []
            weight_sum = 0
            movies_with_vectors = 0
            movies_without_vectors = 0
            
            for movie_id, rating in user_ratings.items():
                weight = rating - 3.0
                
                if movie_id not in movie_vectors:
                    movies_without_vectors += 1
                    continue
                else:
                    movies_with_vectors += 1
                
                if weight != 0:
                    weighted_vectors.append(movie_vectors[movie_id] * weight)
                    weight_sum += abs(weight)
            
            if weighted_vectors and weight_sum > 0:
                user_vector = np.sum(weighted_vectors, axis=0) / weight_sum
                
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
        
        gc.collect()
    
    del user_ratings_dict
    gc.collect()
    
    return user_vectors

def calculate_user_movie_similarity(user_vectors, movie_vectors, threshold=0.3, batch_size=50):
    start_time = time.time()
    
    user_movie_similarities = {}
    
    user_ids = list(user_vectors.keys())
    total_users = len(user_ids)
    total_movies = len(movie_vectors)
    
    for batch_start in range(0, total_users, batch_size):
        batch_end = min(batch_start + batch_size, total_users)
        batch_user_ids = user_ids[batch_start:batch_end]
        
        for user_id in batch_user_ids:
            user_vector = user_vectors[user_id]
            user_sims = {}
            user_similarities = 0
            user_above_threshold = 0
            
            user_vector_array = np.array(user_vector).reshape(1, -1)
            
            movie_ids = list(movie_vectors.keys())
            movie_chunk_size = 1000
            
            for movie_chunk_start in range(0, len(movie_ids), movie_chunk_size):
                movie_chunk_end = min(movie_chunk_start + movie_chunk_size, len(movie_ids))
                chunk_movie_ids = movie_ids[movie_chunk_start:movie_chunk_end]
                
                movie_vectors_array = np.array([movie_vectors[mid] for mid in chunk_movie_ids])
                
                similarities = np.dot(user_vector_array, movie_vectors_array.T)[0]
                
                for i, sim in enumerate(similarities):
                    if sim > threshold:
                        movie_id = chunk_movie_ids[i]
                        user_sims[movie_id] = float(sim)
                        user_above_threshold += 1
                    user_similarities += 1
            
            user_movie_similarities[user_id] = user_sims
        
        elapsed = time.time() - start_time
        progress = batch_end / total_users * 100
        remaining = (elapsed / (batch_end - batch_start)) * (total_users - batch_end) if batch_end < total_users else 0
        
        gc.collect()
    
    return user_movie_similarities

def get_user_rated_movies(user_id, train_ratings, cached_rated_movies=None):
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
    if user_id not in user_movie_similarities:
        return []
    
    rated_movies = get_user_rated_movies(user_id, train_ratings, cached_rated_movies)
    
    user_sims = user_movie_similarities[user_id]
    
    candidates = [(movie_id, sim) for movie_id, sim in user_sims.items() 
                 if movie_id not in rated_movies]
    
    recommendations = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    return recommendations[:n]

def predict_rating(user_id, movie_id, user_movie_similarities, train_ratings):
    if user_id not in user_movie_similarities:
        return 3.0
    
    user_train = train_ratings[train_ratings['userId'] == user_id]
    user_avg_rating = user_train['rating'].mean() if len(user_train) > 0 else 3.0
    
    if movie_id not in user_movie_similarities[user_id]:
        return user_avg_rating
    
    sim_score = user_movie_similarities[user_id][movie_id]
    predicted_rating = 0.5 + 4.5 * sim_score
    
    return predicted_rating

def generate_recommendations_for_all_users(user_movie_similarities, train_ratings, movie_features, n=10, batch_size=100):
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
        
        gc.collect()
    
    return all_recommendations

def evaluate_with_rmse_mae(user_movie_similarities, train_ratings, test_ratings, batch_size=100):
    print("Evaluating recommendation model using RMSE and MAE with batching...")
    start_time = time.time()
    
    test_users = set(test_ratings['userId'].unique())
    train_users = set(train_ratings['userId'].unique())
    users_to_evaluate = test_users.intersection(train_users)
    
    print(f"Users in test set: {len(test_users)}")
    print(f"Users in training set: {len(train_users)}")
    print(f"Users to evaluate (intersection): {len(users_to_evaluate)}")
    
    if len(users_to_evaluate) == 0:
        print("Using user-based split - no common users between train and test.")
        print("Evaluating using average rating for all predictions instead.")
        
        avg_rating = train_ratings['rating'].mean()
        
        total_predictions = len(test_ratings)
        
        squared_errors_sum = ((test_ratings['rating'] - avg_rating) ** 2).sum()
        absolute_errors_sum = (abs(test_ratings['rating'] - avg_rating)).sum()
        
        overall_rmse = np.sqrt(squared_errors_sum / total_predictions)
        overall_mae = absolute_errors_sum / total_predictions
        
        print("\nEvaluation results using average rating baseline:")
        print(f"Total predictions: {total_predictions}")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        
        metrics = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'num_predictions': total_predictions,
            'evaluation_method': 'average_rating_baseline'
        }
        
        return metrics
    
    users_with_similarity = set(user_movie_similarities.keys())
    users_in_test_with_similarity = users_to_evaluate.intersection(users_with_similarity)
    
    print(f"Users in test set with similarity data: {len(users_in_test_with_similarity)}/{len(users_to_evaluate)} ({len(users_in_test_with_similarity)/len(users_to_evaluate)*100:.1f}%)")
    
    squared_errors_sum = 0
    absolute_errors_sum = 0
    total_predictions = 0
    users_evaluated = 0
    
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
            user_test = test_ratings[test_ratings['userId'] == user_id]
            
            if len(user_test) == 0:
                continue
            
            user_train = train_ratings[train_ratings['userId'] == user_id]
            user_avg_rating = user_train['rating'].mean() if len(user_train) > 0 else 3.0
            
            for _, row in user_test.iterrows():
                movie_id = row['movieId']
                true_rating = row['rating']
                
                if movie_id in user_movie_similarities.get(user_id, {}):
                    sim_score = user_movie_similarities[user_id][movie_id]
                    predicted_rating = 0.5 + 4.5 * sim_score
                else:
                    predicted_rating = user_avg_rating
                
                squared_error = (predicted_rating - true_rating) ** 2
                absolute_error = abs(predicted_rating - true_rating)
                
                batch_squared_errors += squared_error
                batch_absolute_errors += absolute_error
                batch_predictions += 1
            
            users_evaluated += 1
        
        squared_errors_sum += batch_squared_errors
        absolute_errors_sum += batch_absolute_errors
        total_predictions += batch_predictions
        
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        progress = batch_end / len(user_list) * 100
        remaining = batch_time * ((len(user_list) - batch_end) / len(batch_users)) if batch_end < len(user_list) else 0
        
        if batch_predictions > 0:
            batch_rmse = np.sqrt(batch_squared_errors / batch_predictions)
            batch_mae = batch_absolute_errors / batch_predictions
            print(f"Batch metrics - RMSE: {batch_rmse:.4f}, MAE: {batch_mae:.4f}, Predictions: {batch_predictions}")
        
        print(f"Processed {batch_end}/{len(user_list)} users ({progress:.1f}%) - Elapsed: {elapsed:.2f}s - Est. remaining: {remaining:.2f}s")
        
        gc.collect()
    
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
    
    metrics = {
        'rmse': overall_rmse,
        'mae': overall_mae,
        'num_users_evaluated': users_evaluated,
        'num_predictions': total_predictions
    }
    
    return metrics

data = load_data()

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
    
    gc.collect()

if 'movie_features' in data and 'corpus_word_counts' in data:
    movie_ll_values = calculate_log_likelihood(data['movie_features'], data['corpus_word_counts'], batch_size=100)
    data['movie_ll_values'] = movie_ll_values
    
    with open(os.path.join(output_path, 'movie_ll_values.pkl'), 'wb') as f:
        pickle.dump(movie_ll_values, f)
    
    gc.collect()

if 'movie_features' in data:
    word2vec_model = train_word2vec(data['movie_features'], word2vec_dim)
    data['word2vec_model'] = word2vec_model
    
    word2vec_path = os.path.join(output_path, 'word2vec_model')
    word2vec_model.save(word2vec_path)
    
    gc.collect()

if 'word2vec_model' in data and 'movie_ll_values' in data:
    movie_vectors = generate_movie_vectors(
        data['movie_ll_values'], 
        data['word2vec_model'],
        data['movie_features'],
        batch_size=100
    )
    data['movie_vectors'] = movie_vectors
    
    with open(os.path.join(output_path, 'movie_vectors.pkl'), 'wb') as f:
        pickle.dump(movie_vectors, f)
    
    movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(movie_vectors.keys())}
    data['movie_id_to_idx'] = movie_id_to_idx
    
    with open(os.path.join(output_path, 'movie_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(movie_id_to_idx, f)
    
    gc.collect()

if 'movie_vectors' in data and 'train_ratings' in data:
    user_vectors = generate_user_vectors(data['movie_vectors'], data['train_ratings'], batch_size=100)
    data['user_vectors'] = user_vectors
    
    with open(os.path.join(output_path, 'user_vectors.pkl'), 'wb') as f:
        pickle.dump(user_vectors, f)
    
    user_id_to_idx = {user_id: i for i, user_id in enumerate(user_vectors.keys())}
    data['user_id_to_idx'] = user_id_to_idx
    
    with open(os.path.join(output_path, 'user_id_to_idx.pkl'), 'wb') as f:
        pickle.dump(user_id_to_idx, f)
    
    gc.collect()

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
    
    gc.collect()

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
            
            if chunk_start == 0:
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='w')
            else:
                chunk_df.to_csv(os.path.join(output_path, 'content_based_recommendations.csv'), index=False, mode='a', header=False)
    
    del recommendations_list
    gc.collect()

if 'user_movie_similarities' in data and 'train_ratings' in data and 'test_ratings' in data:
    print("Running evaluation with RMSE and MAE metrics...")
    evaluation_metrics = evaluate_with_rmse_mae(
        data['user_movie_similarities'],
        data['train_ratings'],
        data['test_ratings'],
        batch_size=100
    )
    
    data['evaluation_metrics'] = evaluation_metrics
    
    print("\nStored evaluation metrics:")
    for key, value in evaluation_metrics.items():
        print(f"  {key}: {value}")
    
    evaluation_results = pd.DataFrame([evaluation_metrics])
    evaluation_results.to_csv(os.path.join(output_path, 'content_based_evaluation.csv'), index=False)
    
    if 'user_metrics' in data:
        user_metrics_df = pd.DataFrame.from_dict(data['user_metrics'], orient='index')
        user_metrics_df.reset_index(inplace=True)
        user_metrics_df.rename(columns={'index': 'userId'}, inplace=True)
        user_metrics_df.to_csv(os.path.join(output_path, 'user_metrics.csv'), index=False)
    
    gc.collect()
