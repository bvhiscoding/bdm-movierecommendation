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
import ast
import json
import os

# Create directory if it doesn't exist
os.makedirs('./processed', exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

print("Libraries imported successfully!")

# Define country to region mapping
def map_country_to_region(country):
    """
    Maps a country to its geographical region.
    Returns a list of regions the country belongs to.
    """
    # North America
    north_america = [
        "United States of America", "USA", "US", "Canada", "Mexico"
    ]
    
    # Europe
    europe = [
        "United Kingdom", "UK", "France", "Germany", "Italy", "Spain", 
        "Netherlands", "Belgium", "Switzerland", "Sweden", "Norway", 
        "Denmark", "Finland", "Ireland", "Austria", "Greece", "Portugal",
        "Poland", "Czech Republic", "Hungary", "Romania", "Bulgaria",
        "Croatia", "Slovakia", "Slovenia", "Serbia", "Ukraine", "Russia"
    ]
    
    # East Asia
    east_asia = [
        "Japan", "China", "South Korea", "North Korea", "Taiwan", "Hong Kong"
    ]
    
    # South Asia
    south_asia = [
        "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Bhutan"
    ]
    
    # Southeast Asia
    southeast_asia = [
        "Thailand", "Vietnam", "Indonesia", "Malaysia", "Philippines", 
        "Singapore", "Myanmar", "Cambodia", "Laos", "Brunei"
    ]
    
    # Oceania
    oceania = [
        "Australia", "New Zealand", "Fiji", "Papua New Guinea"
    ]
    
    # Middle East
    middle_east = [
        "Iran", "Iraq", "Israel", "Saudi Arabia", "Turkey", "United Arab Emirates",
        "Qatar", "Kuwait", "Lebanon", "Syria", "Jordan", "Oman", "Yemen"
    ]
    
    # Africa
    africa = [
        "South Africa", "Nigeria", "Egypt", "Morocco", "Kenya", "Tanzania",
        "Ethiopia", "Ghana", "Senegal", "Algeria", "Tunisia", "Uganda"
    ]
    
    # Latin America
    latin_america = [
        "Brazil", "Argentina", "Chile", "Colombia", "Peru", "Venezuela",
        "Bolivia", "Ecuador", "Uruguay", "Paraguay", "Cuba"
    ]
    
    regions = []
    
    if country in north_america:
        regions.append("North America")
    if country in europe:
        regions.append("Europe")
    if country in east_asia:
        regions.append("East Asia")
    if country in south_asia:
        regions.append("South Asia")
    if country in southeast_asia:
        regions.append("Southeast Asia")
    if country in oceania:
        regions.append("Oceania")
    if country in middle_east:
        regions.append("Middle East")
    if country in africa:
        regions.append("Africa")
    if country in latin_america:
        regions.append("Latin America")
    
    # Default to "Other" if no match
    if not regions:
        regions.append("Other")
    
    return regions

# Step 1: Movie Text Feature Extraction
print("\n" + "="*80)
print("STEP 1: MOVIE TEXT FEATURE EXTRACTION")
print("="*80)
# Load MovieLens movie data
movies_df = pd.read_csv('./extracted_data/extracted_movies.csv')
print(f"Loaded {len(movies_df)} movies from MovieLens dataset")
print(movies_df.head(3))

# Load TMDB data (containing movie overviews, cast, director)
tmdb_df = pd.read_csv('./extracted_data/extracted_tmdb.csv') 
print(f"\nLoaded {len(tmdb_df)} movies from TMDB dataset")
print(tmdb_df.head(3)[['id', 'tmdb_title', 'overview']])

# Load links data to connect MovieLens IDs with TMDB IDs
links_df = pd.read_csv('./extracted_data/extracted_links.csv')
print(f"\nLoaded {len(links_df)} movie links")
print(links_df.head(3))

# Load tags data for additional text information
tags_df = pd.read_csv('./extracted_data/extracted_tags.csv')
print(f"\nLoaded {len(tags_df)} movie tags")
print(tags_df.head(3))

# Merge movie data with TMDB data via links_df
movie_data = pd.merge(movies_df, links_df, on='movieId', how='left')
movie_data = pd.merge(movie_data, tmdb_df, left_on='tmdbId', right_on='id', how='left')

# Function to parse JSON-like strings in production_countries
def safe_parse_json_list(json_str, field_name=None):
    """Parse a JSON-like string into a list of items safely."""
    if pd.isnull(json_str) or json_str == '':
        return []
    
    try:
        # Try parsing as proper JSON
        items = json.loads(json_str)
        if field_name and isinstance(items, list):
            return [item.get(field_name, '') for item in items if field_name in item]
        return items
    except:
        try:
            # Try evaluating as Python literal
            items = ast.literal_eval(json_str)
            if field_name and isinstance(items, list):
                return [item.get(field_name, '') for item in items if field_name in item]
            return items
        except:
            # Fallback to regex extraction for production_countries format
            if field_name == 'name':
                matches = re.findall(r'"name":\s*"([^"]+)"', json_str)
                return matches
            return []

# Extract production countries and map to regions
if 'production_countries' in movie_data.columns:
    movie_data['countries'] = movie_data['production_countries'].apply(
        lambda x: safe_parse_json_list(x, 'name') if not pd.isnull(x) else []
    )
    
    # Create regions list for each movie
    movie_data['regions'] = movie_data['countries'].apply(
        lambda countries: [region for country in countries 
                          for region in map_country_to_region(country)]
    )
else:
    # If production_countries column doesn't exist, create empty lists
    movie_data['countries'] = [[] for _ in range(len(movie_data))]
    movie_data['regions'] = [[] for _ in range(len(movie_data))]

# Create text corpus for each movie
movie_data['text_corpus'] = ""

# Add title to corpus
movie_data['text_corpus'] += movie_data['title'].fillna("")

# Add TMDB overview to corpus
movie_data['text_corpus'] += " " + movie_data['overview'].fillna("")

# Add keywords to corpus with higher weight (repeat them to increase their importance)
if 'keywords' in movie_data.columns:
    # Extract keywords and add them with higher weight
    movie_data['keywords_extracted'] = movie_data['keywords'].apply(
        lambda x: ' '.join(safe_parse_json_list(x, 'name') * 3) if not pd.isnull(x) else ""
    )
    movie_data['text_corpus'] += " " + movie_data['keywords_extracted'].fillna("")

# Aggregate tags by movieId
tags_by_movie = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.fillna(''))).reset_index()

# Merge tags with movies
movie_data = pd.merge(movie_data, tags_by_movie, on='movieId', how='left')

# Add tags to corpus with higher weight
movie_data['text_corpus'] += " " + movie_data['tag'].fillna("") + " " + movie_data['tag'].fillna("")

# Add cast to corpus if available
if 'cast' in movie_data.columns:
    movie_data['cast_extracted'] = movie_data['cast'].apply(
        lambda x: ' '.join(safe_parse_json_list(x, 'name')) if not pd.isnull(x) else ""
    )
    movie_data['text_corpus'] += " " + movie_data['cast_extracted'].fillna("")

# Add director to corpus if available
if 'crew' in movie_data.columns:
    def extract_directors(crew_str):
        try:
            crew = safe_parse_json_list(crew_str)
            directors = [person.get('name', '') for person in crew 
                        if person.get('job', '').lower() == 'director']
            return ' '.join(directors)
        except:
            return ""
    
    movie_data['director'] = movie_data['crew'].apply(extract_directors)
    movie_data['text_corpus'] += " " + movie_data['director'].fillna("")

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

def preserve_full_names(text):
    """Preserve full names as single tokens by replacing spaces with underscores"""
    if not isinstance(text, str):
        return ""
    
    # Pattern to identify potential names (two or more capitalized words in sequence)
    # This will match names like "Tom Hanks", "Robert De Niro", etc.
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    
    # Find all matches
    matches = re.findall(name_pattern, text)
    
    # Replace spaces with underscores in the matched names
    for name in matches:
        text = text.replace(name, name.replace(' ', '_'))
    
    return text

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # First preserve full names
    text = preserve_full_names(text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep underscores (for preserved names)
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = re.sub(r'\d+', ' ', text)  # Remove digits
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Enhanced preprocessing with smarter stopword handling
def preprocess_text(text, stop_words, lemmatizer):
    """Tokenize, remove stopwords, and lemmatize text"""
    if not isinstance(text, str) or text == "":
        return []
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize text
    tokens = word_tokenize(cleaned_text)
    
    # Custom stopwords based on movie domain
    movie_specific_stopwords = {'movie', 'film', 'character', 'scene', 'trailer', 'cinema', 'watch'}
    enhanced_stop_words = stop_words.union(movie_specific_stopwords)
    
    # Keep important words even if they're in stopwords
    important_words = {'not', 'no', 'never', 'good', 'bad', 'great', 'best', 'worst', 'love', 'hate'}
    final_stop_words = enhanced_stop_words - important_words
    
    # Remove stopwords and short words, but keep tokens with underscores (names)
    tokens = [word for word in tokens if (word not in final_stop_words and len(word) > 1) or '_' in word]
    
    try:
        # Try lemmatizing tokens with POS tagging, but don't lemmatize names with underscores
        lemmatized_tokens = []
        for word in tokens:
            if '_' in word:
                # Don't lemmatize names, just replace underscores with spaces
                lemmatized_tokens.append(word.replace('_', ' '))
            else:
                # Get POS tag for regular words
                pos = pos_tag([word])
                lemmatized_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(pos[0][1])))
    except LookupError:
        # Fallback to simple lemmatization without POS tagging
        lemmatized_tokens = []
        for word in tokens:
            if '_' in word:
                lemmatized_tokens.append(word.replace('_', ' '))
            else:
                lemmatized_tokens.append(lemmatizer.lemmatize(word))
    
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
plt.savefig('./processed/token_distribution.png')
print("\nToken distribution plot saved as 'token_distribution.png'")
plt.close()

# Step 3: Data Normalization
print("\n" + "="*80)
print("STEP 3: DATA NORMALIZATION")
print("="*80)

# Load user ratings data
ratings_df = pd.read_csv('./extracted_data/extracted_ratings.csv')
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

# Enhanced normalization that preserves the relative scale while dealing with outliers
def normalize_ratings(ratings_df):
    """
    Normalize ratings using z-score normalization followed by min-max scaling to [0-1]
    This helps with users who have very skewed rating patterns
    """
    # Create a copy to avoid affecting the original data
    result_df = ratings_df.copy()
    
    # Group by userId to normalize per user
    user_groups = result_df.groupby('userId')
    
    normalized_dfs = []
    for user_id, group in user_groups:
        user_mean = group['rating'].mean()
        user_std = group['rating'].std()
        
        # Handle case where user gave the same rating to all movies
        if user_std == 0:
            normalized_rating = 0.5 if user_mean <= 3 else 0.8  # Default values based on mean
            group['normalized_rating'] = normalized_rating
        else:
            # Z-score normalization
            z_scores = (group['rating'] - user_mean) / user_std
            
            # Convert to 0-1 range using sigmoid function
            group['normalized_rating'] = 1 / (1 + np.exp(-z_scores))
            
            # For extremely negative z-scores, set a minimum value
            group.loc[group['normalized_rating'] < 0.01, 'normalized_rating'] = 0.01
            
            # For extremely positive z-scores, set a maximum value
            group.loc[group['normalized_rating'] > 0.99, 'normalized_rating'] = 0.99
            
            # Scale back to [0.5-5.0] for compatibility with existing models
            group['normalized_rating'] = 0.5 + (group['normalized_rating'] * 4.5)
        
        normalized_dfs.append(group)
    
    # Combine all normalized user groups
    result_df = pd.concat(normalized_dfs)
    
    return result_df[['userId', 'movieId', 'rating', 'normalized_rating']]

# Apply normalization
print("\nNormalizing ratings...")
normalized_ratings = normalize_ratings(ratings_df)

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
plt.savefig('./processed/rating_normalization.png')
print("\nRating normalization plot saved as 'rating_normalization.png'")
plt.close()

# Output: Normalized ratings data
print("\nNormalized ratings data:")
print(normalized_ratings.head())

# Step 4: Genre and Region Encoding (Binary Representation)
print("\n" + "="*80)
print("STEP 4: GENRE AND REGION ENCODING")
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

# Define country to region mapping function
def map_country_to_regions(country):
    """
    Maps a country to its geographical/cultural regions.
    Returns a list of regions this country belongs to.
    """
    # Define regions with their member countries
    north_america = ['United States of America', 'Canada', 'Mexico']
    
    latin_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 
                     'Cuba', 'Dominican Republic', 'Ecuador', 'Guatemala', 'Haiti', 
                     'Jamaica', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Puerto Rico', 
                     'Trinidad and Tobago', 'Uruguay', 'Venezuela']
                     
    caribbean = ['Aruba', 'Bahamas', 'Cayman Islands', 'Grenada', 'Guadaloupe', 
                 'Martinique', 'Netherlands Antilles', 'St. Pierre and Miquelon']
    
    western_europe = ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 
                      'Greece', 'Iceland', 'Ireland', 'Italy', 'Liechtenstein', 
                      'Luxembourg', 'Malta', 'Monaco', 'Netherlands', 'Norway', 
                      'Portugal', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']
    
    eastern_europe = ['Albania', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 
                      'Croatia', 'Czech Republic', 'Czechoslovakia', 'East Germany', 
                      'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Macedonia', 
                      'Montenegro', 'Poland', 'Romania', 'Russia', 'Serbia', 
                      'Serbia and Montenegro', 'Slovakia', 'Slovenia', 'Soviet Union', 
                      'Ukraine', 'Yugoslavia']
    
    middle_east = ['Cyprus', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 
                   'Palestinian Territory', 'Qatar', 'Saudi Arabia', 'Syrian Arab Republic', 
                   'Turkey', 'United Arab Emirates']
    
    north_africa = ['Algeria', 'Egypt', 'Libya', 'Libyan Arab Jamahiriya', 'Morocco', 'Tunisia']
    
    sub_saharan_africa = ['Angola', 'Botswana', 'Burkina Faso', 'Cameroon', 'Central African Republic', 
                          'Chad', 'Congo', "Cote D'Ivoire", 'Ethiopia', 'Ghana', 'Kenya', 'Liberia', 
                          'Mali', 'Mauritania', 'Namibia', 'Nigeria', 'Rwanda', 'Senegal', 'Somalia', 
                          'South Africa', 'Swaziland', 'Tanzania', 'Uganda', 'Zaire', 'Zimbabwe']
    
    east_asia = ['China', 'Hong Kong', 'Japan', 'Macao', 'Mongolia', 'North Korea', 
                 'South Korea', 'Taiwan']
    
    south_asia = ['Afghanistan', 'Bangladesh', 'Bhutan', 'British Indian Ocean Territory', 
                  'India', 'Nepal', 'Pakistan', 'Sri Lanka']
    
    southeast_asia = ['Cambodia', 'Indonesia', 'Lao People\'s Democratic Republic', 'Malaysia', 
                      'Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Vietnam']
    
    central_asia = ['Armenia', 'Georgia', 'Kazakhstan', 'Kyrgyz Republic', 'Tajikistan', 'Uzbekistan']
    
    oceania = ['Australia', 'French Polynesia', 'New Caledonia', 'New Zealand', 
               'Papua New Guinea', 'Solomon Islands']
    
    # Create a regions list for this country
    regions = []
    
    if country in north_america:
        regions.append('North America')
    if country in latin_america or country in caribbean:
        regions.append('Latin America/Caribbean')
    if country in western_europe:
        regions.append('Western Europe')
    if country in eastern_europe:
        regions.append('Eastern Europe')
    if country in middle_east:
        regions.append('Middle East')
    if country in north_africa:
        regions.append('North Africa')
    if country in sub_saharan_africa:
        regions.append('Sub-Saharan Africa')
    if country in east_asia:
        regions.append('East Asia')
    if country in south_asia:
        regions.append('South Asia')
    if country in southeast_asia:
        regions.append('Southeast Asia')
    if country in central_asia:
        regions.append('Central Asia')
    if country in oceania:
        regions.append('Oceania')
        
    # If no region was assigned, use "Other"
    if not regions:
        regions.append('Other')
        
    return regions

# Function to extract countries from production_countries field
def extract_countries(production_countries):
    """Extract country names from production_countries string"""
    if pd.isnull(production_countries) or production_countries == '':
        return []
    
    # Handle different formats
    countries = []
    
    # Format 1: JSON-like string with "name" field
    if isinstance(production_countries, str) and '"name":' in production_countries:
        matches = re.findall(r'"name":\s*"([^"]+)"', production_countries)
        if matches:
            countries.extend(matches)
            return countries
    
    # Format 2: Simple comma-separated string
    if isinstance(production_countries, str):
        try:
            # Try to parse as JSON first
            import json
            data = json.loads(production_countries)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        countries.append(item['name'])
                return countries
        except:
            # If not JSON, try comma-separated
            parts = production_countries.replace('"', '').split(',')
            countries = [part.strip() for part in parts if part.strip()]
            return countries
    
    return countries

# Define all regions we'll use
all_regions = ['North America', 'Latin America/Caribbean', 'Western Europe', 
               'Eastern Europe', 'Middle East', 'North Africa', 'Sub-Saharan Africa', 
               'East Asia', 'South Asia', 'Southeast Asia', 'Central Asia', 'Oceania', 'Other']

print(f"\nClassifying countries into {len(all_regions)} regions")

# Process the movie data
if 'production_countries' in movie_data.columns:
    # Extract countries from production_countries
    movie_data['extracted_countries'] = movie_data['production_countries'].apply(extract_countries)
    
    # Map countries to regions
    movie_data['regions'] = movie_data['extracted_countries'].apply(
        lambda countries: [region for country in countries for region in map_country_to_regions(country)]
    )
    
    # Remove duplicates from regions list
    movie_data['regions'] = movie_data['regions'].apply(lambda regions: list(set(regions)))
    
    # Create region data for one-hot encoding
    region_data = []
    for _, movie in movie_data.iterrows():
        movie_id = movie['movieId']
        regions = movie['regions'] if isinstance(movie['regions'], list) else []
        
        for region in regions:
            region_data.append({'movieId': movie_id, 'region': region})
    
    # Convert to DataFrame
    if region_data:
        region_df = pd.DataFrame(region_data)
        
        # Create pivot table for one-hot encoding
        region_one_hot = pd.pivot_table(
            region_df, 
            index='movieId', 
            columns='region', 
            aggfunc=lambda x: 1, 
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        region_one_hot.columns.name = None
        
        print("\nOne-hot encoded regions (sample):")
        print(region_one_hot.head())
        
        # Count occurrences of each region
        region_counts = region_df['region'].value_counts()
        
        # Print region counts
        print("\nMovies by region:")
        for region, count in region_counts.items():
            print(f"- {region}: {count} movies")
        
        # Plot region distribution
        plt.figure(figsize=(14, 7))
        plt.bar(region_counts.index, region_counts.values, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Movies by Region')
        plt.xlabel('Region')
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('./processed/region_distribution.png')
        print("\nRegion distribution plot saved as 'region_distribution.png'")
        plt.close()
    else:
        print("\nNo valid region data extracted")
        # Create empty region one-hot encoding with all regions
        region_one_hot = pd.DataFrame({'movieId': movie_data['movieId'].unique()})
        for region in all_regions:
            region_one_hot[region] = 0
else:
    print("\nNo 'production_countries' column found in movie data")
    # Create empty region one-hot encoding with all regions
    region_one_hot = pd.DataFrame({'movieId': movie_data['movieId'].unique()})
    for region in all_regions:
        region_one_hot[region] = 0

# Plot genre distribution
genre_counts = {}
for genre in all_genres:
    if genre in genre_one_hot.columns:
        genre_counts[genre] = genre_one_hot[genre].sum()

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
plt.savefig('./processed/genre_distribution.png')
print("\nGenre distribution plot saved as 'genre_distribution.png'")
plt.close()

# Merge genre and region features with movie data
movie_features = pd.merge(
    movies_df[['movieId', 'title']], 
    genre_one_hot, 
    on='movieId', 
    how='left'
)

# Merge with region features
movie_features = pd.merge(
    movie_features,
    region_one_hot,
    on='movieId',
    how='left'
)

# Fill NaN values with 0 for genres
for genre in all_genres:
    if genre in movie_features.columns:
        movie_features[genre] = movie_features[genre].fillna(0).astype(int)

# Fill NaN values with 0 for regions
for region in all_regions:
    if region in movie_features.columns:
        movie_features[region] = movie_features[region].fillna(0).astype(int)

print("\nFinal movie features with genre and region encoding (sample):")
print(movie_features.head())

# Print feature set statistics
genre_features = [col for col in movie_features.columns if col in all_genres]
region_features = [col for col in movie_features.columns if col in all_regions]

print(f"\nFeature set summary:")
print(f"- Total features: {len(movie_features.columns) - 2}")  # Subtract movieId and title
print(f"- Genre features: {len(genre_features)}")
print(f"- Region features: {len(region_features)}")

# Plot genre distribution
genre_counts = {}
for genre in all_genres:
    if genre in movie_features.columns:
        genre_counts[genre] = movie_features[genre].sum()

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
plt.savefig('./processed/genre_distribution.png')
print("\nGenre distribution plot saved as 'genre_distribution.png'")
plt.close()

# Plot region distribution if we have region data
region_counts = {}
for region in ['North America', 'Europe', 'East Asia', 'South Asia', 'Southeast Asia', 
               'Oceania', 'Middle East', 'Africa', 'Latin America', 'Other']:
    if region in movie_features.columns:
        region_counts[region] = movie_features[region].sum()

# Sort regions by count
sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(14, 7))
plt.bar([x[0] for x in sorted_regions], [x[1] for x in sorted_regions], color='lightgreen', edgecolor='black')
plt.title('Distribution of Movies by Region')
plt.xlabel('Region')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./processed/region_distribution.png')
print("\nRegion distribution plot saved as 'region_distribution.png'")
plt.close()

# Final Output: Combined Movie Features
print("\n" + "="*80)
print("FINAL OUTPUT: COMBINED MOVIE FEATURES")
print("="*80)

# Combine all features into one DataFrame
movie_features = pd.merge(
    movie_features,  # Contains movieId, title, and genre/region encodings
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
movie_features.to_csv('./processed/processed_movie_features.csv', index=False)
normalized_ratings.to_csv('./processed/normalized_ratings.csv', index=False)

print("\nProcessed data saved to 'processed_movie_features.csv' and 'normalized_ratings.csv'")

# Summary of the data processing pipeline
print("\n" + "="*80)
print("SUMMARY OF STAGE 1 DATA PROCESSING")
print("="*80)
print(f"1. Extracted text features for {len(movie_data)} movies")
print(f"2. Preprocessed text resulting in a vocabulary of {len(corpus_word_counts)} unique words")
print(f"3. Normalized {len(normalized_ratings)} ratings from {len(user_stats)} users")
print(f"4. Created one-hot encodings for {len(all_genres)} genres")
print(f"5. Added region classification based on production countries")
print(f"6. Final dataset contains {len(movie_features)} movies with complete feature sets")
print("="*80)

# Count the number of actor names preserved in the tokens
actor_name_count = 0
total_tokens = 0

for tokens in movie_features['tokens']:
    if isinstance(tokens, list):
        for token in tokens:
            total_tokens += 1
            if ' ' in token:  # Tokens with spaces are preserved actor names
                actor_name_count += 1

actor_name_percentage = (actor_name_count / total_tokens) * 100 if total_tokens > 0 else 0
print(f"\nActor name statistics:")
print(f"- Total tokens: {total_tokens}")
print(f"- Actor name tokens: {actor_name_count} ({actor_name_percentage:.2f}%)")
print("="*80)