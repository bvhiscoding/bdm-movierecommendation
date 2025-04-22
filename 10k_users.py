import pandas as pd
import os

# Function to create output directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set input and output paths
input_path = "./data/20M/"  # Adjust this to your MovieLens data location
output_path = "./extracted_data/"
ensure_dir(output_path)

print("Starting data extraction process...")

# Load ratings data
print("Loading ratings data...")
ratings_df = pd.read_csv(f"{input_path}rating.csv")

# Define user ID range to extract (1-10)
min_user_id = 1
max_user_id = 1000

print(f"Filtering for users with IDs from {min_user_id} to {max_user_id}...")

# Filter ratings for target users
filtered_ratings = ratings_df[
    (ratings_df['userId'] >= min_user_id) & 
    (ratings_df['userId'] <= max_user_id)
]

# Check if we have data after filtering
if filtered_ratings.empty:
    print("No users found in the specified ID range.")
    exit(1)

# Get the list of unique users that were actually found
found_users = filtered_ratings['userId'].unique()
print(f"Found {len(found_users)} users in the specified range: {found_users}")

# Get the unique movies rated by these users
movies_to_keep = filtered_ratings['movieId'].unique()
print(f"These users have rated {len(movies_to_keep)} unique movies")

# Load movie data
print("Loading movie data...")
movies_df = pd.read_csv(f"{input_path}26k_movies.csv")

# Filter movies to only those rated by our target users
filtered_movies = movies_df[movies_df['movieId'].isin(movies_to_keep)]
print(f"Filtered movies dataset now contains {len(filtered_movies)} movies")

# Optional: Load and filter tags data if available
try:
    tags_df = pd.read_csv(f"{input_path}tag.csv")
    filtered_tags = tags_df[
        (tags_df['userId'].isin(found_users)) & 
        (tags_df['movieId'].isin(movies_to_keep))
    ]
    print(f"Filtered tags dataset now contains {len(filtered_tags)} tag entries")
    # Save filtered tags
    filtered_tags.to_csv(f"{output_path}extracted_tags.csv", index=False)
    print(f"Saved filtered tags to {output_path}extracted_tags.csv")
except Exception as e:
    print(f"Could not process tags file: {str(e)}")

# Optional: Load and filter links data if available
try:
    links_df = pd.read_csv(f"{input_path}link.csv")
    filtered_links = links_df[links_df['movieId'].isin(movies_to_keep)]
    print(f"Filtered links dataset now contains {len(filtered_links)} link entries")
    # Save filtered links
    filtered_links.to_csv(f"{output_path}extracted_links.csv", index=False)
    print(f"Saved filtered links to {output_path}extracted_links.csv")
except Exception as e:
    print(f"Could not process links file: {str(e)}")

# Save filtered ratings and movies
filtered_ratings.to_csv(f"{output_path}extracted_ratings.csv", index=False)
filtered_movies.to_csv(f"{output_path}extracted_movies.csv", index=False)

print(f"Saved filtered ratings to {output_path}extracted_ratings.csv")
print(f"Saved filtered movies to {output_path}extracted_movies.csv")

print("Data extraction complete!")

# Print some statistics about the extracted data
print("\nExtraction Summary:")
print(f"Total Users: {len(found_users)}")
print(f"Total Movies: {len(filtered_movies)}")
print(f"Total Ratings: {len(filtered_ratings)}")
user_rating_counts = filtered_ratings.groupby('userId').size()
print(f"Average ratings per user: {user_rating_counts.mean():.2f}")
movie_rating_counts = filtered_ratings.groupby('movieId').size()
print(f"Average ratings per movie: {movie_rating_counts.mean():.2f}")