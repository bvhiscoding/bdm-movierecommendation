# extract_countries.py

import pandas as pd
import re
import json
import ast
from collections import Counter

def extract_countries():
    """Extract an array of all unique countries from the movie dataset"""
    
    print("Loading movie data...")
    
    # Load TMDB data containing production countries
    try:
        tmdb_df = pd.read_csv('./extracted_data/extracted_tmdb.csv')
        print(f"Loaded {len(tmdb_df)} movies from TMDB dataset")
    except FileNotFoundError:
        print("Error: TMDB data file not found at './extracted_data/extracted_tmdb.csv'")
        return []
    
    # Function to extract countries from different formats
    def parse_countries(production_countries):
        if pd.isnull(production_countries) or production_countries == '':
            return []
        
        # Try different parsing methods
        try:
            # Check for JSON format with "name" field
            if isinstance(production_countries, str) and '"name"' in production_countries:
                countries = re.findall(r'"name":\s*"([^"]+)"', production_countries)
                if countries:
                    return countries
            
            # Try parsing as JSON or Python dict
            try:
                data = json.loads(production_countries)
                if isinstance(data, list):
                    return [item.get('name', '') for item in data if 'name' in item]
            except:
                try:
                    data = ast.literal_eval(production_countries)
                    if isinstance(data, list):
                        return [item.get('name', '') for item in data if 'name' in item]
                except:
                    pass
            
            # Try simple comma-separated string
            if isinstance(production_countries, str):
                countries = production_countries.replace('"', '').split(',')
                return [country.strip() for country in countries if country.strip()]
            
            return []
        except:
            return []
    
    print("Extracting countries...")
    
    # Extract countries from each movie
    all_countries = []
    for _, row in tmdb_df.iterrows():
        if 'production_countries' in row:
            countries = parse_countries(row['production_countries'])
            all_countries.extend(countries)
    
    # Get unique countries
    unique_countries = sorted(list(set([c for c in all_countries if c])))
    
    # Count frequency for each country
    country_counts = Counter(all_countries)
    
    # Print summary
    print(f"Found {len(unique_countries)} unique countries")
    print("\nMost common countries:")
    for country, count in country_counts.most_common(10):
        print(f"- {country}: {count} movies")
    
    # Save the country array to a file
    with open('./countries_array.json', 'w') as f:
        json.dump(unique_countries, f, indent=2)
    
    print(f"\nSaved array of {len(unique_countries)} countries to 'countries_array.json'")
    
    return unique_countries

if __name__ == "__main__":
    countries = extract_countries()
    print("\nCountry array:")
    print(countries)