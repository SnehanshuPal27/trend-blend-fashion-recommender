import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import json
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
   
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_product_categories(precomputed_file):
    
    with open(precomputed_file, 'rb') as f:
        data = pickle.load(f)
    
    products_df = data['data']
    return {
        'genders': sorted(products_df['gender'].unique().tolist()),
        'categories': sorted(products_df['masterCategory'].unique().tolist())
    }

def get_recommendations(user_vector, tfidf_matrix, products_df, indices, num_recommendations):
    
    if len(indices) == 0:
        return []
        
    filtered_tfidf = tfidf_matrix[indices]
    temp_knn = NearestNeighbors(n_neighbors=min(num_recommendations, len(indices)), 
                               algorithm='brute', metric='cosine')
    temp_knn.fit(filtered_tfidf)
    
    distances, neighbor_indices = temp_knn.kneighbors(user_vector)
    final_indices = [indices[i] for i in neighbor_indices[0]]
    
    recommendations = products_df.iloc[final_indices].copy()
    recommendations['similarity_score'] = 1 - distances[0]
    return recommendations

def recommend_products(user_description, gender_filter, category_filter, precomputed_file, output_file, num_recommendations=5):

    try:
        with open(precomputed_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Precomputed file {precomputed_file} not found. Please run precompute_fashion.py first.")
        return None

    products_df = data['data']

    
    products_df = products_df.dropna(subset=['gender', 'masterCategory'])
    if products_df.shape[0] < data['data'].shape[0]:
        logging.warning("Dropped rows with missing gender or masterCategory.")

    similarity_vectorizer = data['similarity_vectorizer']
    tfidf_matrix = data['tfidf_matrix']

    
    processed_description = preprocess_text(user_description)
    user_vector = similarity_vectorizer.transform([processed_description])

    all_recommendations = []

    
    if gender_filter or category_filter:
        filter_conditions = []
        if gender_filter:
            filter_conditions.append(f"gender == '{gender_filter}'")
        if category_filter:
            filter_conditions.append(f"masterCategory == '{category_filter}'")

        query = ' & '.join(filter_conditions)
        filtered_indices = products_df.query(query).index

        if len(filtered_indices) == 0:
            logging.warning("No products found for specified filters. Using the entire dataset for recommendations.")
            filtered_indices = products_df.index  

        recommendations = get_recommendations(user_vector, tfidf_matrix, products_df, filtered_indices, num_recommendations)
        if len(recommendations) > 0:
            all_recommendations.append(recommendations)
    else:
        
        logging.info("No filters applied. Using the entire dataset for recommendations.")
        all_indices = products_df.index
        recommendations = get_recommendations(user_vector, tfidf_matrix, products_df, all_indices, num_recommendations)
        if len(recommendations) > 0:
            all_recommendations.append(recommendations)

    if not all_recommendations:
        logging.warning("No recommendations found.")
        return []

    final_recommendations = pd.concat(all_recommendations)
    final_recommendations = final_recommendations.sort_values('similarity_score', ascending=False)
    final_recommendations = final_recommendations.drop_duplicates(subset=['id'])
    final_recommendations = final_recommendations.head(num_recommendations)

    recommendations_list = final_recommendations[[ 
        'id', 'name', 'description', 'brand', 'price', 
        'gender', 'masterCategory', 'similarity_score' 
    ]].to_dict(orient='records')

    with open(output_file, 'w') as f:
        json.dump(recommendations_list, f, indent=4)

    logging.info(f"Found {len(recommendations_list)} recommendations.")
    logging.info(f"Recommendations saved to {output_file}.")
    return recommendations_list
