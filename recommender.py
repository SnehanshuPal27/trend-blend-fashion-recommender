import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging

# --- Global variables to hold loaded models and data ---
# This prevents reloading on every request, which is crucial for performance.
artifacts = {}

def load_artifacts():
    """
    Loads all necessary artifacts (models, data, index) into memory.
    This function should be called once when the Flask app starts.
    """
    global artifacts
    if not artifacts: # Only load if they haven't been loaded yet
        logging.info("Loading recommendation artifacts for the first time...")
        
        # Load the main data file
        with open("fashion_data_bert_faiss.pkl", "rb") as f:
            data = pickle.load(f)
            artifacts['products_df'] = data['data']

        # Load the Faiss index
        artifacts['faiss_index'] = faiss.read_index("fashion_faiss.index")

        # Load the trained LightGBM ranker model
        with open("ranker_model.pkl", "rb") as f:
            artifacts['ranker_model'] = pickle.load(f)

        # Load the SentenceTransformer model for encoding text
        artifacts['st_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        logging.info("✅ All artifacts loaded successfully.")

def get_recommendations(description: str, gender_filter: str = None, category_filter: str = None, num_results: int = 8):
    """
    Performs the full two-stage recommendation.
    
    Args:
        description (str): The user's text description of the desired product.
        gender_filter (str, optional): A gender to filter by.
        category_filter (str, optional): A master category to filter by.
        num_results (int, optional): The final number of recommendations to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a recommended product.
    """
    if not artifacts:
        load_artifacts()

    products_df = artifacts['products_df']
    st_model = artifacts['st_model']
    faiss_index = artifacts['faiss_index']
    ranker_model = artifacts['ranker_model']

    # --- STAGE 1: RETRIEVAL ---
    # Retrieve a large number of candidates for the ranker to process.
    num_candidates = 100
    
    logging.info(f"Retrieving {num_candidates} candidates for description: '{description}'")
    query_embedding = st_model.encode([description], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k=num_candidates)

    candidate_df = products_df.iloc[indices[0]].copy()
    candidate_df['retrieval_score'] = 1 - (distances[0]**2 / 2) # Convert L2 distance to cosine similarity

    # --- Filtering (Optional) ---
    if gender_filter:
        candidate_df = candidate_df[candidate_df['gender'].str.lower() == gender_filter.lower()]
    if category_filter:
        candidate_df = candidate_df[candidate_df['masterCategory'].str.lower() == category_filter.lower()]
    
    if candidate_df.empty:
        logging.warning("No candidates found after filtering.")
        return []

    # --- STAGE 2: RANKING ---
    logging.info(f"Ranking {len(candidate_df)} candidates...")
    
    # Prepare features for the ranker model
    features_for_ranking = candidate_df[['retrieval_score', 'price']]
    
    # Predict a new, more accurate score using the ranker
    ranking_scores = ranker_model.predict_proba(features_for_ranking)[:, 1]
    candidate_df['ranking_score'] = ranking_scores
    
    # Sort by the new ranking score and select the top results
    final_recommendations = candidate_df.sort_values('ranking_score', ascending=False).head(num_results)

    logging.info(f"Returning {len(final_recommendations)} final recommendations.")
    return final_recommendations.to_dict(orient='records')