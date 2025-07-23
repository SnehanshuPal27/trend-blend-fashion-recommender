import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
import os
from google.cloud import storage

# --- Configuration ---
BUCKET_NAME = 'fashion-recommender-models' # <--- IMPORTANT: CHANGE TO YOUR BUCKET NAME
MODEL_FILES = {
    "data": "fashion_data_bert_faiss.pkl",
    "index": "fashion_faiss.index",
    "ranker": "ranker_model.pkl"
}
LOCAL_MODEL_PATH = "/app/models" # A directory inside the container

# --- Global variables to hold loaded models and data ---
artifacts = {}

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    if os.path.exists(destination_file_name):
        logging.info(f"File {destination_file_name} already exists. Skipping download.")
        return
    
    logging.info(f"Downloading {source_blob_name} from bucket {bucket_name} to {destination_file_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} successfully.")


def load_artifacts():
    """
    Downloads models from GCS if they don't exist locally,
    then loads all artifacts into memory.
    """
    global artifacts
    if not artifacts:
        logging.info("Loading recommendation artifacts...")

        # Download all model files from GCS
        for key, filename in MODEL_FILES.items():
            download_blob(BUCKET_NAME, filename, os.path.join(LOCAL_MODEL_PATH, filename))

        # Load the main data file
        with open(os.path.join(LOCAL_MODEL_PATH, MODEL_FILES["data"]), "rb") as f:
            data = pickle.load(f)
            artifacts['products_df'] = data['data']

        # Load the Faiss index
        artifacts['faiss_index'] = faiss.read_index(os.path.join(LOCAL_MODEL_PATH, MODEL_FILES["index"]))

        # Load the trained LightGBM ranker model
        with open(os.path.join(LOCAL_MODEL_PATH, MODEL_FILES["ranker"]), "rb") as f:
            artifacts['ranker_model'] = pickle.load(f)

        # Load the SentenceTransformer model
        artifacts['st_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        logging.info("✅ All artifacts loaded successfully.")

# The rest of your recommender.py file remains the same...
def get_recommendations(description: str, gender_filter: str = None, category_filter: str = None, num_results: int = 8):
    # ... (no changes needed in this function)
    if not artifacts:
        load_artifacts()

    products_df = artifacts['products_df']
    st_model = artifacts['st_model']
    faiss_index = artifacts['faiss_index']
    ranker_model = artifacts['ranker_model']

    # --- STAGE 1: RETRIEVAL ---
    num_candidates = 100
    
    logging.info(f"Retrieving {num_candidates} candidates for description: '{description}'")
    query_embedding = st_model.encode([description], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k=num_candidates)

    candidate_df = products_df.iloc[indices[0]].copy()
    candidate_df['retrieval_score'] = 1 - (distances[0]**2 / 2)

    # --- Filtering ---
    if gender_filter:
        candidate_df = candidate_df[candidate_df['gender'].str.lower() == gender_filter.lower()]
    if category_filter:
        candidate_df = candidate_df[candidate_df['masterCategory'].str.lower() == category_filter.lower()]
    
    if candidate_df.empty:
        logging.warning("No candidates found after filtering.")
        return []

    # --- STAGE 2: RANKING ---
    logging.info(f"Ranking {len(candidate_df)} candidates...")
    
    features_for_ranking = candidate_df[['retrieval_score', 'price']]
    
    ranking_scores = ranker_model.predict_proba(features_for_ranking)[:, 1]
    candidate_df['ranking_score'] = ranking_scores
    
    final_recommendations = candidate_df.sort_values('ranking_score', ascending=False).head(num_results)

    logging.info(f"Returning {len(final_recommendations)} final recommendations.")
    return final_recommendations.to_dict(orient='records')
