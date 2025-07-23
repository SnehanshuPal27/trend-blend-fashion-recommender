import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import json
from pathlib import Path
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor

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

def process_json_file(file_path):
  
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        product = data.get('data', {})
        
        if 'description' not in product.get('productDescriptors', {}):
            logging.warning(f"Skipping file {file_path}: Missing description field")
            return None
        
        return {
            'id': product.get('id', ''),
            'name': product.get('productDisplayName', ''),
            'description': product['productDescriptors']['description']['value'],
            'brand': product.get('brandName', ''),
            'gender': product.get('gender', ''),
            'masterCategory': product.get('masterCategory', {}).get('typeName', ''),
            'price': product.get('price', 0),  
            'file': file_path.name
        }
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def preprocess_json_directory(data_directory, output_file, max_threads=4):
  
    directory = Path(data_directory)
    json_files = list(directory.glob("*.json"))
    total_files = len(json_files)
    logging.info(f"Processing {total_files} files from {data_directory}...")

    products = []

    with ThreadPoolExecutor(max_threads) as executor:
        results = executor.map(process_json_file, json_files)

        for idx, product in enumerate(results):
            if product:
                products.append(product)
            if idx % 100 == 0:
                logging.info(f"Processed {idx}/{total_files} files...")

    logging.info(f"Finished processing. Successfully processed {len(products)} products.")

    products_df = pd.DataFrame(products)
    if products_df.empty:
        logging.warning("No valid products found. Exiting preprocessing.")
        return

    
    products_df['processed_text'] = products_df['description'].apply(preprocess_text)

    
    similarity_vectorizer = TfidfVectorizer()
    tfidf_matrix = similarity_vectorizer.fit_transform(products_df['processed_text'])

    knn = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='cosine')
    knn.fit(tfidf_matrix)

    
    processed_data = {
        'data': products_df,
        'tfidf_matrix': tfidf_matrix,
        'similarity_vectorizer': similarity_vectorizer,
        'knn_model': knn,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    logging.info(f"Preprocessing complete. Processed data saved to {output_file}.")
    
    
    logging.info("Available Gender Categories:")
    logging.info(products_df['gender'].unique())
    logging.info("Available Master Categories:")
    logging.info(products_df['masterCategory'].unique())

if __name__ == "__main__":
    input_directory = "fashion-dataset/fashion-dataset/styles"  
    output_file = "preprocessed_fashion_data.pkl"  
    
    preprocess_json_directory(input_directory, output_file, max_threads=8)
