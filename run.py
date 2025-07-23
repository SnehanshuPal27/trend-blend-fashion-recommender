from flask import Flask, request, jsonify, render_template
import logging
from recommender import load_artifacts, get_recommendations

# --- App Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load all the ML models and data once on startup
# This happens when the Gunicorn server starts the application.
with app.app_context():
    load_artifacts()

# --- Routes ---

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/results')
def results():
    """Renders the results page."""
    return render_template('results.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    The main API endpoint for getting recommendations.
    Accepts a JSON payload with a description and optional filters.
    """
    try:
        data = request.get_json()
        description = data.get('description')
        gender = data.get('gender')
        category = data.get('category')

        if not description:
            return jsonify({'error': 'Description is required.'}), 400

        recommendations = get_recommendations(
            description=description,
            gender_filter=gender if gender != 'all' else None,
            category_filter=category if category != 'all' else None
        )
        
        if not recommendations:
            return jsonify({'message': 'No matching products found.'}), 404
            
        return jsonify(recommendations)

    except Exception as e:
        logging.error(f"An error occurred in /recommend: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500
