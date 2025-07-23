import os
from flask import Flask, request, jsonify, render_template
from newRec import preprocess_text, recommend_products
import json

app = Flask(__name__, static_folder='images')


@app.route('/')
def index():
    print('home')
    return render_template('index.html')  
 




@app.route('/recommendationbyDescp', methods=['POST'])
def recommendation_by_description():
    try:
        print("Raw request data:", request.data)
        print("Request content type:", request.content_type)
        print("Request headers:", dict(request.headers))

        try:
            
            request_data = request.get_json(force=True)
        except json.JSONDecodeError as je:
            print(f"JSON parsing error. Raw data: {request.data.decode('utf-8')}")
            raw_data = request.data.decode('utf-8').strip()
            try:
                request_data = json.loads(raw_data)
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Invalid JSON format',
                    'raw_data': raw_data,
                    'details': str(je)
                }), 400

        print("Parsed request data:", request_data)

        
        if not request_data or 'description' not in request_data:
            print("Error: Invalid input, 'description' not in request data.")
            return jsonify({
                'error': 'Invalid input: Description is required',
                'received_data': request_data
            }), 400

        user_description = request_data['description'].strip()
        if not user_description:
            print("Error: Description is empty.")
            return jsonify({'error': 'Description cannot be empty'}), 400

        
        gender_filter = request_data.get('gender')
        category_filter = request_data.get('masterCategory')

        print("Processed description:", user_description)
        print("Gender filter:", gender_filter if gender_filter else "None provided (defaulting to None)")
        print("Category filter:", category_filter if category_filter else "None provided (defaulting to None)")

        
        processed_description = preprocess_text(user_description)

        
        precomputed_file = 'preprocessed_fashion_data.pkl'
        output_file = 'recommendations.json'

        
        recommendations = recommend_products(
            user_description=processed_description,
            gender_filter=gender_filter if gender_filter else None,
            category_filter=category_filter if category_filter else None,
            precomputed_file=precomputed_file,
            output_file=output_file,
            num_recommendations=5
        )

        
        if not recommendations:
            print("No recommendations found.")
            return jsonify({'message': 'No recommendations found'}), 404

        
        return jsonify(recommendations)

    except json.JSONDecodeError as e:
        print(f"JSON decoding error occurred. Request data: {request.data}")
        return jsonify({
            'error': 'JSON decoding error',
            'details': str(e),
            'raw_data': request.data.decode('utf-8') if request.data else None
        }), 400

    except Exception as e:
        print(f"Server error occurred. Error: {str(e)}")
        print(f"Request data: {request.data}")
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'raw_data': request.data.decode('utf-8') if request.data else None
        }), 500



@app.route('/test-recommendation', methods=['POST'])
def test_recommendation():
    """Test endpoint to verify JSON parsing"""
    try:
        data = request.get_json(force=True)
        return jsonify({
            'received': data,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'raw_data': request.data.decode('utf-8') if request.data else None
        }), 400

    
@app.route('/show-recommendations')
def show_recommendations():
    """
    Render the recommendations page.
    """
    return render_template('recommendText.html')


if __name__ == '__main__':
    
    app.run(debug=True)
