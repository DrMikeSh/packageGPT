from flask import Flask, request, jsonify
import requests
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from openai import OpenAI
from pinecone import Pinecone



app = Flask(__name__)

# Setup rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"]
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Bearer token-based authentication
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        token = auth_header.split(" ")[1]
        if token != os.getenv('ACCESS_TOKEN'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/get_results', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")  # Rate limit for this endpoint
def get_results():
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    PICONE_API_KEY = os.getenv('PINECONE_KEY')
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    
    # Input validation
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Text must be a non-empty string'}), 400

    try:
        client = OpenAI(api_key = OPENAI_API_KEY)
        # Call OpenAI API with timeout
        text = text.replace("\n", " ")
        openai_response = client.embeddings.create(input=[text], model="text-embedding-3-large", dimensions=1024).data[0].embedding
    except Exception as e:
        return jsonify({'error': f'Error calling OpenAI API: {str(e)}'}), 500
    
    try:
        logging.info(f'pinecone key:{PICONE_API_KEY}')
        # Call Picone API with timeout
        pc = Pinecone(api_key=PICONE_API_KEY)
        index = pc.Index("packagegpt")
        picone_response = index.query(
            vector=openai_response,
            include_metadata = True,
            top_k=10
        )     

        # Check if the response is valid
        if not picone_response:
            raise ValueError("Empty response from Picone API")

    except Exception as e:
        logging.error(f"Picone API call failed: {e}")
        return jsonify({'error': 'Error calling Picone API'}), 500

    results = picone_response.matches
    all_pages = ''
    for res in results:
        all_pages += res.metadata['page']

    return jsonify({'all_pages': all_pages})

# Privacy Policy Route
@app.route('/privacy_policy', methods=['GET'])
def privacy_policy():
    privacy_text = """
    Privacy Policy:

    Your privacy is important to us. This application collects and processes data
    in accordance with applicable data protection laws. The information we collect
    includes text inputs you provide to our APIs and may be used for improving our services.

    We do not sell, share, or misuse your data. For any concerns or inquiries
    about your privacy, please contact our support team.

    By using this GPT, you agree to this Privacy Policy.
    """
    return jsonify({'privacy_policy': privacy_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
