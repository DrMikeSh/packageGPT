from flask import Flask, request, jsonify
import requests
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from openai import OpenAI
from pinecone import Pinecone
# from pydantic import BaseModel
from functions import main_topics

########################Set up the app########################

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

########################Set up the functions########################


#Basic function  - gets a single entity - returns docs related to the entity
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
    text = text.replace("\n", " ")
    
    # Input validation
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Text must be a non-empty string'}), 400

    # Connect to OpenAI API
    try:
        client = OpenAI(api_key = OPENAI_API_KEY)
    except Exception as e:
        return jsonify({'error': f'Error calling OpenAI API: {str(e)}'}), 500

    #Get topics from gpt
    topics = main_topics(text, client)

    #search topics in pinecone and write to final_output
    pc = Pinecone(api_key=PICONE_API_KEY)
    final_output = ''

    topics_list = topics.split(';')
    logging.error(f"Topics list {topics_list}, topics {topics}")
    n_topics = len(topics_list)
    for topic in topics_list[:11]: #limit to 10 topics
        topic_embedding = client.embeddings.create(input=[text], model="text-embedding-3-large", dimensions=1024).data[0].embedding
        try:
            # Call Picone API with timeout
            index = pc.Index("packagegpt")
            picone_response = index.query(
                vector=topic_embedding,
                include_metadata = True,
                top_k=min(20//len(n_topics),6)
            )
            results = picone_response.matches
            final_output += f"Topic: {topic}\n" 
            for res in results:
                final_output += res.metadata['text']     

            # Check if the response is valid
            if not picone_response:
                raise ValueError("Empty response from Picone API")

        except Exception as e:
            logging.error(f"Picone API call failed: {e}")
            return jsonify({'error': 'Error calling Picone API'}), 500
        
    return jsonify({'final_output': final_output})


######################### Privacy Policy Route ########################

#OpenAI requires any GPT have a privacy policy to be distributed by a link. 
#This is a simple privacy policy, for any serious use case, please consult a legal professional.


@app.route('/privacy_policy', methods=['GET'])
def privacy_policy():
    policy_file = 'privacy_policy.txt'
    try:
        with open(policy_file, 'r') as file:
            privacy_text = file.read()
    except FileNotFoundError:
        return jsonify({'error': 'Privacy policy file not found.'}), 500
    except Exception as e:
        logging.error(f"Error reading privacy policy file: {e}")
        return jsonify({'error': 'An error occurred while loading the privacy policy.'}), 500
    
    return jsonify({'privacy_policy': privacy_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
