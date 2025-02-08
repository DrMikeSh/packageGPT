from flask import Flask, request, jsonify
import requests
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel


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
        # Call Picone API with timeout
        pc = Pinecone(api_key=PICONE_API_KEY)
        index = pc.Index("packagegpt")
        picone_response = index.query(
            vector=openai_response,
            include_metadata = True,
            top_k=5
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


#Planner function  - Gets the whole request, seperates the request into parts, return docs for each part
@app.route('/api/get_planned_results', methods=['POST'])
@require_auth
@limiter.limit("20 per minute")  # Rate limit for this endpoint
def get_planned_results():
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    PICONE_API_KEY = os.getenv('PINECONE_KEY')
    client = OpenAI(api_key = OPENAI_API_KEY)

    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    
    # Input validation
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Text must be a non-empty string'}), 400


    #Step 1: seperate the request into parts
    try:
        class Instruction_list(BaseModel):
            a: list[str]

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", 
                "content": '''
                You will receive a user-provided text describing the desired code outcome. 
                Your task is to generate a step-by-step plan that outlines how to achieve the specified outcome. 

                Each step in the plan should be expressed as a simple and clear sentence, such as: 
                - "Create a class to model the data."
                - "Create a text box next to the text"

                The final output must be a list of simple sentences summarizing the plan.

                Example Input:
                "I want to build a web-based calculator that can handle basic arithmetic operations like addition, subtraction, multiplication, and division."

                Example Output:
                ["Create a front-end interface with buttons for numbers and operations.","Design a back-end API to process arithmetic operations.","Write a function to validate user inputs.", "Integrate the front-end and back-end for real-time calculations.","Test the calculator for edge cases like division by zero."

                Make sure the plan covers all major steps to accomplish the desired outcome, expressed in a concise manner.
                '''
                },
                {"role": "user", "content":text}
            ],
            response_format=Instruction_list,
        )

        output_plan = response.choices[0].message.parsed.a
    except Exception as e:
        logging.error(f"OpenAI API call failed for planning: {e}")
        return jsonify({'error': f'Error calling OpenAI API for planning'}), 500
    
    
    #Add the plan steps to the output
    final_output = 'Plan:\n'
    for step in output_plan: 
        final_output += f'Step: {step}\n'


    return jsonify({'Plan': final_output})


######################### Privacy Policy Route ########################

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
