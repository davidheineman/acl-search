import os, math, re, requests, io, time, sys
from typing import List, Optional, Union
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from flask import Flask, abort, request, render_template, jsonify
from functools import lru_cache

from constants import INDEX_PATH, VENUES
from search import ColBERT
from db import create_database, query_paper_metadata
from utils import download_index_from_hf

import PyPDF2
from openai import OpenAI

PORT = int(os.getenv("PORT", 8080))
app = Flask(__name__)

@lru_cache(maxsize=1000000)
def api_search_query(query):
    print(f"Query={query}")

    # Use ColBERT to find passages related to the query
    pids, scores = colbert.search(query)

    # Softmax output probs
    probs = [math.exp(s) for s in scores]
    probs = [p / sum(probs) for p in probs]

    # Sort and return results as a dict
    topk = [{'pid': pid, 'score': score, 'prob': prob} for pid, score, prob in zip(pids, scores, probs)]
    topk = sorted(topk, key=lambda p: (p['score'], p['pid']), reverse=True)

    response = {"query" : query, "topk": topk}

    return response


def is_valid_query(query):
    return re.match(r'^[a-zA-Z0-9 ]*$', query) and len(query) <= 256


@app.route("/api/colbert", methods=["GET"])
def api_search():
    if request.method == "GET":
        query = str(request.args.get('query'))
        if not is_valid_query(query): abort(400, "Invalid query :(")
        return api_search_query(query)
    return ('', 405)


@app.route('/api/search', methods=['POST', 'GET'])
def query():
    query: str
    start_year: Optional[int]
    end_year: Optional[int]
    venue_type: Optional[Union[VENUES, List[VENUES]]]

    if request.method in ["POST", "GET"]:
        args = request.form if request.method == "POST" else request.args
        query       = args.get('query')
        start_year  = args.get('start_year', None)
        end_year    = args.get('end_year', None)
        venue_type  = args.getlist('venue_type', None)
    
    if not is_valid_query(query): 
        abort(400, "Invalid query :(")

    # Get top passage IDs from ColBERT
    colbert_response = api_search_query(query)

    # Query database for paper information
    pids = [r['pid'] for r in colbert_response["topk"]]
    server_response = query_paper_metadata(
        pids, 
        start_year=start_year,
        end_year=end_year,
        venue_type=venue_type
    )

    K = 1000
    server_response = server_response[:K]

    return server_response


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/table', methods=['POST', 'GET'])
def table():
    return render_template('table.html')


@lru_cache(maxsize=1000)
def get_pdf_text(pdf_url: str) -> str:
    """Cache and retrieve PDF text content."""
    response = requests.get(pdf_url)
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)


def get_openai_client():
    if os.path.exists('.openai-api-key'):
        with open('.openai-api-key', 'r') as f:
            api_key = f.read().strip()
        return OpenAI(api_key=api_key)
    return None


@app.route('/api/llm', methods=['POST'])
def query_llm():
    print(f'Started a new query!')
    client = get_openai_client()
    if client is None:
        return jsonify({'error': 'No OpenAI API key'}), 500
    data = request.json
    title = data['title']
    abstract = data['abstract'] 
    question = data['question']
    pdf_url = data['pdf_url'] if 'pdf_url' in data else None

    try:
        start_time = time.time()
        pdf_text = get_pdf_text(pdf_url)
        print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")

        prompt = f"""Paper Title: {title}
Abstract: {abstract}
Paper Text: {pdf_text}
Question: {question}
Please provide a concise answer."""

        # print_estimate_cost(prompt, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6, estimated_output_toks=100)
        
        start_llm_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing academic papers. Please only respond with the direct answer (e.g., Yes, No) and no explanation or additional details unless it is absolutely necessary. Please respond with a phrase instead of a full sentence when possible. If the question does not apply, simply reply 'N/A'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        print(f"LLM inference took {time.time() - start_llm_time:.2f} seconds")
        
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Failed to process request'}), 500


def initalize_backend():
    with app.app_context():
        global colbert
        download_index_from_hf()
        create_database()
        colbert = ColBERT(index_path=INDEX_PATH)
        # Test queries
        print(colbert.search('text simplificaiton'))
        print(api_search_query("text simplification")['topk'][:5])


if __name__ == "__main__":
    """
    Example usage:
    python server.py
    http://localhost:8080/api/colbert?query=Information retrevial with BERT
    http://localhost:8080/api/search?query=Information retrevial with BERT
    """
    initalize_backend()
    extra_files = [os.path.join(dirname, filename) for dirname, _, files in os.walk('templates') for filename in files]
    app.run("0.0.0.0", PORT, debug=False, extra_files=extra_files)
