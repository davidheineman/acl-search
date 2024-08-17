import os, math, re
from typing import List, Optional, Union

from flask import Flask, abort, request, render_template
from functools import lru_cache

from constants import INDEX_PATH, VENUES

from search import init_colbert, search_colbert
from db import create_database, query_paper_metadata
from utils import download_index_from_hf

PORT = int(os.getenv("PORT", 8080))
app = Flask(__name__)


@lru_cache(maxsize=1000000)
def api_search_query(query):
    print(f"Query={query}")

    # Use ColBERT to find passages related to the query
    pids, scores = search_colbert(query)

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
    is_findings: Optional[bool]

    if request.method in ["POST", "GET"]:
        args = request.form if request.method == "POST" else request.args
        query       = args.get('query')
        start_year  = args.get('start_year', None)
        end_year    = args.get('end_year', None)
        venue_type  = args.getlist('venue_type', None)
        is_findings = args.get('is_findings', None)
    
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
        venue_type=venue_type,
        is_findings=is_findings
    )

    K = 20
    server_response = server_response[:K]

    return server_response


# @app.route('/search', methods=['POST', 'GET'])
# def search_web():
#     return render_template('public/results.html', query=query, year=year, results=results)
    

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    """
    Example usage:
    python server.py
    http://localhost:8080/api/colbert?query=Information retrevial with BERT
    http://localhost:8080/api/search?query=Information retrevial with BERT
    """
    download_index_from_hf()
    create_database()
    init_colbert(index_path=INDEX_PATH)
    app.run("0.0.0.0", PORT) # debug=True
