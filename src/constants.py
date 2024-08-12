import os
from typing import Literal

HF_INDEX_REPO = "davidheineman/colbert-acl"

INDEX_NAME = os.getenv("INDEX_NAME", 'index')
INDEX_ROOT = os.getenv("INDEX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX_PATH = os.path.join(INDEX_ROOT, INDEX_NAME)
ANTHOLOGY_PATH = os.path.join(INDEX_ROOT, 'anthology.bib')
DATASET_PATH = os.path.join(INDEX_ROOT, 'dataset.json')

DB_NAME = 'anthology'
DB_HOSTNAME = 'mysql_db' # localhost
DB_PORT = 3306 # None

VENUES = Literal['workshop', 'journal', 'short', 'demo', 'tutorial', 'industry', 'findings', 'main']