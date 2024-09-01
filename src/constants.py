import os
from typing import Literal

HF_INDEX_REPO = "davidheineman/colbert-acl"

INDEX_NAME = os.getenv("INDEX_NAME", 'index')
INDEX_ROOT = os.getenv("INDEX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX_PATH = os.path.join(INDEX_ROOT, INDEX_NAME)
DATA_PATH  = os.path.join(INDEX_ROOT, 'data')

ANTHOLOGY_RAW_PATH = os.path.join(DATA_PATH, 'acl_data')
ANTHOLOGY_BIB_PATH = os.path.join(DATA_PATH, 'anthology.bib')
ANTHOLOGY_PATH  = os.path.join(DATA_PATH, 'anthology.json')
OPENREVIEW_PATH = os.path.join(DATA_PATH, 'openreview.json')

DATASET_PATH    = os.path.join(DATA_PATH, 'papers.json')
DB_FILENAME     = os.path.join(DATA_PATH, 'papers.db')

VENUES = Literal['workshop', 'journal', 'short', 'demo', 'tutorial', 'industry', 'findings', 'main']