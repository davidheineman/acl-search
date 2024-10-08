import json
from typing import List, Optional, Union

import sqlite3
from constants import DATASET_PATH, DB_FILENAME, VENUES


def read_dataset():
    print("Reading dataset...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.loads(f.read())
    dataset = [d for d in dataset if 'abstract' in d.keys()]
    return dataset


def create_database():
    db: sqlite3.Connection = sqlite3.connect(
        database = DB_FILENAME
    )
    cursor = db.cursor()

    # Create table
    print('Creating new table...')
    cursor.execute(f'DROP TABLE IF EXISTS paper')
    cursor.execute("""
    CREATE TABLE paper (
        pid INT PRIMARY KEY, 
        title VARCHAR(1024), 
        abstract TEXT(12800), 
        year INT, 
        url VARCHAR(150), 
        pdf VARCHAR(150), 
        author VARCHAR(2170), 
        venue VARCHAR(500), 
        venueid VARCHAR(500), 
        bibtex TEXT(12800), 
        bibkey VARCHAR(100), 
        invitation VARCHAR(100), 
        venue_type VARCHAR(150)
    )
    """)

    acl_data = read_dataset()

    vals = []
    paper: dict
    for pid, paper in enumerate(acl_data):
        title       = paper.get('title', '')
        abstract    = paper.get('abstract', '')
        year        = paper.get('year', '')
        url         = paper.get('url', '')
        pdf         = paper.get('pdf', '')
        author      = paper.get('authors', '')
        venue       = paper.get('venue', '')
        venueid     = paper.get('venueid', '')
        bibtex      = paper.get('_bibtex', '')
        bibkey      = paper.get('_bibkey', '')
        invitation  = paper.get('invitation', '')
        venue_type  = paper.get('venue_type', '')

        author = str(author)

        if not abstract: continue

        vals += [(pid, title, abstract, year, url, pdf, author, venue, venueid, bibtex, bibkey, invitation, venue_type)]

    sql = """
    INSERT INTO paper (
        pid, title, abstract, year, url, pdf, author, venue, venueid, bibtex, bibkey, invitation, venue_type
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    """

    print('Writing entries to table...')
    cursor.executemany(sql, vals)
    db.commit()


def parse_results(results):
    parsed_results = {}

    for result in results:
        pid, title, abstract, year, url, pdf, author, venue, venueid, bibtex, bibkey, invitation, venue_type = result

        title    = title.replace("{", "").replace("}", "")
        author  = author.replace("{", "").replace("}", "").replace('\\"', "")
        abstract = abstract.replace("{", "").replace("}", "").replace("\\", "")

        # Parse list of strings
        import ast
        author = ast.literal_eval(str(author))
        
        parsed_results[int(pid)] = {
            'title': title, 
            'abstract': abstract, 
            'year': year, 
            'url': url, 
            'pdf': pdf, 
            'author': author, 
            'venue': venue, 
            'venueid': venueid, 
            'bibtex': bibtex, 
            'bibkey': bibkey, 
            'invitation': invitation, 
            'venue_type': venue_type
        }

    return parsed_results


def query_paper_metadata(
        pids: List[int], 
        start_year: int = None, 
        end_year: int = None, 
        venue_type: Union[VENUES, List[VENUES]] = None
    ):
    PAPER_QUERY = """
    SELECT * 
    FROM paper 
    WHERE pid IN ({query_arg_str}){constraints_str};
    """

    if not isinstance(venue_type, list): venue_type = [venue_type]
    
    db: sqlite3.Connection = sqlite3.connect(
        database = DB_FILENAME
    )

    cursor = db.cursor()

    pids_str = ', '.join(['?'] * len(pids))

    constraints_str = ""
    if start_year: constraints_str += f" AND year >= {start_year}"
    if end_year: constraints_str += f" AND year <= {end_year}"
    if venue_type: 
        venue_str = ','.join([f'"{venue}"' for venue in venue_type])
        constraints_str += f" AND venue_type IN ({venue_str})"

    query = PAPER_QUERY.format(
        query_arg_str=pids_str, 
        constraints_str=constraints_str
    )

    # print(PAPER_QUERY.format(query_arg_str=', '.join([str(p) for p in pids]), year=year))
    
    cursor.execute(query, pids)
    results = cursor.fetchall()
    
    if len(results) == 0: return []

    parsed_results = parse_results(results)

    # Restore original ordering of PIDs from ColBERT
    results = [parsed_results[pid] for pid in pids if pid in parsed_results.keys()]

    return results


if __name__ == '__main__': create_database()
