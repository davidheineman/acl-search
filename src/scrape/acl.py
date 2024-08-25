import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(CURRENT_DIR)

import locale

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

import json
from tqdm import tqdm

from constants import ANTHOLOGY_PATH
from anthology import Anthology, Paper, PersonName


ANTHOLOGY_RAW_PATH = os.path.join(CURRENT_DIR, 'acl_data')


def preprocess_acl(anthology_path):
    anthology = Anthology(importdir=ANTHOLOGY_RAW_PATH)

    dataset = []

    paper: Paper
    for id_, paper in tqdm(anthology.papers.items(), desc='Processing ACL'):
        paper_dict: dict = paper.as_dict()

        year = int(paper_dict['year'])
        url = paper_dict['url']

        venue_type, is_findings = get_venue_type(year, url)

        authors = [person for person in paper.iter_people()]
        authors = [author for author, id, type_ in authors]

        for i, author in enumerate(authors):
            if not isinstance(author, str):
                authors[i] = str(author)

        formatted_entry = {
            'title':    paper.get_title(form='plain'),
            'abstract': paper.get_abstract(form='plain'),

            'year':     year,
            'url':      url,
            'pdf':      paper_dict.get('pdf'),
            'authors':  authors,
            'venue':    paper_dict['booktitle'],
            'venueid':  paper.get_venue_acronym(),
            '_bibtex':  paper.as_bibtex(concise=True),
            '_bibkey':  paper_dict['bibkey'],

            'invitation': None, # used by OpenReview

            'venue_type': venue_type,
            'findings': is_findings,

            'area': 'nlp'

            # 'TL;DR':    None,
        }

        dataset += [formatted_entry]

    os.makedirs(os.path.dirname(anthology_path), exist_ok=True)
    with open(anthology_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))

    return dataset


def get_venue_type(year: int, url: str):
    """
    Very rough attempt at using ACL URLs to infer their venues. Bless this mess. 
    """
    if year < 2020: 
        return None, None

    if 'https://aclanthology.org/' in url:
        url = url.split('https://aclanthology.org/')[1]
    elif 'http://www.lrec-conf.org/proceedings/' in url:
        url = url.split('http://www.lrec-conf.org/proceedings/')[1]

    if year >= 2020:
        # new URL format

        url_new = '.'.join(url.split('.')[:-1])
        if url_new != '': url = url_new

        # For most new venues, the format is "2023.eacl-tutorials" -> "eacl-tutorials"
        url_new = '.'.join(url.split('.')[1:])
        if url_new != '': url = url_new

        # 'acl-main' -> 'acl-long'?
        # 'acl-main' -> 'acl-short'?

        # 'eacl-demo' -> 'eacl-demos'
        # 'emnlp-tutorial' -> 'emnlp-tutorials'
        url = url.replace('-demos', '-demo')
        url = url.replace('-tutorials', '-tutorial')

    elif year >= 2016:
        # old URL format
        # P17-1001 -> P17

        url = url.split('-')[0]

        raise RuntimeError('not working')

    # Extract paper type from URL
    _type = None
    if any(venue in url for venue in ['parlaclarin', 'nlpcovid19', 'paclic']):
        _type = 'workshop'
    elif not any(venue in url for venue in ['aacl', 'naacl', 'acl', 'emnlp', 'eacl', 'tacl']):
        _type = 'workshop'
    elif 'tacl' in url: _type = 'journal'
    elif 'srw' in url: _type = 'workshop'
    elif 'short' in url: _type = 'short'
    elif 'demo' in url: _type = 'demo'
    elif 'tutorial' in url: _type = 'tutorial'
    elif 'industry' in url: _type = 'industry'
    elif 'findings' in url: _type = 'findings'
    elif 'main' in url or 'long' in url: _type = 'main'
    else:
        print(f'Could not parse: {url}')

    findings = ('findings' in url)

    return _type, findings


if __name__ == '__main__': preprocess_acl(ANTHOLOGY_PATH)