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
    no_abs = 0
    all_venue_ids = []

    paper: Paper
    for id_, paper in tqdm(anthology.papers.items(), desc='Processing ACL'):
        paper_dict: dict = paper.as_dict()

        title = paper.get_title(form='plain')
        year = int(paper_dict['year'])
        url = paper_dict['url']

        venueid = paper.get_venue_acronym()
        venue_type = get_venue_type(year, url, venueid, title)

        all_venue_ids += [venueid]

        authors = [person for person in paper.iter_people()]
        authors = [author for author, id, type_ in authors]

        for i, author in enumerate(authors):
            if not isinstance(author, str):
                authors[i] = str(author)

        abstract = paper.get_abstract(form='plain')
        if paper.has_abstract and abstract == '':
            abstract = paper.get_abstract(form='text')

        if abstract == '':
            no_abs += 1

        formatted_entry = {
            'title':    title,
            'abstract': abstract,
            'year':     year,
            'url':      url,
            'pdf':      paper_dict.get('pdf'),
            'authors':  authors,
            'venue':    paper_dict['booktitle'],
            'venueid':  venueid,
            '_bibtex':  paper.as_bibtex(concise=True),
            '_bibkey':  paper_dict['bibkey'],
            'invitation': None, # used by OpenReview
            'venue_type': venue_type,
            'area': 'nlp'
            # 'TL;DR':    None,
        }

        dataset += [formatted_entry]

    print(f'Papers without abstracts: {no_abs} / {len(anthology.papers)}')
    # print(set(all_venue_ids))

    os.makedirs(os.path.dirname(anthology_path), exist_ok=True)
    with open(anthology_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))

    return dataset


def get_venue_type(year: int, url: str, venueid: str, title: str):
    '''
    Very rough attempt at using ACL URLs to infer their venues. Bless this mess. 
    '''
    if 'https://aclanthology.org/' in url:
        url = url.split('https://aclanthology.org/')[1]
    elif 'http://www.lrec-conf.org/proceedings/' in url:
        url = url.split('http://www.lrec-conf.org/proceedings/')[1]

    if year >= 2020: # parse using url
        url_new = '.'.join(url.split('.')[:-1])
        if url_new != '': url = url_new

        # For most new venues, the format is '2023.eacl-tutorials' -> 'eacl-tutorials'
        url_new = '.'.join(url.split('.')[1:])
        if url_new != '': url = url_new

        # 'acl-main' -> 'acl-long'?
        # 'acl-main' -> 'acl-short'?

        # 'eacl-demo' -> 'eacl-demos'
        # 'emnlp-tutorial' -> 'emnlp-tutorials'
        url = url.replace('-demos', '-demo')
        url = url.replace('-tutorials', '-tutorial')
        
        venueid = url

        # Extract paper type from id
        if any(venue in venueid for venue in ['parlaclarin', 'nlpcovid19', 'paclic']):
            _type = 'workshop'
        elif not any(venue in venueid for venue in ['aacl', 'naacl', 'acl', 'emnlp', 'eacl', 'tacl']):
            _type = 'workshop'
        elif 'tacl' in venueid or 'cl' in venueid: _type = 'journal'
        elif 'srw' in venueid: _type = 'workshop'
        elif 'short' in venueid: _type = 'short'
        elif 'demo' in venueid: _type = 'demo'
        elif 'tutorial' in venueid: _type = 'tutorial'
        elif 'industry' in venueid: _type = 'industry'
        elif 'findings' in venueid: _type = 'findings'
        elif 'main' in venueid or 'long' in venueid: _type = 'main'
        else:
            raise RuntimeError(f'Could not parse: {venueid}')

    else: # parse using venueid
        venueid = venueid.lower()

        MAIN_CONFS = ['aacl', 'naacl', 'acl', 'emnlp', 'eacl', 'lrec-coling', 'eacl', 'coling', 'conll', 'acl-eacl', 'acl', 'hlt-emnlp', 'lrec', 'aacl', 'ijcnlp', 'emnlp-ijcnlp']
        
        JOURNALS = ['tacl', 'cl']
        
        # TODO: make this correct
        _type = 'workshop'
        if any(venue == venueid for venue in MAIN_CONFS):
            if 'short' in title.lower():
                _type = 'short'
            elif 'demo' in title.lower():
                _type = 'demo'
            elif 'tutorial' in title.lower():
                _type = 'tutorial'
            elif 'industry' in title.lower():
                _type = 'industry'
            else:
                _type = 'main'
        elif any(venue == venueid for venue in JOURNALS):
            _type = 'journal'

    return _type


if __name__ == '__main__': preprocess_acl(ANTHOLOGY_PATH)