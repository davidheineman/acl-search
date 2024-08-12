import bibtexparser, json

from constants import ANTHOLOGY_PATH, DATASET_PATH


def parse_bibtex(anthology_path, dataset_path):
    with open(anthology_path, 'r', encoding='utf-8') as f:
        bib = bibtexparser.load(f)
    dataset = bib.entries

    print(f'Found {len(dataset)} articles with keys: {dataset[0].keys()}')
    paper: dict
    for paper in dataset[:2]:
        print(f"{paper.get('author')}\n{paper.get('title')}\n{paper.get('url')}\n")

    # Remove any entries without abstracts, since we index on abstracts
    dataset = [paper for paper in dataset if 'abstract' in paper.keys()]

    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))

    return dataset


def preprocess_acl_entries(dataset_path):
    """
    Very rough attempt at using ACL URLs to infer their venues. Bless this mess. 
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.loads(f.read())

    venues = []
    for id, paper in enumerate(dataset):
        url = paper['url']
        year = int(paper['year'])

        if year < 2020: 
            dataset[id]['findings'] = None
            dataset[id]['venue_type'] = None
            continue

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

        venues += [url]

        # Extract paper type from URL
        _type = None
        if any(venue in url for venue in ['parlaclarin', 'nlpcovid19']):
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

        dataset[id]['findings'] = findings
        dataset[id]['venue_type'] = _type

    # print(set(venues))

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))

    return dataset


def main():
    # 1) Parse and save the anthology dataset
    dataset = parse_bibtex(ANTHOLOGY_PATH, DATASET_PATH)

    # 2) Pre-process the ACL anthology
    dataset = preprocess_acl_entries(DATASET_PATH)


if __name__ == '__main__': main()