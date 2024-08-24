import json, os

from constants import ANTHOLOGY_PATH, OPENREVIEW_PATH, DATASET_PATH

from scrape.acl import preprocess_acl
from scrape.openrev import download_openreview


def preprocess_openreview(openreview_path):
    dataset = []

    with open(openreview_path, 'r', encoding='utf-8') as f:
        openreview = json.loads(f.read())

    openreview = openreview['conference']

    for conf_name, conf_entries in openreview.items():
        year = conf_name.split('/')[1]
        for conf_entry in conf_entries:
            # raise RuntimeError(conf_entry['content'])
            try:
                formatted_entry = {
                    'title':    conf_entry['content']['title'],
                    'abstract': conf_entry['content']['abstract'], # some failures
                    'year':     year,
                    'url':      'https://openreview.net' + conf_entry['content']['pdf'],
                    'pdf':      'https://openreview.net' + conf_entry['content']['pdf'],
                    'authors':  conf_entry['content']['authors'],
                    # 'TL;DR':    conf_entry['content']['TL;DR'], # some failures
                    'venue':    conf_entry['content']['venue'],
                    'venueid':  conf_entry['content']['venueid'],
                    '_bibtex':  conf_entry['content']['_bibtex'], # some failures

                    'invitation': conf_entry['invitation'],

                    'findings': False,
                    'venue_type': 'main'
                }

                dataset += [formatted_entry]
            except KeyError as e:
                print(e)

    return dataset


def main():
    dataset = []

    # 1) Pre-process the ACL anthology
    dataset += preprocess_acl(ANTHOLOGY_PATH)

    # 2) Pre-process OpenReview
    if not os.path.exists(OPENREVIEW_PATH):
        download_openreview(OPENREVIEW_PATH)
    dataset += preprocess_openreview(OPENREVIEW_PATH)

    # Unfortunately, remove papers without abstracts
    dataset = [paper for paper in dataset if paper['abstract'] != '']

    # Save dataset
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))


if __name__ == '__main__': main()