import json, os

from constants import ANTHOLOGY_PATH, ANTHOLOGY_RAW_PATH, OPENREVIEW_PATH, DATASET_PATH

from scrape.acl import download_acl, preprocess_acl
from scrape.openrev import download_openreview, preprocess_openreview


def main(use_cache=True):
    dataset = []

    # 1) Pre-process the ACL anthology
    if not os.path.exists(ANTHOLOGY_RAW_PATH) or not use_cache:
        download_acl(ANTHOLOGY_RAW_PATH)
    dataset += preprocess_acl(ANTHOLOGY_RAW_PATH, ANTHOLOGY_PATH)

    # 2) Pre-process OpenReview
    if not os.path.exists(OPENREVIEW_PATH) or not use_cache:
        download_openreview(OPENREVIEW_PATH)
    dataset += preprocess_openreview(OPENREVIEW_PATH)

    # Unfortunately, remove papers without abstracts and titles
    for paper in dataset:
        if paper['abstract'] == None: paper['abstract'] = ''
        if paper['title'] == None:    paper['title'] = ''
    dataset = [paper for paper in dataset if (paper['abstract'] != '' or paper['title'] != '')]

    # Save dataset
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))


if __name__ == '__main__': 
    # main()
    main(use_cache=False)