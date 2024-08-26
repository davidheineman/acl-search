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
        skipped = 0

        year = conf_name.split('/')[1]
        for conf_entry in conf_entries:
            # raise RuntimeError(conf_entry)

            try:
                for k, v in conf_entry['content'].items():
                    if isinstance(v, dict) and 'value' in v.keys():
                        conf_entry['content'][k] = v['value']

                bibtex = conf_entry['content'].get('_bibtex', '') # some failures
                if bibtex != '':
                    bibkey = bibtex.split('{')[1].split(',')[0].replace('\n', '')
                # else:
                #     raise ValueError(conf_entry)

                venue = conf_entry['content'].get('venue', 'Submitted')

                venueid = conf_entry['content'].get('venueid')
                if venueid: 
                    venueid = venueid.split('.cc')[0]

                # if int(year) == 2019:
                #     raise RuntimeError(conf_entry)

                if 'Submitted' in venue:
                    venue_type = 'rejected'
                    skipped += 1
                    continue
                elif 'spotlight' in venue.lower() or 'notable top 25%' in venue:
                    venue_type = 'spotlight'
                elif 'oral' in venue.lower() or 'notable top 5%' in venue:
                    venue_type = 'oral'
                elif 'accept' in venue.lower() or 'poster' in venue.lower():
                    venue_type = 'poster'
                elif 'invite' in venue.lower():
                    venue_type = 'invite'
                elif len(venue.split(' ')) == 2: 
                    venue_type = 'poster' # no type specified (e.g., "ICLR 2020")
                else:
                    raise ValueError(venue)

                assert venue_type in ['spotlight', 'oral', 'poster', 'invite', 'rejected'], (venue, venue_type)

                if 'invitation' in conf_entry:
                    invitation = conf_entry['invitation']
                elif 'invitations' in conf_entry:
                    invitation = str(conf_entry['invitations'])

                formatted_entry = {
                    'title':    conf_entry['content']['title'],
                    'abstract': conf_entry['content'].get('abstract'),
                    'year':     int(year),
                    'url':      'https://openreview.net/forum?id=' + conf_entry['id'], # 'forum', 'original'
                    'pdf':      'https://openreview.net' + conf_entry['content']['pdf'],
                    'authors':  conf_entry['content']['authors'],
                    # 'TL;DR':    conf_entry['content']['TL;DR'], # some failures
                    'venue':    venue,
                    'venueid':  venueid,
                    '_bibtex':  bibtex,
                    '_bibkey':  bibkey,
                    'invitation': invitation,
                    'venue_type': venue_type,
                    'area': 'ml'
                }

                dataset += [formatted_entry]
            except KeyError as e:
                skipped += 1
                print((e, conf_entry))

        print(f'Processed {len(conf_entries)-skipped} / {len(conf_entries)} entries for {conf_name}')

    return dataset


def main():
    dataset = []

    # 1) Pre-process the ACL anthology
    dataset += preprocess_acl(ANTHOLOGY_PATH)

    # 2) Pre-process OpenReview
    if not os.path.exists(OPENREVIEW_PATH):
        download_openreview(OPENREVIEW_PATH)
    dataset += preprocess_openreview(OPENREVIEW_PATH)

    for paper in dataset:
        if paper['abstract'] == None: paper['abstract'] = ''
        if paper['title'] == None: paper['title'] = ''

    # Unfortunately, remove papers without abstracts and titles
    dataset = [paper for paper in dataset if (paper['abstract'] != '' or paper['title'] != '')]

    # Save dataset
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset, indent=4))


if __name__ == '__main__': main()