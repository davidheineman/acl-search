import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OPENREVIEW_PATH, INDEX_ROOT

import json, os

from openreview import Client
from openreview.api import OpenReviewClient


def get_venues(client: Client, confs: list[str], years: list[int]) -> list[str]:
    def filter_year(venue):
        return next((str(year) for year in years if str(year) in venue), None)

    venues = client.get_group(id="venues").members
    filtered_venues = [venue for venue in venues if filter_year(venue)]

    read_venues = [
        venue
        for venue in filtered_venues
        if any(conf.lower() in venue.lower() for conf in confs)
    ]

    return [venue for venue in read_venues if filter_year(venue)]


def group_venues(venues: list[str], bins: list[str]) -> dict:
    """
    Group a list of venues into "bins", or their venue type

    {'conference': ['NeurIPS.cc/2021/Conference']}
    """
    bins_dict = {bin: [] for bin in bins}

    for venue in venues:
        for bin in bins:
            if bin.lower() in venue.lower():
                bins_dict[bin].append(venue)
                break

    return bins_dict


def get_grouped_venue_papers(clients: list[Client], grouped_venue: dict, only_accepted: bool=True):
    papers = {}
    for venue in grouped_venue:
        print(f'Getting: {venue}')
        papers[venue] = []
        for client in clients:
            if len(papers[venue]) > 0: continue
            if only_accepted:
                submissions = client.get_all_notes(
                    content={"venueid": venue}, details="directReplies"
                )
            else:
                single_blind_submissions = client.get_all_notes(
                    invitation=f"{venue}/-/Submission", details="directReplies"
                )
                double_blind_submissions = client.get_all_notes(
                    invitation=f"{venue}/-/Blind_Submission", details="directReplies"
                )
                submissions = single_blind_submissions + double_blind_submissions
            papers[venue] += submissions
    return papers


def get_papers(clients: list[Client], grouped_venues: dict, only_accepted: bool=True):
    papers = {}
    for group, grouped_venue in grouped_venues.items():
        papers[group] = get_grouped_venue_papers(clients, grouped_venue, only_accepted)
    return papers


def init_client(OPENREVEW_SECRET_PATH=os.path.join(INDEX_ROOT, ".openreview")):
    file_path = os.path.abspath(OPENREVEW_SECRET_PATH)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        lines = file.readlines()
        username = lines[0].strip()
        password = lines[1].strip()

    old_client = Client(
        baseurl="https://api.openreview.net",
        username=username,
        password=password,
    )

    new_client = OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )

    return old_client, new_client


def download_openreview(openreview_path):
    clientv1, clientv2 = init_client()

    only_accepted = True

    years = list(range(2013, 2026))
    conferences = ["NeurIPS", "ICLR", "ICML", "colmweb"] # https://openreview.net/group?id=NeurIPS.cc
    groups = ["conference"]

    venues = get_venues(clientv1, conferences, years)
    
    print(venues)
    
    grouped_venues = group_venues(venues, groups)
    
    print(grouped_venues)
    
    papers = get_papers([clientv1, clientv2], grouped_venues, only_accepted)

    for i, t in enumerate(papers):
        for j, c in enumerate(papers[t]):
            for k, p in enumerate(papers[t][c]):
                papers[t][c][k] = p.to_json()

    os.makedirs(os.path.dirname(openreview_path), exist_ok=True)
    with open(openreview_path, "w", encoding="utf-8") as json_file:
        json.dump(papers, json_file, indent=4)


def preprocess_openreview(openreview_path):
    dataset = []

    with open(openreview_path, 'r', encoding='utf-8') as f:
        openreview = json.loads(f.read())

    openreview = openreview['conference']

    for conf_name, conf_entries in openreview.items():
        skipped = 0

        if 'COLM' in conf_name:
            year = conf_name.split('/')[2]
            area = 'nlp'
        else:
            year = conf_name.split('/')[1]
            area = 'ml'

        for conf_entry in conf_entries:
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
                if '.cc' in venueid: 
                    venueid = venueid.split('.cc')[0]
                elif 'colmweb' in venueid:
                    venueid = 'COLM'

                # if int(year) == 2019:
                #     raise RuntimeError(conf_entry)

                if venue == 'COLM':
                    venue_type = 'colm'
                elif 'Submitted' in venue:
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

                assert venue_type in ['colm', 'spotlight', 'oral', 'poster', 'invite', 'rejected'], (venue, venue_type)

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
                    'area': area
                }

                dataset += [formatted_entry]
            except KeyError as e:
                skipped += 1
                print((e, conf_entry))

        print(f'Processed {len(conf_entries)-skipped} / {len(conf_entries)} entries for {conf_name}')

    return dataset


if __name__ == "__main__": download_openreview(OPENREVIEW_PATH)
