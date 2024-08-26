import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OPENREVIEW_PATH, INDEX_ROOT

import json, os

from openreview import Client


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


def get_grouped_venue_papers(client: Client, grouped_venue: dict, only_accepted: bool=True):
    papers = {}
    for venue in grouped_venue:
        papers[venue] = []
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


def get_papers(client: Client, grouped_venues: dict, only_accepted: bool=True):
    papers = {}
    for group, grouped_venue in grouped_venues.items():
        papers[group] = get_grouped_venue_papers(client, grouped_venue, only_accepted)
    return papers


def init_client(OPENREVEW_SECRET_PATH=os.path.join(INDEX_ROOT, ".openreview")):
    file_path = os.path.abspath(OPENREVEW_SECRET_PATH)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        lines = file.readlines()
        username = lines[0].strip()
        password = lines[1].strip()

    return Client(
        baseurl="https://api.openreview.net",
        username=username,
        password=password,
    )


def download_openreview(openreview_path):
    client = init_client()

    only_accepted = True

    years = list(range(2013, 2026))
    conferences = ["NeurIPS", "ICLR", "ICML"] # https://openreview.net/group?id=NeurIPS.cc
    groups = ["conference"]

    venues = get_venues(client, conferences, years)
    
    print(venues)
    
    grouped_venues = group_venues(venues, groups)
    
    print(grouped_venues)
    
    papers = get_papers(client, grouped_venues, only_accepted)

    for i, t in enumerate(papers):
        for j, c in enumerate(papers[t]):
            for k, p in enumerate(papers[t][c]):
                papers[t][c][k] = p.to_json()

    os.makedirs(os.path.dirname(openreview_path), exist_ok=True)
    with open(openreview_path, "w", encoding="utf-8") as json_file:
        json.dump(papers, json_file, indent=4)


if __name__ == "__main__": download_openreview(OPENREVIEW_PATH)
