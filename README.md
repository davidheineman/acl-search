## ACL Search

Use ColBERT as a search engine for the [ACL Anthology](https://aclanthology.org/). (Parse any bibtex, and store in a MySQL service)

<div align="center">
    <img src="./src/static/demo.jpg" width="600" />
</div>

## Setup

**Setup ColBERT**
```sh
git clone https://github.com/davidheineman/acl-search

# install dependencies
# torch==1.13.1 required (conda install -y -n [env] python=3.10)
pip install -r requirements.txt
brew install mysql
```

**Search with ColBERT**

```sh
# start flask server
python server.py

# or start a production API endpoint
gunicorn -w 4 -b 0.0.0.0:8080 server:app

# Then visit:
# http://localhost:8080
# or use the API:
# http://localhost:8080/api/search?query=Information retrevial with BERT
```

**(Optional) Parse & Index the Anthology**

This step allows indexing the anthology manually. This can be skipped, since the parsed/indexed anthology will be downloaded from [huggingface.co/davidheineman/colbert-acl](https://huggingface.co/davidheineman/colbert-acl).

*You can also include you own papers by adding to the `anthology.bib` file!*

```sh
# get up-to-date abstracts in bibtex
curl -O https://aclanthology.org/anthology+abstracts.bib.gz
gunzip anthology+abstracts.bib.gz
mv anthology+abstracts.bib anthology.bib

# parse .bib -> .json
python parse.py

# index with ColBERT 
# (note sometimes there is a silent failure if the CPP extensions do not exist)
python index.py
```

**Deploy as a Docker App**
```sh
docker-compose build --no-cache
docker-compose up --build
```

## Example notebooks

To see an example of search, visit:
[colab.research.google.com/drive/1-b90_8YSAK17KQ6C7nqKRYbCWEXQ9FGs](https://colab.research.google.com/drive/1-b90_8YSAK17KQ6C7nqKRYbCWEXQ9FGs?usp=sharing)

<!-- ## Notes
- See: 
    - https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/index_updater.py
    - https://github.com/stanford-futuredata/ColBERT/issues/111

- TODO:
    - On UI
        - Colors: make the colors resemble the ACL page much closer
            - There's still a bunch of blue from the bootstrap themeing
        - Smaller line spacing for abstract text
        - Add "PDF" button
        - Justify the result metadata (Year, venue, etc.) so the content all starts at the same vertical position
        - Add a "Expand" button at the end of the abstract
        - Make the results scrollable, without scrolling the rest of the page
        - Put two sliders on the year range (and make the years selectable, with the years at both ends of the bar)
        - If the user selects certain venues, remember these venues
        - Add a dropdown under the "Workshop" box to select specific workshops

    - Include the title in the indexing
    
    - (First seperate github and HF, then deploy as a container app)
    - Deploy: https://console.cloud.google.com/run/create?enableapi=true&hl=en&project=light-lambda-256623
    - https://docs.docker.com/language/python/configure-ci-cd/

    - Have articles before 2020

    - Maybe make the UI more compressed like this: https://aclanthology.org/events/eacl-2024/#2024eacl-long

    - Put query in URL (?q=XXX)

    - Make indexing much easier 
        (currently, the setup involves manually copying the CPP files becuase there is a silent failure, this also should be possible to do on Google Collab, or even MPS)
        - Make index save in parent folder
        - Fix "sanity check" in index.py
    - Profile bibtexparser.load(f) (why so slow)
    - Ship as a containerized service
    - Scrape: 
        - https://proceedings.neurips.cc/
        - https://dblp.uni-trier.de/db/conf/iclr/index.html
        - openreview
 -->