## ACL Search

Use ColBERT as a search engine for the [ACL Anthology](https://aclanthology.org/) (or any .bib file!).

<div align="center">
    <img src="./src/static/demo.jpg" width="600" />
</div>

## Setup

**Setup ColBERT**
```sh
git clone https://github.com/davidheineman/acl-search

# install dependencies
# conda install -y -n aclsearch python=3.10 # (torch==1.13.1 required)
pip install -r requirements.txt
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
# pull from openreview
echo "[email]\n[password]" > .openreview
python src/scrape/openrev.py

# pull from acl anthology
git clone https://github.com/acl-org/acl-anthology src/acl-anthology
cp -r src/acl-anthology/bin/anthology src/scrape/anthology
cp -r src/acl-anthology/data src/scrape/acl_data
pip install -r src/acl-anthology/bin/requirements.txt
rm -rf src/acl-anthology
python src/scrape/acl.py # parse acl_data/ -> .json

# create unified dataset
python src/parse.py 

# index with ColBERT 
# (note sometimes there is a silent failure if the CPP extensions do not exist)
python src/index.py
```

**Deploy as a Docker App**
```sh
# Build and run locally
docker build . -t acl-search:main
docker run -p 8080:8080 acl-search:main

# Or pull the hosted container
docker pull ghcr.io/davidheineman/acl-search:main # add for macos: --platform linux/amd64 
docker run -p 8080:8080 ghcr.io/davidheineman/acl-search:main

# Lauch it as a web service!
brew install flyctl
fly launch
```

## Example notebooks

To see an example of search, visit:
[colab.research.google.com/drive/1-b90_8YSAK17KQ6C7nqKRYbCWEXQ9FGs](https://colab.research.google.com/drive/1-b90_8YSAK17KQ6C7nqKRYbCWEXQ9FGs?usp=sharing)

<!-- ## Notes
- See: 
    - https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/index_updater.py
    - https://github.com/stanford-futuredata/ColBERT/issues/111

- To generate favicon:
    cd src/static
    inkscape favicon.svg --export-type=png --export-background-opacity=0 --export-filename=favicon.png
    convert favicon.png -resize 256x256 favicon.ico
    rm favicon.png

- TODO:
    - On UI
        - Single click "copy" for full bibtex, and bib key!
        - Maybe make the UI more compressed like this: https://aclanthology.org/events/eacl-2024/#2024eacl-long
        - Colors: make the colors resemble the ACL page much closer
            - There's still a bunch of blue from the bootstrap themeing
        - Smaller line spacing for abstract text
        - Add "PDF" button
        - Justify the result metadata (Year, venue, etc.) so the content all starts at the same vertical position
        - Add a "Expand" button at the end of the abstract
        - Put two sliders on the year range (and make the years selectable, with the years at both ends of the bar)
        - If the user selects certain venues, remember these venues
        - Add a dropdown under the "Workshop" box to select specific workshops

    - On search quality
        - Use https://aclanthology.org/info/development/
        - Add openreview scrape
        - Include the title in the indexing
        - Have articles before 2020
        - Put query in URL (?q=XXX)

    - On indexing
        - Make indexing code better 
            (currently, the setup involves manually copying the CPP files becuase there is a silent failure, this also should be possible to do on Google Collab, or even MPS)
            - Make index save in parent folder
            - Fix "sanity check" in index.py
            - Make it possible to do a one-click re-indexing as a GitHub action (potentially when building the container? Or re-build the container when HF is updated)
        - Profile bibtexparser.load(f) (why so slow)
        - Make this a GitHub action

    - On deployment
        - Reduce batch batch size to help RAM usage (https://fly.io/docs/about/pricing/#started-fly-machines)
 -->