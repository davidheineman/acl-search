import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents deadlocks in ColBERT tokenization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # Allows multiple libraries in OpenMP runtime. This can cause unexected behavior, but allows ColBERT to work

import json

from constants import INDEX_NAME, DATASET_PATH

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig


nbits      = 2     # encode each dimension with 2 bits
doc_maxlen = 512   # truncate passages
checkpoint = 'colbert-ir/colbertv2.0' # ColBERT model to use


def index_anthology(collection, index_name):
    with Run().context(RunConfig(nranks=2, experiment='notebook')): # nranks specifies the number of GPUs to use
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen, 
            nbits=nbits, 
            kmeans_niters=4, # specifies the number of iterations of k-means clustering; 4 is a good and fast default.
            index_path=INDEX_NAME,
            bsize=256, # for A40 GPU. need to apply in Indexer.index() directly
            centroid_score_threshold=0.8,
            ndocs=8192
        ) 

        indexer = Indexer(
            checkpoint=checkpoint, 
            config=config
        )

        indexer.index(
            name=index_name, 
            collection=collection, 
            overwrite=True
        )


def search_anthology(query, collection, index_name):
    """ Default ColBERT search function """
    with Run().context(RunConfig(nranks=0, experiment='notebook')):
        searcher = Searcher(index=index_name, collection=collection)

    results = searcher.search(query, k=3)

    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")


def main():
    # Load the parsed anthology
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.loads(f.read())

    # dataset = dataset[:5000] # 5K in 48s/iter on 2 A40s (67K in 2hr)
    
    # Get the abstracts for indexing
    collection = [e['abstract'] for e in dataset]

    # Run ColBERT indexer
    index_anthology(collection, index_name=INDEX_NAME)

    # Sanity check
    # query = ["What are some recent examples of grammar checkers?"]
    # search_anthology(query, collection, index_name=INDEX_NAME)


if __name__ == '__main__': main()
