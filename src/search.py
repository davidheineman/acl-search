import os, ujson, tqdm
import torch
import torch.nn.functional as F

from colbert import Checkpoint
from colbert.infra.config import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from colbert.search.strided_tensor import StridedTensor
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.indexing.codecs.residual import ResidualCodec

from constants import INDEX_PATH


NCELLS = 1  # Number of centroids to use in PLAID
CENTROID_SCORE_THRESHOLD = 0.5 # How close a document has to be to a centroid to be considered
NDOCS = 512  # Number of closest documents to consider


def init_colbert(index_path, load_index_with_mmap=False):
    """
    Load all tensors necessary for running ColBERT
    """
    global index_checkpoint, scorer, centroids, embeddings, ivf, doclens, nbits, bucket_weights, codec, offsets

    # index_checkpoint: Checkpoint    
    
    use_gpu = torch.cuda.is_available()
    if use_gpu: 
        device = 'cuda'
    else:
        device = 'cpu'

    # Load index checkpoint
    from colbert.infra.run import Run
    initial_config    = ColBERTConfig.from_existing(None, Run().config)
    index_config      = ColBERTConfig.load_from_index(index_path)
    checkpoint_path = index_config.checkpoint
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint_path)
    config: ColBERTConfig = ColBERTConfig.from_existing(checkpoint_config, index_config, initial_config)

    index_checkpoint = Checkpoint(checkpoint_path, colbert_config=config)
    index_checkpoint = index_checkpoint.to(device)
    
    load_index_with_mmap = config.load_index_with_mmap
    if load_index_with_mmap and use_gpu:
        raise ValueError(f"Memory-mapped index can only be used with CPU!")

    scorer = IndexScorer(index_path, use_gpu, load_index_with_mmap)

    with open(os.path.join(index_path, 'metadata.json')) as f:
        metadata = ujson.load(f)
    nbits = metadata['config']['nbits']

    centroids = torch.load(os.path.join(index_path, 'centroids.pt'), map_location=device)
    centroids = centroids.float()

    ivf, ivf_lengths = torch.load(os.path.join(index_path, "ivf.pid.pt"), map_location=device)
    ivf = StridedTensor(ivf, ivf_lengths, use_gpu=False)

    embeddings = ResidualCodec.Embeddings.load_chunks(
        index_path,
        range(metadata['num_chunks']),
        metadata['num_embeddings'],
        load_index_with_mmap=load_index_with_mmap,
    )

    doclens = []
    for chunk_idx in tqdm.tqdm(range(metadata['num_chunks'])):
        with open(os.path.join(index_path, f'doclens.{chunk_idx}.json')) as f:
            chunk_doclens = ujson.load(f)
            doclens.extend(chunk_doclens)
    doclens = torch.tensor(doclens)

    buckets_path = os.path.join(index_path, 'buckets.pt')
    bucket_cutoffs, bucket_weights = torch.load(buckets_path, map_location=device)
    bucket_weights = bucket_weights.float()

    codec = ResidualCodec.load(index_path)

    if load_index_with_mmap:
        assert metadata['num_chunks'] == 1
        offsets = torch.cumsum(doclens, dim=0)
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64), offsets))
    else:
        embeddings_strided = ResidualEmbeddingsStrided(codec, embeddings, doclens)
        offsets = embeddings_strided.codes_strided.offsets


def colbert_score(Q: torch.Tensor, D_padded: torch.Tensor, D_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes late interaction between question (Q) and documents (D)
    See Figure 1: https://aclanthology.org/2022.naacl-main.272.pdf#page=3
    """
    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores_padded = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values
    scores = scores.sum(-1)

    return scores


def get_candidates(Q: torch.Tensor, ivf: StridedTensor) -> torch.Tensor:
    """
    First find centroids closest to Q, then return all the passages in all
    centroids.

    We can replace this function with a k-NN search finding the closest passages
    using BERT similarity.
    """
    Q = Q.squeeze(0)

    # Get the closest centroids via a matrix multiplication + argmax
    centroid_scores: torch.Tensor = (centroids @ Q.T)
    if NCELLS == 1:
        cells = centroid_scores.argmax(dim=0, keepdim=True).permute(1, 0)
    else:
        cells = centroid_scores.topk(NCELLS, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)
    cells = cells.flatten().contiguous()  # (32 * ncells,)
    cells = cells.unique(sorted=False)

    # Given the relevant clusters, get all passage IDs in each cluster
    # Note, this may return duplicates since passages can exist in multiple clusters
    pids, _ = ivf.lookup(cells)

    # Sort and retun values
    pids = pids.sort().values
    pids, _ = torch.unique_consecutive(pids, return_counts=True)
    return pids, centroid_scores


def _calculate_colbert(Q: torch.Tensor):
    """
    Multi-stage ColBERT pipeline. Implemented using the PLAID engine, see fig. 5:
    https://arxiv.org/pdf/2205.09707#page=5
    """
    # Stage 1 (Initial Candidate Generation): Find the closest candidates to the Q centroid score
    unfiltered_pids, centroid_scores = get_candidates(Q, ivf)
    print(f'Stage 1 candidate generation: {unfiltered_pids.shape}')

    # Stage 2 and 3 (Centroid Interaction with Pruning, then without Pruning)
    idx = centroid_scores.max(-1).values >= CENTROID_SCORE_THRESHOLD

    # C++ : Filter pids under the centroid score threshold
    pids_true = scorer.filter_pids(
        unfiltered_pids, centroid_scores, embeddings.codes, doclens, offsets, idx, NDOCS
    )
    pids = pids_true
    assert torch.equal(pids_true, pids), f'\n{pids_true}\n{pids}'
    print('Stage 2 filtering:', unfiltered_pids.shape, '->', pids.shape) # (n_docs) -> (n_docs/4)

    # Stage 3.5 (Decompression) - Get the true passage embeddings for calculating maxsim
    D_packed = scorer.decompress_residuals(
        pids, doclens, offsets, bucket_weights, codec.reversed_bit_map,
        codec.decompression_lookup_table, embeddings.residuals, embeddings.codes, 
        centroids, codec.dim, nbits
    )
    D_packed = F.normalize(D_packed.to(torch.float32), p=2, dim=-1)
    D_mask = doclens[pids.long()]
    D_padded, D_lengths = StridedTensor(D_packed, D_mask, use_gpu=False).as_padded_tensor()
    print('Stage 3.5 decompression:', pids.shape, '->', D_padded.shape) # (n_docs/4) -> (n_docs/4, num_toks, hidden_dim)

    # Stage 4 (Final Ranking w/ Decompression) - Calculate the final (expensive) maxsim scores with ColBERT 
    scores = colbert_score(Q, D_padded, D_lengths)
    print('Stage 4 ranking:', D_padded.shape, '->', scores.shape)

    return scores, pids


def encode(text, full_length_search=False) -> torch.Tensor:
    queries = text if isinstance(text, list) else [text]
    bsize = 128 if len(queries) > 128 else None

    Q = index_checkpoint.queryFromText(
        queries, 
        bsize=bsize, 
        to_cpu=True, 
        full_length_search=full_length_search
    )

    QUERY_MAX_LEN = index_checkpoint.query_tokenizer.query_maxlen
    Q = Q[:, :QUERY_MAX_LEN] # Cut off query to maxlen tokens
    
    return Q


def search_colbert(query):
    """
    ColBERT search with a query.
    """
    # Encode query using ColBERT model, using the appropriate [Q], [D] tokens
    Q = encode(query)    

    scores, pids = _calculate_colbert(Q)

    # Sort values
    scores_sorter = scores.sort(descending=True)
    pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

    return pids, scores


if __name__ == "__main__":
    # Text-run ColBERT (useful for docker to download the model)
    init_colbert(index_path=INDEX_PATH)