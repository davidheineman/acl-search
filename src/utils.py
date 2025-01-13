import torch
import tqdm
import os
import shutil

from constants import INDEX_PATH, INDEX_ROOT, DATA_PATH, HF_INDEX_REPO


def download_index_from_hf():
    """ Download the pre-built ColBERT index from HF if one does not exist """
    from huggingface_hub import snapshot_download

    if not os.path.isdir(INDEX_PATH):
        print(f'Did not find "{INDEX_PATH}", loading pre-built index from HuggingFace')
        snapshot_download(repo_id=HF_INDEX_REPO, local_dir=INDEX_ROOT, ignore_patterns=["README.md"])

        # Move papers.json -> data/papers.json
        os.makedirs(os.path.dirname(os.path.join(DATA_PATH, 'papers.json')), exist_ok=True)
        shutil.move(os.path.join(INDEX_ROOT, 'papers.json'), os.path.join(DATA_PATH, 'papers.json'))


def maxsim(pids, centroid_scores, codes, doclens, offsets, idx, nfiltered_docs):
    ncentroids, nquery_vectors = centroid_scores.shape
    centroid_scores = centroid_scores.flatten()
    scores = []

    for i in tqdm.tqdm(range(len(pids)), desc='Calculating maxsim over centroids...'):
        seen_codes = set()
        per_doc_scores = torch.full((nquery_vectors,), -9999, dtype=torch.float32)

        pid = pids[i]
        for j in range(doclens[pid]):
            code = codes[offsets[pid] + j]
            assert code < ncentroids
            if idx[code] and code not in seen_codes:
                for k in range(nquery_vectors):
                    per_doc_scores[k] = torch.max(
                        per_doc_scores[k], 
                        centroid_scores[code * nquery_vectors + k]
                    )
                seen_codes.add(code)

        score = torch.sum(per_doc_scores[:nquery_vectors]).item()
        scores += [(score, pid)]

    # Sort and return scores
    global_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    filtered_pids = [pid for _, pid in global_scores[:nfiltered_docs]]
    filtered_pids = torch.tensor(filtered_pids, dtype=torch.int32)

    return filtered_pids


def filter_pids(pids, centroid_scores, codes, doclens, offsets, idx, nfiltered_docs):
    filtered_pids = maxsim(
        pids, centroid_scores, codes, doclens, offsets, idx, nfiltered_docs
    )

    print('Stage 2 filtering:', pids.shape, '->', filtered_pids.shape) # (all_docs) -> (n_docs/4)

    nfinal_filtered_docs = int(nfiltered_docs / 4)
    ones = [True] * centroid_scores.size(0)

    final_filtered_pids = maxsim(
        filtered_pids, centroid_scores, codes, doclens, offsets, ones, nfinal_filtered_docs
    )

    print('Stage 3 filtering:', filtered_pids.shape, '->', final_filtered_pids.shape) # (n_docs) -> (n_docs/4)

    return final_filtered_pids


def decompress_residuals(pids, doclens, offsets, bucket_weights, reversed_bit_map,
        bucket_weight_combinations, binary_residuals, codes, 
        centroids, dim, nbits):
    npacked_vals_per_byte = 8 // nbits
    packed_dim = dim // npacked_vals_per_byte
    cumulative_lengths = [0 for _ in range(len(pids)+1)]
    noutputs = 0
    for i in range(len(pids)):
        noutputs += doclens[pids[i]]
        cumulative_lengths[i + 1] = cumulative_lengths[i] + doclens[pids[i]]

    output = []

    binary_residuals = binary_residuals.flatten()
    centroids = centroids.flatten()

    # Iterate over all documents
    for i in range(len(pids)):
        pid = pids[i]

        # Offset into packed list of token vectors for the given document
        offset = offsets[pid]

        # For each document, iterate over all token vectors
        for j in range(doclens[pid]):
            code = codes[offset + j]

            # For each token vector, iterate over the packed (8-bit) residual values
            for k in range(packed_dim):
                x = binary_residuals[(offset + j) * packed_dim + k]
                x = reversed_bit_map[x]

                # For each packed residual value, iterate over the bucket weight indices.
                # If we use n-bit compression, that means there will be (8 / n) indices per packed value.
                for l in range(npacked_vals_per_byte):
                    output_dim_idx = k * npacked_vals_per_byte + l
                    bucket_weight_idx = bucket_weight_combinations[x * npacked_vals_per_byte + l]
                    output[(cumulative_lengths[i] + j) * dim + output_dim_idx] = \
                        bucket_weights[bucket_weight_idx] + centroids[code * dim + output_dim_idx]

    return output


def print_estimate_cost(prompt: list[str], model: str="gpt-4o", input_cost: float=5, output_cost: float=15, estimated_output_toks: int=None):
    """
    See: https://openai.com/api/pricing
    """
    from tiktoken import encoding_for_model
    enc = encoding_for_model(model)

    input_toks = 0
    for p in prompt:
        encoding = enc.encode(p)
        input_toks += len(encoding)

    input_cost  = (input_toks * (input_cost / 1_000_000))

    # STRONG ASSUMPTION -- our output will be 
    if estimated_output_toks is None: estimated_output_toks = input_toks
    output_cost = (estimated_output_toks * (output_cost / 1_000_000)) 

    cost = input_cost + output_cost

    print(f'Cost: ${cost:.4f} on "{model}" for {input_toks} input toks + {estimated_output_toks} output toks.')