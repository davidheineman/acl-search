import os, ujson, tqdm
import torch as th
import torch.nn.functional as F

from colbert.infra.config import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from colbert.search.strided_tensor import StridedTensor
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.indexing.codecs.residual import ResidualCodec

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager

from constants import INDEX_PATH


NCELLS = 1  # Number of self.centroids to use in PLAID
CENTROID_SCORE_THRESHOLD = 0.5 # How close a document has to be to a centroid to be considered
NDOCS = 512  # Number of closest documents to consider


class ColBERT():
    def __init__(self, index_path: str):
        """
        Load all tensors necessary for running ColBERT
        """
        if th.cuda.is_available(): 
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            from colbert.modeling.colbert import ColBERT
            ColBERT.try_load_torch_extensions(False)
        
        # Load configuation
        index_config: ColBERTConfig = ColBERTConfig.load_from_index(index_path)
        checkpoint_path: str        = index_config.checkpoint # 'colbert-ir/colbertv2.0'
        load_index_with_mmap: bool  = index_config.load_index_with_mmap

        # Load tokenizers
        self.query_tokenizer: QueryTokenizer    = QueryTokenizer(index_config)
        self.doc_tokenizer: DocTokenizer        = DocTokenizer(index_config)
        self.amp_manager: MixedPrecisionManager = MixedPrecisionManager(True)

        # Load model and index
        self.load_model(checkpoint_path, index_config)
        self.load_index(index_path, load_index_with_mmap)


    def load_model(self, checkpoint_path: str, config: ColBERTConfig):
        from colbert.modeling.hf_colbert import class_factory
        from transformers import AutoTokenizer

        try:
            HFModelFactory = class_factory(checkpoint_path)
        except:
            checkpoint_path = 'bert-base-uncased'
            HFModelFactory = class_factory(checkpoint_path)
        
        model = HFModelFactory.from_pretrained(checkpoint_path, colbert_config=config)
        raw_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        model.colbert_config = config
        model.eval()
        self.pad_token = raw_tokenizer.pad_token_id
        if config.mask_punctuation:
            import string
            model.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [symbol, raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }
        self.colbert_model = model


    def load_index(self, index_path: str, load_index_with_mmap: bool=False):
        """ Utilities for PLAID indexing """
        use_gpu = 'cuda' in self.device
        
        load_index_with_mmap = load_index_with_mmap
        if load_index_with_mmap and self.device != 'cpu':
            raise ValueError(f"Memory-mapped index can only be used with CPU!")

        self.scorer: IndexScorer = IndexScorer(index_path, use_gpu, load_index_with_mmap)

        with open(os.path.join(index_path, 'metadata.json')) as f:
            metadata = ujson.load(f)
        self.nbits = metadata['config']['nbits']

        self.centroids = th.load(os.path.join(index_path, 'centroids.pt'), map_location=self.device)
        self.centroids: th.FloatTensor = self.centroids.float()

        self.ivf, self.ivf_lengths = th.load(os.path.join(index_path, "ivf.pid.pt"), map_location=self.device)
        self.ivf: StridedTensor = StridedTensor(self.ivf, self.ivf_lengths, use_gpu=False)

        self.embeddings: ResidualCodec = ResidualCodec.Embeddings.load_chunks(
            index_path,
            range(metadata['num_chunks']),
            metadata['num_embeddings'],
            load_index_with_mmap=load_index_with_mmap,
        )

        self.doclens = []
        for chunk_idx in tqdm.tqdm(range(metadata['num_chunks'])):
            with open(os.path.join(index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)
                self.doclens.extend(chunk_doclens)
        self.doclens: th.Tensor = th.tensor(self.doclens)

        buckets_path = os.path.join(index_path, 'buckets.pt')
        bucket_cutoffs, self.bucket_weights = th.load(buckets_path, map_location=self.device)
        self.bucket_weights: th.FloatTensor = self.bucket_weights.float()

        self.codec: ResidualCodec = ResidualCodec.load(index_path)

        if load_index_with_mmap:
            assert metadata['num_chunks'] == 1
            self.offsets = th.cumsum(self.doclens, dim=0)
            self.offsets = th.cat((th.zeros(1, dtype=th.int64), self.offsets))
        else:
            self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)
            self.offsets = self.embeddings_strided.codes_strided.offsets


    def colbert_score(self, Q: th.Tensor, D_padded: th.Tensor, D_mask: th.Tensor) -> th.Tensor:
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


    def get_candidates(self, Q: th.Tensor, ivf: StridedTensor) -> th.Tensor:
        """
        First find self.centroids closest to Q, then return all the passages in all
        self.centroids.

        We can replace this function with a k-NN search finding the closest passages
        using BERT similarity.
        """
        Q = Q.squeeze(0)

        # Get the closest self.centroids via a matrix multiplication + argmax
        centroid_scores: th.Tensor = (self.centroids @ Q.T)
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
        pids, _ = th.unique_consecutive(pids, return_counts=True)
        return pids, centroid_scores


    def _calculate_colbert(self, Q: th.Tensor):
        """
        Multi-stage ColBERT pipeline. Implemented using the PLAID engine, see fig. 5:
        https://arxiv.org/pdf/2205.09707#page=5
        """
        # Stage 1 (Initial Candidate Generation): Find the closest candidates to the Q centroid score
        unfiltered_pids, centroid_scores = self.get_candidates(Q, self.ivf)
        print(f'Stage 1 candidate generation: {unfiltered_pids.shape}')

        # Stage 2 and 3 (Centroid Interaction with Pruning, then without Pruning)
        idx = centroid_scores.max(-1).values >= CENTROID_SCORE_THRESHOLD

        # C++ : Filter pids under the centroid score threshold
        pids_true = self.scorer.filter_pids(
            unfiltered_pids, centroid_scores, self.embeddings.codes, self.doclens, self.offsets, idx, NDOCS
        )
        pids = pids_true
        assert th.equal(pids_true, pids), f'\n{pids_true}\n{pids}'
        print('Stage 2 filtering:', unfiltered_pids.shape, '->', pids.shape) # (n_docs) -> (n_docs/4)

        # Stage 3.5 (Decompression) - Get the true passage self.embeddings for calculating maxsim
        D_packed = self.scorer.decompress_residuals(
            pids, self.doclens, self.offsets, self.bucket_weights, self.codec.reversed_bit_map,
            self.codec.decompression_lookup_table, self.embeddings.residuals, self.embeddings.codes, 
            self.centroids, self.codec.dim, self.nbits
        )
        D_packed = F.normalize(D_packed.to(th.float32), p=2, dim=-1)
        D_mask = self.doclens[pids.long()]
        D_padded, D_lengths = StridedTensor(D_packed, D_mask, use_gpu=False).as_padded_tensor()
        print('Stage 3.5 decompression:', pids.shape, '->', D_padded.shape) # (n_docs/4) -> (n_docs/4, num_toks, hidden_dim)

        # Stage 4 (Final Ranking w/ Decompression) - Calculate the final (expensive) maxsim scores with ColBERT 
        scores = self.colbert_score(Q, D_padded, D_lengths)
        print('Stage 4 ranking:', D_padded.shape, '->', scores.shape)

        return scores, pids
    

    def _mask(self, input_ids: th.Tensor, skiplist: th.Tensor) -> th.Tensor:
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask
    

    def _query_forward(self, input_ids: th.Tensor, attention_mask: th.Tensor) -> th.Tensor:
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.colbert_model.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.colbert_model.linear(Q)

        mask = th.tensor(self._mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        Q = F.normalize(Q, p=2, dim=2)
        Q = Q.to(self.device)
        
        return Q


    def encode(self, queries: list[str], full_length_search=False, context=None, bsize=128) -> th.Tensor:
        bsize = bsize if len(queries) > bsize else None

        if bsize:
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize, full_length_search=full_length_search)
            batches = [self._query_forward(input_ids, attention_mask) for input_ids, attention_mask in batches]
            return th.cat(batches)
        else:
            input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context, full_length_search=full_length_search)

            with th.no_grad():
                with self.amp_manager.context():
                    Q = self._query_forward(input_ids, attention_mask)

        QUERY_MAX_LEN = self.query_tokenizer.query_maxlen
        Q = Q[:, :QUERY_MAX_LEN] # Cut off query to maxlen tokens
        
        return Q


    def search(self, queries):
        """
        ColBERT search with a query.
        """
        queries = queries if isinstance(queries, list) else [queries]

        # Encode query using ColBERT model, using the appropriate [Q], [D] tokens
        Q = self.encode(queries)    

        scores, pids = self._calculate_colbert(Q)

        # Sort values
        scores_sorter = scores.sort(descending=True)
        pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores


if __name__ == "__main__":
    # Test-run ColBERT (useful for docker to download the model)
    colbert = ColBERT(index_path=INDEX_PATH)
    print(colbert.search('text simplificaiton'))