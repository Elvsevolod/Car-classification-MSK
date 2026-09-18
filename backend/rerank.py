"""Streaming-safe k-reciprocal reranking over one query and a static gallery."""
import numpy as np

from .core import normalize
from .scoring import ranked_queries

ACTIVE_K1 = 20
ACTIVE_K2 = 3
ACTIVE_LAMBDA = .5


class KReciprocalReranker:
    """Precompute a gallery graph, then rerank each query independently.

    This is a streaming adaptation of Zhong et al. (CVPR 2017): gallery
    k-reciprocal encodings are static, and no other query is ever visible.
    """

    def __init__(self, gallery_vectors, k1=20, k2=6):
        self.gallery = normalize(gallery_vectors)
        self.k1 = int(k1)
        self.k2 = int(k2)
        count = len(self.gallery)
        if not 1 <= self.k1 < count:
            raise ValueError("k1 must be between 1 and gallery_size - 1")
        if not 1 <= self.k2 <= self.k1 + 1:
            raise ValueError("k2 must be between 1 and k1 + 1")

        cosine = np.clip(self.gallery @ self.gallery.T, -1, 1)
        raw_distance = 1 - cosine
        scale = np.maximum(raw_distance.max(axis=1, keepdims=True), 1e-12)
        self.distance = (raw_distance / scale).astype(np.float32)
        self.initial_rank = np.argsort(self.distance, axis=1, kind="stable")
        self.base_encoding = self._gallery_encoding()
        if self.k2 > 1:
            neighbors = self.initial_rank[:, :self.k2]
            self.encoding = self.base_encoding[neighbors].mean(axis=1).astype(np.float32)
        else:
            self.encoding = self.base_encoding
        self.inverted = [np.flatnonzero(self.encoding[:, column]) for column in range(count)]
        cutoff_position = min(self.k1, count - 1)
        self.reciprocal_cutoff = raw_distance[np.arange(count), self.initial_rank[:, cutoff_position]]

    def _reciprocal(self, index, size):
        width = min(size + 1, len(self.gallery))
        forward = self.initial_rank[index, :width]
        backward = self.initial_rank[forward, :width]
        return forward[np.any(backward == index, axis=1)]

    def _gallery_encoding(self):
        count = len(self.gallery)
        encoding = np.zeros((count, count), dtype=np.float32)
        half = int(np.around(self.k1 / 2))
        for index in range(count):
            reciprocal = self._reciprocal(index, self.k1)
            expansion = [reciprocal]
            for candidate in reciprocal:
                candidate_set = self._reciprocal(int(candidate), half)
                overlap = np.intersect1d(candidate_set, reciprocal, assume_unique=True)
                if len(overlap) > 2 / 3 * len(candidate_set):
                    expansion.append(candidate_set)
            expanded = np.unique(np.concatenate(expansion))
            weights = np.exp(-self.distance[index, expanded])
            encoding[index, expanded] = weights / weights.sum()
        return encoding

    def components(self, query_vector):
        """Return normalized raw and Jaccard distances for one query."""
        query = normalize(query_vector)
        raw_distance = 1 - np.clip(self.gallery @ query, -1, 1)
        raw = raw_distance / max(float(raw_distance.max()), 1e-12)
        order = np.argsort(raw_distance, kind="stable")
        reciprocal = order[:self.k1]
        reciprocal = reciprocal[raw_distance[reciprocal] <= self.reciprocal_cutoff[reciprocal]]

        query_encoding = np.zeros(len(self.gallery), dtype=np.float32)
        if len(reciprocal):
            half = int(np.around(self.k1 / 2))
            expansion = [reciprocal]
            reciprocal_set = set(reciprocal.tolist())
            for candidate in reciprocal:
                candidate_set = self._reciprocal(int(candidate), half)
                overlap = sum(int(value) in reciprocal_set for value in candidate_set)
                if overlap > 2 / 3 * len(candidate_set):
                    expansion.append(candidate_set)
            expanded = np.unique(np.concatenate(expansion))
            weights = np.exp(-raw[expanded])
            # The original algorithm also assigns weight 1 to the query itself.
            query_encoding[expanded] = weights / (1 + weights.sum())

        if self.k2 > 1:
            neighbors = order[:self.k2 - 1]
            query_encoding = np.vstack([query_encoding, self.base_encoding[neighbors]]).mean(axis=0)

        minimum = np.zeros(len(self.gallery), dtype=np.float32)
        for column in np.flatnonzero(query_encoding):
            rows = self.inverted[int(column)]
            minimum[rows] += np.minimum(query_encoding[column], self.encoding[rows, column])
        jaccard = 1 - minimum / np.maximum(2 - minimum, 1e-12)
        return raw.astype(np.float32), jaccard.astype(np.float32)

    def distances(self, query_vector, lambda_value=.3):
        if not 0 <= lambda_value <= 1:
            raise ValueError("lambda_value must be in [0, 1]")
        raw, jaccard = self.components(query_vector)
        return ((1 - lambda_value) * jaccard + lambda_value * raw).astype(np.float32)


def rerank_protocol(queries, gallery, embeddings, k1=ACTIVE_K1, k2=ACTIVE_K2,
                    lambda_value=ACTIVE_LAMBDA):
    """Rerank a labeled local protocol and return raw-cosine refusal scores."""
    gallery_vectors = np.stack([embeddings[row["image_id"]] for row in gallery])
    reranker = KReciprocalReranker(gallery_vectors, k1, k2)
    scores = []
    for query in queries:
        vector = embeddings[query["image_id"]]
        distances = reranker.distances(vector, lambda_value)
        scores.append(-distances)
    ranked = ranked_queries(queries, gallery, embeddings, scores)
    return ranked, ranked.confidence
