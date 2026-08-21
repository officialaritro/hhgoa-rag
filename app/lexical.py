"""BM25 lexical retrieval over the passage store.

Why lexical at all: MS MARCO is a keyword-heavy web-search corpus, and BM25
fails on different queries than a dense bi-encoder does. That difference is the
only reason fusing them can beat either alone -- two rankings that agree add
nothing. The measured evidence for trying it is in docs/CHUNKING_REPORT.md,
where every dense splitting strategy lands within noise of every other, so the
remaining headroom is in scoring rather than segmentation.

Why hand-rolled: sklearn and scipy are already dependencies, and BM25 reduces to
one precomputed sparse weight matrix plus a column sum. A new package would mean
a lock update and a CI resolution on a deadline for about forty lines of
well-understood arithmetic.

Neither torch nor faiss is imported here, so this module is safe to use from
either half of the two-process build.
"""

import pickle
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Standard Okapi BM25 parameters. k1 controls how quickly repeated terms
# saturate; b controls how much document length is penalised.
K1 = 1.5
B = 0.75

PICKLE_PROTOCOL = 4


@dataclass
class BM25Index:
    """A BM25 index as a precomputed weight matrix.

    `weights[d, t]` already contains that term's full contribution to document
    d's score, so scoring a query is `weights[:, query_terms].sum(axis=1)` -- no
    per-query length normalisation or IDF lookup.
    """

    weights: sp.csc_matrix
    vocabulary: dict[str, int]
    n_documents: int

    @classmethod
    def build(cls, documents: list[str]) -> "BM25Index":
        vectorizer = CountVectorizer(lowercase=True)
        try:
            counts = vectorizer.fit_transform(documents).tocsr().astype("float32")
        except ValueError:
            # sklearn raises when the vocabulary comes out empty, which happens
            # for a corpus whose tokens are all shorter than its default token
            # pattern accepts. Degenerate, but the build must not die on
            # unexpected corpus content -- an index that scores everything zero
            # is recoverable, a crashed build at 3am is not.
            return cls(
                weights=sp.csc_matrix((len(documents), 0), dtype="float32"),
                vocabulary={},
                n_documents=len(documents),
            )
        n_docs = counts.shape[0]

        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = float(lengths.mean()) if n_docs else 0.0

        # df per term, then the standard BM25 idf with the +0.5 smoothing that
        # keeps a term appearing in every document from going negative.
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(
            1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype("float32")

        # Saturation and length normalisation applied to the nonzeros in place:
        #   weight = idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len_d / avgdl))
        weights = counts.tocoo()
        denominator_per_doc = K1 * (
            1.0 - B + B * (lengths / average_length if average_length else lengths)
        )
        frequencies = weights.data
        weights.data = (
            idf[weights.col]
            * frequencies
            * (K1 + 1.0)
            / (frequencies + denominator_per_doc[weights.row])
        ).astype("float32")

        return cls(
            weights=weights.tocsc(),
            vocabulary={term: int(i) for term, i in vectorizer.vocabulary_.items()},
            n_documents=int(n_docs),
        )

    def _query_columns(self, query: str) -> list[int]:
        # Tokenized the same way CountVectorizer did, so "The Coati," matches
        # the indexed token "coati".
        tokenizer = CountVectorizer(lowercase=True).build_analyzer()
        return [
            self.vocabulary[token]
            for token in tokenizer(query)
            if token in self.vocabulary
        ]

    def scores(self, query: str) -> np.ndarray:
        """BM25 score per document. Zero for every document on an unknown term,
        rather than raising -- an out-of-vocabulary query word is normal."""
        columns = self._query_columns(query)
        if not columns:
            return np.zeros(self.n_documents, dtype="float32")
        return np.asarray(self.weights[:, columns].sum(axis=1)).ravel()

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """The k best-matching documents, best first.

        Zero-scoring documents are omitted rather than padding the list: a
        document that matched no query term is not a lexical candidate, and
        handing RRF an arbitrary rank for it would fuse noise.
        """
        scores = self.scores(query)
        matched = np.flatnonzero(scores > 0)
        if matched.size == 0:
            return []
        top = matched[np.argsort(-scores[matched])[:k]]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=PICKLE_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)
