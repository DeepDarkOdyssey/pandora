import numpy as np
import pickle
from copy import copy
from typing import Callable, List, Union, Tuple, Dict, Optional
from .vocab import BaseVocab, Tokenizer
from .utils import gram_schmidt_process


class GEM(object):
    def __init__(
        self, window_size: int = 7, top_k: int = 45, top_r: int = 17, power: int = 3
    ):
        self.window_size = window_size
        self.top_k = top_k
        self.top_r = top_r
        self.power = power
        self.corpus_singular_vectors, self.corpus_singular_values = None, None

    def build_corpus_principles(
        self,
        corpus: List[str],
        vocab: BaseVocab,
        tokenizer: Tokenizer,
        save_to: Optional[str] = None,
    ):
        assert len(corpus) >= self.top_k
        coarse_sentence_embeddings = np.zeros((vocab.embed_size, len(corpus)))
        for i, text in enumerate(corpus):
            token_embeddings = vocab.text2embed(text, tokenizer)
            U, s, _ = np.linalg.svd(token_embeddings, full_matrices=False)
            coarse_sen_emb = U @ (s ** self.power)
            coarse_sentence_embeddings[:, i] = coarse_sen_emb

        U, s, _ = np.linalg.svd(coarse_sentence_embeddings, full_matrices=False)
        self.corpus_singular_vectors = U[:, : self.top_k]
        self.corpus_singular_values = s[: self.top_k]

        if save_to:
            with open(save_to, "wb") as f:
                pickle.dump(
                    {
                        "singular_vectors": self.corpus_singular_vectors,
                        "singular_values": self.corpus_singular_values,
                    },
                    f,
                )

    def load_corpus_principles(self, load_from: str):
        with open(load_from, "rb") as f:
            principles = pickle.load(f)
        self.corpus_singular_vectors = principles["singular_vectors"]
        self.corpus_singular_values = principles["singular_values"]

    def rerank_principles(
        self, token_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.corpus_singular_values is None:
            raise ValueError("corpus principles haven't been initialized")
        rank_values: np.ndarray = (
            np.linalg.norm(
                token_embeddings.T @ self.corpus_singular_vectors, ord=2, axis=0
            )
            * self.corpus_singular_values
        )
        rank_indexes = np.argsort(rank_values)[::-1][: self.top_r]
        return (
            self.corpus_singular_vectors[:, rank_indexes],
            self.corpus_singular_values[rank_indexes],
        )

    def window_matrix(self, i: int, token_embeddings: np.ndarray) -> np.ndarray:
        left_context = token_embeddings[:, i - self.window_size : i]
        right_context = token_embeddings[:, i + 1 : i + self.window_size + 1]
        window_matrix = np.hstack(
            [left_context, right_context, token_embeddings[:, [i]]]
        )
        return window_matrix

    @staticmethod
    def get_novelty_score(r_i: np.ndarray):
        return np.math.exp(r_i[-1] / np.linalg.norm(r_i, ord=2) + 1e-18)

    @staticmethod
    def get_significance_score(r_i: np.ndarray, window_size: int):
        return r_i[-1] / window_size

    @staticmethod
    def get_uniqueness_score(
        q_i: np.ndarray, singular_vectors: np.ndarray, singular_values: np.ndarray
    ):
        uniqueness_score = np.math.exp(
            -np.linalg.norm(singular_values * (q_i @ singular_vectors), ord=2)
            / singular_values.shape[0]
        )
        return uniqueness_score

    def encode_text(self, text: str, vocab: BaseVocab, tokenizer: Tokenizer):
        token_embeddings = vocab.text2embed(text, tokenizer)
        singular_vectors, singular_values = self.rerank_principles(token_embeddings)
        token_weights = []
        for i in range(token_embeddings.shape[1]):
            window_matrix = self.window_matrix(i, token_embeddings)
            # Q_i, R_i = np.linalg.qr(window_matrix)
            Q_i, R_i = gram_schmidt_process(window_matrix)
            q_i = Q_i[:, -1]
            r_i = R_i[:, -1]
            novelty_score = self.get_novelty_score(r_i)
            significance_score = self.get_significance_score(
                r_i, window_matrix.shape[1]
            )
            uniqueness_score = self.get_uniqueness_score(
                q_i, singular_vectors, singular_values
            )
            token_weight = novelty_score + significance_score + uniqueness_score
            token_weights.append(token_weight)

        text_embedding = token_embeddings @ np.array(token_weights)
        text_embedding -= singular_vectors @ singular_vectors.T @ text_embedding
        return np.ravel(text_embedding)
