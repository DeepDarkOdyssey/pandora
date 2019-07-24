import numpy as np
from typing import Callable, List, Union
from loguru import logger


class GEM(object):
    def __init__(
        self,
        string: str,
        tokenizer: Callable[[str], List[str]],
        token2emb: Callable[[str], np.array],
        corpus_singular_vectors: np.array,
        corpus_singular_values: np.array,
        window_size: int = 3,
        top_r: int = 3,
    ):
        tokens = tokenizer(string)
        self.num_tokens = len(tokens)
        self.token_embeddings = np.hstack([token2emb(token) for token in tokens])
        self.window_size = window_size
        self.top_r = top_r
        self.corpus_singular_vectors = corpus_singular_vectors
        self.corpus_singular_values = corpus_singular_values

    @staticmethod
    def build_corpus_principles(
        corpus: List[str],
        tokenizer: Callable[[str], List[str]],
        token2emb: Callable[[str], np.array],
        top_k: int = 3,
    ):
        assert len(corpus) >= top_k
        sentence_embeddings = []
        for sentence in corpus:
            tokens = tokenizer(sentence)
            token_matrix = np.hstack([token2emb(token) for token in tokens])
            u, s, _ = np.linalg.svd(token_matrix)
            coarse_sen_emb = np.sum(s ** 3 * u[:, : s.shape[0]], axis=1).reshape(-1, 1)
            sentence_embeddings.append(coarse_sen_emb)
        sentence_embedding_matrix = np.hstack(sentence_embeddings)

        u, s, _ = np.linalg.svd(sentence_embedding_matrix)
        return u[:, :top_k], s[:top_k]

    @staticmethod
    def rerank_principles(
        token_embeddings: np.ndarray,
        singular_vectors: np.ndarray,
        singular_values: np.ndarray,
        top_r: int,
    ):
        rank_values = (
            np.linalg.norm(
                np.matmul(token_embeddings.T, singular_vectors), ord=2, axis=0
            )
            * singular_values
        )
        sorted_sgl_vectors, sorted_sgl_values = [], []
        for _, sgl_vec, sgl_val in sorted(
            zip(rank_values, singular_vectors, singular_values), reverse=True
        ):
            sorted_sgl_vectors.append(sgl_vec)
            sorted_sgl_values.append(sgl_val)

        return (
            np.array(sorted_sgl_vectors[: top_r]),
            np.array(sorted_sgl_values[: top_r]),
        )

    def get_window_matrix(self, i: int):
        left_context = self.token_embeddings[:, i - self.window_size : i]
        right_context = self.token_embeddings[:, i + 1 : i + self.window_size + 1]
        window_matrix = np.hstack(
            [left_context, right_context, self.token_embeddings[:, i]]
        )
        return window_matrix

    def get_novelty_score(self, r_i: np.ndarray):
        return np.math.exp(r_i[-1] / np.linalg.norm(r_i, 2))

    def get_significance_score(self, r_i: np.ndarray):
        return r_i / (2 * self.window_size + 1)

    def get_uniqueness_score(
        self, q_i: np.ndarray, singular_vectors: np.ndarray, singular_values: np.ndarray
    ):
        uniqueness_score = np.math.exp(
            -np.linalg.norm(singular_values * np.matmul(q_i.T, singular_vectors), ord=2)
            / q_i.shape[0]
        )
        return uniqueness_score

    def build_sentence_vector(
        self,
        token_embeddings: np.ndarray,
        singular_vectors: np.ndarray,
        singular_values: np.ndarray,
    ):
        sentence_vector = np.zeros((token_embeddings.shape[0], 1))
        for i in range(token_embeddings.shape[1]):
            window_matrix = self.get_window_matrix(i)
            Q_i, R_i = np.linalg.qr(window_matrix)
            q_i = Q_i[:, -1]
            r_i = R_i[:, -1]
            novelty_score = self.get_novelty_score(r_i)
            significance_score = self.get_significance_score(r_i)
            uniqueness_score = self.get_uniqueness_score(
                q_i, singular_vectors, singular_values
            )
            token_weight = novelty_score + significance_score + uniqueness_score
            sentence_vector += token_weight * token_embeddings[:, i]

        sentence_vector -= singular_values @ singular_values.T @ sentence_vector
        return sentence_vector


def gram_schmidt_process():
    pass
