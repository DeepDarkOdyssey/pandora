import numpy as np
from typing import Callable, List, Union, Tuple, Dict
from .vocab import BaseVocab, Tokenizer


class GEM(object):
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        token2embed: Dict[str, np.array],
        corpus_singular_vectors: np.array,
        corpus_singular_values: np.array,
        window_size: int = 7,
        top_k: int = 45,
        top_r: int = 17,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.top_k = top_k
        self.top_r = top_r
        self.num_tokens = len(tokens)
        self.window_size = window_size
        self.embed_size = iter(token2embed.values()).shape[0]
        self.corpus_singular_vectors, self.corpus_singular_values = None, None

    def build_corpus_principles(
        self, corpus: List[str], vocab: BaseVocab, tokenizer: Tokenizer, power: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(corpus) >= self.top_k
        coarse_sentence_embeddings = np.zeros((self.embed_size, len(corpus)))
        for i, text in enumerate(corpus):
            token_embeddings = vocab.text2embed(text, tokenizer)
            U, s, _ = np.linalg.svd(token_embeddings, full_matrices=False)
            coarse_sen_emb = U @ (s ** power)
            coarse_sentence_embeddings[:, i] = coarse_sen_emb

        U, s, _ = np.linalg.svd(coarse_sentence_embeddings, full_matrices=False)
        self.corpus_singular_vectors = U[:, : self.top_k]
        self.corpus_singular_values = s[: self.top_k]
        return U[:, : self.top_k], s[: self.top_k]

    def rerank_principles(
        self, token_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rank_values: np.ndarray = (
            np.linalg.norm(
                token_embeddings.T @ self.corpus_singular_vectors, ord=2, axis=0
            )
            * self.corpus_singular_values
        )
        rank_indexes = np.argsort(rank_values)[::-1][: self.top_r]
        return (
            self.corpus_singular_vectors[:, rank_indexes],
            self.corpus_singular_values[:, rank_indexes],
        )

    def window_matrix(self, i: int, token_embeddings: np.ndarray) -> np.ndarray:
        left_context = token_embeddings[:, i - self.window_size : i]
        right_context = token_embeddings[:, i + 1 : i + self.window_size + 1]
        window_matrix = np.hstack(
            [left_context, right_context, token_embeddings[:, [i]]]
        )
        return window_matrix

    def get_novelty_score(self, r_i: np.ndarray):
        return np.math.exp(r_i[-1] / np.linalg.norm(r_i, ord=2) + 1e-18)

    def get_significance_score(self, r_i: np.ndarray, window_size: int):
        return r_i[-1] / window_size

    def get_uniqueness_score(
        self, q_i: np.ndarray, singular_vectors: np.ndarray, singular_values: np.ndarray
    ):
        uniqueness_score = np.math.exp(
            -np.linalg.norm(singular_values * (q_i @ singular_vectors), ord=2)
            / singular_values.shape[0]
        )
        return uniqueness_score

    def encode_text(
        self,
        token_embeddings: np.ndarray,
        top_r: int,
    ):
        singular_vectors, singular_values = self.rerank_principles(token_embeddings)
        token_weights = []
        for i in range(token_embeddings.shape[1]):
            window_matrix = self.window_matrix(i, token_embeddings)
            Q_i, R_i = np.linalg.qr(window_matrix)
            q_i = Q_i[:, -1]
            r_i = R_i[:, -1]
            novelty_score = self.get_novelty_score(r_i)
            significance_score = self.get_significance_score(r_i, window_matrix.shape[1])
            uniqueness_score = self.get_uniqueness_score(
                q_i, singular_vectors, singular_values
            )
            token_weight = novelty_score + significance_score + uniqueness_score
            token_weights.append(token_weight)
        
        text_embedding = np.array(token_weights) @ token_embeddings
        text_embedding -= singular_vectors @ singular_vectors.T @ text_embedding
        return np.ravel(text_embedding)


def gram_schmidt_process():
    pass