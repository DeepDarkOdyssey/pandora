from typing import List
from .utils import generate_hash_fn, preprocess, reproducible_randoms
import numpy as np


class MinHash(object):
    def __init__(self,
                 text: str,
                 ngrams: int = 4,
                 seed: int = 0,
                 hash_size: int = 200):
        self.text = text
        self.tokens = preprocess(text)
        self.ngrams = ngrams
        self.hash_fns = []
        for rand_bytes in reproducible_randoms(hash_size, seed):
            self.hash_fns.append(generate_hash_fn(salt=rand_bytes))

    @property
    def shingles(self):
        shingles = []
        for i in range(len(self.tokens) - self.ngrams + 1):
            shingles.append(self.tokens[i:i + self.ngrams])
        return shingles

    @property
    def signature(self):
        signature = []
        for hash_fn in self.hash_fns:
            min_value = float('inf')
            for shingle in self.shingles:
                value = hash_fn(' '.join(shingle))
                min_value = min(min_value, value)
            signature.append(min_value)
        return signature

    @classmethod
    def build_signature_matrix(cls, min_hashes) -> np.ndarray:
        hash_values = []
        for min_hash_obj in min_hashes:
            hash_values.append(min_hash_obj.min_hash)
        return np.array(hash_values)

    @classmethod
    def hash_sim(cls, min_hash_a, min_hash_b) -> float:
        assert len(min_hash_a.min_hash) == len(min_hash_b.min_hash)
        count = 0
        for a, b in zip(min_hash_a.min_hash, min_hash_b.min_hash):
            if a == b:
                count += 1
        return count / len(min_hash_a.min_hash)

    @classmethod
    def find_near_duplicates(cls, hash_vector: np.ndarray,
                             hash_matrix: np.matrix, threshold: float) -> list:
        assert len(hash_vector) == hash_matrix.shape[1]
        sims = np.sum(hash_vector == hash_matrix, axis=1) / len(hash_vector)
        index = np.where(sims > threshold)
        return index[0].tolist()