from .utils import get_hash, preprocess
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')


class MinHash(object):
    def __init__(self, text: str, ngrams: int = 4, hash_size: int = 200):
        self.text = text
        self.tokens = preprocess(text)
        self.ngrams = ngrams
        self.hash_size = hash_size

    @property
    def shingles(self):
        shingles = []
        for i in range(len(self.tokens) - self.ngrams + 1):
            shingles.append(self.tokens[i:i + self.ngrams])
        return shingles

    @property
    def min_hash(self):
        min_hash = []
        for i in range(self.hash_size):
            min_value = float('inf')
            new_hash = get_hash(str(i))
            for shingle in self.shingles:
                value = new_hash(' '.join(shingle))
                min_value = min(min_value, value)
            min_hash.append(min_value)
        return min_hash


def hash_sim(min_hash_a: MinHash, min_hash_b: MinHash) -> float:
    assert len(min_hash_a.min_hash) == len(min_hash_b.min_hash)
    count = 0
    for a, b in zip(min_hash_a.min_hash, min_hash_b.min_hash):
        if a == b:
            count += 1
    return count / len(min_hash_a.min_hash)


def find_near_duplicates(hash_vector, hash_matrix, threshold: float) -> list:
    assert len(hash_vector) == hash_matrix.shape[1]
    sims = np.sum(hash_vector == hash_matrix, axis=1) / len(hash_vector)
    index = np.where(sims > threshold)
    return index[0].tolist()