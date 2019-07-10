import numpy as np
from collections import defaultdict
from .utils import generate_hash_fn, reproducible_randoms


class LSH(object):
    def __init__(self, feature_matrix: np.ndarray, num_bands: int,
                 num_buckets: int):
        self.feature_matrix = feature_matrix
        self.hash_table = defaultdict(list)
        cols_per_band = feature_matrix.shape[1] // num_bands + 1

        for i, rand_bytes in enumerate(reproducible_randoms(num_bands)):
            band = self.feature_matrix[:, i * cols_per_band:(i + 1) *
                                       cols_per_band]
            hash_fn = generate_hash_fn(salt=rand_bytes)
            hashed_band = np.apply_along_axis(lambda x: hash_fn(x.tobytes()),
                                              axis=1,
                                              arr=band)
            for j in range(self.feature_matrix.shape[0]):
                self.hash_table[hashed_band[j]].append((j, i))