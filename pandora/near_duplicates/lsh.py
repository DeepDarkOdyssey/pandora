import numpy as np
from typing import Dict, DefaultDict, Optional
from collections import defaultdict
from .utils import generate_hash_fn, reproducible_randoms


class LSH(object):
    def __init__(
        self,
        feature_matrix: np.ndarray,
        num_bands: Optional[int] = None,
        num_cols: Optional[int] = None,
        num_buckets: int = 20,
        random_seed: int = 0,
    ):
        self.feature_matrix = feature_matrix
        if num_cols:
            self.cols_per_band = num_cols
        elif num_bands:
            self.cols_per_band = self.feature_matrix.shape[1] // num_bands + 1
        else:
            raise TypeError("Either 'num_bands' or 'num_cols should be specified")

        self.hash_fns = []
        for rand_bytes in reproducible_randoms(num_bands, random_seed, "bytes"):
            self.hash_fns.append(generate_hash_fn(num_buckets, salt=rand_bytes))

        self.hash_table = self.build_hash_table()

    def build_hash_table(self) -> DefaultDict[int, Dict[int, int]]:
        hash_table = defaultdict(dict)
        for band_id, hash_fn in enumerate(self.hash_fns):
            band = self.feature_matrix[
                :, band_id * self.cols_per_band : (band_id + 1) * self.cols_per_band
            ]
            hashed_band = np.apply_along_axis(
                lambda x: hash_fn(x.tobytes()), axis=1, arr=band
            )
            for row_id, hash_val in enumerate(hashed_band):
                # TODO: figure out how to deal with hash collision
                hash_table[hash_val][row_id] = band_id
        return hash_table

    def get_candidates(self, inputs: np.ndarray, threshold: int = 1) -> list:
        candidates = defaultdict(int)
        for band_id, hash_fn in enumerate(self.hash_fns):
            band = inputs[
                band_id * self.cols_per_band : (band_id + 1) * self.cols_per_band
            ]
            hashed_band = hash_fn(band.tobytes())

            if hashed_band in self.hash_table:
                for row_id in self.hash_table[hashed_band].keys():
                    candidates[row_id] += 20
        candidates = [
            row_id for row_id, count in candidates.items() if count >= threshold
        ]
        return candidates
