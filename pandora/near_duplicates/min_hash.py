""" MinHash module

Ref: 
https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation
https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
"""

from typing import List, Callable, Iterable, Tuple
from functools import reduce
from .utils import generate_hash_fn, reproducible_randoms, preprocess
from loguru import logger
import numpy as np
import time
import sys


PRIME = 2 ** 31 - 1


class MinHash(object):
    def __init__(
        self,
        tokens: List[str],
        ngrams: int = 4,
        permutation_way: str = "perm",
        hash_fns: List[Callable] = [],
        random_seed: int = 0,
        signature_len: int = 200,
    ):
        self.tokens = tokens
        self.signature_len = signature_len

        tick = time.time()
        self.shingles = self.get_shingles(ngrams)
        tock = time.time()
        logger.trace(f"Time consumed by generating singles {tock - tick:.6f}s")

        tick = time.time()
        if permutation_way == "perm":
            self.perms = np.array(
                [
                    (rand_a, rand_b)
                    for rand_a, rand_b in zip(
                        reproducible_randoms(signature_len, random_seed),
                        reproducible_randoms(signature_len, random_seed + 1),
                    )
                ],
                dtype=int,
            ).T
            self.hash_fn = np.vectorize(
                generate_hash_fn(salt=b"perm"), otypes=[float], cache=True
            )
            self.hash_fns = None
        elif permutation_way == "hash":
            if len(hash_fns) > 0:
                self.hash_fns = hash_fns
            else:
                self.hash_fns = []
                for rand_bytes in reproducible_randoms(
                    signature_len, random_seed, "bytes"
                ):
                    self.hash_fns.append(generate_hash_fn(salt=rand_bytes))
            self.perms = None
        else:
            raise ValueError("Permutation way only support 'perm' or 'hash'.")
        tock = time.time()
        logger.trace(f"Time consumed by generating hash functions {tock - tick:.6f}s")

        tick = time.time()
        self.signature = self.get_signature()
        tock = time.time()
        logger.trace(f"Time consumed by generating signature {tock - tick:.6f}s")

    def __len__(self):
        return self.signature_len

    def get_shingles(self, ngrams: int) -> List[str]:
        shingles = []
        for i in range(len(self.tokens) - ngrams + 1):
            shingles.append(" ".join(self.tokens[i : i + ngrams]))
        return shingles

    def get_signature(self) -> np.ndarray:
        if not self.perms is None:
            hashed_shingles = self.hash_fn(np.array(self.shingles)).reshape(-1, 1)
            signature = np.full(len(self), float("inf"))
            a, b = self.perms
            signature = np.min(np.mod(hashed_shingles * a + b, PRIME), axis=0)

        elif not self.hash_fns is None:
            hash_shingles = lambda hash_fn: reduce(
                min, [hash_fn(shingle) for shingle in self.shingles]
            )
            signature = map(lambda hash_fn: hash_shingles(hash_fn), self.hash_fns)
            signature = np.array(list(signature))
        else:
            raise ValueError("No available functions to get signature")
        return signature.astype(int)

    def find_near_duplicates(
        self, signature_matrix: np.matrix, threshold: float
    ) -> List[Tuple[int, float]]:

        assert len(self) == signature_matrix.shape[1]
        sims = np.sum(self.signature == signature_matrix, axis=1) / len(self)
        index = np.where(sims > threshold)
        return [(i, sims[i]) for i in index[0]]

    def similarity_with(self, min_hash) -> float:
        assert len(self) == len(min_hash)
        return np.sum(self.signature == min_hash.signature) / len(self)


def build_signature_matrix(
    samples: Iterable[Tuple[str, int]],
    tokenizer: Callable[[str], List[str]],
    output_path: str,
    ngrams: int = 3,
    permutation_way: str = "perm",
    signature_len: int = 200,
    random_seed: int = 0,
) -> np.ndarray:
    logger.info("Building signature matrix...")

    min_hashes = []
    tick = time.time()
    logger.info("Building MinHash for each sample...")
    for sample_id, sample in enumerate(samples):
        min_hashes.append(
            MinHash(
                tokenizer(sample[0]),
                ngrams,
                permutation_way,
                random_seed=random_seed,
                signature_len=signature_len,
            )
        )
        if sample_id % 100 == 0:
            checkpoint_time = time.time()
            print(
                f"\r{sample_id}/{len(samples)}----"
                f"total time: {checkpoint_time - tick:.2f}s----"
                f"avg time: {(checkpoint_time - tick)/(sample_id + 1):.4f}s",
                end="",
            )

    print()
    logger.info("\nBuilding signature matrix using MinHashes...")
    signature_matrix = np.vstack([min_hash.signature for min_hash in min_hashes])
    tock = time.time()
    logger.info(f"Finshed, total time cost by building signature matrix {tock - tick:.2f}s")
    np.save(output_path, signature_matrix)
    logger.info(f"Signature matrix has beed saved to {output_path}")
    return signature_matrix
