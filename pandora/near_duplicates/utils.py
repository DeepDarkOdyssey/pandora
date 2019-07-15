from typing import Callable, List, Union, Optional
from hashlib import blake2b, md5
from string import punctuation
from math import ceil
import numpy as np
import random
import spacy
import time


MAX_INT = 2 ** 32 - 1


def generate_hash_fn(
    bit_len: int = 32, salt: bytes = b'', encoding: str = "utf8"
) -> Callable[[Union[str, bytes]], int]:
    digest_size = ceil(bit_len / 8)

    def hash_fn(inputs: Union[str, bytes]) -> int:
        if type(inputs) is str:
            inputs = bytes(inputs.encode(encoding))
        elif not type(inputs) is bytes:
            raise TypeError("Hash inputs only support `string` and `bytes`")
        blake = blake2b(inputs, digest_size=digest_size, salt=salt)
        return int(blake.hexdigest(), 16)

    return hash_fn


def reproducible_randoms(
    nums: int, seed: int = 0, dtype="int"
) -> Union[int, str, bytes]:
    random.seed(seed)
    for _ in range(nums):
        rand = random.randint(0, MAX_INT)
        if dtype == "int":
            pass
        elif dtype == "str":
            rand = str(rand)
        elif dtype == "bytes":
            rand = bytes(str(rand).encode())
        else:
            raise TypeError("dtype argument only support 'int', 'str' or 'bytes'.")
        yield rand


def preprocess(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    result = []
    entity = ""
    for token in doc:
        if token.ent_iob_ == "B":
            entity = token.ent_type_
        elif token.ent_iob_ == "I":
            assert entity == token.ent_type_  # Sanity Check
        elif token.ent_iob_ == "O":
            if entity:
                result.append(entity)
                entity = ""
            if not token.text.strip():
                continue
            if token.is_punct:
                continue
            if token.is_upper:
                result.append(token.text)
            else:
                result.append(token.lemma_)
        else:
            raise TypeError('Entity type not in "BIO"')

    return result


def jaccard_similarity(set_a: set, set_b: set) -> float:
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
