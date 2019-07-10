from typing import Callable
from hashlib import blake2b
from string import punctuation
import random
import spacy

nlp = spacy.load('en_core_web_sm')


def generate_hash_fn(bit_len: int = 64,
                     salt: bytes = b'',
                     encoding: str = 'utf8') -> Callable:
    digest_size = int(bit_len / 8)

    def hash_fn(inputs):
        if type(inputs) is str:
            inputs = bytes(inputs.encode(encoding))
        elif not type(inputs) is bytes:
            raise TypeError('Hash inputs only support `string` and `bytes`')
        blake = blake2b(inputs, digest_size=digest_size, salt=salt)
        return int(blake.hexdigest(), 16)

    return hash_fn


def reproducible_randoms(nums: int, seed: int = 0):
    random.seed(seed)
    for _ in range(nums):
        yield str(random.random())[2:].encode()[:16]


def preprocess(text: str):
    doc = nlp(text)
    new_text = ''
    prev_end_char = 0
    if len(doc.ents) > 0:
        for ent in doc.ents:
            new_text += text[prev_end_char:ent.start_char] + ent.label_
            prev_end_char = ent.end_char
        new_text += text[prev_end_char:]
    else:
        new_text = text
    doc = nlp(new_text)
    result = []
    for token in doc:
        if not token.text.strip():
            continue
        if token.text in punctuation:
            continue
        if token.text.isupper():
            result.append(token.text)
        else:
            result.append(token.lemma_)

    return result