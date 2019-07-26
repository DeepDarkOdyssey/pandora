import numpy as np
from .vocab import Tokenizer, BaseVocab


def mean_pooling(text: str, vocab: BaseVocab, tokenizer: Tokenizer) -> np.ndarray:
    token_embeddings = vocab.text2embed(text, tokenizer)
    return np.mean(token_embeddings, axis=1)


def max_pooling(text: str, vocab: BaseVocab, tokenizer: Tokenizer) -> np.ndarray:
    token_embeddings = vocab.text2embed(text, tokenizer)
    return np.max(token_embeddings, axis=1)


def sum_pooling(text: str, vocab: BaseVocab, tokenizer: Tokenizer) -> np.ndarray:
    token_embeddings = vocab.text2embed(text, tokenizer)
    return np.sum(token_embeddings, axis=1)
