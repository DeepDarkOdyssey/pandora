from pandora.shallow_text_encoders.vocab import LexVecVocab
from pandora.shallow_text_encoders.utils import spacy_tokenizer
from pandora.shallow_text_encoders.pooling_embeddings import (
    mean_pooling,
    max_pooling,
    sum_pooling,
)
from .utils import sts_sentence_pairs, STS_PAIRS, lexvec_vocab, lexvec_dict
from scipy.spatial.distance import cosine
import numpy as np


def test_mean_pooling(sts_sentence_pairs: STS_PAIRS, lexvec_vocab: LexVecVocab):
    scores = []
    cos_sims = []
    for score, sentence1, sentence2 in sts_sentence_pairs:
        scores.append(score)
        encode1 = mean_pooling(sentence1, lexvec_vocab, spacy_tokenizer)
        encode2 = mean_pooling(sentence2, lexvec_vocab, spacy_tokenizer)
        cos_sim = 1 - cosine(encode1, encode2)
        cos_sims.append(cos_sim)
    corr = np.corrcoef(cos_sims, scores)
    print("\nOn the whole dev set of STS")
    print(f"Pearson correlation between mean-pooling and human labels: {corr[0][1]}")


def test_max_pooling(sts_sentence_pairs: STS_PAIRS, lexvec_vocab: LexVecVocab):
    scores = []
    cos_sims = []
    for score, sentence1, sentence2 in sts_sentence_pairs:
        scores.append(score)
        encode1 = max_pooling(sentence1, lexvec_vocab, spacy_tokenizer)
        encode2 = max_pooling(sentence2, lexvec_vocab, spacy_tokenizer)
        cos_sim = 1 - cosine(encode1, encode2)
        cos_sims.append(cos_sim)
    corr = np.corrcoef(cos_sims, scores)
    print("\nOn the whole dev set of STS")
    print(f"Pearson correlation between max-pooling and human labels: {corr[0][1]}")


def test_sum_pooling(sts_sentence_pairs: STS_PAIRS, lexvec_vocab: LexVecVocab):
    scores = []
    cos_sims = []
    for score, sentence1, sentence2 in sts_sentence_pairs:
        scores.append(score)
        encode1 = sum_pooling(sentence1, lexvec_vocab, spacy_tokenizer)
        encode2 = sum_pooling(sentence2, lexvec_vocab, spacy_tokenizer)
        cos_sim = 1 - cosine(encode1, encode2)
        cos_sims.append(cos_sim)
    corr = np.corrcoef(cos_sims, scores)
    print("\nOn the whole dev set of STS")
    print(f"Pearson correlation between sum-pooling and human labels: {corr[0][1]}")
