import os
import pytest
import numpy as np
from typing import List, Dict, Union, Iterable, Tuple
from scipy.spatial.distance import cosine
from pandora.shallow_text_encoders.geometric_embedding import GEM
from pandora.shallow_text_encoders.vocab import LexVecVocab
from pandora.shallow_text_encoders.utils import spacy_tokenizer
from .utils import sts_sentence_pairs, STS_PAIRS, lexvec_dict, lexvec_vocab, score2sim


corpus_principles_path = "./tmp/corpus_principles.pkl"


def test_lexvec_vocab(lexvec_vocab: LexVecVocab):
    assert len(lexvec_vocab) == 369001
    assert lexvec_vocab.embed_size == 300
    pad_token, pad_embed = lexvec_vocab[0]
    assert pad_token == "<PAD>"
    assert all(pad_embed == np.zeros(lexvec_vocab.embed_size))


def test_gem_sts_first_pair_lexvec(
    sts_sentence_pairs: STS_PAIRS, lexvec_vocab: LexVecVocab
):
    scores, sts_sentences = [], []
    for pair in sts_sentence_pairs:
        scores.append(pair[0])
        sts_sentences.extend(pair[1:])

    gem = GEM()
    if os.path.exists(corpus_principles_path):
        gem.load_corpus_principles(corpus_principles_path)
    else:
        gem.build_corpus_principles(
            sts_sentences, lexvec_vocab, spacy_tokenizer, save_to=corpus_principles_path
        )
    score, sentence1, sentence2 = sts_sentence_pairs[0]
    encode1 = gem.encode_text(sentence1, lexvec_vocab, spacy_tokenizer)
    encode2 = gem.encode_text(sentence2, lexvec_vocab, spacy_tokenizer)

    sim = 1 - cosine(encode1, encode2)
    print("\nOn first sample of dev set of STS")
    print(f"Similarity between GEM vectors: {sim:.4f} [-1 to 1]")
    print(f"Human rated similarity : {score:.4f} [0 to 5]")
    assert abs(sim - score2sim(score)) < 0.2


def test_gem_sts_all_lexvec(sts_sentence_pairs: STS_PAIRS, lexvec_vocab: LexVecVocab):
    scores, sts_sentences = [], []
    for pair in sts_sentence_pairs:
        scores.append(pair[0])
        sts_sentences.extend(pair[1:])
    gem = GEM()

    if os.path.exists(corpus_principles_path):
        gem.load_corpus_principles(corpus_principles_path)
    else:
        gem.build_corpus_principles(
            sts_sentences, lexvec_vocab, spacy_tokenizer, save_to=corpus_principles_path
        )
    cos_sims = []
    for _, sentence1, sentence2 in sts_sentence_pairs:
        encode1 = gem.encode_text(sentence1, lexvec_vocab, spacy_tokenizer)
        encode2 = gem.encode_text(sentence2, lexvec_vocab, spacy_tokenizer)
        cos_sim = 1 - cosine(encode1, encode2)
        cos_sims.append(cos_sim)
    corr = np.corrcoef(cos_sims, scores)
    print("\nOn the whole dev set of STS")
    print(f"Pearson correlation between GEM results and human labels: {corr[0][1]}")
    assert corr[0][1] >= 0.75
