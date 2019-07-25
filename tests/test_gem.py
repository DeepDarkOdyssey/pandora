import numpy as np
import spacy
import pytest
from typing import Iterable, List, Dict
from text_encoder.geometric_embedding import GEM
from text_encoder.vocab import LexVecVocab, LexVecDict


nlp = spacy.load("en_core_web_sm")
np.random.seed(0)
random_embed_size = 100
lexvec_vocab_size, lexvec_embed_size = 0, 0


def spacy_tokenizer(string: str) -> List[str]:
    doc = nlp(string)
    return [token.lower_ for token in doc]


@pytest.fixture(scope="module")
def sts_sentence_pairs() -> Iterable[List[str]]:
    pairs = []
    with open("./tmp/stsbenchmark/sts-dev.csv", encoding="utf8") as f:
        for line in f:
            sentence1, sentence2 = line.split("\t")[-2:]
            pairs.append([sentence1, sentence2])
    return pairs


@pytest.fixture(scope="module")
def lexvec_dict() -> LexVecDict:
    token2embed = {}
    with open("./tmp/lexvec.enwiki+newscrawl.300d.W.pos.vectors") as f:
        global lexvec_vocab_size, lexvec_embed_size
        lexvec_vocab_size, lexvec_embed_size = f.readline()
        for line in f:
            values = line.split()
            token = values[0]
            embed = np.array(values[1:], dtype=np.float)
            token2embed[token] = embed
    return token2embed


@pytest.fixture(scope="module")
def lexvec_vocab(lexvec_dict) -> LexVecVocab:
    vocab = LexVecVocab.create_from(lexvec_dict)
    return vocab


def test_lexvec_vocab(lexvec_vocab: LexVecVocab):
    assert len(lexvec_vocab) == lexvec_vocab_size
    assert lexvec_vocab.embed_size == lexvec_embed_size
    pad_token, pad_embed = lexvec_vocab[0]
    assert pad_token == "<PAD>"
    assert all(pad_embed == np.zeros(lexvec_vocab.embed_size))


def test_gem_sts_lexvec(sts_sentence_pairs, lexvec_vocab):
    sts_sentences = []
    for pair in sts_sentence_pairs:
        sts_sentences.extend(pair)
    token2embed_fn = lambda token: np.array(lexvec_vocab.token2embed[token])
    corpus_singular_vectors, corpus_singular_values = GEM.build_corpus_principles(
        sts_sentences, spacy_tokenizer, token2embed_fn
    )
    sentence1, sentence2 = sts_sentence_pairs[0]

    gem1 = GEM(
        sentence1,
        spacy_tokenizer,
        token2embed_fn,
        corpus_singular_vectors,
        corpus_singular_values,
    )
    gem2 = GEM(
        sentence2,
        spacy_tokenizer,
        token2embed_fn,
        corpus_singular_vectors,
        corpus_singular_values,
    )
    mean_embed1 = np.mean(
        np.vstack(
            [lexvec_vocab.token2embed[token] for token in spacy_tokenizer(sentence1)]
        ),
        axis=0,
    )
    mean_embed2 = np.mean(
        np.vstack(
            [lexvec_vocab.token2embed[token] for token in spacy_tokenizer(sentence2)]
        ),
        axis=0,
    )
    print()
    print(np.corrcoef(gem1.text_embedding, gem2.text_embedding))
    print(np.corrcoef(mean_embed1, mean_embed2))
