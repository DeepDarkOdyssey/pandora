import numpy as np
import spacy
from typing import List, Iterable
from functools import partial
from pandora.shallow_text_encoders.geometric_embedding import GEM
from pandora.shallow_text_encoders.vocab import STSVocab

nlp = spacy.load("en_core_web_sm")
np.random.seed(0)
embed_size = 100


def sts_sentence_generator(file_path: str) -> Iterable[str]:
    with open(file_path, encoding="utf8") as f:
        for line in f:
            sentence1, sentence2 = line.split("\t")[-2:]
            yield sentence1
            yield sentence2


def spacy_tokenizer(string: str) -> List[str]:
    doc = nlp(string)
    return [token.lower_ for token in doc]


def embed_generator(token: str) -> np.ndarray:
    return np.random.rand(embed_size)


corpus = sts_sentence_generator("./tmp/stsbenchmark/sts-train.csv")
vocab_with_random_embed = STSVocab.create_from(corpus, spacy_tokenizer, embed_generator)


def test_vocab():
    corpus = sts_sentence_generator("./tmp/stsbenchmark/sts-train.csv")
    vocab = STSVocab.create_from(corpus, spacy_tokenizer, embed_generator)
    assert vocab.embed_size == embed_size
    assert all(vocab[0][1] == np.zeros(embed_size))


def test_gem_with_random_embeddings():
    corpus = sts_sentence_generator("./tmp/stsbenchmark/sts-train.csv")
    vocab_with_random_embed = STSVocab.create_from(
        corpus, spacy_tokenizer, embed_generator
    )

    singular_vectors, singular_values = GEM.build_corpus_principles(
        corpus, spacy_tokenizer, lambda x: vocab_with_random_embed.token2embed[x], 10
    )
    new_corpus = sts_sentence_generator("./tmp/stsbenchmark/sts-train.csv")
    gem = GEM(
        next(new_corpus),
        spacy_tokenizer,
        lambda x: vocab_with_random_embed.token2embed[x],
        singular_vectors,
        singular_values,
        top_r=5,
    )
    print(gem.sentence_embedding)