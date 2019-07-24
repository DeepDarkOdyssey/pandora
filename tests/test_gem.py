import numpy as np
import spacy
from pandora.shallow_text_encoders.geometric_embedding import GEM

nlp = spacy.load("en_core_web_sm")


def test_corpus_principles():
    np.random.seed(0)
    corpus = [
        "test functionality of gem algorithem",
        "just a test sentence",
        "just another test sentence",
    ]
    tokenizer = lambda x: x.split()
    token2emb = lambda x: np.random.rand(10, 1)
    singular_vectors, singular_values = GEM.build_corpus_principles(
        corpus, tokenizer, token2emb
    )
    # sorted_rank_values, sorted_sgl_values = GEM(
    #     "test rerank corpus principles",
    #     tokenizer,
    #     token2emb,
    #     singular_vectors,
    #     singular_values,
    # ).rerank_principles()
    # print(sorted_rank_values)
    # print(sorted_sgl_values)


def test_corpus():
    vocab = []
    count = 0
    with open(
        "./tests/tmp/lexvec.enwiki+newscrawl.300d.W.pos.vectors", encoding="utf8"
    ) as f:
        f.readline()
        for line in f:
            token: str = line.split()[0]
            vocab.append(token)
            if not token.islower():
                count += 1

    tokens = set()
    with open('./tests/tmp/stsbenchmark/sts-dev.csv', encoding='utf8') as f:
        for line in f:
            score, sentence1, sentence2 = line.split('\t')[-3:]
            doc = nlp(sentence1)
            for token in doc:
                tokens.add(token.lower_)
            doc = nlp(sentence2)
            for token in doc:
                tokens.add(token.lower_)
    print(tokens - set(vocab))
