import numpy as np
import pandas as pd
import pytest
from typing import List, Tuple, Union
from pandora.shallow_text_encoders.utils import LexVecVocab, LexVecDict

STS_PAIRS = List[Tuple[Union[str, int]]]
lexvec_vocab_size, lexvec_embed_size = 0, 0


@pytest.fixture(scope="session")
def sts_sentence_pairs() -> STS_PAIRS:
    dev_df = pd.read_csv(
        "./tmp/stsbenchmark/sts-dev.csv",
        sep="\t",
        encoding="utf8",
        header=None,
        names=["genre", "file", "year", "id", "score", "sentence1", "sentence2"],
    )
    dev_df = dev_df.dropna()
    pairs = list(
        zip(
            dev_df["score"].values,
            dev_df["sentence1"].values,
            dev_df["sentence2"].values,
        )
    )

    return pairs


@pytest.fixture(scope="session")
def lexvec_dict() -> LexVecDict:
    token2embed = {}
    with open("./tmp/lexvec.enwiki+newscrawl.300d.W.pos.vectors") as f:
        f.readline()
        for line in f:
            values = line.split()
            token = values[0]
            embed = np.array(values[1:], dtype=np.float)
            token2embed[token] = embed
    return token2embed


@pytest.fixture(scope="session")
def lexvec_vocab(lexvec_dict) -> LexVecVocab:
    vocab = LexVecVocab.create_from(lexvec_dict)
    return vocab


def score2sim(score: float) -> float:
    return score / 5 * 2 - 1
