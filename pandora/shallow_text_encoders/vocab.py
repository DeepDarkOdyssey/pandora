from typing import List, Optional, Union, Iterable, Callable, Generator, Dict
from collections import Counter, defaultdict
import numpy as np

Tokenizer = Callable[[str], List[str]]
LexVecDict = Dict[str, np.ndarray]


class BaseVocab(object):
    def __init__(
        self,
        tokens: List,
        embedding_dict: Optional[Dict[str, np.ndarray]],
        unk: Optional[str] = None,
        pad: Optional[str] = None,
    ):
        self.id2token = tokens
        offset = 0
        if unk:
            self.id2token.insert(0, unk)
            offset += 1
        if pad:
            self.id2token.insert(0, pad)
            offset += 1

        self.token2id = dict(((token, i) for i, token in enumerate(self.id2token)))

        if embedding_dict:
            self.embedding_matrix = np.zeros(
                (len(self), iter(embedding_dict.values()).shape[0])
            )

            if unk:
                unk_embed = np.mean(np.array(list(embedding_dict.values())), axis=0)
                self.token2embed = defaultdict(lambda: unk_embed)
            else:
                self.token2embed = {}
            for token in self.id2token:
                self.embedding_matrix[self.token2id[token]] = embedding_dict[token]

    @classmethod
    def create_from(cls, *args, **kwargs):
        raise NotImplementedError()

    def tokens2indexes(self, tokens: List[str]) -> List[int]:
        return [self.token2id[token] for token in tokens]

    def text2indexes(self, text: str, tokenizer: Tokenizer) -> List[int]:
        tokens = tokenizer(text)
        return self.tokens2indexes(tokens)

    def text2embed(self, text: str, tokenizer: Tokenizer) -> np.ndarray:
        token_indexes = self.text2indexes(text, tokenizer)
        return self.embedding_matrix[token_indexes].T

    def __len__(self):
        return len(self.id2token)

    @property
    def embed_size(self):
        return self.embedding_matrix.shape[1]

    def __contains__(self, id_or_string: Union[str, int]):
        if isinstance(id_or_string, str):
            return id_or_string in self.token2id
        elif isinstance(id_or_string, int):
            return id_or_string < len(self)
        else:
            raise TypeError("Only support `str` and `int`")

    def __getitem__(self, id_or_string: Union[int, str]):
        if isinstance(id_or_string, int):
            token = self.id2token[id_or_string]
            if self.token2embed:
                return token, self.token2embed[token]
            else:
                return token
        elif isinstance(id_or_string, str):
            if self.token2embed:
                return self.token2id[id_or_string], self.token2embed[id_or_string]
            else:
                return self.token2id[id_or_string]
        else:
            raise TypeError("Only support `str` and `int`")


class LexVecVocab(BaseVocab):
    @classmethod
    def create_from(cls, lexvec_dict: LexVecDict) -> BaseVocab:
        tokens = list(lexvec_dict.keys())
        return cls(tokens, lexvec_dict, unk="<UNK>", pad="<PAD>")

