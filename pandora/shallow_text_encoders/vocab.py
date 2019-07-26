from typing import List, Optional, Union, Iterable, Callable, Generator, Dict
from collections import Counter, defaultdict
from .helpers import LazyDict
import numpy as np

Tokenizer = Callable[[str], List[str]]
LexVecDict = Dict[str, np.ndarray]


class BaseVocab(object):
    def __init__(
        self,
        tokens: List,
        embedding_dict: Optional[Dict[str, np.ndarray]] = None,
        unk: Optional[str] = None,
        pad: Optional[str] = None,
    ):
        self.id2token = tokens
        if unk:
            self.id2token.insert(0, unk)
        if pad:
            self.id2token.insert(0, pad)

        if unk:
            self.token2id = defaultdict(lambda: self.id2token.index(unk))
            for i, token in enumerate(self.id2token):
                self.token2id[token] = i
        else:
            self.token2id = dict(((token, i) for i, token in enumerate(self.id2token)))

        if embedding_dict:
            self.embedding_matrix = np.zeros(
                (len(self), next(iter(embedding_dict.values())).shape[0])
            )

            unk_embed = np.mean(np.array(list(embedding_dict.values())), axis=0)
            for i, token in enumerate(self.id2token):
                if token == pad:
                    continue
                if token in embedding_dict:
                    self.embedding_matrix[i] = embedding_dict[token]
                else:
                    self.embedding_matrix[i] = unk_embed
            self.token2embed = LazyDict(
                lambda token: self.embedding_matrix[self.token2id[token]]
            )
        else:
            self.embedding_matrix = None
            self.token2embed = None

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
            if self.token2embed is None:
                return token
            else:
                return token, self.token2embed[token]
        elif isinstance(id_or_string, str):
            token_id = self.token2id[id_or_string]
            if self.token2embed is None:
                return token_id
            else:
                return token_id, self.embedding_matrix[token_id]
        else:
            raise TypeError("Only support `str` and `int`")


class LexVecVocab(BaseVocab):
    @classmethod
    def create_from(cls, lexvec_dict: LexVecDict) -> BaseVocab:
        tokens = list(lexvec_dict.keys())
        return cls(tokens, lexvec_dict, unk="<UNK>", pad="<PAD>")
