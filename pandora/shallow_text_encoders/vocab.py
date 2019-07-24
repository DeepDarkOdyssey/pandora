from typing import List, Optional, Union, Iterable, Callable, Generator
from collections import Counter
import numpy as np

Embed = Union[List[List[int]], List[np.array], np.ndarray]


class BaseVocab(object):
    def __init__(
        self,
        tokens: List,
        embed_generator: Optional[Callable[[str], np.array]] = None,
        embeddings: Optional[Embed] = None,
        unk: Optional[str] = None,
        pad: Optional[str] = None,
    ):
        self.embeddings = None
        if embed_generator:
            self.embeddings = [embed_generator(token) for token in tokens]
        elif embeddings:
            if isinstance(embeddings, np.ndarray):
                self.embeddings = list(embeddings)
            elif isinstance(embeddings, list):
                self.embeddings = [np.array(embedding) for embedding in embeddings]
            else:
                raise TypeError("Only support `list` or `numpy.ndarray`.")

        self.id2token = []
        if pad:
            self.id2token.append(pad)
            if self.embeddings:
                self.embeddings.insert(0, np.zeros(self.embed_size))
        if unk:
            self.id2token.append(unk)
            if self.embeddings:
                self.embeddings.insert(1, np.random.rand(self.embed_size))

        self.id2token.extend(tokens)
        self.token2id = dict(((token, i) for i, token in enumerate(self.id2token)))

        if self.embeddings:
            self.token2embed = {}
            for i, embedding in enumerate(self.embeddings):
                self.token2embed[self.id2token[i]] = embedding

    @classmethod
    def create_from(cls, *args, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.id2token)

    @property
    def embed_size(self):
        return self.embeddings[0].shape[0]

    def __contains__(self, string_or_embed: Union[str, Embed]):
        if isinstance(string_or_embed, str):
            return string_or_embed in self.token2id
        elif isinstance(string_or_embed, (list, np.ndarray)):
            return string_or_embed in self.embed2token
        else:
            raise TypeError("Only support `str`, `list` or `numpy.ndarrary`")

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
            raise TypeError("Can only get token string or id")


class STSVocab(BaseVocab):
    @classmethod
    def create_from(
        cls,
        corpus: Iterable[str],
        tokenizer: Callable[[str], List[str]],
        embed_generator: Callable[[str], np.ndarray],
    ) -> BaseVocab:
        token_counter = Counter()
        for text in corpus:
            token_counter.update(tokenizer(text))
        tokens = [token for token, count in token_counter.most_common()]
        return cls(tokens, embed_generator=embed_generator, unk="<UNK>", pad='<PAD>')
