from copy import copy
from typing import List, Tuple
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


def gram_schmidt_process(A: np.array) -> Tuple[np.ndarray]:
    d, n = A.shape
    Q = np.zeros((d, n))
    R = np.zeros((n, n))
    for i in range(n):
        v_i = A[:, i]
        qs = Q[:, :i]
        rs = v_i @ qs
        R[:i, i] = rs
        q_i = v_i - np.sum((v_i @ qs) / np.sum((qs ** 2), axis=0) * qs, axis=1)
        norm = np.linalg.norm(q_i, ord=2)
        Q[:, i] = q_i / norm
        R[i, i] = norm
    return Q, R


def spacy_tokenizer(string: str) -> List[str]:
    doc = nlp(string)
    return [token.lower_ for token in doc]
