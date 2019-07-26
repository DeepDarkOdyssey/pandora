from pandora.shallow_text_encoders.utils import gram_schmidt_process
import numpy as np


def test_gram_schmidt():
    A = np.random.rand(13, 10) * 100
    Q, R = gram_schmidt_process(A)
    assert np.abs(np.sum(Q.T @ Q)) - Q.shape[0] < 1e-6
    assert all(Q[:, 0] == A[:, 0] / np.linalg.norm(A[:, 0]))
    assert np.sum(np.abs(A - Q.dot(R))) < 1e-6