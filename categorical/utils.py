import numpy as np


def kullback_leibler(p1, p2):
    return np.sum(p1 * np.log(p1 / p2), axis=-1)


def logsumexp(s):
    smax = np.amax(s, axis=-1)
    return smax + np.log(
        np.sum(np.exp(s - np.expand_dims(smax, axis=-1)), axis=-1))


def logit2proba(s):
    return np.exp(s - np.expand_dims(logsumexp(s), axis=-1))


def proba2logit(p):
    s = np.log(p)
    s -= np.mean(s, axis=-1, keepdims=True)
    return s


def test_proba2logit():
    p = np.random.dirichlet(np.ones(50), size=300)
    s = proba2logit(p)
    assert np.allclose(0, np.sum(s, axis=-1))

    q = logit2proba(s)
    assert np.allclose(1, np.sum(q, axis=-1)), q
    assert np.allclose(p, q), p - q


if __name__ == "__main__":
    test_proba2logit()