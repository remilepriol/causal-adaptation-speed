import numpy as np


def causal2anti(pa, cond_pba):
    """Return marginal pb and conditional pab such that pab*pb = pba*pa.

    pa is n*k array.
    pba is n*k*k array.
    """
    joint = cond_pba * np.expand_dims(pa, axis=-1)
    joint = np.swapaxes(joint, -1, -2)
    pb = np.sum(joint, axis=-1)
    cond_pab = joint / np.expand_dims(pb, axis=-1)

    return pb, cond_pab


def test_causal2anti():
    pa = np.array([.5, .5])
    pba = np.array([[.5, .5], [0, 1]])

    anspb = np.array([.25, .75])
    anspab = np.array([[1, 0], [1 / 3, 2 / 3]])
    pb, pab = causal2anti(pa, pba)

    assert np.allclose(pb, anspb)
    assert np.allclose(pab, anspab)


test_causal2anti()


def proba2logit(p):
    s = np.log(p)
    s -= np.mean(s, axis=-1, keepdims=True)
    return s


def normal2anti(p):
    pass


def categorical_distances_cause(k, n, concentration=1):
    """Sample n mechanisms from a dirichlet of order k then evaluate the distance after
    intervention on the cause between causal and anticausal models.
    """

    # sample mechanisms
    joint = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k))
    pa = np.sum(joint, axis=1)
    pba = joint / pa[:, None, :]

    tpa = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k)).sum(
        axis=1)

    # pb = np.sum(joint, axis=2)
    # pab = joint / pb[:, :, None]

    # pa = np.random.dirichlet(concentration * np.ones(k), size=n)
    # tpa = np.random.dirichlet(concentration * np.ones(k), size=n)  # transfer / intervention
    # pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])

    # evaluate anticausal probabilities
    pb, pab = causal2anti(pa, pba)
    tpb, tpab = causal2anti(tpa, pba)

    # compute distances
    distances = np.zeros([n, 8])

    distances[:, 0] = np.sum((pa - tpa) ** 2, axis=-1)
    bmarginaldistance = np.sum((pb - tpb) ** 2, axis=-1)
    distances[:, 1] = bmarginaldistance + np.sum((pab - tpab) ** 2, axis=(-1, -2))
    distances[:, 2] = distances[:, 1] / distances[:, 0]

    distances[:, 6] = bmarginaldistance

    # in the score / logits space, we have to consider logits which sum to 0
    sa = proba2logit(pa)
    tsa = proba2logit(tpa)
    # sba = proba2logit(pba)

    sb = proba2logit(pb)
    sab = proba2logit(pab)

    tsb = proba2logit(tpb)
    tsab = proba2logit(tpab)

    # distances
    distances[:, 3] = np.sum((sa - tsa) ** 2, axis=-1)  # causal
    bmarginaldistance = np.sum((sb - tsb) ** 2, axis=-1)
    distances[:, 4] = bmarginaldistance + np.sum((sab - tsab) ** 2, axis=(-1, -2))  # anticausal
    distances[:, 5] = distances[:, 4] / distances[:, 3]  # ratio

    distances[:, 7] = bmarginaldistance

    return distances


categorical_distances_cause(3, 4)


def categorical_distances_effect(k, n, concentration=1):
    """Sample n mechanisms from a dirichlet of order k then evaluate the distance after
    intervention on the effect between causal and anticausal models.
    """

    # sample causal mechanisms
    joint = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k))
    pa = np.sum(joint, axis=1)
    pba = joint / pa[:, None, :]

    # pa = np.random.dirichlet(concentration * np.ones(k), size=n)
    # pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])
    # # evaluate anticausal probabilities
    pb, pab = causal2anti(pa, pba)

    # intervention on the effect. tp = transfer distribution. tp(a,b) = tp(a)*tp(b)
    tpb = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k)).sum(
        axis=1)
    tpa = pa
    tpba = np.expand_dims(tpb, axis=1)
    tpab = np.expand_dims(tpa, axis=1)

    # compute distances
    distances = np.zeros([n, 8])

    distances[:, 0] = np.sum((pba - tpba) ** 2, axis=(1, 2))  # causal
    bmarginaldistance = np.sum((pb - tpb) ** 2, axis=1)
    distances[:, 1] = bmarginaldistance + np.sum((pab - tpab) ** 2, axis=(1, 2))
    distances[:, 2] = distances[:, 1] / distances[:, 0]

    distances[:, 6] = bmarginaldistance

    # in the score / logits space, we have to consider logits which sum to 0
    sba = proba2logit(pba)
    tsba = proba2logit(tpba)

    sb = proba2logit(pb)
    tsb = proba2logit(tpb)

    sab = proba2logit(pab)
    tsab = proba2logit(tpab)

    # distances
    distances[:, 3] = np.sum((sba - tsba) ** 2, axis=(1, 2))
    bmarginaldistance = np.sum((sb - tsb) ** 2, axis=1)
    distances[:, 4] = bmarginaldistance + np.sum((sab - tsab) ** 2, axis=(1, 2))
    distances[:, 5] = distances[:, 4] / distances[:, 3]

    distances[:, 7] = bmarginaldistance

    return distances


categorical_distances_effect(3, 4)


