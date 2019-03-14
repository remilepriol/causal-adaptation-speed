import numpy as np


def proba2logit(p):
    s = np.log(p)
    s -= np.mean(s, axis=-1, keepdims=True)
    return s


def joint2conditional(joint):
    marginal = np.sum(joint, axis=-1)
    conditional = joint / np.expand_dims(marginal, axis=-1)

    return ConditionalCategorical(marginal, conditional)


def sample_joint(k, n, concentration, symmetric=True):
    """Sample n causal mechanisms of categorical variables of dimension K."""
    if symmetric:
        joint = np.random.dirichlet(concentration * np.ones(k ** 2), size=n).reshape((n, k, k))
        return joint2conditional(joint)
    else:
        pa = np.random.dirichlet(concentration * np.ones(k), size=n)
        pba = np.random.dirichlet(concentration * np.ones(k), size=[n, k])
        return ConditionalCategorical(pa, pba)


class ConditionalCategorical():
    """Represent n categorical distributions of variables (a,b) of dimension k each."""

    def __init__(self, marginal, conditional):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n = marginal.shape[0]
        self.k = marginal.shape[1]

        s = conditional.shape
        assert s[0] == self.n
        assert s[1] == self.k and s[2] == self.k
        assert np.allclose(marginal.sum(axis=-1), np.ones(self.n))
        assert np.allclose(conditional.sum(axis=-1), np.ones((self.n, self.k)))

        self.marginal = marginal
        self.conditional = conditional

        self.sa = proba2logit(self.marginal)
        self.sba = proba2logit(self.conditional)

    def to_joint(self):
        return self.conditional * np.expand_dims(self.marginal, axis=-1)

    def reverse(self):
        """Return conditional from b to a.
        Compute marginal pb and conditional pab such that pab*pb = pba*pa.
        """
        joint = self.to_joint()
        joint = np.swapaxes(joint, -1, -2)  # invert variables
        return joint2conditional(joint)

    def sqdistance(self, other):
        """Return the squared euclidean distance between self and other"""
        pd = np.sum((self.marginal - other.marginal) ** 2, axis=(-1))
        pd += np.sum((self.conditional - other.conditional) ** 2, axis=(-1, -2))

        sd = np.sum((self.sa - other.sa) ** 2, axis=(-1))
        sd += np.sum((self.sba - other.sba) ** 2, axis=(-1, -2))
        return pd, sd

    def intervention(self, on, concentration, fromjoint=True):
        # sample new marginal
        if fromjoint:
            newmarginal = sample_joint(self.k, self.n, concentration, symmetric=True).marginal
        else:
            newmarginal = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)

        # replace the cause or the effect by this marginal
        if on == 'cause':
            return ConditionalCategorical(newmarginal, self.conditional)
        else:  # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return ConditionalCategorical(self.marginal, newconditional)


def test_causal2anti():
    pa = np.array([[.5, .5]])
    pba = np.array([[[.5, .5], [1 / 3, 2 / 3]]])
    anspb = np.array([[5 / 12, 7 / 12]])
    anspab = np.array([[[3 / 5, 2 / 5], [3 / 7, 4 / 7]]])

    test = ConditionalCategorical(pa, pba).reverse()
    answer = ConditionalCategorical(anspb, anspab)

    probadist, scoredist = test.sqdistance(answer)
    assert probadist < 1e-4, probadist
    # score dist will be nan because of the 0 probabilities


test_causal2anti()


def experiment(k, n, concentration, intervention,
               symmetric_init=True, symmetric_intervention=True):
    """Sample n mechanisms of order k and for each of them sample an intervention on the desired
    mechanism. Return the distance between the original distribution and the intervened
    distribution in the causal parameter space and in the anticausal parameter space.

    :param intervention: takes value 'cause' or 'effect'
    :param symmetric_init: sample the causal parameters such that the marginals on a and b are
    drawn from the same distribution
    :param symmetric_intervention: sample the intervention parameters from the same law as the
    initial parameters
    """
    # causal parameters
    causal = sample_joint(k, n, concentration, symmetric_init)
    transfer = causal.intervention(
        on=intervention,
        concentration=concentration,
        fromjoint=symmetric_intervention
    )
    cpd, csd = causal.sqdistance(transfer)

    # anticausal parameters
    anticausal = causal.reverse()
    antitransfer = transfer.reverse()
    apd, asd = anticausal.sqdistance(antitransfer)

    return np.array([cpd, apd, csd, asd])


def test_experiment():
    for intervention in ['cause', 'effect']:
        for symmetric_init in [True, False]:
            for symmetric_intervention in [True, False]:
                experiment(2, 3, 1, intervention, symmetric_init, symmetric_intervention)


test_experiment()

