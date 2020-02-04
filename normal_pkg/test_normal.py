import unittest

import numpy as np
import scipy

from normal_pkg import normal

np.random.seed(1)


class TestNormals(unittest.TestCase):

    def setUp(self):
        self.nat = normal.sample_natural(dim=3)
        self.cho = normal.sample_cholesky(dim=3)

    # change of representation
    def test_nat2mean2nat(self):
        self.assertAlmostEqual(0, self.nat.to_mean().to_natural().distance(self.nat))

    def test_nat2joint2nat(self):
        self.assertAlmostEqual(0, self.nat.to_joint().to_conditional().distance(self.nat))

    def test_nat2joint2mean2cond2nat(self):
        self.assertAlmostEqual(0, self.nat.to_joint().to_mean().to_conditional().to_natural()
                               .distance(self.nat))

    def test_nat2mean2joint2nat2cond(self):
        self.assertAlmostEqual(0, self.nat.to_mean().to_joint().to_natural().to_conditional()
                               .distance(self.nat))

    def test_nat2cho2nat(self):
        self.assertAlmostEqual(0, self.nat.to_cholesky().to_natural().distance(self.nat))

    def test_cho2nat2cho(self):
        self.assertAlmostEqual(0, self.cho.to_natural().to_cholesky().distance(self.cho))

    def test_nat2joint2cho2nat2cond(self):
        self.assertAlmostEqual(0, self.nat.to_joint().to_cholesky().to_natural()
                               .to_conditional().distance(self.nat))

    # change of direction
    def test_reversereverse(self):
        self.assertAlmostEqual(0, self.nat.reverse().reverse().distance(self.nat))

    # misc
    def test_interventions(self):
        self.nat.intervention(on='cause')
        self.nat.intervention(on='effect')

    def test_meanjoint(self):
        meanjoint = self.nat.to_joint().to_mean()
        meanjoint.sample(5)
        encoder = scipy.stats.ortho_group.rvs(meanjoint.mean.shape[0])
        meanjoint.encode(encoder)

    def test_meancond(self):
        self.nat.to_mean().sample(5)

    def test_natjoint(self):
        natjoint = self.nat.to_joint()
        natjoint.logpartition
        natjoint.negativeloglikelihood(np.random.randn(10, natjoint.eta.shape[0]))


if __name__ == '__main__':
    unittest.main()
