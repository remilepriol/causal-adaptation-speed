import unittest

import numpy as np
import scipy

import normal

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

    # change of direction
    def test_reversereverse(self):
        self.assertAlmostEqual(0, self.nat.reverse().reverse().distance(self.nat))

    # misc
    def test_compile(self):
        self.nat.to_mean().sample(5)
        meanjoint = self.nat.to_joint().to_mean()
        meanjoint.sample(5)
        self.nat.intervention(on='cause')
        self.nat.intervention(on='effect')
        encoder = scipy.stats.ortho_group.rvs(meanjoint.mean.shape[0])
        meanjoint.encode(encoder)


if __name__ == '__main__':
    unittest.main()
