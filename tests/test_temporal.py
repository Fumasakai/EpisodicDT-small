import unittest
import numpy as np
from src.evaluation.temporal import temporal_statistics


class TemporalTests(unittest.TestCase):
    def test_known_series_and_constants(self):
        lags, acf, change = temporal_statistics(np.array([[0, 1, 2, 3], [9, 9, 9, 9]]))
        np.testing.assert_array_equal(lags, [0, 1, 2])
        np.testing.assert_allclose(acf[0], [1, .25, -.3])
        np.testing.assert_allclose(change[0], [0, 1, 4])
        self.assertTrue(np.isnan(acf[1]).all())
        np.testing.assert_array_equal(change[1], [0, 0, 0])

    def test_offset_and_scale_and_episode_boundaries(self):
        x = np.array([[1, -1, 1, -1], [100, 101, 102, 103]])
        _, acf, change = temporal_statistics(x)
        _, shifted_acf, scaled_change = temporal_statistics(3 * x + 20)
        np.testing.assert_allclose(acf, shifted_acf)
        np.testing.assert_allclose(scaled_change, 9 * change)
        np.testing.assert_allclose(change[:, 1], [4, 1])
        np.testing.assert_allclose(acf[0, 1], -.75)
