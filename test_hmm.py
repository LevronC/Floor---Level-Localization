import unittest
import math
from hmm_model import safe_log, normal_pdf, normal_cdf, truncated_normal_pdf, HMM, viterbi_decode, naive_snap
from simulation import is_sequence_physically_valid

class TestHMMMathAndAlgorithms(unittest.TestCase):
    
    def test_safe_log(self):
        self.assertEqual(safe_log(0), float('-inf'))
        self.assertEqual(safe_log(-5.0), float('-inf'))
        self.assertAlmostEqual(safe_log(math.e), 1.0)
        self.assertAlmostEqual(safe_log(1.0), 0.0)

    def test_normal_pdf(self):
        # Peak of standard normal should be 1 / sqrt(2*pi)
        expected_peak = 1.0 / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(normal_pdf(0.0, 0.0, 1.0), expected_peak)
        # Symmetrical property
        self.assertAlmostEqual(normal_pdf(1.0, 0.0, 1.0), normal_pdf(-1.0, 0.0, 1.0))
        # Zero std deviation
        self.assertEqual(normal_pdf(1.0, 1.0, 0.0), 1.0)
        self.assertEqual(normal_pdf(2.0, 1.0, 0.0), 0.0)

    def test_normal_cdf(self):
        # Median of normal distribution
        self.assertAlmostEqual(normal_cdf(0.0, 0.0, 1.0), 0.5)
        # Limits
        self.assertGreater(normal_cdf(5.0, 0.0, 1.0), 0.99)
        self.assertLess(normal_cdf(-5.0, 0.0, 1.0), 0.01)
        # Zero std deviation
        self.assertEqual(normal_cdf(1.0, 1.0, 0.0), 1.0)
        self.assertEqual(normal_cdf(0.5, 1.0, 0.0), 0.0)

    def test_truncated_normal_pdf(self):
        # If outside [a, b], density must be 0
        self.assertEqual(truncated_normal_pdf(-1.0, 0.0, 1.0, 0.0, 2.0), 0.0)
        self.assertEqual(truncated_normal_pdf(3.0, 0.0, 1.0, 0.0, 2.0), 0.0)
        
        # If within [a, b], density must be greater than standard normal density since denominator is < 1
        a, b = 0.0, 1.0
        standard_density = normal_pdf(0.5, 0.0, 1.0)
        truncated_density = truncated_normal_pdf(0.5, 0.0, 1.0, a, b)
        self.assertGreater(truncated_density, standard_density)

    def test_hmm_predecessors(self):
        # 3 floors, transition matrix where floor 1 and 3 cannot reach each other (banded)
        pi = [1/3, 1/3, 1/3]
        A = [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2],
            [0.0, 0.3, 0.7]
        ]
        dummy_emission = lambda s, o: 1.0
        hmm = HMM(3, pi, A, dummy_emission)
        
        # Predecessors for Floor 1: can only be 1 or 2
        # (Since A[0][0] = 0.8, A[1][0] = 0.1, A[2][0] = 0.0)
        self.assertEqual(set(hmm.get_predecessors(1)), {1, 2})
        
        # Predecessors for Floor 2: can be 1, 2, or 3
        # (Since A[0][1] = 0.2, A[1][1] = 0.7, A[2][1] = 0.3)
        self.assertEqual(set(hmm.get_predecessors(2)), {1, 2, 3})
        
        # Predecessors for Floor 3: can only be 2 or 3
        # (Since A[0][2] = 0.0, A[1][2] = 0.2, A[2][2] = 0.7)
        self.assertEqual(set(hmm.get_predecessors(3)), {2, 3})

    def test_physical_validity(self):
        self.assertTrue(is_sequence_physically_valid([]))
        self.assertTrue(is_sequence_physically_valid([1]))
        self.assertTrue(is_sequence_physically_valid([1, 2, 3, 2, 1, 1]))
        
        # Jumps of 2 or more floors are invalid
        self.assertFalse(is_sequence_physically_valid([1, 3, 2]))
        self.assertFalse(is_sequence_physically_valid([3, 1, 2]))
        self.assertFalse(is_sequence_physically_valid([1, 2, 5, 4]))

if __name__ == "__main__":
    unittest.main()
