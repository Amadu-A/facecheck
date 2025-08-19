# verification/tests/test_services.py
import numpy as np
from django.test import SimpleTestCase
from verification.services.matcher import cosine_similarity, match

class ServicesTest(SimpleTestCase):
    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=6)

    def test_match_threshold(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.6, 0.8], dtype=np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        res = match(a, b, threshold=0.7)
        self.assertFalse(res.verified)
        res2 = match(a, b, threshold=0.5)
        self.assertTrue(res2.verified)
