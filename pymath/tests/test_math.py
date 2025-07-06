import unittest
from pymath.lib.math import fibonacci

class TestMath(unittest.TestCase):

    def test_fibonacci_zero(self):
        self.assertEqual(fibonacci(0), 0)

    def test_fibonacci_one(self):
        self.assertEqual(fibonacci(1), 1)

    def test_fibonacci_positive(self):
        self.assertEqual(fibonacci(2), 1)
        self.assertEqual(fibonacci(3), 2)
        self.assertEqual(fibonacci(4), 3)
        self.assertEqual(fibonacci(10), 55)

    def test_fibonacci_negative_input(self):
        # Assuming fibonacci is defined to return None or raise error for negative
        # Adjust based on actual implementation
        with self.assertRaises(ValueError): # Or TypeError, or custom error
            fibonacci(-1)

if __name__ == '__main__':
    unittest.main()
