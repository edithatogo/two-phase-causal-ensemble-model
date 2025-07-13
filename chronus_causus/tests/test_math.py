import unittest
from chronus_causus.lib.math import fibonacci

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
        with self.assertRaisesRegex(ValueError, "Input must be a non-negative integer."):
            fibonacci(-1)
        with self.assertRaisesRegex(ValueError, "Input must be a non-negative integer."):
            fibonacci(-5)

    def test_fibonacci_non_integer_input(self):
        with self.assertRaisesRegex(TypeError, "Input must be an integer."):
            fibonacci(1.5)
        with self.assertRaisesRegex(TypeError, "Input must be an integer."):
            fibonacci("test")
        with self.assertRaisesRegex(TypeError, "Input must be an integer."):
            fibonacci([1, 2])

if __name__ == '__main__':
    unittest.main()
