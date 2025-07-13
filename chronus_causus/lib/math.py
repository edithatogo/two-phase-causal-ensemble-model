def fibonacci(n):
  """
  Calculates the n-th Fibonacci number.

  Args:
    n: A non-negative integer.

  Returns:
    The n-th Fibonacci number.

  Raises:
    TypeError: If n is not an integer.
    ValueError: If n is negative.
  """
  if not isinstance(n, int):
    raise TypeError("Input must be an integer.")
  if n < 0:
    raise ValueError("Input must be a non-negative integer.")
  elif n == 0:
    return 0
  elif n == 1:
    return 1
  else:
    # This is a recursive solution, which is clear but can be inefficient
    # for large n due to repeated calculations and recursion depth limits.
    # An iterative solution would be more performant for large inputs.
    return fibonacci(n - 1) + fibonacci(n - 2)
