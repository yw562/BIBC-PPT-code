# solutions.py
"""Programming test solutions for Xantium Recruiting Team.

This module contains three fully‑vectorised NumPy implementations that
match the performance and interface constraints described in the
assignment.  All functions work on very large arrays (≈1e8 elements)
without explicit Python‑level loops over every data point, achieving
`O(N)` time.

Functions
---------
problem1_distance_to_prev_true(arr_bool):
    Distance to the most‑recent ``True`` (or ``‑1`` when none).

problem2_compute_B(A, w, h1, h2):
    Windowed, exponentially damped cosine dot product.

problem3_compute_X(A, S):
    Prefix‑windowed, exponentially‑weighted sums with varying lower
    bounds.

A unit test for **Problem 3** (``N = 20``) is included under the main
guard.  Run

    $ python solutions.py

and the test will execute automatically via ``unittest``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "problem1_distance_to_prev_true",
    "problem2_compute_B",
    "problem3_compute_X",
]

###############################################################################
# Problem 1 – distance to the previous True
###############################################################################

def problem1_distance_to_prev_true(arr_bool: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Return the distance to the last ``True`` at or before each index.

    Parameters
    ----------
    arr_bool : (N,) bool array
        Input mask.

    Returns
    -------
    dist : (N,) int64 array
        ``dist[i] = i - t`` where *t* is the largest index ``≤ i`` such
        that ``arr_bool[t]`` is ``True``; ``‑1`` if no such *t* exists.
    """
    if arr_bool.dtype != np.bool_:
        raise TypeError("Input must be a boolean array.")

    n = arr_bool.size
    # ``idx`` is *t* at positions where arr_bool is True, else ‑1.
    idx = np.where(arr_bool, np.arange(n, dtype=np.int64), -1)
    # ``np.maximum.accumulate`` propagates the most‑recent True index
    # forward so that every position holds its preceding True.
    last_true = np.maximum.accumulate(idx)

    dist = np.arange(n, dtype=np.int64) - last_true
    dist[last_true == -1] = -1  # no True encountered yet.
    return dist

###############################################################################
# Problem 2 – damped‑cosine rolling sum
###############################################################################

def problem2_compute_B(
    A: NDArray[np.float64],
    w: int,
    h1: float,
    h2: float,
) -> NDArray[np.float64]:
    """Compute the specified windowed sum in **O(N)** time using NumPy.

    Notes
    -----
    Define the complex constant ``c = exp((-1/h1) + 1j/h2)`` and its
    inverse‑conjugate ``c̄⁻¹``.  Writing the target sum in complex form
    reveals the *sliding‑window* can be expressed through two prefix
    arrays followed by element‑wise arithmetic –– all vectorisable and
    linear‑time.
    """
    n = A.size
    if n == 0:
        return A.copy()

    idx = np.arange(n, dtype=np.float64)

    # Complex ratio and its inverse‑conjugate in one shot for speed.
    c_log = (-1.0 / h1) + 1j * (1.0 / h2)
    c_powers = np.exp(idx * c_log)          # c ** idx
    inv_powers = np.exp(idx * (-c_log))     # (c̄⁻¹) ** idx

    # Prefix sum of the transformed sequence.
    P = np.cumsum(A * inv_powers, dtype=np.complex128)

    # ``P_shift`` is P[i‑w‑1] (zero where the index would be <0).
    P_shift = np.empty_like(P, dtype=np.complex128)
    P_shift[: w + 1] = 0.0
    P_shift[w + 1 :] = P[: -w - 1]

    result = np.real(c_powers * (P - P_shift))
    return result.astype(np.float64, copy=False)

###############################################################################
# Problem 3 – variable‑lower‑bound weighted prefix sums
###############################################################################

def problem3_compute_X(
    A: NDArray[np.float64],
    S: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute weighted sums ``X[i]`` without Python loops.

    The core idea is to build an **inclusive prefix sum** of
    ``A * exp(-50*k/N)`` and then subtract the running total immediately
    *before* index ``S[i]``.
    """
    if A.shape != S.shape:
        raise ValueError("A and S must have the same length.")

    n = A.size
    if n == 0:
        return A.copy()

    idx = np.arange(n, dtype=np.float64)
    weights = A * np.exp(-50.0 * idx / n)

    # Inclusive prefix sums of the weighted sequence.
    prefix = np.cumsum(weights, dtype=np.float64)

    # Total up to (but not including) S[i].
    prev = np.where(S > 0, prefix[S - 1], 0.0)
    return prefix - prev

###############################################################################
# Lightweight validation – run `python solutions.py`
###############################################################################

if __name__ == "__main__":
    import unittest

    class _TestProblem3(unittest.TestCase):
        """Compare vectorised vs. naive implementations for a small N."""

        def test_small_N(self) -> None:  # noqa: D401 – simple name OK.
            N = 20
            rng = np.random.default_rng(42)
            A = rng.uniform(1.0, 2.0, size=N)
            S = rng.integers(0, np.arange(N) + 1)

            fast = problem3_compute_X(A, S)

            # Explicit Python loop reference implementation.
            ref = np.empty(N, dtype=np.float64)
            for i in range(N):
                total = 0.0
                for k in range(S[i], i + 1):
                    total += A[k] * np.exp(-50.0 * k / N)
                ref[i] = total

            # Two‑stage tolerance: absolute for small numbers, relative otherwise.
            abs_err = np.max(np.abs(fast - ref))
            rel_mask = np.abs(ref) > 1e-8
            rel_err = (
                np.max(np.abs(fast[rel_mask] - ref[rel_mask]) / np.abs(ref[rel_mask]))
                if np.any(rel_mask)
                else 0.0
            )

            self.assertLess(abs_err, 1e-10)
            self.assertLess(rel_err, 1e-8)

    unittest.main()
