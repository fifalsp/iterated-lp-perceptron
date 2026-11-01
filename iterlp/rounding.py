"""The rounding loss D_{a,T}(q) and the finishing test of Section 8.

After the LP rounds, the remaining free coordinates are rounded by repeated
partial colouring. With the constants of Section 8.1 (T = 21/5), the total
loss in row margin is

    D_{a,T}(q) = sqrt(2 T q) * (2 sqrt(log(200 a / q)) + log 4 / sqrt(log(200 a / q))).

The reserve at round t is kappa_t - D_{1,21/5}(q_t). A run stops at the first
accepted, numerically resolved round whose reserve exceeds 10^-3.
"""
import math

import numpy as np

MINIMUM_RESERVE = 1e-3


def rounding_loss(q, a=1):
    """D_{a,21/5}(q) for a scalar or an array of free fractions q."""
    if np.ndim(q) == 0:
        L = math.log(200 * a / q)
        return math.sqrt(42 * q / 5) * (2 * math.sqrt(L) + math.log(4) / math.sqrt(L))
    q = np.asarray(q)
    L = np.log(200 * a / q)
    return np.sqrt(42 * q / 5) * (2 * np.sqrt(L) + np.log(4) / np.sqrt(L))


def is_resolved(record):
    """Scale checks that a round must pass before it may stop the run.

    0.01 < tau < 10^3, q > 10^-10, conditional survival probability above
    10^-5, and q_t > 100 * (largest scaled residual) * q_{t-1}. They exclude
    unresolved scales; they do not bound integration or resampling error.
    """
    return bool(.01 < record['tau'] < 1e3
                and record['q'] > 1e-10
                and record['survival_ratio'] > 1e-5
                and record['q'] > 100 * record['maximum_scaled_residual'] * record['previous_q'])


def add_finishing_test(record):
    """Add the rounding loss, the reserve and the finishing test to an accepted round record."""
    record['rounding_loss'] = rounding_loss(record['q'])
    record['reserve'] = record['kappa'] - record['rounding_loss']
    record['resolved'] = is_resolved(record)
    record['passes_finishing_test'] = record['resolved'] and record['reserve'] > MINIMUM_RESERVE
    return record
