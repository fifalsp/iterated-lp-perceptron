"""Iterated-LP on finite matrices (Algorithm 1 of the paper), solved with HiGHS.

In round t we maximize v_t . x over x in [-1, 1]^N subject to A x >= kappa_t,
with the coordinates frozen in earlier rounds held fixed. Coordinates of the
optimizer with |x_j| >= 1 - 1e-9 are then frozen at their signs.
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

FREEZE_TOLERANCE = 1e-9      # |x_j| >= 1 - FREEZE_TOLERANCE freezes coordinate j
MARGIN_TOLERANCE = 1e-7      # tolerance for row margins (feasibility and tight rows)
LP_FEASIBILITY_TOLERANCE = 1e-9


def gaussian_matrix(rng, m, n):
    """M x N matrix with iid N(0, 1/N) entries."""
    return rng.standard_normal((m, n)) / np.sqrt(n)


def rademacher_matrix(rng, m, n):
    """M x N matrix with iid entries +-1/sqrt(N)."""
    return (2 * rng.integers(0, 2, size=(m, n)) - 1) / np.sqrt(n)


def solve_round(matrix, x, free, objective, kappa, time_limit):
    """Solve the LP of one round.

    Maximizes objective . x over the free coordinates, with frozen coordinates
    fixed at their current values, subject to matrix @ x >= kappa and the box
    [-1, 1]. Returns the indices of the free coordinates and the linprog result.
    """
    m = matrix.shape[0]
    free_index = np.flatnonzero(free)
    frozen_index = np.flatnonzero(~free)
    rhs = kappa - (matrix[:, frozen_index] @ x[frozen_index] if len(frozen_index) else np.zeros(m))
    with warnings.catch_warnings():
        # 'threads' is passed to HiGHS but SciPy reports it as unrecognized.
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
        fit = linprog(-objective[free_index], A_ub=-matrix[:, free_index], b_ub=-rhs,
                      bounds=(-1, 1), method='highs',
                      options={'primal_feasibility_tolerance': LP_FEASIBILITY_TOLERANCE,
                               'dual_feasibility_tolerance': LP_FEASIBILITY_TOLERANCE,
                               'threads': 1, 'time_limit': time_limit})
    return free_index, fit


def freeze(x, free, free_index, solution):
    """Write the LP solution into x and freeze coordinates that reached the boundary.

    Modifies x and free in place and returns the indices frozen in this round.
    """
    x[free_index] = solution
    newly_frozen = free_index[np.abs(x[free_index]) >= 1 - FREEZE_TOLERANCE]
    x[newly_frozen] = np.sign(x[newly_frozen])
    free[newly_frozen] = False
    return newly_frozen


def sign_vector(x):
    """Coordinatewise signs of x, with sign(0) = +1."""
    return np.where(x >= 0, 1., -1.)
