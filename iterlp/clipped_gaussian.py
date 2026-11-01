"""Moments of a Gaussian score clipped to the interval [-tau, tau].

In round t a free coordinate is updated to Y = clip(Z / tau, -1, 1), where,
given the history, Z is Gaussian with mean `mean` and standard deviation `sd`.
The population recursion needs E[Y], E[Y^2], P(|Z| < tau) and the two
unnormalized interior moments E[Z 1{|Z|<tau}] and E[Z^2 1{|Z|<tau}]. All of them
have closed forms in terms of the normal density and distribution function.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

SQRT_2PI = math.sqrt(2 * math.pi)

# Below this ratio tau/sd the interior moments are computed with a fixed
# 12-point Gauss-Legendre rule on [-tau, tau]; the closed-form second moment
# loses precision to cancellation on very narrow intervals.
NARROW_INTERVAL_RATIO = 0.03
LEGENDRE_NODES, LEGENDRE_WEIGHTS = leggauss(12)


def normal_pdf(x):
    """Standard normal density."""
    return np.exp(-0.5 * np.asarray(x) ** 2) / SQRT_2PI


def clipped_moments(mean, sd, tau):
    """Moments of Y = clip(Z / tau, -1, 1) for Z ~ N(mean, sd^2), elementwise.

    Returns (E[Y], E[Y^2], P(|Z| < tau), E[Z 1{|Z|<tau}], E[Z^2 1{|Z|<tau}]).
    """
    lo = (-tau - mean) / sd
    hi = (tau - mean) / sd
    if tau / sd < NARROW_INTERVAL_RATIO:
        # Quadrature on z = tau * node, node in [-1, 1].
        density = normal_pdf((tau * LEGENDRE_NODES[None, :] - mean[:, None]) / sd) * (tau / sd) * LEGENDRE_WEIGHTS
        mass = density.sum(axis=1)
        first_inside = tau * (density @ LEGENDRE_NODES)
        second_inside = tau * tau * (density @ (LEGENDRE_NODES ** 2))
        mean_y = first_inside / tau + ndtr(-hi) - ndtr(lo)
        mean_y2 = second_inside / tau ** 2 + ndtr(lo) + ndtr(-hi)
        return mean_y, mean_y2, mass, first_inside, second_inside

    pdf_lo = normal_pdf(lo)
    pdf_hi = normal_pdf(hi)
    # When both limits are positive, subtract upper tails to avoid cancellation.
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    first_inside = mean * mass + sd * (pdf_lo - pdf_hi)
    second_inside = ((mean ** 2 + sd ** 2) * mass
                     + 2 * mean * sd * (pdf_lo - pdf_hi)
                     + sd ** 2 * (lo * pdf_lo - hi * pdf_hi))
    mean_y = first_inside / tau + ndtr(-hi) - ndtr(lo)
    mean_y2 = second_inside / tau ** 2 + ndtr(lo) + ndtr(-hi)
    return mean_y, mean_y2, mass, first_inside, second_inside
