"""Numerical companion to "State Evolution and Universality of Iterated Linear
Programs for the Binary Perceptron" (Jianan Zhang and Shuangping Li).

Modules
-------
clipped_gaussian  closed-form moments of a Gaussian clipped to [-1, 1]
population        particle approximation of the population (state-evolution) recursion
schedules         margin schedules kappa_t
rounding          the rounding loss D_{a,T}(q) and the finishing test of Section 8
finite_lp         Iterated-LP on finite matrices, solved with HiGHS
io                small helpers for JSON files and run environments

Importing this package limits BLAS to one thread (unless the variables are
already set). This keeps runs reproducible and lets several LP workers share a
laptop. Import it before NumPy.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
