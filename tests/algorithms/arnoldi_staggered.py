#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# staggered parameters
p = {
    "mass": 0,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

# grid
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# staggered operator
stagg = g.qcd.fermion.reference.staggered(U,p))

# test works for different colors and representations
def run_test(U):
    return
    Nc = U[0].otype.Nc
    if "adjoint" in U[0].otype.__name__:
        ndim = Nc*Nc-1
    else:
        ndim = Nc
    otype = g.ot_vector_color(ndim)
    # start vector
    start = g.vector_color(U[0].grid, ndim)
    start[:] = g.vector_color([1 for i in range(ndim)], ndim)
    # arnoldi with modest convergence criterion
    a = g.algorithms.eigen.arnoldi(Nmin=50, Nmax=120, Nstep=10, Nstop=1, resid=1e-5)
    evec, evals = a(stagg, start)
    # expect the largest eigenvector to have converged somewhat
    expected_largest_eigenvalue = 7.437868841644861 + 0.012044335728622612j
    evals_test = g.algorithms.eigen.evals(w, evec[-1:], check_eps2=1e5 * evals[-1] ** 2.0)
    assert abs(evals_test[-1] - expected_largest_eigenvalue) < 1e-3

##########################################
# run tests
##########################################

# SU(3) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"))
run_test(U)

# SU(2) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_fundamental())
run_test(U)

# SU(2) adjoint
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_adjoint())
run_test(U)
