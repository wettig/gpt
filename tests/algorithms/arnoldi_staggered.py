#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#
# Desc.: Test reference implementation of staggered Dirac operator
#
# general properties
# * the eigenvalues are imaginary and occur in pairs +/- i\lambda
# * for SU(2) fundamental, the eigenvalues are doubly degenerate
# * the total number of eigenvalues is Nrep*V
#   - Nrep = Nc (fundamental) or Nc^2-1 (adjoint)
#   - V = lattice volume
# * sum rule:  tr(-D^2) = 2*Nrep*V

import gpt as g
import numpy as np
import sys

def run_test(U, Nadd):
    """ 
    Compute all eigenvalues of D_staggered for different colors and
    representations and check the exact sum rule for tr(D^2)
    Inputs: U    = gauge field
            Nadd = additional eigenvalues for Arnoldi (see below) 
    Output: imaginary parts of the eigenvalues
    """

    # dimension of vector on which D acts depends on Nc and on fermion representation
    Nc = U[0].otype.Nc
    if "adjoint" in U[0].otype.__name__:
        Nrep = Nc*Nc-1
        rep  = "adjoint"
    else:
        Nrep = Nc
        rep = "fundamental"
    Volume = np.prod(U[0].grid.fdimensions)
    Nev = Nrep*Volume

    # staggered operator
    stagg = g.qcd.fermion.reference.staggered(U,p)

    # start vector
    start = g.vector_color(U[0].grid, Nrep)
    start[:] = g.vector_color([1 for i in range(Nrep)], Nrep)

    # Arnoldi with modest convergence criterion
    # * with Nmin = Nmax = Nstop = Nev, Arnoldi misses some eigenvalues
    #   -> sum rule tr(-D^2) = 2*Nev is not satisfied
    # * if a few (Nadd) "additional" eval are requested, Arnoldi finds all evals
    #   (and produces extraneous evals near zero) -> sum rule satisfied
    Neval = Nev+Nadd
    a = g.algorithms.eigen.arnoldi(Nmin=Neval, Nmax=Neval, Nstep=10, Nstop=Neval, resid=1e-5)
    evec, evals = a(stagg, start)

    # extract imaginary parts and delete extraneous eigenvalues near zero
    ev = evals.imag
    ev.sort()
    ev = np.concatenate([ev[:Nev//2], ev[-Nev//2:]])

    # check sum rule
    sum = (ev*ev).sum()
    g.message(f"tr(-D^2): {sum}, expected: {2*Nev}")
    assert abs(sum - 2*Nev) <1e-3

    g.message(f"Test passed for SU({Nc}) {rep}.")
    return ev

########################################################
# tests for different gauge groups and representations #
########################################################

# staggered parameters
p = {
    "mass": 0,
    "mu5": 0,
    "hop": 1,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

# grid (each dimensions must be at least 4 to get correct sum rule; why?)
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# SU(3) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"))
ev = run_test(U, 36)

# SU(2) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_fundamental())
ev = run_test(U, 28)

# SU(2) adjoint
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_adjoint())
ev = run_test(U, 45)
