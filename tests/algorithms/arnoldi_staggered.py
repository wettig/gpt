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
# * sum rule:  tr(-D^2) = 2*Nrep*V = 2*Nev

import gpt as g
import numpy as np
import sys

def run_test(U):
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
    Volume = U[0].grid.fsites
    Nev = Nrep*Volume

    # staggered operator
    stagg = g.qcd.fermion.reference.staggered(U,p)

    # start vector
    start = g.vector_color(U[0].grid, Nrep)
    start[:] = g.vector_color([1 for i in range(Nrep)], Nrep)

    # Arnoldi with modest convergence criterion
    # Typically the Krylov space needs to be larger than Nev to find all evals.
    # The extraneous evals found by Arnoldi are essentially zero.
    # * Nstop = number of evals to be computed
    # * Nmin = when to start checking convergence (must be >= Nstop)
    # * Nstep = interval for convergence checks (should not be too small to minimize cost of check)
    # * Nmax = when to abort
    Neval = Nev + 50 
    # The following doesn't work.
    a = g.algorithms.eigen.arnoldi(Nmin=Nev, Nmax=Neval, Nstep=20, Nstop=Nev, resid=1e-5)
    # The following works.
    a = g.algorithms.eigen.arnoldi(Nmin=Neval, Nmax=Neval, Nstep=20, Nstop=Neval, resid=1e-5)
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

# grid (each dimension must be at least 4 to get correct sum rule)
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# SU(3) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"))
ev = run_test(U)

# SU(2) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_fundamental())
ev = run_test(U)

# SU(2) adjoint
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_adjoint())
ev = run_test(U)
