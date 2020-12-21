#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#
# Desc.: Test reference implementation of staggered Dirac operator
#
# general properties
# * the eigenvalues of D_hop and D_5 are imaginary and occur in pairs +/- i\lambda
# * for SU(2) fundamental, the eigenvalues are doubly degenerate
# * the total number of eigenvalues is Nrep*V
#   - Nrep = Nc (fundamental) or Nc^2-1 (adjoint)
#   - V = lattice volume
# * sum rules:  tr(-D_hop^2) = 2*Nrep*V = 2*Nev
#               tr(-D_5^2)   = < tr \bar U_delta \bar U_delta^\dagger > * V/2

import gpt as g
import numpy as np
import sys
from itertools import permutations


def compute_sum(U, p):
    """ 
    Compute sum of squared eigenvalues of staggered operator (with m=0)
    Inputs: U = gauge field
            p = staggered parameters
    Output: tr(-D^2)
    """
    # dimension of vector on which D acts depends on Nc and on fermion representation
    Nc = U[0].otype.Nc
    if "adjoint" in U[0].otype.__name__:
        Nrep = Nc*Nc-1
    else:
        Nrep = Nc
    Volume = U[0].grid.fsites
    Nev = Nrep*Volume

    # staggered operator
    stagg = g.qcd.fermion.reference.staggered(U, p)

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
    a = g.algorithms.eigen.arnoldi(Nmin=Neval, Nmax=Neval, Nstep=20, Nstop=Neval, resid=1e-5)
    evec, evals = a(stagg, start)

    # return tr(-D^2)
    return -sum(evals*evals)


def Udelta_average(U):
    """
    compute < tr Udelta * Udelta^\dagger >
    """
    Volume = float(U[0].grid.fsites)
    Udelta = g.lattice(U[0].grid, U[0].otype)
    Udelta[:] = 0.0

    for [i, j, k] in permutations([0, 1, 2]):
        Udelta += U[i] * g.cshift(U[j], i, 1) * g.cshift(g.cshift(U[k], i, 1), j, 1)

    return g.sum(g.trace(Udelta * g.adj(Udelta))).real / Volume / 36.0


def test_sumrule(U, p):
    """ 
    Check exact sum rule for tr(D_hop^2) or tr(D_5^2)
    Inputs: U = gauge field
            p = staggered parameters
    """
    # sanity checks
    mass = p["mass"]
    hop = p["hop"]
    mu5 = p["mu5"]
    assert hop == 1, "Hop parameter must be 1."

    # extract parameters for output message
    Nc = U[0].otype.Nc
    if "adjoint" in U[0].otype.__name__:
        rep  = "adjoint"
        Nrep = Nc*Nc-1
    else:
        rep = "fundamental"
        Nrep = Nc

    # compute sum
    summe = compute_sum(U, p)

    # check sum rule
    Volume = U[0].grid.fsites
    expected1 = - mass**2 * Nrep
    expected2 = 2 * Nrep
    expected3 = .5 * mu5**2 * Udelta_average(U)
    expected  = (expected1 + expected2 + expected3) * Volume
    
    g.message("-----------------------------------------------------------")
    g.message(f"Test for SU({Nc}) {rep}.")
    g.message(f"tr(-D^2): {summe}, expected: {expected}")
    assert abs(summe - expected) / Volume < 1e-8
    g.message(f"Test passed for SU({Nc}) {rep}.")
    g.message("-----------------------------------------------------------")
    
    return 1


#################################################################
# test sum rules for different gauge groups and representations #
#################################################################

# staggered parameters
p = {
    "mass": .897,
    "hop": 1,
    "mu5": complex(1.23, .537),
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

# grid (each dimension must be at least 4 to get correct sum rule)
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)

# SU(2) fundamental
U = g.qcd.gauge.random(grid_dp, g.random("test"), otype=g.ot_matrix_su2_fundamental())
test_sumrule(U, p)

# SU(2) adjoint
U = g.qcd.gauge.random(grid_dp, g.random("test"), otype=g.ot_matrix_su2_adjoint())
test_sumrule(U, p)

# SU(3) fundamental
U = g.qcd.gauge.random(grid_dp, g.random("test"))
test_sumrule(U, p)
