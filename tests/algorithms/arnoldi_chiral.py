import gpt as g
import numpy as np
import sys
from itertools import permutations


def compute_evals(U):
    """ 
    Compute all eigenvalues of D_5 (staggered)
    Input:  U = gauge field
    Output: imaginary parts of the eigenvalues
    """
    # dimension of vector on which D_5 acts depends on Nc and on fermion representation
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

    # extract imaginary parts and delete extraneous eigenvalues near zero
    ev = evals.imag
    ev.sort()
    ev = np.concatenate([ev[:Nev//2], ev[-Nev//2:]])

    return ev


def Udelta_average(U):
    """
    compute < Udelta * Udelta^\dagger >
    """
    vol = float(U[0].grid.fsites)
    Udelta = g.lattice(U[0].grid, U[0].otype)
    Udelta[:] = 0.0

    for [i, j, k] in permutations([0, 1, 2]):
        Udelta += U[i] * g.cshift(U[j], i, 1) * g.cshift(g.cshift(U[k], i, 1), j, 1)

    return g.sum(g.trace(Udelta * g.adj(Udelta))).real / vol / 36.0


def run_test(U):
    """ 
    Check the exact sum rule for tr(D_5^2)
    Inputs: U    = gauge field
    Output: imaginary parts of the eigenvalues
    """
    # extract representation (for output message)
    Nc = U[0].otype.Nc
    if "adjoint" in U[0].otype.__name__:
        rep  = "adjoint"
    else:
        rep = "fundamental"

    # compute imaginary parts of eigenvalues
    ev = compute_evals(U)
    summe = sum(ev*ev)

    # expected sum
    Volume = U[0].grid.fsites
    expected = Udelta_average(U) * Volume / 2.0

    # check sum rule
    g.message(f"tr(-D_5^2): {summe}, expected: {expected}")
    assert abs(summe - expected)/Volume < 1e-4
    g.message(f"Test passed for SU({Nc}) {rep}.")

    return 1


########################################################
# tests for different gauge groups and representations #
########################################################

# staggered parameters
p = {
    "mass": 0,
    "mu5": 1,
    "hop": 0,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

# grid (each dimension must be at least 4 to get correct sum rule)
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

# SU(2) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_fundamental())
ev = run_test(U)

# SU(2) adjoint
U = g.qcd.gauge.random(grid_sp, g.random("test"), otype=g.ot_matrix_su2_adjoint())
run_test(U)

# SU(3) fundamental
U = g.qcd.gauge.random(grid_sp, g.random("test"))
run_test(U)
