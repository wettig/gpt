#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2021
#
# Desc.: Test reference implementation of staggered Dirac operator.
#        For properties of the eigenvalues, see arXiv:xxx.

import gpt as g
import numpy as np
from itertools import permutations


def compute_evals_numpy(U, p):
    r"""
    The eigenvalues come in pairs (\lambda+m, -\lambda+m). Therefore we compute
    only half of the spectrum, which asymptotically saves 8x in runtime.
    """
    # operator
    stagg = g.qcd.fermion.reference.staggered(U, p)
    op = stagg.Meooe * stagg.Meooe + stagg.Mooee * stagg.Mooee
    # sizes
    Nrep = U[0].otype.Ndim
    N = Nrep * U[0].grid.fsites // 2
    grid_eo = U[0].grid.checkerboarded(g.redblack)
    # allocate gpt vectors and numpy arrays (including dense matrix for operator)
    xg = g.vector_color(grid_eo, Nrep)
    yg = g.vector_color(grid_eo, Nrep)
    xn = np.zeros(N, complex)
    yn = np.empty(N, complex)
    D = np.empty((N, N), complex, order="F")  # "F" for column-major storage
    # create copy_plans between gpt and numpy
    g2n = g.copy_plan(yn, yg)
    g2n.source += yg.view[:]
    g2n.destination += g.global_memory_view(
        yg.grid,
        [[yg.grid.processor, yn, 0, yn.nbytes]] if yn.nbytes > 0 else None,
    )
    g2n = g2n()
    n2g = g.copy_plan(xg, xn)
    n2g.source += g.global_memory_view(
        xg.grid,
        [[xg.grid.processor, xn, 0, xn.nbytes]] if xn.nbytes > 0 else None,
    )
    n2g.destination += xg.view[:]
    n2g = n2g()
    # compute matrix D of staggered operator
    for i in range(N):
        xn[i] = 1
        n2g(xg, xn)
        xn[i] = 0
        yg @= op * xg
        g2n(yn, yg)
        D[:, i] = yn
    # compute eigenvalues
    evals = np.linalg.eigvals(D)
    mass = p["mass"] if "mass" in p else 0.0
    return np.sqrt(evals - mass ** 2) + mass


def Udelta_average(U):
    r"""
    Compute the average <\bar U_\delta(x)\bar U_\delta(x)^\dagger>,
    see Eq. (2.3) in arXiv:1503.06670 but with my change of notation
    """
    Volume = float(U[0].grid.fsites)
    Udelta = g.lattice(U[0].grid, U[0].otype)
    Udelta[:] = 0.0
    for [i, j, k] in permutations([0, 1, 2]):
        Udelta += U[i] * g.cshift(U[j], i, 1) * g.cshift(g.cshift(U[k], i, 1), j, 1)
    return g.sum(g.trace(Udelta * g.adj(Udelta))).real / Volume / 36.0


def sumrule_expected(U, p):
    """
    Expected sum of staggered eigenvalues squared.
    """
    Nrep = U[0].otype.Ndim
    V = float(U[0].grid.fsites)
    expected = -2.0 * Nrep
    if "mass" in p:
        expected += Nrep * p["mass"] ** 2
    if "mu5" in p:
        expected -= 0.5 * p["mu5"] ** 2 * Udelta_average(U)
    if "chiral_U1" in p:
        theta = p["chiral_U1"]
        theta_sq_ave = sum([g.sum(theta[mu] * theta[mu]) for mu in range(4)]) / 4.0 / V
        expected -= 2.0 * Nrep * (theta_sq_ave - 1.0)
    return expected * V


def compute_and_check_eigenvalues(U, p):
    r"""
    The eigenvalues come in pairs (\lambda+m, -\lambda+m). Therefore we compute
    only half of the spectrum, which asymptotically saves 8x in runtime.
    """
    evals = compute_evals_numpy(U, p)
    mass = p["mass"] if "mass" in p else 0.0
    actual = 2.0 * sum((evals - mass) ** 2 + mass ** 2)
    expected = sumrule_expected(U, p)
    try:
        assert abs((actual - expected) / expected) < 1e-8
        g.message("sumrule satisfied")
    except AssertionError:
        g.message("sumrule violated")


def run_test(U, p, src, test_sumrule=False):
    # stagg = g.qcd.fermion.reference.staggered(U, p)
    # dst @= stagg * src
    if test_sumrule:
        compute_and_check_eigenvalues(U, p)


def convert_U1_chiral(phi):
    """
    Input: U(1) gauge field phi
    Output: chiral U(1) gauge field theta
    * on even grid, theta = phi
    * on odd grid, theta = adj(phi)
    """
    grid = phi[0].grid
    theta = [g.complex(grid) for i in range(4)]
    grid_eo = phi[0].grid.checkerboarded(g.redblack)
    tmp_eo = g.complex(grid_eo)
    for mu in range(4):
        g.pick_checkerboard(g.even, tmp_eo, phi[mu])
        g.set_checkerboard(theta[mu], tmp_eo)
        g.pick_checkerboard(g.odd, tmp_eo, phi[mu])
        g.set_checkerboard(theta[mu], g.eval(g.adj(tmp_eo)))
    return theta


def main(test_sumrule=False):
    # grid (every dimension must be larger than 2 to get correct sum rule)
    L = [4, 4, 4, 4]
    grid = g.grid(L, g.double)
    rng = g.random("test", "vectorized_ranlux24_24_64")

    # chiral U(1) field
    theta = [g.complex(grid) for i in range(4)]
    rng.uniform_real(theta, min=0, max=2.0 * np.pi)
    for mu in range(4):
        theta[mu] = g.component.exp((g.eval(1j * theta[mu])))
    theta = convert_U1_chiral(theta)

    # staggered parameters
    mass = 1.23
    mu5 = 2.13 + 3.12 * 1j
    boundary_phases = [1.0, 1.0, 1.0, 1.0]
    stagg_params = {
        "plain": {"boundary_phases": boundary_phases},
        "mass": {"mass": mass, "boundary_phases": boundary_phases},
        "mu5": {"mu5": mu5, "boundary_phases": boundary_phases},
        "chiral_U1": {"chiral_U1": theta, "boundary_phases": boundary_phases},
        "all": {
            "mass": mass,
            "mu5": mu5,
            "chiral_U1": theta,
            "boundary_phases": boundary_phases,
        },
    }

    reps = [
        eval("g.ot_matrix_" + rep)
        for rep in [
            "su_n_fundamental_group(2)",
            "su_n_adjoint_group(2)",
            "su_n_fundamental_group(3)",
        ]
    ]
    for rep in reps:
        U = g.qcd.gauge.random(grid, g.random("test"), otype=rep)
        src = rng.cnormal(g.vector_color(grid, rep.Ndim))
        for params in stagg_params.values():
            run_test(U, params, src, test_sumrule)


if __name__ == "__main__":
    # use test_sumrule=True to test sumrules (takes long)
    main(test_sumrule=True)
