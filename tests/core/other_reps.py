#!/usr/bin/env python3
#
# Authors: Tilo Wettig 2020
#
# test code for new gauge field types
#
import gpt as g
import numpy as np
import sys, cgpt

# grid
L = [8, 4, 4, 4]
grid_dp = g.grid(L, g.double)
grid_sp = g.grid(L, g.single)

################################################################################
# Test SU(2) fundamental and conversion to adjoint
################################################################################

rng = g.random("test")
U = g.qcd.gauge.random(grid_sp, rng, otype=g.ot_matrix_su2_fundamental())
eps = abs(g.qcd.gauge.plaquette(U) - 0.8813162545363108)
g.message("Test SU(2) fundamental single", eps)
assert eps < 1e-7
V = g.qcd.gauge.fund2adj(U)
eps = abs(g.qcd.gauge.plaquette(V) - 0.7126823001437717)
g.message("Test SU(2) fundamental to adjoint single", eps)
assert eps < 1e-7

rng = g.random("test")
U = g.qcd.gauge.random(grid_dp, rng, otype=g.ot_matrix_su2_fundamental())
eps = abs(g.qcd.gauge.plaquette(U) - 0.8813162591343201)
g.message("Test SU(2) fundamental double", eps)
assert eps < 1e-14
V = g.qcd.gauge.fund2adj(U)
eps = abs(g.qcd.gauge.plaquette(V) - 0.7126822868786024)
g.message("Test SU(2) fundamental to adjoint double", eps)
assert eps < 1e-14

################################################################################
# Test SU(2) adjoint
################################################################################

rng = g.random("test")
U = g.qcd.gauge.random(grid_sp, rng, otype=g.ot_matrix_su2_adjoint())
eps = abs(g.qcd.gauge.plaquette(U) - 0.712682286898295)
g.message("Test SU(2) adjoint single", eps)
assert eps < 1e-7

rng = g.random("test")
U = g.qcd.gauge.random(grid_dp, rng, otype=g.ot_matrix_su2_adjoint())
eps = abs(g.qcd.gauge.plaquette(U) - 0.7126822868786024)
g.message("Test SU(2) adjoint double", eps)
assert eps < 1e-14
