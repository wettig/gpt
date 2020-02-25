#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
cg=g.algorithms.iterative.cg
powit=g.algorithms.iterative.power_iteration

m0crit=4.0
m0=g.default.get_float("--mass",0.1) + m0crit

grid=g.grid(g.default.grid, g.single)

src=g.complex(grid)
src[:]=0
src[0,0,0,0]=1

# Create a free Klein-Gordon operator (spectrum from mass^2-16 .. mass^2)
def A(dst,src,mass):
    g.eval(dst,(mass**2.)*src)
    for i in range(4):
        g.eval(dst, dst + g.cshift(src, i, 1) + g.cshift(src, i, -1) - 2*src )

# find largest eigenvalue
g.message("Largest eigenvalue: ", powit(lambda i,o: A(o,i,m0),src,1e-6,100)[0])

# Perform CG
psi=g.lattice(src)
psi[:]=0
cg(lambda i,o: A(o,i,m0),src,psi,1e-8,1000)

g.meminfo()

# Test CG
tmp=g.lattice(psi)
A(tmp,psi,m0)
g.message("True residuum:", g.norm2(g.eval(tmp - src)))
