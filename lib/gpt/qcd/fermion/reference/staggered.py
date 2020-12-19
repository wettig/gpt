#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#    2020 Tilo Wettig <tilo.wettig@ur.de>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g
from gpt.params import params_convention
from gpt.core.covariant import shift
from gpt import matrix_operator
from itertools import permutations


class staggered(shift, matrix_operator):
    '''
    M_xy = 1/2 sum_mu eta_mu(x) [ U_mu(x) delta_{x+mu,y} - U_mu^dagger(x-mu) delta_{x-mu,y} ] + m delta_xy
    * gauge field could also be SU(2) or adjoint (see otype)
    * nohop: if nonzero, turns off "hopping" term (for checks)
    * mu5: coefficient of chiral chemical potential, see Eqs. (2.2) and (2.3) in
      https://link.springer.com/content/pdf/10.1007/JHEP06(2015)094.pdf
    * theta5: if nonzero, chiral U(1) gauge field is present, passed in the U list after the SU field
    '''
    @params_convention()
    def __init__(self, U, params):

        shift.__init__(self, U, params)

        Nc = U[0].otype.Nc
        if "adjoint" in U[0].otype.__name__:
            otype = g.ot_vector_color(Nc*Nc-1)
        else:
            otype = g.ot_vector_color(Nc)
        grid = U[0].grid
        self.mass  = params["mass"]
        self.nohop = params["nohop"]
        self.mu5   = params["mu5"]
        self.theta = params["theta"]
        
        self.Mooee = g.matrix_operator(
            lambda dst, src: self._Mdiag(dst, src), otype=otype, grid=grid
        )
        self.Meooe = g.matrix_operator(
            lambda dst, src: self._Mshift(dst, src), otype=otype, grid=grid
        )
        matrix_operator.__init__(
            self, lambda dst, src: self._M(dst, src), otype=otype, grid=grid
        )

        # staggered phases
        # see also Grid/Grid/qcd/action/fermion/StaggeredImpl.h
        self.phases = [g.complex(grid) for i in range(4)]
        for mu in range(4):
            self.phases[mu][:] = 1.0
        for x in range(0,grid.fdimensions[0],2):
            self.phases[1][x+1, :, :, :] = -1.0
            for y in range(0,grid.fdimensions[1],2):
                self.phases[2][x, y+1, :, :] = -1.0
                self.phases[2][x+1, y, :, :] = -1.0
                for z in range(0,grid.fdimensions[2],2):
                    self.phases[3][x, y, z+1, :] = -1.0
                    self.phases[3][x, y+1, z, :] = -1.0
                    self.phases[3][x+1, y, z, :] = -1.0
                    self.phases[3][x+1, y+1, z+1, :] = -1.0
        # use stride > 1 once it is implemented:
        # self.phases[1][1::2, :, :, :] = -1.0
        # self.phases[2][0::2, 1::2, :, :] = -1.0
        # self.phases[2][1::2, 0::2, :, :] = -1.0
        # self.phases[3][0::2, 0::2, 1::2, :] = -1.0
        # self.phases[3][0::2, 1::2, 0::2, :] = -1.0
        # self.phases[3][1::2, 0::2, 0::2, :] = -1.0
        # self.phases[3][1::2, 1::2, 1::2, :] = -1.0

        # s(x) is defined between (2.2) and (2.3) in
        # https://link.springer.com/content/pdf/10.1007/JHEP06(2015)094.pdf
        self.s = g.complex(grid)
        for y in range(0,grid.fdimensions[1],2):
            self.s[:, y, :, :] = 1.0
            self.s[:, y+1, :, :] = -1.0
            # self.s[:, :, :, y] = 1.0
            # self.s[:, :, :, y+1] = -1.0
        # use stride > 1 once it is implemented:
        # self.s[1][:, 0::2, :, :] = 1.0
        # self.s[1][:, 1::2, :, :] = -1.0

        # eps(x) = (-1)^{x1+x2+x3+x4}
        # simplify this once even-odd structure is implemented
        # core/grid.py: redblack
        # core/lattice.py: checkerboard
        # core/checkerboard.py
        # core/coordinates.py
        self.eps = g.complex(grid)
        

    def _Mooee(self, dst, src):
        assert dst != src
        if (self.mass != 0):
            dst @= self.mass * src
        else:
            dst[:] = 0

    def _Meooe(self, dst, src):
        assert dst != src
        dst[:] = 0
        if (self.hop != 0):
            for mu in range(4):
                src_plus = g.eval(self.forward[mu] * src)
                src_minus = g.eval(self.backward[mu] * src)
                dst += self.phases[mu] * ( src_plus - src_minus ) / 2.0 
        if (self.mu5 != 0):
            for [i, j, k] in permutations([0, 1, 2]):
                src_plus = g.eval(self.forward[i] * self.forward[j] * self.forward[k] * src)
                src_minus = g.eval(self.backward[k] * self.backward[j] * self.backward[i] * src)
                dst += self.s * ( src_plus + src_minus ) * self.mu5 / (-12.0)

    def _M(self, dst, src):
        assert dst != src
        dst @= self.Mooee * src + self.Meooe * src
