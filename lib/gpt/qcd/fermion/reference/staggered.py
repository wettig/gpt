#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class staggered(shift, matrix_operator):
    # M_xy = 1/2 sum_mu eta_mu(x) [ U_mu(x) delta_{x+mu,y} - U_mu^dagger(x-mu) delta_{x-mu,y} ] + m delta_xy
    @params_convention()
    def __init__(self, U, params):

        shift.__init__(self, U, params)

        Nc = U[0].otype.Nc
        otype = g.ot_vector_color(Nc)
        grid = U[0].grid
        self.mass = params["mass"]

        self.Mshift = g.matrix_operator(
            lambda dst, src: self._Mshift(dst, src), otype=otype, grid=grid
        )
        self.Mdiag = g.matrix_operator(
            lambda dst, src: self._Mdiag(dst, src), otype=otype, grid=grid
        )
        matrix_operator.__init__(
            self, lambda dst, src: self._M(dst, src), otype=otype, grid=grid
        )

        # staggered phases
        self.phases = g.complex(grid)[4]
        
    
    def _Mshift(self, dst, src):
        assert dst != src
        dst[:] = 0
        for mu in range(4):
            src_plus = g.eval(self.forward[mu] * src)
            src_minus = g.eval(self.backward[mu] * src)
            dst += phases[mu] * ( src_plus - src_minus ) / 2.0 

    def _Mdiag(self, dst, src):
        assert dst != src
        dst @= self.mass * src

    def _M(self, dst, src):
        assert dst != src
        dst @= self.Mshift * src + self.Mdiag * src
