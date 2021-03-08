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
from gpt.core.covariant import shift_eo
from gpt import matrix_operator
from itertools import permutations


class staggered(shift_eo, matrix_operator):
    """
    D_xy = 1/2 sum_mu eta_mu(x) [ U_mu(x) delta_{x+mu,y} - U_mu^dagger(x-mu) delta_{x-mu,y} ] + m delta_xy
    * gauge field could also be adjoint (see otype)
    * extensions (passed in params):
      - mu5: coefficient of chiral chemical potential, see Eqs. (2.2) and (2.3) in
        https://link.springer.com/content/pdf/10.1007/JHEP06(2015)094.pdf
      - chiral_U1: chiral U(1) gauge field
    """

    @params_convention()
    def __init__(self, U, params):

        assert U[0].grid.nd == 4, "Only 4 dimensions implemented for now."

        shift_eo.__init__(self, U, params)

        # stuff that's needed later on
        Ndim = U[0].otype.Ndim
        otype = g.ot_vector_color(Ndim)
        grid = U[0].grid
        grid_eo = grid.checkerboarded(g.redblack)
        self.F_grid = grid
        self.U_grid = grid
        self.F_grid_eo = grid_eo
        self.U_grid_eo = grid_eo

        self.src_e = g.vector_color(grid_eo, Ndim)
        self.src_o = g.vector_color(grid_eo, Ndim)
        self.dst_e = g.vector_color(grid_eo, Ndim)
        self.dst_o = g.vector_color(grid_eo, Ndim)
        self.dst_e.checkerboard(g.even)
        self.dst_o.checkerboard(g.odd)

        self.mass = (
            params["mass"] if "mass" in params and params["mass"] != 0.0 else None
        )
        self.mu5 = params["mu5"] if "mu5" in params and params["mu5"] != 0.0 else None
        self.chiral_U1 = params["chiral_U1"] if "chiral_U1" in params else None

        # matrix operators
        self.Mooee = g.matrix_operator(
            lambda dst, src: self._Mooee(dst, src), otype=otype, grid=grid_eo
        )
        self.Meooe = g.matrix_operator(
            lambda dst, src: self._Meooe(dst, src), otype=otype, grid=grid_eo
        )
        matrix_operator.__init__(
            self, lambda dst, src: self._M(dst, src), otype=otype, grid=grid
        )
        self.Mdiag = g.matrix_operator(
            lambda dst, src: self._Mdiag(dst, src), otype=otype, grid=grid
        )

        # staggered phases
        # see also Grid/Grid/qcd/action/fermion/StaggeredImpl.h
        _phases = [g.complex(grid) for i in range(4)]
        for mu in range(4):
            _phases[mu][:] = 1.0
        for x in range(0, grid.fdimensions[0], 2):
            _phases[1][x + 1, :, :, :] = -1.0
            for y in range(0, grid.fdimensions[1], 2):
                _phases[2][x, y + 1, :, :] = -1.0
                _phases[2][x + 1, y, :, :] = -1.0
                for z in range(0, grid.fdimensions[2], 2):
                    _phases[3][x, y, z + 1, :] = -1.0
                    _phases[3][x, y + 1, z, :] = -1.0
                    _phases[3][x + 1, y, z, :] = -1.0
                    _phases[3][x + 1, y + 1, z + 1, :] = -1.0
        # use stride > 1 once it is implemented:
        # _phases[1][1::2, :, :, :] = -1.0
        # _phases[2][0::2, 1::2, :, :] = -1.0
        # _phases[2][1::2, 0::2, :, :] = -1.0
        # _phases[3][0::2, 0::2, 1::2, :] = -1.0
        # _phases[3][0::2, 1::2, 0::2, :] = -1.0
        # _phases[3][1::2, 0::2, 0::2, :] = -1.0
        # _phases[3][1::2, 1::2, 1::2, :] = -1.0
        self.phases = {}
        for cb in [g.even, g.odd]:
            _phases_eo = [g.lattice(grid_eo, _phases[0].otype) for i in range(4)]
            for mu in range(4):
                g.pick_checkerboard(cb, _phases_eo[mu], _phases[mu])
            self.phases[cb] = _phases_eo

        # theta is the chiral U(1) gauge field
        if self.chiral_U1:
            # for now, allow both mu5 and chiral U(1) field for testing purposes
            # assert "mu5" not in params, "should not have both mu5 and chiral_U1 in params"
            assert (
                len(self.chiral_U1) == 4
            ), "chiral U(1) field should be list of length 4"
            self.theta = {}
            for cb in [g.even, g.odd]:
                _theta_eo = [
                    g.lattice(grid_eo, self.chiral_U1[0].otype) for i in range(4)
                ]
                for mu in range(4):
                    g.pick_checkerboard(cb, _theta_eo[mu], self.chiral_U1[mu])
                self.theta[cb] = _theta_eo

        # s(x) is defined between (2.2) and (2.3) in
        # https://link.springer.com/content/pdf/10.1007/JHEP06(2015)094.pdf
        if self.mu5:
            self.s = {}
            _s = g.complex(grid)
            for y in range(0, grid.fdimensions[1], 2):
                _s[:, y, :, :] = 1.0
                _s[:, y + 1, :, :] = -1.0
            for cb in [g.even, g.odd]:
                _s_eo = g.lattice(grid_eo, _s.otype)
                g.pick_checkerboard(cb, _s_eo, _s)
                self.s[cb] = _s_eo
        # use stride > 1 once it is implemented:
        # self.s[1][:, 0::2, :, :] = 1.0
        # self.s[1][:, 1::2, :, :] = -1.0

    def _Mooee(self, dst, src):
        assert dst != src
        cb = src.checkerboard()
        dst.checkerboard(cb)
        if self.mass:
            dst @= self.mass * src
        else:
            dst[:] = 0

    def _Meooe(self, dst, src):
        assert dst != src
        cb = src.checkerboard()
        scb = self.checkerboard[cb]
        scbi = self.checkerboard[cb.inv()]
        phases_cb = self.phases[cb.inv()]
        dst.checkerboard(cb.inv())
        dst[:] = 0
        if self.chiral_U1:
            theta_cb = self.theta[cb]
            theta_cbi = self.theta[cb.inv()]
        for mu in range(4):
            src_plus = g.eval(scbi.forward[mu] * src)
            src_minus = g.eval(scb.backward[mu] * src)
            if self.chiral_U1:
                src_plus = g.eval(theta_cbi[mu] * src_plus)
                src_minus = g.eval(g.cshift(theta_cb[mu], mu, -1) * src_minus)
            dst += phases_cb[mu] * (src_plus - src_minus) / 2.0
        if self.mu5:
            s_cbi = self.s[cb.inv()]
            for [i, j, k] in permutations([0, 1, 2]):
                src_plus = g.eval(
                    scbi.forward[i] * scb.forward[j] * scbi.forward[k] * src
                )
                src_minus = g.eval(
                    scb.backward[k] * scbi.backward[j] * scb.backward[i] * src
                )
                dst += s_cbi * (src_plus + src_minus) * self.mu5 / (-12.0)

    def _M(self, dst, src):
        assert dst != src

        g.pick_checkerboard(g.even, self.src_e, src)
        g.pick_checkerboard(g.odd, self.src_o, src)

        self.dst_o @= self.Meooe * self.src_e + self.Mooee * self.src_o
        self.dst_e @= self.Meooe * self.src_o + self.Mooee * self.src_e

        g.set_checkerboard(dst, self.dst_o)
        g.set_checkerboard(dst, self.dst_e)

    def _Mdiag(self, dst, src):
        assert dst != src

        g.pick_checkerboard(g.even, self.src_e, src)
        g.pick_checkerboard(g.odd, self.src_o, src)

        self.dst_o @= self.Mooee * self.src_o
        self.dst_e @= self.Mooee * self.src_e

        g.set_checkerboard(dst, self.dst_o)
        g.set_checkerboard(dst, self.dst_e)
