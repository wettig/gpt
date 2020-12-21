#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020 Tilo Wettig
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
import gpt, sys
import numpy

# need a basic container to store the group/algebra data
from gpt.core.object_type import ot_matrix_color, ot_vector_color

# Base class
class ot_matrix_u1_base(ot_matrix_color):
    def __init__(self, name):
        self.Nc = 1
        self.Ndim = 1
        super().__init__(1)  # Ndim x Ndim matrix
        self.__name__ = name
        self.data_alias = lambda: ot_matrix_color(1)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({1})": (lambda: ot_vector_color(1), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }


class ot_matrix_u1_group(ot_matrix_u1_base):
    def __init__(self, name):
        super().__init__(1, 1, name)

    def is_element(self, U):
        I = gpt.identity(U)
        err = (gpt.norm2(U * gpt.adj(U) - I) / gpt.norm2(I)) ** 0.5
        # consider additional determinant check
        return err < U.grid.precision.eps * 10.0
