#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Tilo Wettig
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
import cgpt, gpt, numpy


def convert_representation(first, second):
    """Convert fundamental to adjoint representation"""
    # Code based on transform.py
    if second == gpt.single or second == gpt.double:

        # if first is a list, distribute
        if type(first) == list:
            return [convert_representation(x, second) for x in first]

        # if first is no list, evaluate
        src = gpt.eval(first)
        dst_grid = src.grid.converted(second)
        return convert_representation(gpt.lattice(dst_grid, src.otype), src)

    elif type(first) == gpt.lattice:

        # second may be expression
        second = gpt.eval(second)

        # now second is lattice
        assert len(first.otype.v_idx) == len(second.otype.v_idx)
        for i in first.otype.v_idx:
            cgpt.convert_representation(first.v_obj[i], second.v_obj[i])

        # set checkerboard
        first.checkerboard(second.checkerboard())
        return first

    else:
        print("Argument must be fundamental gauge field ")
        assert 0
