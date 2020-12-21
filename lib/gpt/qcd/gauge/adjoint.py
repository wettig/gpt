#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Tilo Wettig (tilo.wettig@ur.de)
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

import gpt


def fundamental_to_adjoint(U, U_adj):
    """
    For SU(2), convert fundamental to adjoint representation

    Parameters: fundamental gauge field (single matrix, list, or lattice) 
    
    Returns: adjoint gauge field
    """
    #
    # Code based on transform.py


    if type(U) == list:
        return [fundamental_to_adjoint(x) for x in U]

    elif type(U) == gpt.tensor:  # U is a single matrix
        assert(U.otype == gpt.ot_matrix_su2_fundamental)
        dt = type(U[0, 0])
        T = g.ot_matrix_su2_fundamental.generators(U,dt)
        
    elif type(U) == gpt.lattice:
        assert(U[0,0,0,0].otype == gpt.ot_matrix_su2_fundamental)

    else:
        print("Argument must be fundamental SU(2) gauge field ")
        assert 0
    
#    for a in range(3):
#        for b in range(3):
#            U_adj[a, b] = 2 * trace(T[a] * U_fund * T[b] * adj(U_fund))
