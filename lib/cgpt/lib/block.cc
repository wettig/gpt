/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include "lib.h"

EXPORT(block_project,{

    PyObject* _basis;
    void* _coarse,* _fine;
    int idx;
    if (!PyArg_ParseTuple(args, "llOi", &_coarse,&_fine,&_basis,&idx)) {
      return NULL;
    }

    cgpt_Lattice_base* fine = (cgpt_Lattice_base*)_fine;
    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,idx);

    fine->block_project(coarse,basis);

    return PyLong_FromLong(0);
  });

EXPORT(block_promote,{

    PyObject* _basis;
    void* _coarse,* _fine;
    int idx;
    if (!PyArg_ParseTuple(args, "llOi", &_coarse,&_fine,&_basis,&idx)) {
      return NULL;
    }

    cgpt_Lattice_base* fine = (cgpt_Lattice_base*)_fine;
    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,idx);

    fine->block_promote(coarse,basis);

    return PyLong_FromLong(0);
  });

EXPORT(block_orthonormalize,{

    PyObject* _basis;
    void* _coarse;
    if (!PyArg_ParseTuple(args, "lO", &_coarse,&_basis)) {
      return NULL;
    }

    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,0); // TODO: generalize

    ASSERT(basis.size() > 0);
    basis[0]->block_orthonormalize(coarse,basis);

    return PyLong_FromLong(0);
  });


EXPORT(block_maskedInnerProduct,{

    void* _coarse,* _fineMask,* _fineX,* _fineY;
    if (!PyArg_ParseTuple(args, "llll", &_coarse,&_fineMask,&_fineX,&_fineY)) {
      return NULL;
    }

    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;
    cgpt_Lattice_base* fineMask = (cgpt_Lattice_base*)_fineMask;
    cgpt_Lattice_base* fineX = (cgpt_Lattice_base*)_fineX;
    cgpt_Lattice_base* fineY = (cgpt_Lattice_base*)_fineY;

    fineX->block_maskedInnerProduct(coarse,fineMask,fineY);

    return PyLong_FromLong(0);
  });


EXPORT(block_innerProduct,{

    PyObject* _fineX,* _fineY;
    void* _coarse;
    if (!PyArg_ParseTuple(args, "lOO", &_coarse,&_fineX,&_fineY)) {
      return NULL;
    }

    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;

    std::vector<cgpt_Lattice_base*> fineX;
    std::vector<cgpt_Lattice_base*> fineY;
    cgpt_vlattice_fill(fineX,_fineX);
    cgpt_vlattice_fill(fineY,_fineY);

    std::cout << GridLogDebug<< "block_innerProduct: size fineX = " << fineX.size() << " size fineY = " << fineY.size() << std::endl;

    ASSERT(fineX.size() > 0);
    ASSERT(fineY.size() > 0);

    fineX[0]->block_innerProduct(coarse,fineX,fineY);

    return PyLong_FromLong(0);
  });


EXPORT(block_innerProduct_test,{

    void* _coarse,* _fineX,* _fineY;
    if (!PyArg_ParseTuple(args, "lll", &_coarse,&_fineX,&_fineY)) {
      return NULL;
    }

    cgpt_Lattice_base* coarse = (cgpt_Lattice_base*)_coarse;
    cgpt_Lattice_base* fineX = (cgpt_Lattice_base*)_fineX;
    cgpt_Lattice_base* fineY = (cgpt_Lattice_base*)_fineY;

    fineX->block_innerProduct_test(coarse,fineY);

    return PyLong_FromLong(0);
  });


EXPORT(block_zaxpy,{

    void* _fineZ,* _coarseA,* _fineX,* _fineY;
    if (!PyArg_ParseTuple(args, "llll", &_fineZ,&_coarseA,&_fineX,&_fineY)) {
      return NULL;
    }

    cgpt_Lattice_base* fineZ = (cgpt_Lattice_base*)_fineZ;
    cgpt_Lattice_base* coarseA = (cgpt_Lattice_base*)_coarseA;
    cgpt_Lattice_base* fineX = (cgpt_Lattice_base*)_fineX;
    cgpt_Lattice_base* fineY = (cgpt_Lattice_base*)_fineY;

    fineX->block_zaxpy(fineZ,coarseA,fineY);

    return PyLong_FromLong(0);
  });
