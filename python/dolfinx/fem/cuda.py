from __future__ import annotations

import collections
import functools
import typing

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.assemble import create_matrix

@functools.singledispatch
def assemble_matrix_cuda(
    a: typing.Any,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    block_mode: typing.Optional[la.BlockMode] = None,
):
    """Assemble bilinear form into a matrix on the GPU.

    Args:
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on on the matrix diagonal.
        constants: Constants that appear in the form. If not provided,
            any required coefficients will be computed.
         block_mode: Block size mode for the returned space matrix. If
            ``None``, default is used.

    Returns:
        Matrix representation of the bilinear form ``a``.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """

    bcs = [] if bcs is None else bcs
    A: la.MatrixCSR = create_matrix(a, block_mode)
    _assemble_matrix_csr_cuda(A, a, bcs, diagonal, constants, coeffs)
    

    return A

@assemble_matrix_cuda.register
def _assemble_matrix_csr_cuda(
    A: la.MatrixCSR,
    a: Form,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> la.MatrixCSR:
    """Assemble bilinear form into a matrix on the GPU.

    Args:
        A: The matrix to assemble into. It must have been initialized
            with the correct sparsity pattern.
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on on
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
            the matrix diagonal.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """
    # step 1 - create a cuda context
    # TODO - manage this guy properly
    ctx = _cpp.fem.CUDAContext()
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    cuda_bcs = []
    for bc in bcs:
      if type(bc) is _cpp.fem.DirichletBC_float32:
        cuda_bc = _cpp.fem.CUDADirichletBC_float32(ctx, bc.function_space, [bc])
      elif type(bc) is _cpp.fem.DirichletBC_float64:
        cuda_bc = _cpp.fem.CUDADirichletBC_float64(ctx, bc.function_space, [bc])
      else:
        raise ValueError(f"No CUDA wrapper for bc of type {type(bc)}")

      cuda_bcs.append(cuda_bc)
    #constants = _pack_constants(a._cpp_object) if constants is None else constants
    #coeffs = _pack_coefficients(a._cpp_object) if coeffs is None else coeffs
    #_cpp.fem.assemble_matrix(A._cpp_object, a._cpp_object, constants, coeffs, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    #if a.function_spaces[0] is a.function_spaces[1]:
    #    _cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], bcs, diagonal)
    return A



