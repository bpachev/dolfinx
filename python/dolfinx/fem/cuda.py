from __future__ import annotations

import collections
import functools
import typing
import tempfile

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function, FunctionSpace
from petsc4py import PETSc


def to_device(obj: typing.Union[Function, FunctionSpace]):
   """Copy an object needed for assembly to the GPU
   """

   ctx = _cpp.fem.CUDAContext()
   if type(obj) is Function:
       _cpp.fem.copy_function_to_device(ctx, obj._cpp_object)
   elif type(obj) is FunctionSpace:
       _cpp.fem.copy_function_space_to_device(ctx, obj._cpp_object)
   else:
       raise ValueError(f"Cannot copy object of type {type(obj)} to the GPU")


@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None
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

    Returns:
        Matrix representation of the bilinear form ``a``.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """

    bcs = [] if bcs is None else bcs
    A: PETSc.Mat = _cpp.fem.petsc.create_matrix_with_fixed_pattern(a._cpp_object)
    _assemble_matrix_petsc(A, a, bcs, diagonal, constants, coeffs)
    return A

@assemble_matrix.register
def _assemble_matrix_petsc(
    A: PETSc.Mat,
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
    tmpdir = tempfile.TemporaryDirectory()
    assembler = _cpp.fem.CUDAAssembler(ctx, tmpdir.name)
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    # TODO properly handle packing of coefficients and constants
    _cpp.fem.assemble_matrix_on_device(ctx, assembler, a._cpp_object, A, bcs)
    #constants = _pack_constants(a._cpp_object) if constants is None else constants
    #coeffs = _pack_coefficients(a._cpp_object) if coeffs is None else coeffs
    #_cpp.fem.assemble_matrix(A._cpp_object, a._cpp_object, constants, coeffs, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    #if a.function_spaces[0] is a.function_spaces[1]:
    #    _cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], bcs, diagonal)
    return A



