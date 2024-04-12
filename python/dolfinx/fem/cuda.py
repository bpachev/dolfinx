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
from dolfinx.mesh import Mesh
from dolfinx import fem as fe
from dolfinx.fem.petsc import create_vector as create_petsc_vector, create_matrix as create_petsc_matrix
from petsc4py import PETSc
import gc
import numpy as np

def init_device():
  """Initialize PETSc device
  """

  d = PETSc.Device()
  d.create(PETSc.Device.Type.CUDA)
  return d

def create_petsc_cuda_vector(L: Form) -> PETSc.Vec:
  """Create PETSc Vector on device
  """
  index_map = L.function_spaces[0].dofmap.index_map
  bs = L.function_spaces[0].dofmap.index_map_bs
  size = (index_map.size_local * bs, index_map.size_global * bs)
  # we need to provide at least the local CPU array
  arr = np.zeros(size[0])
  return PETSc.Vec().createCUDAWithArrays(cpuarray=arr, size=size, bsize=bs, comm=index_map.comm)

class CUDAAssembler:
  """Class for assembly on the GPU
  """

  def __init__(self):
    """Initialize the assembler
    """

    self._device = init_device()
    self._ctx = _cpp.fem.CUDAContext()
    self._tmpdir = tempfile.TemporaryDirectory()
    self._cpp_object = _cpp.fem.CUDAAssembler(self._ctx, self._tmpdir.name)

  def assemble_matrix(self,
      a: Form,
      mat: typing.Optional[_cpp.fem.CUDAMatrix] = None,
      bcs: typing.Optional[list[DirichletBC]] = None,
      diagonal: float = 1.0
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
    self._copy_form_to_device(a)
    if mat is None:
      mat = self.create_matrix(a)

    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs] 
    _cpp.fem.assemble_matrix_on_device(
       self._ctx, self._cpp_object, a._cuda_form,
       a._cuda_mesh, mat._cpp_object, bcs
    )

    return mat

  def assemble_vector(self,
    b: Form,
    vec: typing.Optional[CUDAVector] = None,
    constants=None, coeffs=None
  ):
    """Assemble linear form into vector on GPU

    Args:
        b: the linear form to use for assembly
        vec: the vector to assemble into. Created if it doesn't exist
        constants: Form constants
        coeffs: Form coefficients
    """

    self._copy_form_to_device(b)
    if vec is None:
      vec = self.create_vector(b)
    
    _cpp.fem.assemble_vector_on_device(self._ctx, self._cpp_object, b._cuda_form,
      b._cuda_mesh, vec._cpp_object)
    return vec

  def create_matrix(self, a: Form) -> CUDAMatrix:
    """Create a CUDAMatrix from a given form
    """
    petsc_mat = _cpp.fem.petsc.create_cuda_matrix(a._cpp_object)
    return CUDAMatrix(self._ctx, petsc_mat)

  def create_vector(self, b: Form) -> CUDAVector:
    """Create a CUDAVector from a given form
    """

    petsc_vec = create_petsc_cuda_vector(b)
    return CUDAVector(self._ctx, petsc_vec)

  def apply_lifting(self,
    b: CUDAVector,
    a: list[Form],
    bcs: list[list[DirichletBC]],
    x0: typing.optional[list[CUDAVector]] = None,
    scale: float = 1.0
  ):
    """GPU equivalent of apply_lifting

    Args:
       b: CUDAVector to modify
       a: list of forms to lift
       bcs: list of boundary condition lists
       x0: optional list of shift vectors for lifting
       scale: scale of lifting
    """
  
    x0 = [] if x0 is None else [x._cpp_object for x in x0]
    _bcs = [[bc._cpp_object for bc in bc_list] for bc_list in bcs]
    cuda_forms = []
    cuda_mesh = None
    for form in a:
      self._copy_form_to_device(form)
      cuda_forms.append(form._cuda_form)
      if cuda_mesh is None: cuda_mesh = form._cuda_mesh

    _cpp.fem.apply_lifting_on_device(
      self._ctx, self._cpp_object,
      cuda_forms, cuda_mesh,
      b._cpp_object, _bcs, x0, scale
    )

  def set_bc(self,
    b: CUDAVector,
    bcs: list[DirichletBC],
    x0: typing.Optional[CUDAVector] = None,
    scale: float = 1.0
  ):
    """Set boundary conditions on device.

    Args:
     b: vector to modify
     x0: optional 
    """

    _bcs = [bc._cpp_object for bc in bcs]
    if x0 is None:
      _cpp.fem.set_bc_on_device(
        self._ctx, self._cpp_object, 
        b._cpp_object, _bcs, scale
      )
    else:
      _cpp.fem.set_bc_on_device(
        self._ctx, self._cpp_object,
        b._cpp_object, _bcs, x0._cpp_object, scale
      )

  def _copy_form_to_device(self, form: Form):
    """Copy all needed assembly data structures to the device.
    """
    
    # prevent duplicate initialization of CUDA data
    if hasattr(form, 'cuda_form'): return

    # now determine the Mesh object corresponding to this form
    form._cuda_mesh = self._copy_mesh_to_device(form.mesh)
    # functionspaces
    coeffs = form._cpp_object.coefficients
    for space in form._cpp_object.function_spaces:
      _cpp.fem.copy_function_space_to_device(self._ctx, space)
    for c in coeffs:
      _cpp.fem.copy_function_space_to_device(self._ctx, c.function_space)

    cpp_form = form._cpp_object
    if type(cpp_form) is _cpp.fem.Form_float32:
      cuda_form = _cpp.fem.CUDAForm_float32(self._ctx, cpp_form)
    elif type(cpp_form) is _cpp.fem.Form_float64:
      cuda_form = _cpp.fem.CUDAForm_float64(self._ctx, cpp_form)
    else:
      raise ValueError(f"Cannot instantiate CUDAForm for Form of type {type(cpp_form)}!")

    # TODO expose these to the user. . . .
    cuda_form.compile(self._ctx, max_threads_per_block=1024, min_blocks_per_multiprocessor=1)
    form._cuda_form = cuda_form

  def _copy_mesh_to_device(self, cpp_mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64]):
    """Copy mesh to device.
    """

    if type(cpp_mesh) is _cpp.mesh.Mesh_float32:
      return _cpp.fem.CUDAMesh_float32(self._ctx, cpp_mesh)
    elif type(cpp_mesh) is _cpp.mesh.Mesh_float64:
      return _cpp.fem.CUDAMesh_float64(self._ctx, cpp_mesh)
    else:
      raise ValueError(f"Cannot instantiate CUDAMesh for Mesh of type {type(cpp_mesh)}!")

    
class CUDAVector:
  """Vector on device
  """

  def __init__(self, ctx, vec):
    """Initialize the vector
    """

    if type(vec) is la.Vector:
      self._petsc_vec = la.create_petsc_vector_wrap(vec)
      # check if vector already has cuda vector
      if vec._cpp_object.has_cuda_vector():
        self._cpp_object = vec._cpp_object.cuda_vector
      else:
        # otherwise create it
        self._cpp_object = _cpp.fem.CUDAVector(ctx, self._petsc_vec)
        vec._cpp_object.set_cuda_vector(self._cpp_object)
    else:
      self._petsc_vec = vec
      self._cpp_object = _cpp.fem.CUDAVector(ctx, self._petsc_vec)

  def vector(self):
    """Return underlying PETSc vector
    """

    return self._petsc_vec

  def __del__(self):
    """Delete the vector and free up GPU resources
    """

    # Ensure that the cpp CUDAVector is taken care of BEFORE the petsc vector. . . .
    del self._cpp_object
    gc.collect()

class CUDAMatrix:
  """Matrix on device
  """

  def __init__(self, ctx, petsc_mat):
    """Initialize the matrix
    """

    self._petsc_mat = petsc_mat
    self._cpp_object = _cpp.fem.CUDAMatrix(ctx, petsc_mat)

  def mat(self):
    """Return underlying CUDA matrix
    """

    return self._petsc_mat

  def __del__(self):
    """Delete the matrix and free up GPU resources
    """

    # make sure we delete the CUDAMatrix before the petsc matrix
    del self._cpp_object
    gc.collect()
