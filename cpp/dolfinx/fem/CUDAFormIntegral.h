// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/FormIntegrals.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <map>
#include <string>
#include <vector>

namespace dolfinx {
namespace mesh {
class CUDAMesh;
}

namespace la {
class CUDAMatrix;
class CUDAVector;
}

namespace fem {
class Form;
class CUDADofMap;
class CUDADirichletBC;
class CUDAFormConstants;
class CUDAFormCoefficients;

/**
 * `assembly_kernel_type` is used to enumerate different kinds of
 * assembly kernels that may be used to perform assembly for a form
 * integral.
 */
enum assembly_kernel_type
{
    /**
     * An assembly kernel that computes element matrices, sometimes
     * referred to as local assembly, on a CUDA device, whereas the
     * global assembly, that is, scattering element matrices to a
     * global matrix, is performed on the host using PETSc's
     * MatSetValues().
     */
    ASSEMBLY_KERNEL_LOCAL = 0,

    /**
     * An assembly kernel that performs global assembly of a CSR
     * matrix using a binary search within a row to locate a non-zero
     * in the sparse matrix of the bilinear form.
     */
    ASSEMBLY_KERNEL_GLOBAL,

    /**
     * An assembly kernel that uses a lookup table to locate non-zeros
     * in the sparse matrix corresponding to a bilinear form.
     */
    ASSEMBLY_KERNEL_LOOKUP_TABLE,

    /**
     * An assembly kernel that assembles a sparse matrix row-by-row.
     */
    ASSEMBLY_KERNEL_ROWWISE,

    /**
     * A final entry, whose value is equal to the number of enum
     * values.
     */
    NUM_ASSEMBLY_KERNEL_TYPES,
};

#if defined(HAS_CUDA_TOOLKIT)
/// A wrapper for a form integral with a CUDA-based assembly kernel
/// and data that is stored in the device memory of a CUDA device.
class CUDAFormIntegral
{
public:
  /// Create an empty form_integral
  CUDAFormIntegral();

  /// Create a form_integral
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form_integrals A collection of form integrals
  /// @param[in] integral_type The type of integral
  /// @param[in] i The number of the integral among the integrals of
  ///              the given type belonging to the collection
  /// @param[in] assembly_kernel_type The type of assembly kernel to use
  /// @param[in] cudasrcdir Path for outputting CUDA C++ code
  CUDAFormIntegral(
      const CUDA::Context& cuda_context,
      CUjit_target target,
      const Form& form,
      FormIntegrals::Type integral_type, int i,
      int32_t max_threads_per_block,
      int32_t min_blocks_per_multiprocessor,
      int32_t num_vertices_per_cell,
      int32_t num_coordinates_per_vertex,
      int32_t num_dofs_per_cell0,
      int32_t num_dofs_per_cell1,
      enum assembly_kernel_type assembly_kernel_type,
      bool debug,
      const char* cudasrcdir,
      bool verbose);

  /// Destructor
  ~CUDAFormIntegral();

  /// Copy constructor
  /// @param[in] form_integral The object to be copied
  CUDAFormIntegral(const CUDAFormIntegral& form_integral) = delete;

  /// Move constructor
  /// @param[in] form_integral The object to be moved
  CUDAFormIntegral(CUDAFormIntegral&& form_integral);

  /// Assignment operator
  /// @param[in] form_integral Another CUDAFormIntegral object
  CUDAFormIntegral& operator=(const CUDAFormIntegral& form_integral) = delete;

  /// Move assignment operator
  /// @param[in] form_integral Another CUDAFormIntegral object
  CUDAFormIntegral& operator=(CUDAFormIntegral&& form_integral);

  /// Get the type of integral
  FormIntegrals::Type integral_type() const { return _integral_type; }

  /// Get the identifier of the integral
  int id() const { return _id; }

  /// Get the number of mesh entities that the integral applies to
  int32_t num_mesh_entities() const { return _num_mesh_entities; }

  /// Get the mesh entities that the integral applies to
  CUdeviceptr mesh_entities() const { return _dmesh_entities; }

  /// Get the number of mesh ghost entities that the integral applies to
  int32_t num_mesh_ghost_entities() const { return _num_mesh_ghost_entities; }

  /// Get the mesh ghost entities that the integral applies to
  CUdeviceptr mesh_ghost_entities() const { return _dmesh_ghost_entities; }

  /// Get the type of assembly kernel
  enum assembly_kernel_type assembly_kernel_type() const {
      return _assembly_kernel_type; }

  /// Get a handle to the assembly kernel
  CUfunction assembly_kernel() const { return _assembly_kernel; }

  /// Assemble a vector from the form integral
  void assemble_vector(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap,
    const dolfinx::fem::CUDADirichletBC& bc,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAVector& cuda_vector,
    bool verbose) const;

  /// Assemble a matrix from the form integral
  void lift_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    double scale,
    const dolfinx::la::CUDAVector& x0,
    dolfinx::la::CUDAVector& b,
    bool verbose) const;

  /// Assemble a matrix from the form integral performing local
  /// assembly on a CUDA device and global assembly on the host
  void assemble_matrix_local(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

  /// Copy element matrices from a CUDA device and perform global
  /// assembly on the host
  void assemble_matrix_local_copy_to_host(
      const CUDA::Context& cuda_context);

  /// Perform global assembly on the host
  void assemble_matrix_local_host_global_assembly(
      const CUDA::Context& cuda_context,
      const dolfinx::fem::CUDADofMap& dofmap0,
      const dolfinx::fem::CUDADofMap& dofmap1,
      dolfinx::la::CUDAMatrix& A);

  /// Assemble a matrix from the form integral
  void assemble_matrix(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

  /// Compute the lookup table that is used during matrix assembly for
  /// kernels that need it
  void compute_lookup_table(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

private:
  void assemble_matrix_global(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

  void assemble_matrix_lookup_table(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

  void assemble_matrix_rowwise(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose);

private:
  /// Type of the integral
  FormIntegrals::Type _integral_type;

  /// Identifier for the integral
  int _id;

  /// A name for the integral assigned to it by UFC
  std::string _name;

  /// Path to use for outputting CUDA C++ code
  const char* _cudasrcdir;

  /// The number of vertices in a mesh cell
  int32_t _num_vertices_per_cell;

  /// The number of coordinates for each vertex
  int32_t _num_coordinates_per_vertex;

  /// The number of degrees of freedom per cell associated with the
  /// first function space
  int32_t _num_dofs_per_cell0;

  /// The number of degrees of freedom per cell associated with the
  /// second function space
  int32_t _num_dofs_per_cell1;

  /// The number of mesh entities that the integral applies to
  int32_t _num_mesh_entities;

  /// Host-side storage for mesh entities that the integral applies to
  const std::vector<std::int32_t>* _mesh_entities;

  /// Device-side storage for mesh entities that the integral applies to
  CUdeviceptr _dmesh_entities;

  /// The number of mesh ghost entities that the integral applies to
  int32_t _num_mesh_ghost_entities;

  /// Host-side storage for mesh ghost entities that the integral applies to
  const std::vector<std::int32_t>* _mesh_ghost_entities;

  /// Device-side storage for mesh ghost entities that the integral applies to
  CUdeviceptr _dmesh_ghost_entities;

  /// Host-side storage for element vector or matrix values, which is
  /// used for kernels that only perform local assembly on a CUDA
  /// device, but perform global assembly on the host.
  std::vector<PetscScalar> _element_values;

  /// Device-side storage for element vector or matrix values, which
  /// is used for kernels that only perform local assembly on a CUDA
  /// device, but perform global assembly on the host.
  CUdeviceptr _delement_values;

  /// The number of entries in the lookup table
  int64_t _num_nonzero_locations;

  /// A lookup table that is used to find the locations of non-zeros
  /// in the sparse matrix for the degrees of freedom of each mesh
  /// entity.
  CUdeviceptr _dnonzero_locations;

  /// Another lookup table that is used during rowwise assembly which
  /// maps each mesh cell containing a degree of freedom of the test
  /// space to the corresponding row of the computed element matrix.
  CUdeviceptr _delement_matrix_rows;

  /// CUDA module contaitning compiled and loaded device code for
  /// assembly kernels based on the form integral
  CUDA::Module _assembly_module;

  /// The type of assembly kernel to use
  enum assembly_kernel_type _assembly_kernel_type;

  /// CUDA kernel for assembly based on the form integral
  CUfunction _assembly_kernel;

  /// CUDA kernel for computing a lookup table for assembling the form integral
  CUfunction _compute_lookup_table_kernel;

  /// CUDA kernel for imposing essential boundary conditions
  CUfunction _lift_bc_kernel;
};

/// Compile assembly kernels for a collection of form integrals and
/// copy the necessary data to a CUDA device
///
/// @param[in] cuda_context A context for a CUDA device
/// @param[in] form A variational form
/// @param[in] assembly_kernel_type The type of assembly kernel to use
/// @param[in] cudasrcdir Path for outputting CUDA C++ code
std::map<FormIntegrals::Type, std::vector<CUDAFormIntegral>>
  cuda_form_integrals(
    const CUDA::Context& cuda_context,
    CUjit_target target,
    const Form& form,
    enum assembly_kernel_type assembly_kernel_type,
    int32_t max_threads_per_block,
    int32_t min_blocks_per_multiprocessor,
    bool debug,
    const char* cudasrcdir,
    bool verbose);

/// CUDA C++ code for a CUDA kernel that performs a binary search
std::string cuda_kernel_binary_search(void);

#endif

} // namespace fem
} // namespace dolfinx
