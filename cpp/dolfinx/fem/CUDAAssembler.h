// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/Form.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif


#include <map>
#include <memory>
#include <vector>
#include <utility>

struct ufc_integral;

namespace dolfinx
{

namespace CUDA
{
class Context;
class Module;
}

namespace function
{
class FunctionSpace;
}

namespace mesh
{
class CUDAMesh;
};

namespace la
{
class CUDAMatrix;
class CUDAVector;
};

namespace fem
{
class CUDADirichletBC;
class CUDADofMap;
class CUDAFormIntegral;
class CUDAFormConstants;
class CUDAFormCoefficients;
class DirichletBC;

#if defined(HAS_CUDA_TOOLKIT)
/// Interface for GPU-accelerated assembly of variational forms.
class CUDAAssembler
{
public:
  /// Create CUDA-based assembler
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] debug Whether or not to compile CUDA C++ code with
  ///                  debug information
  /// @param[in] cudasrcdir Path for outputting CUDA C++ code
  CUDAAssembler(
    const CUDA::Context& cuda_context,
    CUjit_target target,
    bool debug,
    const char* cudasrcdir,
    bool verbose);

  /// Destructor
  ~CUDAAssembler() = default;

  /// Copy constructor
  /// @param[in] assembler The object to be copied
  CUDAAssembler(const CUDAAssembler& assembler) = delete;

  /// Move constructor
  /// @param[in] assembler The object to be moved
  CUDAAssembler(CUDAAssembler&& assembler) = delete;

  /// Assignment operator
  /// @param[in] assembler Another CUDAAssembler object
  CUDAAssembler& operator=(const CUDAAssembler& assembler) = delete;

  /// Move assignment operator
  /// @param[in] assembler Another CUDAAssembler object
  CUDAAssembler& operator=(CUDAAssembler&& assembler) = delete;

  /// Set the entries of a device-side CSR matrix to zero
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] A The device-side CSR matrix
  void zero_matrix_entries(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAMatrix& A) const;

  /// Set the entries of a device-side vector to zero
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] x The device-side vector
  void zero_vector_entries(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAVector& x) const;

  /// Pack coefficient values for a form.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] coefficients Device-side data for form coefficients
  void pack_coefficients(
    const CUDA::Context& cuda_context,
    dolfinx::fem::CUDAFormCoefficients& coefficients,
    bool verbose) const;

  /// Assemble linear form into a vector. The vector must already be
  /// initialised. Does not zero vector and ghost contributions are
  /// not accumulated (sent to owner). Caller is responsible for
  /// calling VecGhostUpdateBegin/End.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap Device-side data for degrees of freedom
  /// @param[in] bc Device-side data for Dirichlet boundary conditions
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in,out] b The device-side vector to assemble the form into
  void assemble_vector(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap,
    const dolfinx::fem::CUDADirichletBC& bc,
    const std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAVector& b,
    bool verbose) const;

  /// Apply Dirichlet boundary conditions to a vector.
  ///
  /// For points where the given boundary conditions apply, the value
  /// is set to
  ///
  ///   b <- b - scale * (g - x0),
  ///
  /// where g denotes the boundary condition values. The boundary
  /// conditions bcs are defined on the test space or a subspace of
  /// the test space of the given linear form that was used to
  /// assemble the vector.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] bcs Device-side data for Dirichlet boundary conditions
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
  void set_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADirichletBC& bcs,
    const dolfinx::la::CUDAVector& x0,
    double scale,
    dolfinx::la::CUDAVector& b) const;

  /// Modify a right-hand side vector `b` to account for essential
  /// boundary conditions,
  ///
  ///   b <- b - scale * A (g - x0),
  ///
  /// where `g` denotes the boundary condition values. The boundary
  /// conditions `bcs1` are defined on the trial space of the given
  /// bilinear form. The test space of the bilinear form must be the
  /// same as the test space of the linear form from which `b` is
  /// assembled, but the trial spaces may differ.
  ///
  /// Ghost contributions are not accumulated (not sent to owner). The
  /// caller is responsible for calling `VecGhostUpdateBegin/End()`.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form. Note
  ///                           that this refers to the form that was
  ///                           used to assemble the coefficient matrix.
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] bcs1 Device-side data for Dirichlet boundary
  ///                 conditions on the trial space
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
  void lift_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    const dolfinx::fem::CUDADirichletBC& bcs1,
    const dolfinx::la::CUDAVector& x0,
    double scale,
    dolfinx::la::CUDAVector& b,
    bool verbose) const;

  /// Modify a right-hand side vector `b` to account for essential
  /// boundary conditions,
  ///
  ///   b <- b - scale * A_j (g_j - x0_j),
  ///
  /// where `g_j` denotes the boundary condition values and `j` is
  /// block (nest) index, or `j=0` if the problem is not blocked. The
  /// boundary conditions `bcs1` are defined on the trial spaces `V_j`
  /// of the `j`-th bilinear form. The test spaces of the bilinear
  /// forms must be the same as the test space of the linear form from
  /// which `b` is assembled, but the trial spaces may differ.
  ///
  /// Ghost contributions are not accumulated (not sent to owner). The
  /// caller is responsible for calling `VecGhostUpdateBegin/End()`.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form. Note
  ///                           that this refers to the form that was
  ///                           used to assemble the coefficient matrix.
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] bcs1 Device-side data for Dirichlet boundary
  ///                 conditions on the trial spaces
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
  void apply_lifting(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const std::vector<const dolfinx::fem::CUDADofMap*>& dofmap1,
    const std::vector<const std::map<IntegralType, std::vector<CUDAFormIntegral>>*>& form_integrals,
    const std::vector<const dolfinx::fem::CUDAFormConstants*>& constants,
    const std::vector<const dolfinx::fem::CUDAFormCoefficients*>& coefficients,
    const std::vector<const dolfinx::fem::CUDADirichletBC*>& bcs1,
    const std::vector<const dolfinx::la::CUDAVector*>& x0,
    double scale,
    dolfinx::la::CUDAVector& b,
    bool verbose) const;

  /// Assemble bilinear form into a matrix. Matrix must already be
  /// initialised. Does not zero or finalise the matrix.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] bc0 Device-side data for Dirichlet boundary
  ///                conditions on the test space
  /// @param[in] bc1 Device-side data for Dirichlet boundary
  ///                conditions on the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in,out] A The device-side CSR matrix that is used to
  ///                  store the assembled form. The matrix must be
  ///                  initialised before calling this function. The
  ///                  matrix is not zeroed.
  void assemble_matrix(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals,
    const dolfinx::fem::CUDAFormConstants& constants,
    const dolfinx::fem::CUDAFormCoefficients& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose) const;

  /// Copy element matrices from CUDA device memory to host memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  void assemble_matrix_local_copy_to_host(
    const CUDA::Context& cuda_context,
    std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals) const;

  /// Perform global assembly on the host.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in,out] A The device-side CSR matrix that is used to
  ///                  store the assembled form. The matrix must be
  ///                  initialised before calling this function. The
  ///                  matrix is not zeroed.
  void assemble_matrix_local_host_global_assembly(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals,
    dolfinx::la::CUDAMatrix& A) const;

  /// Adds a value to the diagonal entries of a matrix that belong to
  /// rows where a Dirichlet boundary condition applies. This function
  /// is typically called after assembly. The assembly function does
  /// not add any contributions to rows and columns that are affected
  /// by Dirichlet boundary conditions. This function adds a value
  /// only to rows that are locally owned, and therefore does not
  /// create a need for parallel communication. For block matrices,
  /// this function should normally be called only on the diagonal
  /// blocks, that is, blocks for which the test and trial spaces are
  /// the same.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in,out] A The matrix to add diagonal values to
  /// @param[in] bc The Dirichlet boundary condtions
  /// @param[in] diagonal The value to add to the diagonal for rows with a
  ///                     boundary condition applied
  void add_diagonal(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAMatrix& A,
    const dolfinx::fem::CUDADirichletBC& bc,
    double diagonal = 1.0) const;

  /// Compute lookup tables that are used during matrix assembly for
  /// kernels that need it
  void compute_lookup_tables(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC& bc0,
    const dolfinx::fem::CUDADirichletBC& bc1,
    std::map<IntegralType, std::vector<CUDAFormIntegral>>& form_integrals,
    dolfinx::la::CUDAMatrix& A,
    bool verbose) const;

private:
  /// Module for various useful device-side functions
  CUDA::Module _util_module;
};
#endif

} // namespace fem
} // namespace dolfinx
