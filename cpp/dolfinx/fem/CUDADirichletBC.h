// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/FunctionSpace.h>
#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <memory>
#include <vector>

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {

//namespace function {
//class FunctionSpace;
//}

namespace fem {
//class DirichletBC;

/// A wrapper for data marking which degrees of freedom that are
/// affected by Dirichlet boundary conditions, with data being stored
/// in the device memory of a CUDA device.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDADirichletBC
{
public:
  /// Create empty Dirichlet boundary conditions
  CUDADirichletBC();

  /// Create Dirichlet boundary conditions
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] V The function space to build dof markers for.
  ///              Boundary conditions are only applied for degrees of
  ///              freedom that belong to the given function space.
  /// @param[in] bcs The boundary conditions to copy to device memory
  CUDADirichletBC(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::FunctionSpace<T>& V,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs);

  /// Destructor
  ~CUDADirichletBC();

  /// Copy constructor
  /// @param[in] bc The object to be copied
  CUDADirichletBC(const CUDADirichletBC<T,U>& bc) = delete;

  /// Move constructor
  /// @param[in] bc The object to be moved
  CUDADirichletBC(CUDADirichletBC<T,U>&& bc);

  /// Assignment operator
  /// @param[in] bc Another CUDADirichletBC object
  CUDADirichletBC& operator=(const CUDADirichletBC<T,U>& bc) = delete;

  /// Move assignment operator
  /// @param[in] bc Another CUDADirichletBC object
  CUDADirichletBC& operator=(CUDADirichletBC<T,U>&& bc);

  /// Get the number of degrees of freedom
  int32_t num_dofs() const { return _num_dofs; }

  /// Get a handle to the device-side dof markers
  CUdeviceptr dof_markers() const { return _ddof_markers; }

  /// Get the number of degrees of freedom owned by the current
  /// process that are subject to boundary conditions
  int32_t num_owned_boundary_dofs() const { return _num_owned_boundary_dofs; }

  /// Get the number of degrees of freedom subject to boundary
  /// conditions
  int32_t num_boundary_dofs() const { return _num_boundary_dofs; }

  /// Get a handle to the device-side dof indices
  CUdeviceptr dof_indices() const { return _ddof_indices; }

  /// Get a handle to the device-side dofs for the values
  CUdeviceptr dof_value_indices() const { return _ddof_indices; }

  /// Get a handle to the device-side dof values
  CUdeviceptr dof_values() const { return _ddof_values; }

private:
  /// The number of degrees of freedom
  int32_t _num_dofs;

  /// The number of degrees of freedom owned by the current process
  /// that are subject to the essential boundary conditions.
  int32_t _num_owned_boundary_dofs;

  /// The number of degrees of freedom that are subject to the
  /// essential boundary conditions, including ghost nodes.
  int32_t _num_boundary_dofs;

  /// Markers for each degree of freedom, indicating whether or not
  /// they are subject to essential boundary conditions
  CUdeviceptr _ddof_markers;

  /// Indices of the degrees of freedom that are subject to essential
  /// boundary conditions
  CUdeviceptr _ddof_indices;

  /// Indices of the degrees of freedom of the boundary value function
  CUdeviceptr _ddof_value_indices;

  /// Values for each degree of freedom, indicating whether or not
  /// they are subject to essential boundary conditions
  CUdeviceptr _ddof_values;
};

} // namespace fem
} // namespace dolfinx

#endif
