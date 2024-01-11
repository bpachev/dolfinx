// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDADirichletBC.h"
#include <dolfinx/common/CUDA.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
template <class T, class U>
CUDADirichletBC<T, U>::CUDADirichletBC()
  : _num_dofs()
  , _num_owned_boundary_dofs()
  , _num_boundary_dofs()
  , _ddof_markers(0)
  , _ddof_indices(0)
  , _ddof_value_indices(0)
  , _ddof_values(0)
{
}
//-----------------------------------------------------------------------------
template <class T, class U>
CUDADirichletBC<T, U>::CUDADirichletBC(
  const CUDA::Context& cuda_context,
  const dolfinx::fem::FunctionSpace<T>& V,
  const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs)
  : _num_dofs()
  , _num_owned_boundary_dofs()
  , _num_boundary_dofs()
  , _ddof_markers(0)
  , _ddof_indices(0)
  , _ddof_value_indices(0)
  , _ddof_values(0)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Count the number of degrees of freedom
  const dolfinx::fem::DofMap& dofmap = *(V.dofmap());
  const common::IndexMap& index_map = *dofmap.index_map;
  // Looks like index_map no longer has block_size
  const int block_size = dofmap.index_map_bs();
  _num_dofs = block_size * (
		  index_map.size_local() + index_map.num_ghosts());

  // Count the number of degrees of freedom affected by boundary
  // conditions
  _num_owned_boundary_dofs = 0;
  _num_boundary_dofs = 0;
  for (auto const& bc : bcs) {
    if (V.contains(*bc->function_space())) {
      _num_owned_boundary_dofs += bc->dofs_owned().rows();
      _num_boundary_dofs += bc->dofs().rows();
    }
  }

  // Build dof markers, indices and values
  char* dof_markers = nullptr;
  std::vector<std::int32_t> dof_indices(_num_boundary_dofs);
  std::vector<std::int32_t> dof_value_indices(_num_boundary_dofs);
  std::vector<T> dof_values;
  _num_owned_boundary_dofs = 0;
  _num_boundary_dofs = 0;
  for (auto const& bc : bcs) {
    if (V.contains(*bc->function_space())) {
      if (!dof_markers) {
        dof_markers = new char[_num_dofs];
        for (int i = 0; i < _num_dofs; i++)
          dof_markers[i] = 0;
        dof_values.assign(_num_dofs, 0.0);
      }
      bc->mark_dofs(_num_dofs, dof_markers);
      auto dofs = bc->dofs();
      for (std::int32_t i = 0; i < dofs.rows(); i++) {
        dof_indices[_num_boundary_dofs + i] = dofs(i,0);
        dof_value_indices[_num_boundary_dofs + i] = dofs(i,1);
      }
      _num_owned_boundary_dofs += bc->dofs_owned().rows();
      _num_boundary_dofs += dofs.rows();
      bc->dof_values(dof_values);
    }
  }

  // Allocate device-side storage for dof markers
  if (dof_markers && _num_dofs > 0) {
    size_t ddof_markers_size = _num_dofs * sizeof(char);
    cuda_err = cuMemAlloc(&_ddof_markers, ddof_markers_size);
    if (cuda_err != CUDA_SUCCESS) {
      delete[] dof_markers;
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy dof markers to device
    cuda_err = cuMemcpyHtoD(
      _ddof_markers, dof_markers, ddof_markers_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddof_markers);
      delete[] dof_markers;
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  if (dof_markers)
    delete[] dof_markers;

  // Allocate device-side storage for dof indices
  if (_num_boundary_dofs > 0) {
    size_t ddof_indices_size = _num_boundary_dofs * sizeof(std::int32_t);
    cuda_err = cuMemAlloc(&_ddof_indices, ddof_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy dof indices to device
    cuda_err = cuMemcpyHtoD(
      _ddof_indices, dof_indices.data(), ddof_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddof_indices);
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for dof indices for the boundary values
  if (_num_boundary_dofs > 0) {
    size_t ddof_value_indices_size = _num_boundary_dofs * sizeof(std::int32_t);
    cuda_err = cuMemAlloc(&_ddof_value_indices, ddof_value_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      if (_ddof_indices)
        cuMemFree(_ddof_indices);
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy dof indices to device
    cuda_err = cuMemcpyHtoD(
      _ddof_value_indices, dof_value_indices.data(), ddof_value_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddof_value_indices);
      if (_ddof_indices)
        cuMemFree(_ddof_indices);
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for dof values
  if (dof_markers && _num_dofs > 0) {
    size_t ddof_values_size = _num_dofs * sizeof(T);
    cuda_err = cuMemAlloc(&_ddof_values, ddof_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      if (_ddof_value_indices)
        cuMemFree(_ddof_value_indices);
      if (_ddof_indices)
        cuMemFree(_ddof_indices);
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy dof values to device
    cuda_err = cuMemcpyHtoD(
      _ddof_values, dof_values.data(), ddof_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddof_values);
      if (_ddof_value_indices)
        cuMemFree(_ddof_value_indices);
      if (_ddof_indices)
        cuMemFree(_ddof_indices);
      if (_ddof_markers)
        cuMemFree(_ddof_markers);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
}
//-----------------------------------------------------------------------------
template <class T, class U>
CUDADirichletBC<T, U>::~CUDADirichletBC()
{
  if (_ddof_values)
    cuMemFree(_ddof_values);
  if (_ddof_value_indices)
    cuMemFree(_ddof_value_indices);
  if (_ddof_indices)
    cuMemFree(_ddof_indices);
  if (_ddof_markers)
    cuMemFree(_ddof_markers);
}
//-----------------------------------------------------------------------------
template <class T, class U>
CUDADirichletBC<T, U>::CUDADirichletBC(CUDADirichletBC<T,U>&& bc)
  : _num_dofs(bc._num_dofs)
  , _num_owned_boundary_dofs(bc._num_owned_boundary_dofs)
  , _num_boundary_dofs(bc._num_boundary_dofs)
  , _ddof_markers(bc._ddof_markers)
  , _ddof_indices(bc._ddof_indices)
  , _ddof_value_indices(bc._ddof_value_indices)
  , _ddof_values(bc._ddof_values)
{
  bc._num_dofs = 0;
  bc._num_owned_boundary_dofs = 0;
  bc._num_boundary_dofs = 0;
  bc._ddof_markers = 0;
  bc._ddof_indices = 0;
  bc._ddof_value_indices = 0;
  bc._ddof_values = 0;
}
//-----------------------------------------------------------------------------
template <class T, class U>
CUDADirichletBC<T, U>& CUDADirichletBC<T, U>::operator=(CUDADirichletBC<T, U>&& bc)
{
  _num_dofs = bc._num_dofs;
  _num_owned_boundary_dofs = bc._num_owned_boundary_dofs;
  _num_boundary_dofs = bc._num_boundary_dofs;
  _ddof_markers = bc._ddof_markers;
  _ddof_indices = bc._ddof_indices;
  _ddof_value_indices = bc._ddof_value_indices;
  _ddof_values = bc._ddof_values;
  bc._num_dofs = 0;
  bc._num_owned_boundary_dofs = 0;
  bc._num_boundary_dofs = 0;
  bc._ddof_markers = 0;
  bc._ddof_indices = 0;
  bc._ddof_value_indices = 0;
  bc._ddof_values = 0;
  return *this;
}
//-----------------------------------------------------------------------------
#endif
