// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDADofMap.h"
#include <dolfinx/common/CUDA.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap()
  : _dofmap(nullptr)
  , _num_dofs()
  , _num_cells()
  , _num_dofs_per_cell()
  , _ddofs_per_cell(0)
  , _dcells_per_dof_ptr(0)
  , _dcells_per_dof(0)
{
}
//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap(
  const CUDA::Context& cuda_context,
  const dolfinx::fem::DofMap& dofmap)
  : _dofmap(&dofmap)
  , _num_dofs()
  , _num_cells()
  , _num_dofs_per_cell()
  , _ddofs_per_cell(0)
  , _dcells_per_dof_ptr(0)
  , _dcells_per_dof(0)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Obtain the cellwise-to-global mapping of the degrees of freedom
  const common::IndexMap& index_map = *dofmap.index_map;
  _num_dofs = index_map.block_size() * (
    index_map.size_local() + index_map.num_ghosts());
  const graph::AdjacencyList<std::int32_t>& dofs = dofmap.list();
  _num_cells = dofs.num_nodes();
  _num_dofs_per_cell = dofmap.element_dof_layout->num_dofs();
  const std::int32_t* dofs_per_cell = dofs.array().data();

  // Allocate device-side storage for degrees of freedom
  if (_num_cells > 0 && _num_dofs_per_cell > 0) {
    size_t ddofs_per_cell_size =
      _num_cells * _num_dofs_per_cell * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_ddofs_per_cell,
      ddofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _ddofs_per_cell, dofs_per_cell, ddofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddofs_per_cell);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Compute mapping from degrees of freedom to cells
  std::vector<int32_t> cells_per_dof_ptr(_num_dofs+1);

  // Count the number cells containing each degree of freedom
  for (int32_t i = 0; i < _num_cells; i++) {
    auto cell_dofs = dofs.links(i);
    for (int32_t l = 0; l < cell_dofs.size(); l++) {
      int32_t j = cell_dofs(l);
      cells_per_dof_ptr[j+1]++;
    }
  }

  // Compute offset to the first cell for each degree of freedom
  for (int32_t i = 0; i < _num_dofs; i++)
    cells_per_dof_ptr[i+1] += cells_per_dof_ptr[i];
  int32_t num_dof_cells = cells_per_dof_ptr[_num_dofs];
  if (num_dof_cells != _num_cells * _num_dofs_per_cell) {
    cuMemFree(_ddofs_per_cell);
    throw std::logic_error(
      "Expected " + std::to_string(_num_cells) + " cells, " +
      std::to_string(_num_dofs_per_cell) + " degrees of freedom per cell, "
      "but the mapping from degrees of freedom to cells contains " +
      std::to_string(num_dof_cells) + " values" );
  }

  // Allocate storage for and compute the cells containing each degree
  // of freedom
  std::vector<int32_t> cells_per_dof(num_dof_cells);
  for (int32_t i = 0; i < _num_cells; i++) {
    auto cell_dofs = dofs.links(i);
    for (int32_t l = 0; l < cell_dofs.size(); l++) {
      int32_t j = cell_dofs(l);
      int32_t p = cells_per_dof_ptr[j];
      cells_per_dof[p] = i;
      cells_per_dof_ptr[j]++;
    }
  }

  // Adjust offsets to first cell
  for (int32_t i = _num_dofs; i > 0; i--)
    cells_per_dof_ptr[i] = cells_per_dof_ptr[i-1];
  cells_per_dof_ptr[0] = 0;

  // Allocate device-side storage for offsets to the first cell
  // containing each degree of freedom
  if (_num_dofs > 0) {
    size_t dcells_per_dof_ptr_size = (_num_dofs+1) * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dcells_per_dof_ptr, dcells_per_dof_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _dcells_per_dof_ptr, cells_per_dof_ptr.data(), dcells_per_dof_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for cells containing each degree of freedom
  if (_num_cells > 0 && _num_dofs_per_cell > 0) {
    size_t dcells_per_dof_size = num_dof_cells * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dcells_per_dof,
      dcells_per_dof_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _dcells_per_dof, cells_per_dof.data(), dcells_per_dof_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
}
//-----------------------------------------------------------------------------
CUDADofMap::~CUDADofMap()
{
  if (_dcells_per_dof)
    cuMemFree(_dcells_per_dof);
  if (_dcells_per_dof_ptr)
    cuMemFree(_dcells_per_dof_ptr);
  if (_ddofs_per_cell)
    cuMemFree(_ddofs_per_cell);
}
//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap(CUDADofMap&& dofmap)
  : _dofmap(dofmap._dofmap)
  , _num_dofs(dofmap._num_dofs)
  , _num_cells(dofmap._num_cells)
  , _num_dofs_per_cell(dofmap._num_dofs_per_cell)
  , _ddofs_per_cell(dofmap._ddofs_per_cell)
  , _dcells_per_dof_ptr(dofmap._dcells_per_dof_ptr)
  , _dcells_per_dof(dofmap._dcells_per_dof)
{
  dofmap._dofmap = nullptr;
  dofmap._num_dofs = 0;
  dofmap._num_cells = 0;
  dofmap._num_dofs_per_cell = 0;
  dofmap._ddofs_per_cell = 0;
  dofmap._dcells_per_dof_ptr = 0;
  dofmap._dcells_per_dof = 0;
}
//-----------------------------------------------------------------------------
CUDADofMap& CUDADofMap::operator=(CUDADofMap&& dofmap)
{
  _dofmap = dofmap._dofmap;
  _num_dofs = dofmap._num_dofs;
  _num_cells = dofmap._num_cells;
  _num_dofs_per_cell = dofmap._num_dofs_per_cell;
  _ddofs_per_cell = dofmap._ddofs_per_cell;
  _dcells_per_dof_ptr = dofmap._dcells_per_dof_ptr;
  _dcells_per_dof = dofmap._dcells_per_dof;
  dofmap._dofmap = nullptr;
  dofmap._num_dofs = 0;
  dofmap._num_cells = 0;
  dofmap._num_dofs_per_cell = 0;
  dofmap._ddofs_per_cell = 0;
  dofmap._dcells_per_dof_ptr = 0;
  dofmap._dcells_per_dof = 0;
  return *this;
}
//-----------------------------------------------------------------------------
#endif
