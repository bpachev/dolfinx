// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAMesh.h"
#include <dolfinx/common/CUDA.h>
#include <dolfinx/mesh/CUDAMeshEntities.h>
#include <dolfinx/mesh/Mesh.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::mesh;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
template <class T>
CUDAMesh<T>::CUDAMesh()
  : _tdim()
  , _num_vertices()
  , _num_coordinates_per_vertex()
  , _dvertex_coordinates(0)
  , _num_cells()
  , _num_vertices_per_cell()
  , _dvertex_indices_per_cell(0)
  , _dcell_permutations(0)
  , _mesh_entities()
{
}
//-----------------------------------------------------------------------------
template <class T>
CUDAMesh<T>::CUDAMesh(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::Mesh<T>& mesh)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  _tdim = mesh.topology().dim();

  // Allocate device-side storage for vertex coordinates
  auto vertex_coordinates = mesh.geometry().x();
  _num_vertices = vertex_coordinates.length() / 3;
  _num_coordinates_per_vertex = mesh.geometry().dim();
  if (_num_vertices > 0 && _num_coordinates_per_vertex > 0) {
    if (_num_coordinates_per_vertex > 3) {
      throw std::runtime_error(
        "Expected at most 3 coordinates per vertex "
        "instead of " + std::to_string(_num_coordinates_per_vertex) + " "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    size_t dvertex_coordinates_size =
      _num_vertices * 3 * sizeof(double);
    cuda_err = cuMemAlloc(
      &_dvertex_coordinates,
      dvertex_coordinates_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy vertex coordinates to device
    cuda_err = cuMemcpyHtoD(
      _dvertex_coordinates,
      vertex_coordinates.data(),
      dvertex_coordinates_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dvertex_coordinates);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Obtain mesh geometry
  auto x_dofmap =
    mesh.geometry().dofmap();

  // Allocate device-side storage for cell vertex indices
  _num_cells = x_dofmap.extent(0);
  _num_vertices_per_cell = x_dofmap.extent(1);
  if (_num_cells > 0 && _num_vertices_per_cell > 0) {
    size_t dvertex_indices_per_cell_size =
      _num_cells * _num_vertices_per_cell * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dvertex_indices_per_cell,
      dvertex_indices_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dvertex_coordinates);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell vertex indices to device
    cuda_err = cuMemcpyHtoD(
      _dvertex_indices_per_cell,
      x_dofmap.data_handle(),
      dvertex_indices_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dvertex_indices_per_cell);
      cuMemFree(_dvertex_coordinates);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Obtain cell permutations
  mesh.topology_mutable().create_entity_permutations();
  auto cell_permutations = mesh.topology().get_cell_permutation_info();

  // Allocate device-side storage for cell permutations
  if (_num_cells > 0) {
    size_t dcell_permutations_size =
      _num_cells * sizeof(uint32_t);
    cuda_err = cuMemAlloc(
      &_dcell_permutations,
      dcell_permutations_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dvertex_indices_per_cell);
      cuMemFree(_dvertex_coordinates);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell permutations to device
    cuda_err = cuMemcpyHtoD(
      _dcell_permutations,
      cell_permutations.data(),
      dcell_permutations_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dcell_permutations);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  for (int dim = 0; dim < _tdim; dim++) {
    _mesh_entities.emplace_back(
      cuda_context, mesh, dim);
  }
}
//-----------------------------------------------------------------------------
template <class T>
CUDAMesh<T>::~CUDAMesh()
{
  if (_dcell_permutations)
    cuMemFree(_dcell_permutations);
  if (_dvertex_indices_per_cell)
    cuMemFree(_dvertex_indices_per_cell);
  if (_dvertex_coordinates)
    cuMemFree(_dvertex_coordinates);
}
//-----------------------------------------------------------------------------
template <class T>
CUDAMesh<T>::CUDAMesh(CUDAMesh<T>&& mesh)
  : _tdim(mesh._tdim)
  , _num_vertices(mesh._num_vertices)
  , _num_coordinates_per_vertex(mesh._num_coordinates_per_vertex)
  , _dvertex_coordinates(mesh._dvertex_coordinates)
  , _num_cells(mesh._num_cells)
  , _num_vertices_per_cell(mesh._num_vertices_per_cell)
  , _dvertex_indices_per_cell(mesh._dvertex_indices_per_cell)
  , _dcell_permutations(mesh._dcell_permutations)
  , _mesh_entities(std::move(mesh._mesh_entities))
{
  mesh._tdim = 0;
  mesh._num_vertices = 0;
  mesh._num_coordinates_per_vertex = 0;
  mesh._dvertex_coordinates = 0;
  mesh._num_cells = 0;
  mesh._num_vertices_per_cell = 0;
  mesh._dvertex_indices_per_cell = 0;
  mesh._dcell_permutations = 0;
}
//-----------------------------------------------------------------------------
template <class T>
CUDAMesh<T>& CUDAMesh<T>::operator=(CUDAMesh<T>&& mesh)
{
  _tdim = mesh._tdim;
  _num_vertices = mesh._num_vertices;
  _num_coordinates_per_vertex = mesh._num_coordinates_per_vertex;
  _dvertex_coordinates = mesh._dvertex_coordinates;
  _num_cells = mesh._num_cells;
  _num_vertices_per_cell = mesh._num_vertices_per_cell;
  _dvertex_indices_per_cell = mesh._dvertex_indices_per_cell;
  _dcell_permutations = mesh._dcell_permutations;
  _mesh_entities = std::move(mesh._mesh_entities);
  mesh._tdim = 0;
  mesh._num_vertices = 0;
  mesh._num_coordinates_per_vertex = 0;
  mesh._dvertex_coordinates = 0;
  mesh._num_cells = 0;
  mesh._num_vertices_per_cell = 0;
  mesh._dvertex_indices_per_cell = 0;
  mesh._dcell_permutations = 0;
  return *this;
}
//-----------------------------------------------------------------------------
#endif
