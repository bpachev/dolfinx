// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/CUDAMeshEntities.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <vector>

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace mesh {

/// A wrapper for mesh data that is stored in the device memory of a
/// CUDA device.
template <std::floating_point T>
class CUDAMesh
{
public:
  /// Create an empty mesh
  CUDAMesh();

  /// Create a mesh
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Data structures for mesh topology and geometry
  CUDAMesh(const CUDA::Context& cuda_context, const dolfinx::mesh::Mesh<T>& mesh);

  /// Destructor
  ~CUDAMesh();

  /// Copy constructor
  /// @param[in] mesh The object to be copied
  CUDAMesh(const CUDAMesh& mesh) = delete;

  /// Move constructor
  /// @param[in] mesh The object to be moved
  CUDAMesh(CUDAMesh&& mesh);

  /// Assignment operator
  /// @param[in] mesh Another CUDAMesh object
  CUDAMesh& operator=(const CUDAMesh& mesh) = delete;

  /// Move assignment operator
  /// @param[in] mesh Another CUDAMesh object
  CUDAMesh& operator=(CUDAMesh&& mesh);

  /// Get the topological dimension of the mesh
  int32_t tdim() const { return _tdim; }

  /// Get the number of vertices
  int32_t num_vertices() const { return _num_vertices; }

  /// Get the number of coordinates per vertex
  int32_t num_coordinates_per_vertex() const {
    return _num_coordinates_per_vertex; }

  /// Get a handle to the device-side vertex coordinates
  CUdeviceptr vertex_coordinates() const {
    return _dvertex_coordinates; }

  /// Get the number of cells
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of vertices per cell
  int32_t num_vertices_per_cell() const {
    return _num_vertices_per_cell; }

  /// Get a handle to the device-side cell vertex indices
  CUdeviceptr vertex_indices_per_cell() const {
    return _dvertex_indices_per_cell; }

  /// Get a handle to the device-side cell permutations
  CUdeviceptr cell_permutations() const {
    return _dcell_permutations; }

  /// Get the mesh entities of each dimension
  const std::vector<CUDAMeshEntities<T>>& mesh_entities() const {
    return _mesh_entities; }

private:
  /// The topological dimension of the mesh, or the largest dimension
  /// of any of the mesh entities
  int32_t _tdim;

  /// The number of vertices in the mesh
  int32_t _num_vertices;

  /// The number of coordinates for each vertex
  int32_t _num_coordinates_per_vertex;

  /// The coordinates of the mesh vertices
  CUdeviceptr _dvertex_coordinates;

  /// The number of cells in the mesh
  int32_t _num_cells;

  /// The number of vertices in each cell
  int32_t _num_vertices_per_cell;

  /// The vertex indices of each cell
  CUdeviceptr _dvertex_indices_per_cell;

  /// Cell permutations
  CUdeviceptr _dcell_permutations;

  /// The mesh entities of each dimension
  std::vector<CUDAMeshEntities<T>> _mesh_entities;
};

} // namespace mesh
} // namespace dolfinx

#endif
