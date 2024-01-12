// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>
#include <dolfinx/mesh/Mesh.h>
#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace mesh {

/// A wrapper for data related to mesh entities of a given dimension,
/// stored in the device memory of a CUDA device.
template <std::floating_point T>
class CUDAMeshEntities
{
public:
  /// Create an empty set of mesh entities
  CUDAMeshEntities();

  /// Create a set of mesh entities from a mesh
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Data structures for mesh topology and geometry
  /// @param[in] dim The dimension of mesh entities
  CUDAMeshEntities(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::Mesh<T>& mesh,
    int dim);

  /// Destructor
  ~CUDAMeshEntities();

  /// Copy constructor
  /// @param[in] mesh_entities The object to be copied
  CUDAMeshEntities(const CUDAMeshEntities& mesh_entities) = delete;

  /// Move constructor
  /// @param[in] mesh_entities The object to be moved
  CUDAMeshEntities(CUDAMeshEntities&& mesh_entities);

  /// Assignment operator
  /// @param[in] mesh_entities Another CUDAMeshEntities object
  CUDAMeshEntities& operator=(const CUDAMeshEntities& mesh_entities) = delete;

  /// Move assignment operator
  /// @param[in] mesh_entities Another CUDAMeshEntities object
  CUDAMeshEntities& operator=(CUDAMeshEntities&& mesh_entities);

  /// Get the topological dimension of the mesh
  int32_t tdim() const { return _tdim; }

  /// Get the dimension of the mesh entities
  int32_t dim() const { return _dim; }

  /// Get the number of cells in the mesh
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of mesh entities
  int32_t num_mesh_entities() const { return _num_mesh_entities; }

  /// Get the number of mesh entities in a cell
  int32_t num_mesh_entities_per_cell() const {
    return _num_mesh_entities_per_cell; }

  /// Get a handle to the device-side mesh entities of each cell
  CUdeviceptr mesh_entities_per_cell() const {
    return _dmesh_entities_per_cell; }

  /// Get a handle to the device-side offsets to the mesh cells of
  /// each mesh entity
  CUdeviceptr cells_per_mesh_entity_ptr() const {
    return _dcells_per_mesh_entity_ptr; }

  /// Get a handle to the device-side mesh cells of each mesh entity
  CUdeviceptr cells_per_mesh_entity() const {
    return _dcells_per_mesh_entity; }

  /// Get a handle to the device-side mesh entity permutations
  CUdeviceptr mesh_entity_permutations() const {
    return _dmesh_entity_permutations; }

private:
  /// The topological dimension of the mesh, or the largest dimension
  /// of any of the mesh entities
  int32_t _tdim;

  /// The dimension of the mesh entities
  int32_t _dim;

  /// The number of cells in the mesh
  int32_t _num_cells;

  /// The number of mesh entities
  int32_t _num_mesh_entities;

  /// The number of mesh entities in a cell
  int32_t _num_mesh_entities_per_cell;

  /// The mesh entities of each cell
  CUdeviceptr _dmesh_entities_per_cell;

  /// Offsets to the first cell containing each mesh entity
  CUdeviceptr _dcells_per_mesh_entity_ptr;

  /// The cells containing each mesh entity
  CUdeviceptr _dcells_per_mesh_entity;

  /// Mesh entity permutations
  CUdeviceptr _dmesh_entity_permutations;
};

} // namespace mesh
} // namespace dolfinx

#endif
