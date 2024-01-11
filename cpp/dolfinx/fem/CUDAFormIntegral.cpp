// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAFormIntegral.h"
#include <dolfinx/common/CUDA.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CUDADirichletBC.h>
#include <dolfinx/fem/CUDADofMap.h>
#include <dolfinx/fem/CUDAFormCoefficients.h>
#include <dolfinx/fem/CUDAFormConstants.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/CUDAMatrix.h>
#include <dolfinx/la/CUDASeqMatrix.h>
#include <dolfinx/la/CUDAVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/CUDAMesh.h>
#include <dolfinx/mesh/CUDAMeshEntities.h>
#include <dolfinx/mesh/Mesh.h>
#include <ufc.h>


#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
namespace {

std::string to_string(IntegralType integral_type)
{
  switch (integral_type) {
  case IntegralType::cell: return "cell";
  case IntegralType::exterior_facet: return "exterior_facet";
  case IntegralType::interior_facet: return "interior_facet";
  default: return "unknown";
  }
}

/// CUDA C++ code for cellwise assembly of a vector from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_vector_cell(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_vertices,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_dofs_per_cell,\n"
    "  const int32_t* __restrict__ dofmap,\n"
    "  const char* __restrict__ bc,\n"
    "  int32_t num_values,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell == " + std::to_string(num_dofs_per_cell) + ");\n"
    "  ufc_scalar_t xe[" + std::to_string(num_dofs_per_cell) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      xe[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element vector\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      xe,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Add element vector values to the global vector,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs = &dofmap[c*" + std::to_string(num_dofs_per_cell) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      int32_t row = dofs[j];\n"
    "      atomicAdd(&values[row], xe[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a vector from a form
/// integral over exterior mesh facets
std::string cuda_kernel_assemble_vector_exterior_facet(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_vertices,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int32_t num_mesh_entities,\n"
    "  int32_t num_mesh_entities_per_cell,\n"
    "  const int32_t* __restrict__ mesh_entities_per_cell,\n"
    "  const int32_t* __restrict__ cells_per_mesh_entity_ptr,\n"
    "  const int32_t* __restrict__ cells_per_mesh_entity,\n"
    "  const uint8_t* __restrict__ mesh_entity_permutations,\n"
    "  int32_t num_active_mesh_entities,\n"
    "  const int32_t* __restrict__ active_mesh_entities,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_dofs_per_cell,\n"
    "  const int32_t* __restrict__ dofmap,\n"
    "  const char* __restrict__ bc,\n"
    "  int32_t num_values,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell == " + std::to_string(num_dofs_per_cell) + ");\n"
    "  ufc_scalar_t xe[" + std::to_string(num_dofs_per_cell) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_mesh_entities;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t e = active_mesh_entities[i];\n"
    "    int32_t c = cells_per_mesh_entity[\n"
    "      cells_per_mesh_entity_ptr[e]];\n"
    "\n"
    "    // Find the local index of the mesh entity with respect to the cell\n"
    "    int32_t local_mesh_entity = 0;\n"
    "    for (; local_mesh_entity < num_mesh_entities_per_cell; local_mesh_entity++) {\n"
    "      if (e == mesh_entities_per_cell[\n"
    "            c*num_mesh_entities_per_cell+local_mesh_entity])\n"
    "        break;\n"
    "    }\n"
    "\n"
    "    // Set element vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      xe[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const uint8_t* quadrature_permutation =\n"
    "      &mesh_entity_permutations[\n"
    "        local_mesh_entity*num_cells+c];\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element vector\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      xe,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      &local_mesh_entity,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Add element vector values to the global vector,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs = &dofmap[c*" + std::to_string(num_dofs_per_cell) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell) + "; j++) {\n"
    "      int32_t row = dofs[j];\n"
    "      atomicAdd(&values[row], xe[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for assembly of a vector from a form integral
std::string cuda_kernel_assemble_vector(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_assemble_vector_cell(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell);
  case IntegralType::exterior_facet:
    return cuda_kernel_assemble_vector_exterior_facet(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for modifying a right-hand side vector to impose
/// essential boundary conditions for integrals over mesh cells
std::string cuda_kernel_lift_bc_cell(
  std::string lift_bc_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "" + lift_bc_kernel_name + "(\n"
    "  int32_t num_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  int num_constant_values,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc_markers1,\n"
    "  const char* __restrict__ bc_values1,\n"
    "  double scale,\n"
    "  int32_t num_columns,\n"
    "  const ufc_scalar_t* x0,\n"
    "  ufc_scalar_t* b)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "  ufc_scalar_t be[" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  for (int c = thread_idx;\n"
    "    c < num_cells;\n"
    "    c += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    // Skip cell if boundary conditions do not apply\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    bool has_bc = false;\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        has_bc = true;\n"
    "        break;\n"
    "      }\n"
    "    }\n"
    "    if (!has_bc)\n"
    "      continue;\n"
    "\n"
    "    // Set element matrix and vector values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "      be[j] = 0.0;\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Compute modified element vector\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "      int32_t column = dofs1[k];\n"
    "      if (bc_markers1 && bc_markers1[column]) {\n"
    "        ufc_scalar_t bc = bc_values1[column];\n"
    "        for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "          be[j] -= Ae[k*" + std::to_string(num_dofs_per_cell1) + "+j] * scale * (bc - x0[column]);\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Add element vector values to the global vector\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      atomicAdd(&b[row], be[j]);\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for assembly of a matrix from a form integral
std::string cuda_kernel_lift_bc(
  std::string kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_lift_bc_cell(
      kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for cellwise, local assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_local(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    ufc_scalar_t* Ae = &values[\n"
    "      i*" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // For degrees of freedom that are subject to essential boundary conditions,\n"
    "    // set the element matrix values to zero.\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "        }\n"
    "        continue;\n"
    "      }\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int32_t column = dofs1[k];\n"
    "        if (bc1 && bc1[column]) {;\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // This kernel does not perform any global assembly. That is,\n"
    "    // only element matrices are computed here. The element matrices\n"
    "    // should be copied to the host, and the host is responsible for\n"
    "    // scattering them to the correct locations in the global matrix.\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_global(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return ""
    "extern \"C\" int printf(const char * format, ...);\n"
    "\n"
    "extern \"C\" void __global__\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_local_rows,\n"
    "  int32_t num_local_columns,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  ufc_scalar_t* __restrict__ values,\n"
    "  const int32_t* __restrict__ offdiag_row_ptr,\n"
    "  const int32_t* __restrict__ offdiag_column_indices,\n"
    "  ufc_scalar_t* __restrict__ offdiag_values,\n"
    "  int32_t num_local_offdiag_columns,\n"
    "  const int32_t* __restrict__ colmap)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) continue;\n"
    "      if (row < num_local_rows) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) continue;\n"
    "          if (column < num_local_columns) {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            assert(!err && \"Failed to find column index!\");\n"
    "            r += row_ptr[row];\n"
    "            atomicAdd(&values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          } else {\n"
    "            /* Search for the correct column index in the column map\n"
    "             * of the off-diagonal part of the local matrix. */\n"
    "            int32_t colmap_idx = -1;\n"
    "            for (int q = 0; q < num_local_offdiag_columns; q++) {\n"
    "              if (column == colmap[q]) {\n"
    "                colmap_idx = q;\n"
    "                break;\n"
    "              }\n"
    "            }\n"
    "            assert(colmap_idx != -1);\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              offdiag_row_ptr[row+1] - offdiag_row_ptr[row],\n"
    "              &offdiag_column_indices[offdiag_row_ptr[row]],\n"
    "              colmap_idx, &r);\n"
    "            assert(!err && \"Failed to find column index!\");\n"
    "            r += offdiag_row_ptr[row];\n"
    "            atomicAdd(&offdiag_values[r],\n"
    "              Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for computing a lookup table for the sparse matrix
/// non-zeros corresponding to the degrees of freedom of each mesh entity
std::string cuda_kernel_compute_lookup_table(
  std::string assembly_kernel_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    "extern \"C\" void __global__\n"
    "compute_lookup_table_" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_dofs,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_rows,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  int64_t num_nonzero_locations,\n"
    "  int32_t* __restrict__ nonzero_locations,\n"
    "  int32_t* __restrict__ element_matrix_rows)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    // Use a binary search to locate non-zeros in the sparse matrix.\n"
    "    // For degrees of freedom that are subject to essential boundary\n"
    "    // conditions, insert a negative value in the lookup table.\n"
    "    int32_t c = active_cells[i];\n"
    "    const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "    const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      int32_t row = dofs0[j];\n"
    "      if (bc0 && bc0[row]) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (((int64_t) (i / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell0) + " + (int64_t) j) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + (int64_t) k) *\n"
    "            warpSize + (i % warpSize);\n"
    "            nonzero_locations[l] = -1;\n"
    "        }\n"
    "      } else {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (((int64_t) (i / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell0) + " + (int64_t) j) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + (int64_t) k) *\n"
    "            warpSize + (i % warpSize);\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) {\n"
    "            nonzero_locations[l] = -1;\n"
    "          } else {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[row+1] - row_ptr[row],\n"
    "              &column_indices[row_ptr[row]],\n"
    "              column, &r);\n"
    "            nonzero_locations[l] = row_ptr[row] + r;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_lookup_table(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    cuda_kernel_compute_lookup_table(
      assembly_kernel_name, num_vertices_per_cell,
      num_coordinates_per_vertex, num_dofs_per_cell0, num_dofs_per_cell1) + "\n"
    "\n"
    "extern \"C\" void __global__\n"
    "__launch_bounds__(" + std::to_string(max_threads_per_block) + ", " + std::to_string(min_blocks_per_multiprocessor) + ")\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ nonzero_locations,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  for (int i = thread_idx;\n"
    "    i < num_active_cells;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = active_cells[i];\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int64_t l = (((int64_t) (i / warpSize) *\n"
    "          " + std::to_string(num_dofs_per_cell0) + "ll + (int64_t) j) *\n"
    "          " + std::to_string(num_dofs_per_cell1) + "ll + (int64_t) k) *\n"
    "          warpSize + (i % warpSize);\n"
    "        int r = nonzero_locations[l];\n"
    "        if (r < 0) continue;\n"
    "        atomicAdd(&values[r],\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for computing a lookup table for the sparse matrix
/// non-zeros corresponding to the degrees of freedom of each mesh entity
std::string cuda_kernel_compute_lookup_table_rowwise(
  std::string assembly_kernel_name,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  // Generate the CUDA C++ code for the assembly kernel
  return
    "extern \"C\" void __global__\n"
    "compute_lookup_table_" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_dofs,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ dofmap0,\n"
    "  const int32_t* __restrict__ dofmap1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const char* __restrict__ bc0,\n"
    "  const char* __restrict__ bc1,\n"
    "  int32_t num_rows,\n"
    "  const int32_t* __restrict__ row_ptr,\n"
    "  const int32_t* __restrict__ column_indices,\n"
    "  int64_t num_nonzero_locations,\n"
    "  int32_t* __restrict__ nonzero_locations,\n"
    "  int32_t* __restrict__ element_matrix_rows)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "  assert(num_dofs == num_rows);\n"
    "\n"
    "  // Iterate over the global degrees of freedom of the test space\n"
    "  for (int i = thread_idx;\n"
    "    i < num_dofs;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    if (bc0 && bc0[i]) {\n"
    "      for (int p = cells_per_dof_ptr[i]; p < cells_per_dof_ptr[i+1]; p++) {\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (int64_t) ((p / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "            p % warpSize;\n"
    "          nonzero_locations[l] = -1;\n"
    "        }\n"
    "      }\n"
    "    } else {\n"
    "      // Iterate over the mesh cells containing the current degree of freedom\n"
    "      // TODO: What if the integral is only over part of the domain?\n"
    "      // How do we iterate over the \"active\" cells?\n"
    "      for (int p = cells_per_dof_ptr[i]; p < cells_per_dof_ptr[i+1]; p++) {\n"
    "        int32_t c = cells_per_dof[p];\n"
    "\n"
    "        const int32_t* dofs0 = &dofmap0[c*" + std::to_string(num_dofs_per_cell0) + "];\n"
    "        const int32_t* dofs1 = &dofmap1[c*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "        // Find the row of the element matrix that contributes to\n"
    "        // the current global degree of freedom\n"
    "        int j = 0;\n"
    "        while (j < " + std::to_string(num_dofs_per_cell0) + " && i != dofs0[j])\n"
    "          j++;\n"
    "        assert(i == dofs0[j]);\n"
    "        element_matrix_rows[p] = j;\n"
    "\n"
    "        for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "          int64_t l = (int64_t) ((p / warpSize) *\n"
    "            " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "            p % warpSize;\n"
    "          int32_t column = dofs1[k];\n"
    "          if (bc1 && bc1[column]) {\n"
    "            nonzero_locations[l] = -1;\n"
    "          } else {\n"
    "            int r;\n"
    "            int err = binary_search(\n"
    "              row_ptr[i+1] - row_ptr[i],\n"
    "              &column_indices[row_ptr[i]],\n"
    "              column, &r);\n"
    "            assert(!err);\n"
    "            nonzero_locations[l] = row_ptr[i] + r;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for rowwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell_rowwise(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1)
{
  return
    cuda_kernel_compute_lookup_table_rowwise(
      assembly_kernel_name, num_vertices_per_cell,
      num_coordinates_per_vertex, num_dofs_per_cell0, num_dofs_per_cell1) + "\n"
    "\n"
    "extern \"C\" void __global__\n"
    "__launch_bounds__(" + std::to_string(max_threads_per_block) + ", " + std::to_string(min_blocks_per_multiprocessor) + ")\n"
    "" + assembly_kernel_name + "(\n"
    "  int32_t num_active_cells,\n"
    "  const int32_t* __restrict__ active_cells,\n"
    "  int num_vertices_per_cell,\n"
    "  const int32_t* __restrict__ vertex_indices_per_cell,\n"
    "  int num_coordinates_per_vertex,\n"
    "  const double* __restrict__ vertex_coordinates,\n"
    "  int num_coeffs_per_cell,\n"
    "  const ufc_scalar_t* __restrict__ coeffs,\n"
    "  const ufc_scalar_t* __restrict__ constant_values,\n"
    "  const uint32_t* __restrict__ cell_permutations,\n"
    "  int num_dofs_per_cell0,\n"
    "  int num_dofs_per_cell1,\n"
    "  const int32_t* __restrict__ cells_per_dof_ptr,\n"
    "  const int32_t* __restrict__ cells_per_dof,\n"
    "  const int32_t* __restrict__ nonzero_locations,\n"
    "  const int32_t* __restrict__ element_matrix_rows,\n"
    "  int32_t num_rows,\n"
    "  ufc_scalar_t* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "\n"
    "  assert(num_vertices_per_cell == " + std::to_string(num_vertices_per_cell) + ");\n"
    "  assert(num_coordinates_per_vertex == " + std::to_string(num_coordinates_per_vertex) + ");\n"
    "  double cell_vertex_coordinates[" + std::to_string(num_vertices_per_cell) + "*" + std::to_string(num_coordinates_per_vertex) + "];\n"
    "\n"
    "  assert(num_dofs_per_cell0 == " + std::to_string(num_dofs_per_cell0) + ");\n"
    "  assert(num_dofs_per_cell1 == " + std::to_string(num_dofs_per_cell1) + ");\n"
    "\n"
    "  ufc_scalar_t Ae[" + std::to_string(num_dofs_per_cell0) + "*" + std::to_string(num_dofs_per_cell1) + "];\n"
    "\n"
    "  // Iterate over the global degrees of freedom of the test space and\n"
    "  // the mesh cells containing them.\n"
    "  // TODO: What if the integral is only over part of the domain?\n"
    "  // How do we iterate over the \"active\" cells?\n"
    "  for (int p = thread_idx;\n"
    "       p < cells_per_dof_ptr[num_rows];\n"
    "       p += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int32_t c = cells_per_dof[p];\n"
    "    const ufc_scalar_t* coeff_cell = &coeffs[c*num_coeffs_per_cell];\n"
    "    int* entity_local_index = NULL;\n"
    "    uint8_t* quadrature_permutation = NULL;\n"
    "    uint32_t cell_permutation = cell_permutations[c];\n"
    "\n"
    "    // Gather cell vertex coordinates\n"
    "    for (int j = 0; j < " + std::to_string(num_vertices_per_cell) + "; j++) {\n"
    "      int vertex = vertex_indices_per_cell[\n"
    "        c*" + std::to_string(num_vertices_per_cell) + "+j];\n"
    "      for (int k = 0; k < " + std::to_string(num_coordinates_per_vertex) + "; k++) {\n"
    "        cell_vertex_coordinates[j*" + std::to_string(num_coordinates_per_vertex) + "+k] =\n"
    "          vertex_coordinates[vertex*3+k];\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Set element matrix values to zero\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k] = 0.0;\n"
    "      }\n"
    "    }\n"
    "\n"
    "    // Compute element matrix\n"
    "    " + tabulate_tensor_function_name + "(\n"
    "      Ae,\n"
    "      coeff_cell,\n"
    "      constant_values,\n"
    "      cell_vertex_coordinates,\n"
    "      entity_local_index,\n"
    "      quadrature_permutation,\n"
    "      cell_permutation);\n"
    "\n"
    "    // Add element matrix values to the global matrix,\n"
    "    // skipping entries related to degrees of freedom\n"
    "    // that are subject to essential boundary conditions.\n"
    "    for (int j = 0; j < " + std::to_string(num_dofs_per_cell0) + "; j++) {\n"
    "      if (j != element_matrix_rows[p]) continue;\n"
    "      for (int k = 0; k < " + std::to_string(num_dofs_per_cell1) + "; k++) {\n"
    "        int64_t l = (int64_t) ((p / warpSize) *\n"
    "          " + std::to_string(num_dofs_per_cell1) + " + k) * warpSize +\n"
    "          p % warpSize;\n"
    "        int r = nonzero_locations[l];\n"
    "        if (r < 0) continue;\n"
    "        atomicAdd(&values[r],\n"
    "          Ae[j*" + std::to_string(num_dofs_per_cell1) + "+k]);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}";
}

/// CUDA C++ code for cellwise assembly of a matrix from a form
/// integral over mesh cells
std::string cuda_kernel_assemble_matrix_cell(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type)
{
  switch (assembly_kernel_type) {
  case ASSEMBLY_KERNEL_LOCAL:
    return cuda_kernel_assemble_matrix_cell_local(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_GLOBAL:
    return cuda_kernel_assemble_matrix_cell_global(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_LOOKUP_TABLE:
    return cuda_kernel_assemble_matrix_cell_lookup_table(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  case ASSEMBLY_KERNEL_ROWWISE:
    return cuda_kernel_assemble_matrix_cell_rowwise(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1);
  default:
    throw std::invalid_argument(
      "Invalid argument at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

/// CUDA C++ code for assembly of a matrix from a form integral
std::string cuda_kernel_assemble_matrix(
  std::string assembly_kernel_name,
  std::string tabulate_tensor_function_name,
  IntegralType integral_type,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type)
{
  switch (integral_type) {
  case IntegralType::cell:
    return cuda_kernel_assemble_matrix_cell(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      assembly_kernel_type);
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

static const char * nvrtc_options_gpuarch(
    CUjit_target target)
{
    switch (target) {
    case CU_TARGET_COMPUTE_30: return "--gpu-architecture=compute_30";
    case CU_TARGET_COMPUTE_32: return "--gpu-architecture=compute_32";
    case CU_TARGET_COMPUTE_35: return "--gpu-architecture=compute_35";
    case CU_TARGET_COMPUTE_37: return "--gpu-architecture=compute_37";
    case CU_TARGET_COMPUTE_50: return "--gpu-architecture=compute_50";
    case CU_TARGET_COMPUTE_52: return "--gpu-architecture=compute_52";
    case CU_TARGET_COMPUTE_53: return "--gpu-architecture=compute_53";
    case CU_TARGET_COMPUTE_60: return "--gpu-architecture=compute_60";
    case CU_TARGET_COMPUTE_61: return "--gpu-architecture=compute_61";
    case CU_TARGET_COMPUTE_62: return "--gpu-architecture=compute_62";
    case CU_TARGET_COMPUTE_70: return "--gpu-architecture=compute_70";
    case CU_TARGET_COMPUTE_72: return "--gpu-architecture=compute_72";
    case CU_TARGET_COMPUTE_75: return "--gpu-architecture=compute_75";
    case CU_TARGET_COMPUTE_80: return "--gpu-architecture=compute_80";
    case CU_TARGET_COMPUTE_86: return "--gpu-architecture=compute_86";
    case CU_TARGET_COMPUTE_87: return "--gpu-architecture=compute_87";
    default: return "";
    }
}

/// Configure compiler options for CUDA C++ code
static const char** nvrtc_compiler_options(
  int* out_num_compile_options,
  CUjit_target target,
  bool debug)
{
  int num_compile_options;
  static const char* default_compile_options[] = {
    "--device-as-default-execution-space",
    nvrtc_options_gpuarch(target)};
  static const char* debug_compile_options[] = {
    "--device-as-default-execution-space",
    nvrtc_options_gpuarch(target),
    "--device-debug",
    "--generate-line-info"};

  const char** compile_options;
  if (debug) {
    compile_options = debug_compile_options;
    num_compile_options =
      sizeof(debug_compile_options) /
      sizeof(*debug_compile_options);
  } else {
    compile_options = default_compile_options;
    num_compile_options =
      sizeof(default_compile_options) /
      sizeof(*default_compile_options);
  }

  *out_num_compile_options = num_compile_options;
  return compile_options;
}

/// Compile assembly kernel for a form integral
CUDA::Module compile_form_integral_kernel(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  int form_rank,
  IntegralType integral_type,
  ufc_integral* integral,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
{
  // Obtain the automatically generated CUDA C++ code for the
  // element matrix kernel (tabulate_tensor).
  int num_program_headers;
  const char** program_headers;
  const char** program_include_names;
  const char* tabulate_tensor_src;
  const char* tabulate_tensor_function_name;
  integral->tabulate_tensor_cuda(
    &num_program_headers, &program_headers,
    &program_include_names, &tabulate_tensor_src,
    &tabulate_tensor_function_name);

  // Generate CUDA C++ code for the assembly kernel
  std::string assembly_kernel_name =
    std::string("assemble_") + std::string(integral->name);
  std::string lift_bc_kernel_name =
    std::string("lift_bc_") + std::string(integral->name);

  std::string assembly_kernel_src =
    std::string(tabulate_tensor_src) + "\n"
    "typedef int int32_t;\n"
    "typedef long long int int64_t;\n"
    "\n";

  switch (form_rank) {
  case 1:
    assembly_kernel_src += cuda_kernel_assemble_vector(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0) + "\n";
    break;
  case 2:
    assembly_kernel_src += cuda_kernel_binary_search() + "\n"
      "\n";
    assembly_kernel_src += cuda_kernel_assemble_matrix(
      assembly_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      max_threads_per_block,
      min_blocks_per_multiprocessor,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1,
      assembly_kernel_type) + "\n"
      "\n";
    assembly_kernel_src += cuda_kernel_lift_bc(
      lift_bc_kernel_name,
      tabulate_tensor_function_name,
      integral_type,
      num_vertices_per_cell,
      num_coordinates_per_vertex,
      num_dofs_per_cell0,
      num_dofs_per_cell1) + "\n";
    break;
  default:
    throw std::runtime_error(
      "Forms of rank " + std::to_string(form_rank) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Configure compiler options
  int num_compile_options;
  const char** compile_options =
    nvrtc_compiler_options(&num_compile_options, target, debug);

  // Compile CUDA C++ code to PTX assembly
  const char* program_name = integral->name;
  std::string ptx = CUDA::compile_cuda_cpp_to_ptx(
    program_name, num_program_headers, program_headers,
    program_include_names, num_compile_options, compile_options,
    assembly_kernel_src.c_str(), cudasrcdir, verbose);

  // Load the PTX assembly as a module
  int num_module_load_options = 0;
  CUjit_option * module_load_options = NULL;
  void ** module_load_option_values = NULL;
  return CUDA::Module(
    cuda_context, ptx, target,
    num_module_load_options,
    module_load_options,
    module_load_option_values,
    verbose,
    debug);
}

} // namespace

//-----------------------------------------------------------------------------
CUDAFormIntegral::CUDAFormIntegral()
  : _integral_type()
  , _id()
  , _name()
  , _cudasrcdir()
  , _num_vertices_per_cell()
  , _num_coordinates_per_vertex()
  , _num_dofs_per_cell0()
  , _num_dofs_per_cell1()
  , _num_mesh_entities()
  , _mesh_entities()
  , _dmesh_entities(0)
  , _num_mesh_ghost_entities()
  , _mesh_ghost_entities()
  , _dmesh_ghost_entities(0)
  , _element_values()
  , _delement_values(0)
  , _num_nonzero_locations()
  , _dnonzero_locations(0)
  , _delement_matrix_rows(0)
  , _assembly_module()
  , _assembly_kernel_type()
  , _assembly_kernel()
  , _compute_lookup_table_kernel()
  , _lift_bc_kernel()
{
}
//-----------------------------------------------------------------------------
CUDAFormIntegral::CUDAFormIntegral(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  const Form& form,
  IntegralType integral_type, int i,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  int32_t num_vertices_per_cell,
  int32_t num_coordinates_per_vertex,
  int32_t num_dofs_per_cell0,
  int32_t num_dofs_per_cell1,
  enum assembly_kernel_type assembly_kernel_type,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
  : _integral_type(integral_type)
  , _id(i)
  , _name(form.integrals().get_ufc_integral(_integral_type, i)->name)
  , _cudasrcdir(cudasrcdir)
  , _num_vertices_per_cell()
  , _num_coordinates_per_vertex()
  , _num_dofs_per_cell0(num_dofs_per_cell0)
  , _num_dofs_per_cell1(num_dofs_per_cell1)
  , _num_mesh_entities()
  , _mesh_entities()
  , _dmesh_entities(0)
  , _num_mesh_ghost_entities()
  , _mesh_ghost_entities()
  , _dmesh_ghost_entities(0)
  , _element_values()
  , _delement_values(0)
  , _num_nonzero_locations()
  , _dnonzero_locations(0)
  , _delement_matrix_rows(0)
  , _assembly_module(compile_form_integral_kernel(
                       cuda_context,
                       target,
                       form.rank(),
                       _integral_type,
                       form.integrals().get_ufc_integral(_integral_type, i),
                       max_threads_per_block,
                       min_blocks_per_multiprocessor,
                       num_vertices_per_cell,
                       num_coordinates_per_vertex,
                       num_dofs_per_cell0,
                       num_dofs_per_cell1,
                       assembly_kernel_type,
                       debug, cudasrcdir, verbose))
  , _assembly_kernel_type(assembly_kernel_type)
  , _assembly_kernel()
  , _compute_lookup_table_kernel()
  , _lift_bc_kernel()
{
  CUresult cuda_err;
  const char * cuda_err_description;

  _assembly_kernel = _assembly_module.get_device_function(
    std::string("assemble_") + _name);

  if (form.rank() == 2 &&
      (_assembly_kernel_type == ASSEMBLY_KERNEL_LOOKUP_TABLE ||
       _assembly_kernel_type == ASSEMBLY_KERNEL_ROWWISE))
  {
    _compute_lookup_table_kernel = _assembly_module.get_device_function(
      std::string("compute_lookup_table_") +
      std::string("assemble_") + _name);
  }

  if (form.rank() == 2) {
    _lift_bc_kernel = _assembly_module.get_device_function(
      std::string("lift_bc_") + _name);
  }

#if 0
  // Set the preferred cache configuration to make more shared memory
  // available to the kernel
  cuda_err = cuFuncSetCacheConfig(
    _assembly_kernel, CU_FUNC_CACHE_PREFER_SHARED);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuFuncSetCacheConfig() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
#endif

  // Allocate device-side storage for mesh entities
  const FormIntegrals& form_integrals = form.integrals();
  _mesh_entities = &form.integrals().integral_domains(_integral_type, i);
  _num_mesh_entities = _mesh_entities->size();
  _mesh_ghost_entities = &form.integrals().integral_domain_ghosts(_integral_type, i);
  _num_mesh_ghost_entities = _mesh_ghost_entities->size();
  if (_num_mesh_entities + _num_mesh_ghost_entities > 0) {
    size_t dmesh_entities_size =
      (_num_mesh_entities + _num_mesh_ghost_entities) * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dmesh_entities, dmesh_entities_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    _dmesh_ghost_entities = (CUdeviceptr) (
      ((char *) _dmesh_entities) + _num_mesh_entities * sizeof(int32_t));

    // Copy mesh entities to device
    cuda_err = cuMemcpyHtoD(
      _dmesh_entities, _mesh_entities->data(),
      _num_mesh_entities * sizeof(int32_t));
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dmesh_entities);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy mesh ghost entities to device
    cuda_err = cuMemcpyHtoD(
      _dmesh_ghost_entities, _mesh_ghost_entities->data(),
      _num_mesh_ghost_entities * sizeof(int32_t));
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dmesh_entities);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate host- and device-side storage for element vector or
  // matrix values
  if (_assembly_kernel_type == dolfinx::fem::ASSEMBLY_KERNEL_LOCAL &&
      (_num_mesh_entities > 0 || _num_mesh_ghost_entities > 0))
  {
    size_t num_element_values =
      (size_t) (_num_mesh_entities + _num_mesh_ghost_entities) *
      _num_dofs_per_cell0 * _num_dofs_per_cell1;
    size_t delement_values_size = num_element_values * sizeof(PetscScalar);
    _element_values = std::vector<PetscScalar>(num_element_values);
    cuda_err = cuMemAlloc(
      &_delement_values, delement_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dmesh_entities);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
}
//-----------------------------------------------------------------------------
CUDAFormIntegral::~CUDAFormIntegral()
{
  if (_delement_matrix_rows)
    cuMemFree(_delement_matrix_rows);
  if (_dnonzero_locations)
    cuMemFree(_dnonzero_locations);
  if (_delement_values)
    cuMemFree(_delement_values);
  _element_values.clear();
  if (_dmesh_entities)
    cuMemFree(_dmesh_entities);
}
//-----------------------------------------------------------------------------
CUDAFormIntegral::CUDAFormIntegral(CUDAFormIntegral&& form_integral)
  : _integral_type(form_integral._integral_type)
  , _id(form_integral._id)
  , _name(form_integral._name)
  , _cudasrcdir(form_integral._cudasrcdir)
  , _num_vertices_per_cell(form_integral._num_vertices_per_cell)
  , _num_coordinates_per_vertex(form_integral._num_coordinates_per_vertex)
  , _num_dofs_per_cell0(form_integral._num_dofs_per_cell0)
  , _num_dofs_per_cell1(form_integral._num_dofs_per_cell1)
  , _num_mesh_entities(form_integral._num_mesh_entities)
  , _mesh_entities(form_integral._mesh_entities)
  , _dmesh_entities(form_integral._dmesh_entities)
  , _num_mesh_ghost_entities(form_integral._num_mesh_ghost_entities)
  , _mesh_ghost_entities(form_integral._mesh_ghost_entities)
  , _dmesh_ghost_entities(form_integral._dmesh_ghost_entities)
  , _element_values(form_integral._element_values)
  , _delement_values(form_integral._delement_values)
  , _num_nonzero_locations(form_integral._num_nonzero_locations)
  , _dnonzero_locations(form_integral._dnonzero_locations)
  , _delement_matrix_rows(form_integral._delement_matrix_rows)
  , _assembly_module(std::move(form_integral._assembly_module))
  , _assembly_kernel_type(form_integral._assembly_kernel_type)
  , _assembly_kernel(form_integral._assembly_kernel)
  , _compute_lookup_table_kernel(form_integral._compute_lookup_table_kernel)
  , _lift_bc_kernel(form_integral._lift_bc_kernel)
{
  form_integral._integral_type = IntegralType::cell;
  form_integral._id = 0;
  form_integral._name = std::string();
  form_integral._cudasrcdir = nullptr;
  form_integral._num_vertices_per_cell = 0;
  form_integral._num_coordinates_per_vertex = 0;
  form_integral._num_dofs_per_cell0 = 0;
  form_integral._num_dofs_per_cell1 = 0;
  form_integral._num_mesh_entities = 0;
  form_integral._mesh_entities = nullptr;
  form_integral._dmesh_entities = 0;
  form_integral._num_mesh_ghost_entities = 0;
  form_integral._mesh_ghost_entities = nullptr;
  form_integral._dmesh_ghost_entities = 0;
  form_integral._element_values = std::vector<PetscScalar>();
  form_integral._delement_values = 0;
  form_integral._num_nonzero_locations = 0;
  form_integral._dnonzero_locations = 0;
  form_integral._delement_matrix_rows = 0;
  form_integral._assembly_module = CUDA::Module();
  form_integral._assembly_kernel_type = ASSEMBLY_KERNEL_GLOBAL;
  form_integral._assembly_kernel = 0;
  form_integral._compute_lookup_table_kernel = 0;
  form_integral._lift_bc_kernel = 0;
}
//-----------------------------------------------------------------------------
CUDAFormIntegral& CUDAFormIntegral::operator=(CUDAFormIntegral&& form_integral)
{
  _integral_type = form_integral._integral_type;
  _id = form_integral._id;
  _name = form_integral._name;
  _cudasrcdir = form_integral._cudasrcdir;
  _num_vertices_per_cell = form_integral._num_vertices_per_cell;
  _num_coordinates_per_vertex = form_integral._num_coordinates_per_vertex;
  _num_dofs_per_cell0 = form_integral._num_dofs_per_cell0;
  _num_dofs_per_cell1 = form_integral._num_dofs_per_cell1;
  _num_mesh_entities = form_integral._num_mesh_entities;
  _mesh_entities = form_integral._mesh_entities;
  _dmesh_entities = form_integral._dmesh_entities;
  _num_mesh_ghost_entities = form_integral._num_mesh_ghost_entities;
  _mesh_ghost_entities = form_integral._mesh_ghost_entities;
  _dmesh_ghost_entities = form_integral._dmesh_ghost_entities;
  _element_values = form_integral._element_values;
  _delement_values = form_integral._delement_values;
  _num_nonzero_locations = form_integral._num_nonzero_locations;
  _dnonzero_locations = form_integral._dnonzero_locations;
  _delement_matrix_rows = form_integral._delement_matrix_rows;
  _assembly_module = std::move(form_integral._assembly_module);
  _assembly_kernel_type = form_integral._assembly_kernel_type;
  _assembly_kernel = form_integral._assembly_kernel;
  _compute_lookup_table_kernel = form_integral._compute_lookup_table_kernel;
  _lift_bc_kernel = form_integral._lift_bc_kernel;
  form_integral._integral_type = IntegralType::cell;
  form_integral._id = 0;
  form_integral._name = std::string();
  form_integral._cudasrcdir = nullptr;
  form_integral._num_vertices_per_cell = 0;
  form_integral._num_coordinates_per_vertex = 0;
  form_integral._num_dofs_per_cell0 = 0;
  form_integral._num_dofs_per_cell1 = 0;
  form_integral._num_mesh_entities = 0;
  form_integral._mesh_entities = nullptr;
  form_integral._dmesh_entities = 0;
  form_integral._num_mesh_ghost_entities = 0;
  form_integral._mesh_ghost_entities = nullptr;
  form_integral._dmesh_ghost_entities = 0;
  form_integral._element_values = std::vector<PetscScalar>();
  form_integral._delement_values = 0;
  form_integral._num_nonzero_locations = 0;
  form_integral._dnonzero_locations = 0;
  form_integral._delement_matrix_rows = 0;
  form_integral._assembly_module = CUDA::Module();
  form_integral._assembly_kernel_type = ASSEMBLY_KERNEL_GLOBAL;
  form_integral._assembly_kernel = 0;
  form_integral._compute_lookup_table_kernel = 0;
  form_integral._lift_bc_kernel = 0;
  return *this;
}
//-----------------------------------------------------------------------------
CUresult launch_cuda_kernel(
  CUfunction kernel,
  unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
  unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
  unsigned int shared_mem_size_per_thread_block,
  CUstream stream,
  void** kernel_parameters,
  void** extra,
  bool verbose)
{
  if (verbose) {
    int max_threads_per_block;
    cuFuncGetAttribute(
      &max_threads_per_block,
      CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      kernel);
    int shared_size_bytes;
    cuFuncGetAttribute(
      &shared_size_bytes,
      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      kernel);
    int max_dynamic_shared_size_bytes;
    cuFuncGetAttribute(
      &max_dynamic_shared_size_bytes,
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      kernel);
    int const_size_bytes;
    cuFuncGetAttribute(
      &const_size_bytes,
      CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
      kernel);
    int local_size_bytes;
    cuFuncGetAttribute(
      &local_size_bytes,
      CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
      kernel);
    int num_regs;
    cuFuncGetAttribute(
      &num_regs,
      CU_FUNC_ATTRIBUTE_NUM_REGS,
      kernel);
    int ptx_version;
    cuFuncGetAttribute(
      &ptx_version,
      CU_FUNC_ATTRIBUTE_PTX_VERSION,
      kernel);
    int binary_version;
    cuFuncGetAttribute(
      &binary_version,
      CU_FUNC_ATTRIBUTE_BINARY_VERSION,
      kernel);
    int cache_mode_ca;
    cuFuncGetAttribute(
      &cache_mode_ca,
      CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
      kernel);
    int preferred_shared_memory_carveout;
    cuFuncGetAttribute(
      &preferred_shared_memory_carveout,
      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
      kernel);

    fprintf(stderr, "Launching kernel with "
            "%dx%dx%d grid of thread blocks (%d thread blocks), "
            "%dx%dx%d threads in each block (max %d threads per block), "
            "%d bytes of statically allocated shared memory per block, "
            "maximum of %d bytes of dynamic shared memory per block, "
            "%d bytes of constant memory, "
            "%d bytes of local memory per thread, "
            "%d registers per thread, "
            "PTX version %d, binary version %d, "
            "%s, preferred shared memory carve-out: %d.\n",
            grid_dim_x, grid_dim_y, grid_dim_z, grid_dim_x*grid_dim_y*grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z, max_threads_per_block,
            shared_size_bytes, max_dynamic_shared_size_bytes,
            const_size_bytes, local_size_bytes, num_regs,
            ptx_version, binary_version,
            cache_mode_ca ? "global loads cached in L1 cache" : "global loads not cached in L1 cache",
            preferred_shared_memory_carveout);
  }

  return cuLaunchKernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, extra);
}
//-----------------------------------------------------------------------------
void assemble_vector_cell(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  std::int32_t num_active_mesh_entities,
  CUdeviceptr dactive_mesh_entities,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
  const dolfinx::fem::CUDADirichletBC& bc,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAVector& cuda_vector,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Mesh vertex coordinates and cells
  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  // Integral constants and coefficients
  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  // Mapping of cellwise to global degrees of freedom and Dirichlet boundary conditions
  int num_dofs_per_cell = dofmap.num_dofs_per_cell();
  CUdeviceptr ddofmap = dofmap.dofs_per_cell();
  CUdeviceptr dbc = bc.dof_markers();

  // Global vector
  std::int32_t num_values = cuda_vector.num_values();
  CUdeviceptr dvalues = cuda_vector.values_write();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_vertices,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &dcell_permutations,
    &num_active_mesh_entities,
    &dactive_mesh_entities,
    &num_constant_values,
    &dconstant_values,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_dofs_per_cell,
    &ddofmap,
    &dbc,
    &num_values,
    &dvalues};

  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  cuda_vector.restore_values_write();
}
//-----------------------------------------------------------------------------
void assemble_vector_exterior_facet(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  std::int32_t num_active_mesh_entities,
  CUdeviceptr dactive_mesh_entities,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
  const dolfinx::fem::CUDADirichletBC& bc,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAVector& cuda_vector,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Mesh vertex coordinates and cells
  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  // Integral constants and coefficients
  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  // Mapping of cellwise to global degrees of freedom and Dirichlet boundary conditions
  int num_dofs_per_cell = dofmap.num_dofs_per_cell();
  CUdeviceptr ddofmap = dofmap.dofs_per_cell();
  CUdeviceptr dbc = bc.dof_markers();

  // Global vector
  std::int32_t num_values = cuda_vector.num_values();
  CUdeviceptr dvalues = cuda_vector.values_write();

  (void) cuda_context;
  // Mesh facets
  std::int32_t tdim = mesh.tdim();
  const dolfinx::mesh::CUDAMeshEntities& facets = mesh.mesh_entities()[tdim-1];
  std::int32_t num_mesh_entities = facets.num_mesh_entities();
  std::int32_t num_mesh_entities_per_cell = facets.num_mesh_entities_per_cell();
  CUdeviceptr dmesh_entities_per_cell = facets.mesh_entities_per_cell();
  CUdeviceptr dcells_per_mesh_entity_ptr = facets.cells_per_mesh_entity_ptr();
  CUdeviceptr dcells_per_mesh_entity = facets.cells_per_mesh_entity();
  CUdeviceptr dmesh_entity_permutations = facets.mesh_entity_permutations();

  void * kernel_parameters[] = {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_vertices,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &dcell_permutations,
    &num_mesh_entities,
    &num_mesh_entities_per_cell,
    &dmesh_entities_per_cell,
    &dcells_per_mesh_entity_ptr,
    &dcells_per_mesh_entity,
    &dmesh_entity_permutations,
    &num_active_mesh_entities,
    &dactive_mesh_entities,
    &num_constant_values,
    &dconstant_values,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_dofs_per_cell,
    &ddofmap,
    &dbc,
    &num_values,
    &dvalues};

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  cuda_vector.restore_values_write();
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_vector(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
  const dolfinx::fem::CUDADirichletBC& bc,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAVector& cuda_vector,
  bool verbose) const
{
  CUresult cuda_err;
  const char * cuda_err_description;

  switch (_integral_type) {
  case IntegralType::cell:
    assemble_vector_cell(
      cuda_context, _assembly_kernel, _num_mesh_entities + _num_mesh_ghost_entities, _dmesh_entities,
      mesh, dofmap, bc, constants, coefficients, cuda_vector, verbose);
    break;
  case IntegralType::exterior_facet:
    assemble_vector_exterior_facet(
      cuda_context, _assembly_kernel, _num_mesh_entities + _num_mesh_ghost_entities, _dmesh_entities,
      mesh, dofmap, bc, constants, coefficients, cuda_vector, verbose);
    break;
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(_integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void lift_bc_cell(
  const CUDA::Context& cuda_context,
  CUfunction kernel,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  double scale,
  const dolfinx::la::CUDAVector& x0,
  dolfinx::la::CUDAVector& b,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  std::int32_t num_cells = mesh.num_cells();
  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  int num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  int num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();

  CUdeviceptr dbc_markers1 = bc1.dof_markers();
  CUdeviceptr dbc_values1 = bc1.dof_values();

  std::int32_t num_constant_values = constants.num_constant_values();
  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  std::int32_t num_columns = dofmap1.num_dofs();
  CUdeviceptr dx0 = x0.values();
  CUdeviceptr db = b.values_write();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_cells,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &num_constant_values,
    &dconstant_values,
    &dcell_permutations,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dbc_markers1,
    &dbc_values1,
    &scale,
    &num_columns,
    &dx0,
    &db};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  b.restore_values_write();
  x0.restore_values();
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::lift_bc(
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
  bool verbose) const
{
  CUresult cuda_err;
  const char * cuda_err_description;

  switch (_integral_type) {
  case IntegralType::cell:
    lift_bc_cell(
      cuda_context, _lift_bc_kernel, mesh, dofmap0, dofmap1,
      bc1, constants, coefficients, scale, x0, b, verbose);
    break;
  default:
    throw std::runtime_error(
      "Forms of type " + to_string(_integral_type) + " are not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_local(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  CUfunction kernel = _assembly_kernel;
  std::int32_t num_mesh_entities = _num_mesh_entities;
  CUdeviceptr mesh_entities = _dmesh_entities;
  CUdeviceptr element_values = _delement_values;

  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();
  CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
  CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

  CUdeviceptr dbc0 = bc0.dof_markers();
  CUdeviceptr dbc1 = bc1.dof_markers();

  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  CUdeviceptr dvalues = _delement_values;

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_mesh_entities,
    &mesh_entities,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &dconstant_values,
    &dcell_permutations,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dbc0,
    &dbc1,
    &dvalues};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_local_copy_to_host(
  const CUDA::Context& cuda_context)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Since the kernel only performed local assembly, we must copy
  // element matrices from the device to the host and perform the
  // global assembly on the host.
  cuda_err = cuMemcpyDtoH(
    _element_values.data(), _delement_values,
    _element_values.size() * sizeof(PetscScalar));
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_local_host_global_assembly(
  const CUDA::Context& cuda_context,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  dolfinx::la::CUDAMatrix& A)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

  PetscErrorCode ierr;
  Mat host_A = A.mat();
  const dolfinx::fem::DofMap& host_dofmap0 = *dofmap0.dofmap();
  const dolfinx::fem::DofMap& host_dofmap1 = *dofmap1.dofmap();
  const graph::AdjacencyList<std::int32_t>& dofs0 = host_dofmap0.list();
  const graph::AdjacencyList<std::int32_t>& dofs1 = host_dofmap1.list();
  for (int i = 0; i < _num_mesh_entities; i++) {
    std::int32_t c = (*_mesh_entities)[i];
    auto element_dofs0 = dofs0.links(c);
    auto element_dofs1 = dofs1.links(c);
    const PetscScalar* Ae = &_element_values[
      i * num_dofs_per_cell0 * num_dofs_per_cell1];
    ierr = MatSetValuesLocal(
      host_A, _num_dofs_per_cell0, element_dofs0.data(),
      _num_dofs_per_cell1, element_dofs1.data(), Ae, ADD_VALUES);
    if (ierr != 0)
      dolfinx::la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_global(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  CUfunction kernel = _assembly_kernel;
  std::int32_t num_mesh_entities =
    _num_mesh_entities + _num_mesh_ghost_entities;
  CUdeviceptr mesh_entities = _dmesh_entities;
  CUdeviceptr element_values = _delement_values;

  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();

  CUdeviceptr dbc0 = bc0.dof_markers();
  CUdeviceptr dbc1 = bc1.dof_markers();

  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  std::int32_t num_local_rows = A.num_local_rows();
  std::int32_t num_local_columns = A.num_local_columns();
  CUdeviceptr drow_ptr = A.diag()->row_ptr();
  CUdeviceptr dcolumn_indices = A.diag()->column_indices();
  CUdeviceptr dvalues = A.diag()->values();
  CUdeviceptr doffdiag_row_ptr = A.offdiag() ? A.offdiag()->row_ptr() : 0;
  CUdeviceptr doffdiag_column_indices = A.offdiag() ? A.offdiag()->column_indices() : 0;
  CUdeviceptr doffdiag_values = A.offdiag() ? A.offdiag()->values() : 0;
  std::int32_t num_local_offdiag_columns = A.num_local_offdiag_columns();
  CUdeviceptr dcolmap = A.colmap();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices and
  // perform global assembly
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_mesh_entities,
    &mesh_entities,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &dconstant_values,
    &dcell_permutations,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dbc0,
    &dbc1,
    &num_local_rows,
    &num_local_columns,
    &drow_ptr,
    &dcolumn_indices,
    &dvalues,
    &doffdiag_row_ptr,
    &doffdiag_column_indices,
    &doffdiag_values,
    &num_local_offdiag_columns,
    &dcolmap};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_lookup_table(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Compute a lookup table, unless it has been computed already.
  if (!_dnonzero_locations) {
    compute_lookup_table(
      cuda_context, dofmap0, dofmap1,
      bc0, bc1, A, verbose);
  }

  CUfunction kernel = _assembly_kernel;
  std::int32_t num_mesh_entities = _num_mesh_entities;
  CUdeviceptr mesh_entities = _dmesh_entities;
  CUdeviceptr nonzero_locations = _dnonzero_locations;

  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
  CUdeviceptr dvalues = A.diag()->values();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_mesh_entities,
    &mesh_entities,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &dconstant_values,
    &dcell_permutations,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &nonzero_locations,
    &dvalues};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix_rowwise(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Compute a lookup table, unless it has been computed already.
  if (!_dnonzero_locations) {
    compute_lookup_table(
      cuda_context, dofmap0, dofmap1,
      bc0, bc1, A, verbose);
  }

  CUfunction kernel = _assembly_kernel;
  std::int32_t num_mesh_entities = _num_mesh_entities;
  CUdeviceptr mesh_entities = _dmesh_entities;
  CUdeviceptr nonzero_locations = _dnonzero_locations;
  CUdeviceptr element_matrix_rows = _delement_matrix_rows;

  std::int32_t num_vertices_per_cell = mesh.num_vertices_per_cell();
  CUdeviceptr dvertex_indices_per_cell = mesh.vertex_indices_per_cell();
  std::int32_t num_vertices = mesh.num_vertices();
  std::int32_t num_coordinates_per_vertex = mesh.num_coordinates_per_vertex();
  CUdeviceptr dvertex_coordinates = mesh.vertex_coordinates();
  CUdeviceptr dcell_permutations = mesh.cell_permutations();

  std::int32_t num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  std::int32_t num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();
  CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
  CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

  CUdeviceptr dconstant_values = constants.constant_values();
  std::int32_t num_coefficient_values_per_cell =
    coefficients.num_packed_coefficient_values_per_cell();
  CUdeviceptr dcoefficient_values = coefficients.packed_coefficient_values();

  /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
  std::int32_t num_rows = A.num_rows();
  CUdeviceptr dvalues = A.diag()->values();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_mesh_entities,
    &mesh_entities,
    &num_vertices_per_cell,
    &dvertex_indices_per_cell,
    &num_coordinates_per_vertex,
    &dvertex_coordinates,
    &num_coefficient_values_per_cell,
    &dcoefficient_values,
    &dconstant_values,
    &dcell_permutations,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &dcells_per_dof_ptr,
    &dcells_per_dof,
    &nonzero_locations,
    &element_matrix_rows,
    &num_rows,
    &dvalues};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::assemble_matrix(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh& mesh,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  const dolfinx::fem::CUDAFormConstants& constants,
  const dolfinx::fem::CUDAFormCoefficients& coefficients,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  switch (_assembly_kernel_type) {
  case ASSEMBLY_KERNEL_LOCAL:
    return assemble_matrix_local(
      cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
      constants, coefficients, A, verbose);
  case ASSEMBLY_KERNEL_GLOBAL:
    return assemble_matrix_global(
      cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
      constants, coefficients, A, verbose);
  case ASSEMBLY_KERNEL_LOOKUP_TABLE:
    return assemble_matrix_lookup_table(
      cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
      constants, coefficients, A, verbose);
  case ASSEMBLY_KERNEL_ROWWISE:
    return assemble_matrix_rowwise(
      cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
      constants, coefficients, A, verbose);
  default:
    throw std::invalid_argument(
      "Invalid argument at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAFormIntegral::compute_lookup_table(
  const CUDA::Context& cuda_context,
  const dolfinx::fem::CUDADofMap& dofmap0,
  const dolfinx::fem::CUDADofMap& dofmap1,
  const dolfinx::fem::CUDADirichletBC& bc0,
  const dolfinx::fem::CUDADirichletBC& bc1,
  dolfinx::la::CUDAMatrix& A,
  bool verbose)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  if (!_compute_lookup_table_kernel)
    return;

  int num_dofs_per_cell0 = dofmap0.num_dofs_per_cell();
  int num_dofs_per_cell1 = dofmap1.num_dofs_per_cell();

  // Get the warp size
  int warp_size;
  cuda_err = cuDeviceGetAttribute(
    &warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuda_context.device());
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuDeviceGetAttribute() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Allocate device-side storage for the lookup table for the sparse
  // matrix non-zeros, rounding the number of mesh cells up to the
  // nearest multiple of the warp size due to coalescing.
  _num_nonzero_locations =
    ((int64_t) ((_num_mesh_entities + (warp_size-1)) / warp_size) * warp_size) *
    num_dofs_per_cell0 * num_dofs_per_cell1;
  if (_num_nonzero_locations <= 0)
    return;

  size_t dnonzero_locations_size = _num_nonzero_locations * sizeof(int32_t);
  cuda_err = cuMemAlloc(&_dnonzero_locations, dnonzero_locations_size);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuMemAlloc() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  size_t delement_matrix_rows_size =
    (int64_t) _num_mesh_entities * _num_dofs_per_cell0 * sizeof(int32_t);
  cuda_err = cuMemAlloc(&_delement_matrix_rows, delement_matrix_rows_size);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuMemAlloc() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  CUfunction kernel = _compute_lookup_table_kernel;
  std::int32_t num_mesh_entities = _num_mesh_entities;
  CUdeviceptr mesh_entities = _dmesh_entities;

  std::int32_t num_dofs = dofmap0.num_dofs();
  CUdeviceptr ddofmap0 = dofmap0.dofs_per_cell();
  CUdeviceptr ddofmap1 = dofmap1.dofs_per_cell();
  CUdeviceptr dcells_per_dof_ptr = dofmap0.cells_per_dof_ptr();
  CUdeviceptr dcells_per_dof = dofmap0.cells_per_dof();

  CUdeviceptr dbc0 = bc0.dof_markers();
  CUdeviceptr dbc1 = bc1.dof_markers();

  /* TODO: Pass in the diagonal and off-diagonal part of the matrix. */
  std::int32_t num_rows = A.num_rows();
  CUdeviceptr drow_ptr = A.diag()->row_ptr();
  CUdeviceptr dcolumn_indices = A.diag()->column_indices();

  // Use the CUDA occupancy calculator to determine a grid and block
  // size for the CUDA kernel
  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  CUstream stream = NULL;

  // Launch device-side kernel to compute element matrices
  (void) cuda_context;
  void * kernel_parameters[] = {
    &num_mesh_entities,
    &mesh_entities,
    &num_dofs,
    &num_dofs_per_cell0,
    &num_dofs_per_cell1,
    &ddofmap0,
    &ddofmap1,
    &dcells_per_dof_ptr,
    &dcells_per_dof,
    &dbc0,
    &dbc1,
    &num_rows,
    &drow_ptr,
    &dcolumn_indices,
    &_num_nonzero_locations,
    &_dnonzero_locations,
    &_delement_matrix_rows};
  cuda_err = launch_cuda_kernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL, verbose);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
std::map<IntegralType, std::vector<CUDAFormIntegral>>
dolfinx::fem::cuda_form_integrals(
  const CUDA::Context& cuda_context,
  CUjit_target target,
  const Form& form,
  enum assembly_kernel_type assembly_kernel_type,
  int32_t max_threads_per_block,
  int32_t min_blocks_per_multiprocessor,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
{
  const FormIntegrals& form_integrals = form.integrals();

  // Get the number of vertices and coordinates
  const mesh::Mesh& mesh = *form.mesh();
  std::int32_t num_vertices_per_cell = mesh.geometry().dofmap().num_links(0);
  std::int32_t num_coordinates_per_vertex = mesh.geometry().dim();

  // Find the number of degrees of freedom per cell
  int32_t num_dofs_per_cell0 = 1;
  int32_t num_dofs_per_cell1 = 1;
  if (form.rank() > 0) {
    const DofMap& dofmap0 = *form.function_space(0)->dofmap();
    num_dofs_per_cell0 = dofmap0.element_dof_layout->num_dofs();
  }
  if (form.rank() > 1) {
    const DofMap& dofmap1 = *form.function_space(1)->dofmap();
    num_dofs_per_cell1 = dofmap1.element_dof_layout->num_dofs();
  }
  std::map<IntegralType, std::vector<CUDAFormIntegral>>
    cuda_form_integrals;

  {
    // Create device-side kernels and data for cell integrals
    IntegralType integral_type = IntegralType::cell;
    int num_integrals = form_integrals.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral>& cuda_cell_integrals =
        cuda_form_integrals[integral_type];
      for (int i = 0; i < num_integrals; i++) {
        cuda_cell_integrals.emplace_back(
          cuda_context, target, form, integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  {
    // Create device-side kernels and data for exterior facet integrals
    IntegralType integral_type = IntegralType::exterior_facet;
    int num_integrals = form_integrals.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral>& cuda_exterior_facet_integrals =
        cuda_form_integrals[integral_type];
      for (int i = 0; i < num_integrals; i++) {
        cuda_exterior_facet_integrals.emplace_back(
          cuda_context, target, form, integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  {
    // Create device-side kernels and data for interior facet integrals
    IntegralType integral_type = IntegralType::interior_facet;
    int num_integrals = form_integrals.num_integrals(integral_type);
    if (num_integrals > 0) {
      std::vector<CUDAFormIntegral>& cuda_interior_facet_integrals =
        cuda_form_integrals[integral_type];
      for (int i = 0; i < num_integrals; i++) {
        cuda_interior_facet_integrals.emplace_back(
          cuda_context, target, form, integral_type, i,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0, num_dofs_per_cell1,
          assembly_kernel_type, debug, cudasrcdir, verbose);
      }
    }
  }

  return cuda_form_integrals;
}
//-----------------------------------------------------------------------------
std::string dolfinx::fem::cuda_kernel_binary_search(void)
{
  return
    "/**\n"
    " * `binary_search()` performs a binary search to find the location\n"
    " * of a given element in a sorted array of integers.\n"
    " */\n"
    "extern \"C\" __device__ int binary_search(\n"
    "  int num_elements,\n"
    "  const int * __restrict__ elements,\n"
    "  int key,\n"
    "  int * __restrict__ out_index)\n"
    "{\n"
    "  if (num_elements <= 0)\n"
    "    return -1;\n"
    "\n"
    "  int p = 0;\n"
    "  int q = num_elements;\n"
    "  int r = (p + q) / 2;\n"
    "  while (p < q) {\n"
    "    if (elements[r] == key) break;\n"
    "    else if (elements[r] < key) p = r + 1;\n"
    "    else q = r - 1;\n"
    "    r = (p + q) / 2;\n"
    "  }\n"
    "  if (elements[r] != key)\n"
    "    return -1;\n"
    "  *out_index = r;\n"
    "  return 0;\n"
    "}\n";
}
//-----------------------------------------------------------------------------
#endif
