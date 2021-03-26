// Copyright (C) 2020-2021 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include <dolfinx/common/span.hpp>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <variant>

namespace dolfinx::fem
{

template <typename T>
class Function;

/// Compute the evaluation points in the physical space at which an
/// expression should be computed to interpolate it in a finite elemenet
/// space.
///
/// @param[in] element The element to be interpolated into
/// @param[in] mesh The domain
/// @param[in] cells Indices of the cells in the mesh to compute
/// interpolation coordinates for
/// @return The coordinates in the physical space at which to evaluate
/// an expression
array2d<double>
interpolation_coords(const fem::FiniteElement& element, const mesh::Mesh& mesh,
                     const tcb::span<const std::int32_t>& cells);

/// Interpolate a finite element Function (on possibly non-matching
/// meshes) in another finite element space
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v);

/// Interpolate an expression in a finite element space
///
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
/// @param[in] x The points at which f should be evaluated, as computed
/// by fem::interpolation_coords. The element used in
/// fem::interpolation_coords should be the same element as associated
/// with u.
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate(Function<T>& u,
                 const std::function<std::variant<std::vector<T>, array2d<T>>(
                     const array2d<double>&)>& f,
                 const array2d<double>& x,
                 const tcb::span<const std::int32_t>& cells);

/// Interpolate an expression f(x)
///
/// @note  This interface uses an expression function f that has an
/// in/out argument for the expression values. It is primarily to
/// support C code implementations of the expression, e.g. using Numba.
/// Generally the interface where the expression function is a pure
/// function, i.e. the expression values are the return argument, should
/// be preferred.
///
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
/// @param[in] x The points at which should be evaluated, as
/// computed by fem::interpolation_coords
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate_c(
    Function<T>& u,
    const std::function<void(array2d<T>&, const array2d<double>&)>& f,
    const array2d<double>& x, const tcb::span<const std::int32_t>& cells);

namespace detail
{

template <typename T>
void interpolate_from_any(Function<T>& u, const Function<T>& v)
{
  assert(v.function_space());
  const auto element = u.function_space()->element();
  assert(element);
  if (v.function_space()->element()->hash() != element->hash())
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not supported.");
  }

  const auto mesh = u.function_space()->mesh();
  assert(mesh);
  assert(v.function_space()->mesh());
  if (mesh->id() != v.function_space()->mesh()->id())
  {
    throw std::runtime_error(
        "Interpolation on different meshes not supported (yet).");
  }
  const int tdim = mesh->topology().dim();

  // Get dofmaps
  assert(v.function_space());
  std::shared_ptr<const fem::DofMap> dofmap_v = v.function_space()->dofmap();
  assert(dofmap_v);
  auto map = mesh->topology().index_map(tdim);
  assert(map);

  std::vector<T>& coeffs = u.x()->mutable_array();

  // Iterate over mesh and interpolate on each cell
  const auto dofmap_u = u.function_space()->dofmap();
  const std::vector<T>& v_array = v.x()->array();
  const int num_cells = map->size_local() + map->num_ghosts();
  const int bs = dofmap_v->bs();
  assert(bs == dofmap_u->bs());
  for (int c = 0; c < num_cells; ++c)
  {
    tcb::span<const std::int32_t> dofs_v = dofmap_v->cell_dofs(c);
    tcb::span<const std::int32_t> cell_dofs = dofmap_u->cell_dofs(c);
    assert(dofs_v.size() == cell_dofs.size());
    for (std::size_t i = 0; i < dofs_v.size(); ++i)
    {
      for (int k = 0; k < bs; ++k)
        coeffs[bs * cell_dofs[i] + k] = v_array[bs * dofs_v[i] + k];
    }
  }
}

} // namespace detail

//----------------------------------------------------------------------------
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v)
{
  assert(u.function_space());
  const auto element = u.function_space()->element();
  assert(element);

  // Check that function ranks match
  if (int rank_v = v.function_space()->element()->value_rank();
      element->value_rank() != rank_v)
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(rank_v)
                             + ") does not match rank of function space ("
                             + std::to_string(element->value_rank()) + ")");
  }

  // Check that function dimension match
  for (int i = 0; i < element->value_rank(); ++i)
  {
    if (int v_dim = v.function_space()->element()->value_dimension(i);
        element->value_dimension(i) != v_dim)
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function (" + std::to_string(v_dim)
          + ") does not match dimension " + std::to_string(i)
          + " of function space(" + std::to_string(element->value_dimension(i))
          + ")");
    }
  }

  detail::interpolate_from_any(u, v);
}
//----------------------------------------------------------------------------
template <typename T>
void interpolate(Function<T>& u,
                 const std::function<std::variant<std::vector<T>, array2d<T>>(
                     const array2d<double>&)>& f,
                 const array2d<double>& x,
                 const tcb::span<const std::int32_t>& cells)
{
  const auto element = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();
  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get the interpolation points on the reference cells
  const array2d<double> X = element->interpolation_points();

  if (X.shape[0] == 0)
    throw std::runtime_error(
        "Interpolation into this space is not yet supported.");

  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Evaluate function at physical points. The returned array has a
  // number of rows equal to the number of components of the function,
  // and the number of columns is equal to the number of evaluation
  // points.

  // TODO: Copies and memory allocation could be avoided with a 'span2d'
  // class, or by just pointing to the data
  array2d<T> values(element->value_size(), x.shape[1]);
  std::variant<std::vector<T>, array2d<T>> values_v = f(x);
  if (std::holds_alternative<array2d<T>>(values_v))
  {
    values = std::get<1>(values_v);
    if (values.shape[0] != element->value_size())
      throw std::runtime_error("Interpolation data has the wrong shape.");
  }
  else
  {
    if (element->value_size() != 1)
      throw std::runtime_error("Interpolation data has the wrong shape.");
    std::copy(std::get<0>(values_v).begin(), std::get<0>(values_v).end(),
              values.data());
  }

  if (values.shape[1] != cells.size() * X.shape[0])
    throw std::runtime_error("Interpolation data has the wrong shape.");

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // Get coordinate map
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const array2d<double>& x_g = mesh->geometry().x();

  // NOTE: The below loop over cells could be skipped for some elements,
  // e.g. Lagrange, where the interpolation is just the identity

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = element->value_size() / element_bs;

  array2d<double> x_cell(X.shape[0], gdim);
  std::vector<double> J(X.shape[0] * gdim * tdim);
  std::vector<double> detJ(X.shape[0]);
  std::vector<double> K(X.shape[0] * tdim * gdim);
  array2d<double> X_ref(X.shape[0], tdim);

  array2d<double> coordinate_dofs(num_dofs_g, gdim);

  array2d<T> reference_data(value_size, X.shape[0]);

  std::vector<T>& coeffs = u.x()->mutable_array();
  std::vector<T> _coeffs(num_scalar_dofs);
  array2d<T> _vals(value_size, X.shape[0]);
  for (std::int32_t c : cells)
  {
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);
    cmap.push_forward(x_cell, X, coordinate_dofs);

    cmap.compute_reference_geometry(X_ref, J, detJ, K, x_cell, coordinate_dofs);

    auto dofs = dofmap->cell_dofs(c);

    for (int k = 0; k < element_bs; ++k)
    {
      // Extract computed expression values for element block k
      for (int m = 0; m < value_size; ++m)
      {
        std::copy_n(&values(k * value_size + m, c * X.shape[0]), X.shape[0],
                    _vals.row(m).begin());
      }

      // Get element degrees of freedom for block
      element->map_pull_back(reference_data.data(), _vals.data(), J.data(),
                             detJ.data(), K.data(), gdim, value_size, 1,
                             X.shape[0]);

      element->interpolate(reference_data, tcb::make_span(_coeffs));
      element->apply_inverse_transpose_dof_transformation(_coeffs.data(),
                                                          cell_info[c], 1);

      assert(_coeffs.size() == num_scalar_dofs);

      // Copy interpolation dofs into coefficient vector
      for (int i = 0; i < num_scalar_dofs; ++i)
      {
        const int dof = i * element_bs + k;
        std::div_t pos = std::div(dof, dofmap_bs);
        coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
      }
    }
  }
}
//----------------------------------------------------------------------------
template <typename T>
void interpolate_c(
    Function<T>& u,
    const std::function<void(array2d<T>&, const array2d<double>&)>& f,
    const array2d<double>& x, const tcb::span<const std::int32_t>& cells)
{
  const auto element = u.function_space()->element();
  assert(element);
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());

  auto fn = [value_size, &f](const array2d<double>& x) {
    array2d<T> values(value_size, x.shape[1]);
    f(values, x);
    return values;
  };

  interpolate<T>(u, fn, x, cells);
}
//----------------------------------------------------------------------------

} // namespace dolfinx::fem