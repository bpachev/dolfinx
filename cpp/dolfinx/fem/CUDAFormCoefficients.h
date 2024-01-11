// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <dolfinx/common/CUDA.h>
#endif

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <petscvec.h>

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace fem {
class Form;

/// A wrapper for a form coefficient with data that is stored in the
/// device memory of a CUDA device.
class CUDAFormCoefficients
{
public:
  /// Scalar Type
  using scalar_type = T;

  /// Create an empty collection coefficient values
  CUDAFormCoefficients();

  /// Create a collection coefficient values from a given form
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form The variational form whose coefficients are used
  /// @param[in] page_lock Whether or not to use page-locked memory
  ///                      for host-side arrays
  CUDAFormCoefficients(
    const CUDA::Context& cuda_context,
    Form* form,
    bool page_lock = true);

  /// Destructor
  ~CUDAFormCoefficients();

  /// Copy constructor
  /// @param[in] form_coefficient The object to be copied
  CUDAFormCoefficients(const CUDAFormCoefficients& form_coefficient) = delete;

  /// Move constructor
  /// @param[in] form_coefficient The object to be moved
  CUDAFormCoefficients(CUDAFormCoefficients&& form_coefficient);

  /// Assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(const CUDAFormCoefficients& form_coefficient) = delete;

  /// Move assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(CUDAFormCoefficients&& form_coefficient);

  /// Get the number of mesh cells that the coefficient applies to
  int32_t num_coefficients() const { return _coefficients->size(); }

  dolfinx::fem::FormCoefficients* coefficients() { return _coefficients; };

  /// Get device-side pointer to number of dofs per cell for each coefficient
  CUdeviceptr dofmaps_num_dofs_per_cell() const { return _dofmaps_num_dofs_per_cell; }

  /// Get device-side pointer to dofs per cell for each coefficient
  CUdeviceptr dofmaps_dofs_per_cell() const { return _dofmaps_dofs_per_cell; }

  /// Get device-side pointer to offsets to coefficient values within
  /// a cell for each coefficient
  CUdeviceptr coefficient_values_offsets() const { return _coefficient_values_offsets; }

  /// Get device-side pointer to an array of pointers to coefficient
  /// values for each coefficient
  CUdeviceptr coefficient_values() const { return _coefficient_values; }

  /// Get the number of mesh cells that the coefficient applies to
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of coefficient values per cell
  int32_t num_packed_coefficient_values_per_cell() const {
      return _num_packed_coefficient_values_per_cell; }

  /// Get the coefficient values that the coefficient applies to
  CUdeviceptr packed_coefficient_values() const { return _dpacked_coefficient_values; }

  /// Update the coefficient values by copying values from host to device.
  /// This must be called before packing the coefficients, if they
  /// have been changed on the host.
  void copy_coefficients_to_device(const CUDA::Context& cuda_context);

  /// Update the coefficient values by copying values from host to device
  void update_coefficient_values() const;

private:
  /// The underlying coefficients on the host
  std::vector<std::shared_ptr<const Function<T, U>>> _coefficients;

  /// Number of dofs per cell for each coefficient
  CUdeviceptr _dofmaps_num_dofs_per_cell;

  /// Dofs per cell for each coefficient
  CUdeviceptr _dofmaps_dofs_per_cell;

  /// Get device-side pointer to offsets to coefficient values within
  /// a cell for each coefficient
  CUdeviceptr _coefficient_values_offsets;

  /// Get device-side pointer to an array of pointers to coefficient
  /// values for each coefficient
  CUdeviceptr _coefficient_values;

  /// The number of cells that the coefficient applies to
  int32_t _num_cells;

  /// The number of packed coefficient values per cell
  int32_t _num_packed_coefficient_values_per_cell;

  /// Whether or not the host-side array of values uses page-locked
  /// (pinned) memory
  bool _page_lock;

  /// Host-side array of coefficient values
  mutable std::vector<T> 
    _host_coefficient_values;

  /// The coefficient values
  CUdeviceptr _dpacked_coefficient_values;
};

} // namespace fem
} // namespace dolfinx
#endif
