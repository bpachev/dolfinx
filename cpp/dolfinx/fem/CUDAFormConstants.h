// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/CUDA.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace fem {
class Form;

/// A wrapper for a form constant with data that is stored in the
/// device memory of a CUDA device.
class CUDAFormConstants
{
public:
  /// Create an empty collection constant values
  CUDAFormConstants();

  /// Create a collection constant values from a given form
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form The variational form whose constants are used
  CUDAFormConstants(
    const CUDA::Context& cuda_context,
    const Form* form);

  /// Destructor
  ~CUDAFormConstants();

  /// Copy constructor
  /// @param[in] form_constant The object to be copied
  CUDAFormConstants(const CUDAFormConstants& form_constant) = delete;

  /// Move constructor
  /// @param[in] form_constant The object to be moved
  CUDAFormConstants(CUDAFormConstants&& form_constant);

  /// Assignment operator
  /// @param[in] form_constant Another CUDAFormConstants object
  CUDAFormConstants& operator=(const CUDAFormConstants& form_constant) = delete;

  /// Move assignment operator
  /// @param[in] form_constant Another CUDAFormConstants object
  CUDAFormConstants& operator=(CUDAFormConstants&& form_constant);

  /// Get the number of constant values that the constant applies to
  int32_t num_constant_values() const { return _num_constant_values; }

  /// Get the constant values that the constant applies to
  CUdeviceptr constant_values() const { return _dconstant_values; }

  /// Update the constant values by copying values from host to device
  void update_constant_values() const;

private:
  // The form that the constant applies to
  const Form* _form;

  /// The number of constant values
  int32_t _num_constant_values;

  /// The constant values
  CUdeviceptr _dconstant_values;
};

} // namespace fem
} // namespace dolfinx

#endif
