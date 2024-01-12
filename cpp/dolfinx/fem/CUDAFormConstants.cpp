// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAFormConstants.h"
#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>


#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
template <class T>
CUDAFormConstants<T>::CUDAFormConstants()
  : _form(nullptr)
  , _num_constant_values()
  , _dconstant_values(0)
{
}
//-----------------------------------------------------------------------------
template <class T>
CUDAFormConstants<T>::CUDAFormConstants(
  const CUDA::Context& cuda_context,
  const Form<T>* form)
  : _form(form)
  , _num_constant_values()
  , _dconstant_values(0)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Pack constants into an array
  if (!_form->all_constants_set()) {
    throw std::runtime_error(
      "Unset constant in Form "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  const std::vector<T>
    constant_values = pack_constants(*_form);

  // Allocate device-side storage for constant values
  _num_constant_values = constant_values.size();
  if (_num_constant_values > 0) {
    size_t dconstant_values_size =
      _num_constant_values * sizeof(T);
    cuda_err = cuMemAlloc(
      &_dconstant_values, dconstant_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy constant values to device
    cuda_err = cuMemcpyHtoD(
      _dconstant_values, constant_values.data(), dconstant_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dconstant_values);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
}
//-----------------------------------------------------------------------------
template <class T>
CUDAFormConstants<T>::~CUDAFormConstants()
{
  if (_dconstant_values)
    cuMemFree(_dconstant_values);
}
//-----------------------------------------------------------------------------
template <class T>
CUDAFormConstants<T>::CUDAFormConstants(CUDAFormConstants<T>&& constants)
  : _form(constants._form)
  , _num_constant_values(constants._num_constant_values)
  , _dconstant_values(constants._dconstant_values)
{
  constants._form = nullptr;
  constants._num_constant_values = 0;
  constants._dconstant_values = 0;
}
//-----------------------------------------------------------------------------
template <class T>
CUDAFormConstants<T>& CUDAFormConstants<T>::operator=(CUDAFormConstants<T>&& constants)
{
  _form = constants._form;
  _num_constant_values = constants._num_constant_values;
  _dconstant_values = constants._dconstant_values;
  constants._form = nullptr;
  constants._num_constant_values = 0;
  constants._dconstant_values = 0;
  return *this;
}
//-----------------------------------------------------------------------------
template <class T>
void CUDAFormConstants<T>::update_constant_values() const
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Pack constants into an array
  const std::vector<T>  
    constant_values = pack_constants(*_form);
  assert(_num_constant_values == constant_values.size());

  // Copy constant values to device
  if (_num_constant_values > 0) {
    size_t dconstant_values_size =
      _num_constant_values * sizeof(T);
    cuda_err = cuMemcpyHtoD(
      _dconstant_values, constant_values.data(), dconstant_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dconstant_values);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
}
//-----------------------------------------------------------------------------
#endif
