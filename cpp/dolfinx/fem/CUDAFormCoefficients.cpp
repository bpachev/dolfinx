// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAFormCoefficients.h"
#if defined(HAS_CUDA_TOOLKIT)
#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/CUDADofMap.h>
#endif
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/common/IndexMap.h>

#include <Eigen/Dense>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
CUDAFormCoefficients::CUDAFormCoefficients()
  : _coefficients(nullptr)
  , _dofmaps_num_dofs_per_cell(0)
  , _dofmaps_dofs_per_cell(0)
  , _coefficient_values_offsets(0)
  , _coefficient_values(0)
  , _num_cells()
  , _num_packed_coefficient_values_per_cell()
  , _page_lock(false)
  , _host_coefficient_values()
  , _dpacked_coefficient_values(0)
{
}
//-----------------------------------------------------------------------------
CUDAFormCoefficients::CUDAFormCoefficients(
  const CUDA::Context& cuda_context,
  Form* form,
  bool page_lock)
  : _coefficients(&form->coefficients())
  , _dofmaps_num_dofs_per_cell(0)
  , _dofmaps_dofs_per_cell(0)
  , _coefficient_values_offsets(0)
  , _coefficient_values(0)
  , _num_cells()
  , _num_packed_coefficient_values_per_cell()
  , _page_lock(false)
  , _host_coefficient_values()
  , _dpacked_coefficient_values(0)
{
  CUresult cuda_err;
  const char * cuda_err_description;
  int num_coefficients = _coefficients->size();
  const std::vector<int>& offsets = _coefficients->offsets();

  // Get the number of cells in the mesh
  std::shared_ptr<const mesh::Mesh> mesh = form->mesh();
  const int tdim = mesh->topology().dim();
  _num_cells = mesh->topology().index_map(tdim)->size_local()
    + mesh->topology().index_map(tdim)->num_ghosts();
  _num_packed_coefficient_values_per_cell = offsets.back();

  // Allocate device-side storage for number of dofs per cell and
  // pointers to the dofs per cell for each coefficient
  if (num_coefficients > 0) {
    std::vector<int> dofmaps_num_dofs_per_cell(num_coefficients);
    std::vector<CUdeviceptr> dofmaps_dofs_per_cell(num_coefficients);
    for (int i = 0; i < num_coefficients; i++) {
      const fem::CUDADofMap* cuda_dofmap =
        _coefficients->get(i)->function_space()->cuda_dofmap().get();
      dofmaps_num_dofs_per_cell[i] = cuda_dofmap->num_dofs_per_cell();
      dofmaps_dofs_per_cell[i] = cuda_dofmap->dofs_per_cell();
    }

    size_t dofmaps_num_dofs_per_cell_size = num_coefficients * sizeof(int);
    cuda_err = cuMemAlloc(
      &_dofmaps_num_dofs_per_cell, dofmaps_num_dofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    cuda_err = cuMemcpyHtoD(
      _dofmaps_num_dofs_per_cell, dofmaps_num_dofs_per_cell.data(),
      dofmaps_num_dofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    size_t dofmaps_dofs_per_cell_size = num_coefficients * sizeof(CUdeviceptr);
    cuda_err = cuMemAlloc(
      &_dofmaps_dofs_per_cell, dofmaps_dofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    cuda_err = cuMemcpyHtoD(
      _dofmaps_dofs_per_cell, dofmaps_dofs_per_cell.data(), dofmaps_dofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dofmaps_dofs_per_cell);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for offsets to and pointers to
  // coefficient values and copy to device
  if (num_coefficients > 0) {
    size_t coefficient_values_offsets_size = offsets.size() * sizeof(int);
    cuda_err = cuMemAlloc(
      &_coefficient_values_offsets, coefficient_values_offsets_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dofmaps_dofs_per_cell);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    cuda_err = cuMemcpyHtoD(
      _coefficient_values_offsets, offsets.data(), coefficient_values_offsets_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_coefficient_values_offsets);
      cuMemFree(_dofmaps_dofs_per_cell);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    size_t coefficient_values_size = num_coefficients * sizeof(CUdeviceptr);
    cuda_err = cuMemAlloc(
      &_coefficient_values, coefficient_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_coefficient_values_offsets);
      cuMemFree(_dofmaps_dofs_per_cell);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for packed coefficient values
  if (_num_cells > 0 && _num_packed_coefficient_values_per_cell > 0) {
    size_t dpacked_coefficient_values_size =
      _num_cells * _num_packed_coefficient_values_per_cell * sizeof(PetscScalar);
    cuda_err = cuMemAlloc(
      &_dpacked_coefficient_values, dpacked_coefficient_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_coefficient_values);
      cuMemFree(_coefficient_values_offsets);
      cuMemFree(_dofmaps_dofs_per_cell);
      cuMemFree(_dofmaps_num_dofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

#if 0
  // Pack coefficients into an array
  _host_coefficient_values = pack_coefficients(*form);
  _num_cells = _host_coefficient_values.rows();
  _num_packed_coefficient_values_per_cell = _host_coefficient_values.cols();

  // Allocate device-side storage for coefficient values
  if (_num_cells > 0 && _num_packed_coefficient_values_per_cell > 0) {
    size_t dpacked_coefficient_values_size =
      _num_cells * _num_packed_coefficient_values_per_cell * sizeof(PetscScalar);
    cuda_err = cuMemAlloc(
      &_dpacked_coefficient_values, dpacked_coefficient_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Register host memory as page-locked before copying
    _page_lock = page_lock;
    if (_page_lock) {
      cuda_err = cuMemHostRegister(
        _host_coefficient_values.data(), dpacked_coefficient_values_size, 0);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dpacked_coefficient_values);
        throw std::runtime_error(
          "cuMemHostRegister() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Copy coefficient values to device
    cuda_err = cuMemcpyHtoD(
      _dpacked_coefficient_values, _host_coefficient_values.data(), dpacked_coefficient_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      if (_page_lock) {
        cuda_err = cuMemHostUnregister(_host_coefficient_values.data());
        if (cuda_err != CUDA_SUCCESS) {
          const char * cuda_err_description;
          cuGetErrorString(cuda_err, &cuda_err_description);
          std::cerr << "cuMemHostUnregister() failed with " << cuda_err_description
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        }
      }
      cuMemFree(_dpacked_coefficient_values);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
#endif
}
//-----------------------------------------------------------------------------
CUDAFormCoefficients::~CUDAFormCoefficients()
{
  CUresult cuda_err;
  const char * cuda_err_description;

#if 0
  if (_page_lock) {
    cuda_err = cuMemHostUnregister(_host_coefficient_values.data());
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      std::cerr << "cuMemHostUnregister() failed with " << cuda_err_description
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    }
  }
#endif

  if (_coefficient_values)
    cuMemFree(_coefficient_values);
  if (_coefficient_values_offsets)
    cuMemFree(_coefficient_values_offsets);
  if (_dofmaps_dofs_per_cell)
    cuMemFree(_dofmaps_dofs_per_cell);
  if (_dofmaps_num_dofs_per_cell)
    cuMemFree(_dofmaps_num_dofs_per_cell);
  if (_dpacked_coefficient_values)
    cuMemFree(_dpacked_coefficient_values);
}
//-----------------------------------------------------------------------------
CUDAFormCoefficients::CUDAFormCoefficients(
  CUDAFormCoefficients&& form_coefficients)
  : _coefficients(form_coefficients._coefficients)
  , _dofmaps_num_dofs_per_cell(form_coefficients._dofmaps_num_dofs_per_cell)
  , _dofmaps_dofs_per_cell(form_coefficients._dofmaps_dofs_per_cell)
  , _coefficient_values_offsets(form_coefficients._coefficient_values_offsets)
  , _coefficient_values(form_coefficients._coefficient_values)
  , _num_cells(form_coefficients._num_cells)
  , _num_packed_coefficient_values_per_cell(form_coefficients._num_packed_coefficient_values_per_cell)
  , _page_lock(form_coefficients._page_lock)
  , _dpacked_coefficient_values(form_coefficients._dpacked_coefficient_values)
{
  form_coefficients._coefficients = nullptr;
  form_coefficients._dofmaps_num_dofs_per_cell = 0;
  form_coefficients._dofmaps_dofs_per_cell = 0;
  form_coefficients._coefficient_values_offsets = 0;
  form_coefficients._coefficient_values = 0;
  form_coefficients._num_cells = 0;
  form_coefficients._num_packed_coefficient_values_per_cell = 0;
  form_coefficients._page_lock = false;
  std::swap(_host_coefficient_values, form_coefficients._host_coefficient_values);
  form_coefficients._dpacked_coefficient_values = 0;
}
//-----------------------------------------------------------------------------
CUDAFormCoefficients& CUDAFormCoefficients::operator=(
  CUDAFormCoefficients&& form_coefficients)
{
  _coefficients = form_coefficients._coefficients;
  _dofmaps_num_dofs_per_cell = form_coefficients._dofmaps_num_dofs_per_cell;
  _dofmaps_dofs_per_cell = form_coefficients._dofmaps_dofs_per_cell;
  _coefficient_values_offsets = form_coefficients._coefficient_values_offsets;
  _coefficient_values = form_coefficients._coefficient_values;
  _num_cells = form_coefficients._num_cells;
  _num_packed_coefficient_values_per_cell = form_coefficients._num_packed_coefficient_values_per_cell;
  _page_lock = form_coefficients._page_lock;
  std::swap(_host_coefficient_values, form_coefficients._host_coefficient_values);
  _dpacked_coefficient_values = form_coefficients._dpacked_coefficient_values;
  form_coefficients._coefficients = nullptr;
  form_coefficients._dofmaps_num_dofs_per_cell = 0;
  form_coefficients._dofmaps_dofs_per_cell = 0;
  form_coefficients._coefficient_values_offsets = 0;
  form_coefficients._coefficient_values = 0;
  form_coefficients._num_cells = 0;
  form_coefficients._num_packed_coefficient_values_per_cell = 0;
  form_coefficients._page_lock = false;
  form_coefficients._dpacked_coefficient_values = 0;
  return *this;
}
//-----------------------------------------------------------------------------
void CUDAFormCoefficients::copy_coefficients_to_device(
  const CUDA::Context& cuda_context)
{
  for (int i = 0; i < _coefficients->size(); i++) {
    // TODO: We should probably try to avoid a const_cast here
    la::CUDAVector& cuda_x = const_cast<la::CUDAVector&>(
      _coefficients->get(i)->cuda_vector(cuda_context));
    cuda_x.copy_vector_values_to_device(cuda_context);
  }
}
//-----------------------------------------------------------------------------
void CUDAFormCoefficients::update_coefficient_values() const
{
  CUresult cuda_err;
  const char * cuda_err_description;

#if 0
  // Pack coefficients into an array
  const std::vector<int>& offsets = _coefficients->offsets();
  std::vector<const fem::DofMap*> dofmaps(_coefficients->size());
  for (int i = 0; i < _coefficients->size(); i++)
    dofmaps[i] = _coefficients->get(i)->function_space()->dofmap().get();

  std::vector<const PetscScalar*> v(_coefficients->size(), nullptr);
  std::vector<Vec> x(_coefficients->size(), nullptr);
  std::vector<Vec> x_local(_coefficients->size(), nullptr);
  for (std::size_t i = 0; i < v.size(); i++) {
    x[i] = _coefficients->get(i)->vector().vec();
    VecGhostGetLocalForm(x[i], &x_local[i]);
    if (x_local[i]) VecGetArrayRead(x_local[i], &v[i]);
    else VecGetArrayRead(x[i], &v[i]);
  }

  if (_coefficients->size() > 0) {
    for (int cell = 0; cell < _num_cells; cell++) {
      for (std::size_t coeff = 0; coeff < dofmaps.size(); coeff++) {
        Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::ConstSegmentReturnType dofs =
          dofmaps[coeff]->cell_dofs(cell);
        const PetscScalar* _v = v[coeff];
        for (Eigen::Index k = 0; k < dofs.size(); k++)
          _host_coefficient_values(cell, k + offsets[coeff]) = _v[dofs[k]];
      }
    }
  }

  for (std::size_t i = 0; i < v.size(); i++) {
    if (x_local[i]) VecRestoreArrayRead(x_local[i], &v[i]);
    else VecRestoreArrayRead(x[i], &v[i]);
    VecGhostRestoreLocalForm(x[i], &x_local[i]);
  }

  // Copy coefficient values to device
  if (_num_cells > 0 && _num_packed_coefficient_values_per_cell > 0) {
    size_t dpacked_coefficient_values_size =
      _num_cells * _num_packed_coefficient_values_per_cell * sizeof(PetscScalar);
    cuda_err = cuMemcpyHtoD(
      _dpacked_coefficient_values, _host_coefficient_values.data(), dpacked_coefficient_values_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_dpacked_coefficient_values);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
#endif
}
//-----------------------------------------------------------------------------
#endif
