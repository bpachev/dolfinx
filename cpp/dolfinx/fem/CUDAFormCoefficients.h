// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/CUDADofMap.h>
#include <dolfinx/la/CUDAVector.h>
#endif
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/common/IndexMap.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <petscvec.h>

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace fem {

/// A wrapper for a form coefficient with data that is stored in the
/// device memory of a CUDA device.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDAFormCoefficients
{
public:
  /// Scalar Type
  using scalar_type = T;

  /// Create an empty collection coefficient values
  //-----------------------------------------------------------------------------
  CUDAFormCoefficients()
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

  /// Create a collection coefficient values from a given form
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form The variational form whose coefficients are used
  /// @param[in] page_lock Whether or not to use page-locked memory
  ///                      for host-side arrays
  //-----------------------------------------------------------------------------
  CUDAFormCoefficients(
    const CUDA::Context& cuda_context,
    Form<T,U>* form,
    bool page_lock=false)
    : _coefficients(form->coefficients())
    , _dofmaps_num_dofs_per_cell(0)
    , _dofmaps_dofs_per_cell(0)
    , _coefficient_values_offsets(0)
    , _coefficient_values(0)
    , _num_cells()
    , _num_packed_coefficient_values_per_cell()
    , _page_lock(page_lock)
    , _host_coefficient_values()
    , _dpacked_coefficient_values(0)
  {
    CUresult cuda_err;
    const char * cuda_err_description;
    int num_coefficients = _coefficients.size();
    std::vector<int> offsets = {0};
    for (const auto & c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      offsets.push_back(offsets.back() + c->function_space()->element()->space_dimension());
    }

    // Get the number of cells in the mesh
    std::shared_ptr<const mesh::Mesh<U>> mesh = form->mesh();
    const int tdim = mesh->topology()->dim();
    _num_cells = mesh->topology()->index_map(tdim)->size_local()
      + mesh->topology()->index_map(tdim)->num_ghosts();
    _num_packed_coefficient_values_per_cell = offsets.back();

    // Allocate device-side storage for number of dofs per cell and
    // pointers to the dofs per cell for each coefficient
    if (num_coefficients > 0) {
      std::vector<int> dofmaps_num_dofs_per_cell(num_coefficients);
      std::vector<CUdeviceptr> dofmaps_dofs_per_cell(num_coefficients);
      for (int i = 0; i < num_coefficients; i++) {
        const fem::CUDADofMap* cuda_dofmap =
          _coefficients[i]->function_space()->cuda_dofmap().get();
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

  }

  /// Destructor
  ~CUDAFormCoefficients()
  {
    CUresult cuda_err;
    const char * cuda_err_description;
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

  /// Copy constructor
  /// @param[in] form_coefficient The object to be copied
  CUDAFormCoefficients(const CUDAFormCoefficients& form_coefficient) = delete;

  /// Assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(const CUDAFormCoefficients& form_coefficient) = delete;


  /// Get the number of mesh cells that the coefficient applies to
  int32_t num_coefficients() const { return _coefficients.size(); }

  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients() { return _coefficients; };

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

  //-----------------------------------------------------------------------------
  /// Move constructor
  /// @param[in] form_coefficient The object to be moved
  CUDAFormCoefficients(
    CUDAFormCoefficients&& form_coefficients)
    : _coefficients(std::move(form_coefficients._coefficients))
    , _dofmaps_num_dofs_per_cell(form_coefficients._dofmaps_num_dofs_per_cell)
    , _dofmaps_dofs_per_cell(form_coefficients._dofmaps_dofs_per_cell)
    , _coefficient_values_offsets(form_coefficients._coefficient_values_offsets)
    , _coefficient_values(form_coefficients._coefficient_values)
    , _num_cells(form_coefficients._num_cells)
    , _num_packed_coefficient_values_per_cell(form_coefficients._num_packed_coefficient_values_per_cell)
    , _page_lock(form_coefficients._page_lock)
    , _dpacked_coefficient_values(form_coefficients._dpacked_coefficient_values)
  {
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
  /// Move assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(
    CUDAFormCoefficients&& form_coefficients)
  {
    _coefficients = std::move(form_coefficients._coefficients);
    _dofmaps_num_dofs_per_cell = form_coefficients._dofmaps_num_dofs_per_cell;
    _dofmaps_dofs_per_cell = form_coefficients._dofmaps_dofs_per_cell;
    _coefficient_values_offsets = form_coefficients._coefficient_values_offsets;
    _coefficient_values = form_coefficients._coefficient_values;
    _num_cells = form_coefficients._num_cells;
    _num_packed_coefficient_values_per_cell = form_coefficients._num_packed_coefficient_values_per_cell;
    _page_lock = form_coefficients._page_lock;
    std::swap(_host_coefficient_values, form_coefficients._host_coefficient_values);
    _dpacked_coefficient_values = form_coefficients._dpacked_coefficient_values;
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
  /// Update the coefficient values by copying values from host to device.
  /// This must be called before packing the coefficients, if they
  /// have been changed on the host.
  void copy_coefficients_to_device(
    const CUDA::Context& cuda_context)
  {
    for (int i = 0; i < _coefficients.size(); i++) {
      std::shared_ptr<const la::Vector<T>> x = _coefficients[i]->x();
      const_cast<la::Vector<T>*>(x.get())->to_device(cuda_context);
    }
  }

  //-----------------------------------------------------------------------------
  /// Update the coefficient values by copying values from host to device
  void update_coefficient_values() const
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
