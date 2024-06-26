// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "numpy_dtype.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#ifdef HAS_CUDA_TOOLKIT
#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/CUDAAssembler.h>
#include <dolfinx/fem/CUDADirichletBC.h>
#include <dolfinx/fem/CUDADofMap.h>
#include <dolfinx/fem/CUDAForm.h>
#include <dolfinx/fem/CUDAFormIntegral.h>
#include <dolfinx/fem/CUDAFormConstants.h>
#include <dolfinx/fem/CUDAFormCoefficients.h>
#include <dolfinx/la/CUDAMatrix.h>
#include <dolfinx/la/CUDAVector.h>
#include <dolfinx/mesh/CUDAMesh.h>
//#include <petscmat.h>
//#include <petscvec.h>
#include "caster_petsc.h"
#include <petsc4py/petsc4py.h>
#include <petscis.h>
#include <cuda.h>
#endif
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <ufcx.h>
#include <utility>

namespace nb = nanobind;

namespace
{
template <typename T, typename = void>
struct geom_type
{
  typedef T value_type;
};
template <typename T>
struct geom_type<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type value_type;
};

template <typename T>
void declare_function_space(nb::module_& m, std::string type)
{
  {
    std::string pyclass_name = "FunctionSpace_" + type;
    nb::class_<dolfinx::fem::FunctionSpace<T>>(m, pyclass_name.c_str(),
                                               "Finite element function space")
        .def(nb::init<std::shared_ptr<const dolfinx::mesh::Mesh<T>>,
                      std::shared_ptr<const dolfinx::fem::FiniteElement<T>>,
                      std::shared_ptr<const dolfinx::fem::DofMap>,
                      std::vector<std::size_t>>(),
             nb::arg("mesh"), nb::arg("element"), nb::arg("dofmap"),
             nb::arg("value_shape"))
        .def("collapse", &dolfinx::fem::FunctionSpace<T>::collapse)
        .def("component", &dolfinx::fem::FunctionSpace<T>::component)
        .def("contains", &dolfinx::fem::FunctionSpace<T>::contains,
             nb::arg("V"))
        .def_prop_ro("element", &dolfinx::fem::FunctionSpace<T>::element)
        .def_prop_ro("mesh", &dolfinx::fem::FunctionSpace<T>::mesh)
        .def_prop_ro("dofmap", &dolfinx::fem::FunctionSpace<T>::dofmap)
        .def_prop_ro(
            "value_shape",
            [](const dolfinx::fem::FunctionSpace<T>& self)
            {
              std::span<const std::size_t> vshape = self.value_shape();
              return nb::ndarray<const std::size_t, nb::numpy>(vshape.data(),
                                                               {vshape.size()});
            },
            nb::rv_policy::reference_internal)
        .def("sub", &dolfinx::fem::FunctionSpace<T>::sub, nb::arg("component"))
        .def("tabulate_dof_coordinates",
             [](const dolfinx::fem::FunctionSpace<T>& self)
             {
               std::vector x = self.tabulate_dof_coordinates(false);
               return dolfinx_wrappers::as_nbarray(std::move(x),
                                                   {x.size() / 3, 3});
             });
  }

  {
    std::string pyclass_name = "FiniteElement_" + type;
    nb::class_<dolfinx::fem::FiniteElement<T>>(m, pyclass_name.c_str(),
                                               "Finite element object")
        .def(
            "__init__",
            [](dolfinx::fem::FiniteElement<T>* self,
               std::uintptr_t ufcx_element)
            {
              ufcx_finite_element* p
                  = reinterpret_cast<ufcx_finite_element*>(ufcx_element);
              new (self) dolfinx::fem::FiniteElement<T>(*p);
            },
            nb::arg("ufcx_element"))
        .def("__eq__", &dolfinx::fem::FiniteElement<T>::operator==)
        .def_prop_ro("dtype", [](const dolfinx::fem::FiniteElement<T>&)
                     { return dolfinx_wrappers::numpy_dtype<T>(); })
        .def_prop_ro("basix_element",
                     &dolfinx::fem::FiniteElement<T>::basix_element,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("num_sub_elements",
                     &dolfinx::fem::FiniteElement<T>::num_sub_elements)
        .def("interpolation_points",
             [](const dolfinx::fem::FiniteElement<T>& self)
             {
               auto [X, shape] = self.interpolation_points();
               return dolfinx_wrappers::as_nbarray(std::move(X), shape.size(),
                                                   shape.data());
             })
        .def_prop_ro("interpolation_ident",
                     &dolfinx::fem::FiniteElement<T>::interpolation_ident)
        .def_prop_ro("space_dimension",
                     &dolfinx::fem::FiniteElement<T>::space_dimension)
        .def(
            "pre_apply_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_dof_transformation(std::span(x.data(), x.size()),
                                                cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def(
            "pre_apply_transpose_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_transpose_dof_transformation(
                  std::span(x.data(), x.size()), cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def(
            "pre_apply_inverse_transpose_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_inverse_transpose_dof_transformation(
                  std::span(x.data(), x.size()), cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def(
            "pre_apply_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_dof_transformation(
                  std::span((std::complex<T>*)x.data(), x.size()),
                  cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def(
            "pre_apply_transpose_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_transpose_dof_transformation(
                  std::span((std::complex<T>*)x.data(), x.size()),
                  cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def(
            "pre_apply_inverse_transpose_dof_transformation",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               std::uint32_t cell_permutation, int dim)
            {
              self.pre_apply_inverse_transpose_dof_transformation(
                  std::span(x.data(), x.shape(0) * x.shape(1)),
                  cell_permutation, dim);
            },
            nb::arg("x"), nb::arg("cell_permutation"), nb::arg("dim"))
        .def_prop_ro("needs_dof_transformations",
                     &dolfinx::fem::FiniteElement<T>::needs_dof_transformations)
        .def("signature", &dolfinx::fem::FiniteElement<T>::signature);
  }
}

// Declare DirichletBC objects for type T
template <typename T>
void declare_objects(nb::module_& m, const std::string& type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  // dolfinx::fem::DirichletBC
  std::string pyclass_name = std::string("DirichletBC_") + type;
  nb::class_<dolfinx::fem::DirichletBC<T, U>> dirichletbc(
      m, pyclass_name.c_str(),
      "Object for representing Dirichlet (essential) boundary "
      "conditions");

  dirichletbc
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             nb::ndarray<const T, nb::c_contig> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            std::vector<std::size_t> shape(g.shape_ptr(),
                                           g.shape_ptr() + g.ndim());
            auto _g = std::make_shared<dolfinx::fem::Constant<T>>(
                std::span(g.data(), g.size()), shape);
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                _g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(), nb::arg("V"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Constant<T>> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(), nb::arg("V"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Function<T, U>> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs)
          {
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                g, std::vector(dofs.data(), dofs.data() + dofs.size()));
          },
          nb::arg("g").noconvert(), nb::arg("dofs"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Function<T, U>> g,
             std::array<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>, 2>
                 V_g_dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            std::array dofs
                = {std::vector(V_g_dofs[0].data(),
                               V_g_dofs[0].data() + V_g_dofs[0].size()),
                   std::vector(V_g_dofs[1].data(),
                               V_g_dofs[1].data() + V_g_dofs[1].size())};
            new (bc) dolfinx::fem::DirichletBC(g, std::move(dofs), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(),
          nb::arg("V").noconvert())
      .def_prop_ro("dtype", [](const dolfinx::fem::Function<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def("dof_indices",
           [](const dolfinx::fem::DirichletBC<T, U>& self)
           {
             auto [dofs, owned] = self.dof_indices();
             return std::pair(nb::ndarray<const std::int32_t, nb::numpy>(
                                  dofs.data(), {dofs.size()}),
                              owned);
           })
      .def_prop_ro("function_space",
                   &dolfinx::fem::DirichletBC<T, U>::function_space)
      .def_prop_ro("value", &dolfinx::fem::DirichletBC<T, U>::value);

  // dolfinx::fem::Function
  std::string pyclass_name_function = std::string("Function_") + type;
  nb::class_<dolfinx::fem::Function<T, U>>(m, pyclass_name_function.c_str(),
                                           "A finite element function")
      .def(nb::init<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>(),
           "Create a function on the given function space")
      .def(nb::init<std::shared_ptr<dolfinx::fem::FunctionSpace<U>>,
                    std::shared_ptr<dolfinx::la::Vector<T>>>())
      .def_rw("name", &dolfinx::fem::Function<T, U>::name)
      .def("sub", &dolfinx::fem::Function<T, U>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<T, U>::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> f,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                      {1, f.size()},
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f"), nb::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> f,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                      {f.shape(0), f.shape(1)},
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f"), nb::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             dolfinx::fem::Function<T, U>& u,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             const std::tuple<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>,
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>,
                 nb::ndarray<const U, nb::ndim<1>, nb::c_contig>,
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>&
                 interpolation_data)
          {
            std::tuple<std::span<const std::int32_t>,
                       std::span<const std::int32_t>, std::span<const U>,
                       std::span<const std::int32_t>>
                _interpolation_data(
                    std::span<const std::int32_t>(
                        std::get<0>(interpolation_data).data(),
                        std::get<0>(interpolation_data).size()),
                    std::span<const std::int32_t>(
                        std::get<1>(interpolation_data).data(),
                        std::get<1>(interpolation_data).size()),
                    std::span<const U>(std::get<2>(interpolation_data).data(),
                                       std::get<2>(interpolation_data).size()),
                    std::span<const std::int32_t>(
                        std::get<3>(interpolation_data).data(),
                        std::get<3>(interpolation_data).size()));
            self.interpolate(u, std::span(cells.data(), cells.size()),
                             _interpolation_data);
          },
          nb::arg("u"), nb::arg("cells"), nb::arg("nmm_interpolation_data"),
          "Interpolate a finite element function")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<T, U>& self, std::uintptr_t addr,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            assert(self.function_space());
            auto element = self.function_space()->element();
            assert(element);

            assert(self.function_space()->mesh());
            const std::vector<U> x = dolfinx::fem::interpolation_coords(
                *element, self.function_space()->mesh()->geometry(),
                std::span(cells.data(), cells.size()));

            const int gdim = self.function_space()->mesh()->geometry().dim();

            // Compute value size
            auto vshape = self.function_space()->value_shape();
            std::size_t value_size = std::reduce(vshape.begin(), vshape.end(),
                                                 1, std::multiplies{});

            std::array<std::size_t, 2> shape{value_size, x.size() / 3};
            std::vector<T> values(shape[0] * shape[1]);
            std::function<void(T*, int, int, const U*)> f
                = reinterpret_cast<void (*)(T*, int, int, const U*)>(addr);
            f(values.data(), shape[1], shape[0], x.data());
            dolfinx::fem::interpolate(self, std::span<const T>(values), shape,
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f_ptr"), nb::arg("cells"),
          "Interpolate using a pointer to an expression with a C signature")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             const dolfinx::fem::Expression<T, U>& expr,
             nb::ndarray<const std::int32_t, nb::c_contig> cells)
          { self.interpolate(expr, std::span(cells.data(), cells.size())); },
          nb::arg("expr"), nb::arg("cells"),
          "Interpolate an Expression on a set of cells")
      .def_prop_ro(
          "x", nb::overload_cast<>(&dolfinx::fem::Function<T, U>::x),
          "Return the vector associated with the finite element Function")
      .def(
          "eval",
          [](const dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const U, nb::ndim<2>, nb::c_contig> x,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             nb::ndarray<T, nb::ndim<2>, nb::c_contig> u)
          {
            // TODO: handle 1d case
            self.eval(std::span(x.data(), x.size()), {x.shape(0), x.shape(1)},
                      std::span(cells.data(), cells.size()),
                      std::span<T>(u.data(), u.size()),
                      {u.shape(0), u.shape(1)});
          },
          nb::arg("x"), nb::arg("cells"), nb::arg("values"),
          "Evaluate Function")
      .def_prop_ro("function_space",
                   &dolfinx::fem::Function<T, U>::function_space);

  // dolfinx::fem::Constant
  std::string pyclass_name_constant = std::string("Constant_") + type;
  nb::class_<dolfinx::fem::Constant<T>>(
      m, pyclass_name_constant.c_str(),
      "Value constant with respect to integration domain")
      .def(
          "__init__",
          [](dolfinx::fem::Constant<T>* cp,
             nb::ndarray<const T, nb::c_contig> c)
          {
            std::vector<std::size_t> shape(c.shape_ptr(),
                                           c.shape_ptr() + c.ndim());
            new (cp)
                dolfinx::fem::Constant<T>(std::span(c.data(), c.size()), shape);
          },
          nb::arg("c").noconvert(), "Create a constant from a value array")
      .def_prop_ro("dtype", [](const dolfinx::fem::Constant<T>)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro(
          "value",
          [](dolfinx::fem::Constant<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(
                self.value.data(), self.shape.size(), self.shape.data());
          },
          nb::rv_policy::reference_internal);

  // dolfinx::fem::Expression
  std::string pyclass_name_expr = std::string("Expression_") + type;
  nb::class_<dolfinx::fem::Expression<T, U>>(m, pyclass_name_expr.c_str(),
                                             "An Expression")
      .def(
          "__init__",
          [](dolfinx::fem::Expression<T, U>* ex,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             nb::ndarray<const U, nb::ndim<2>, nb::c_contig> X,
             std::uintptr_t fn_addr, const std::vector<int>& value_shape,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>
                 argument_function_space)
          {
            auto tabulate_expression_ptr
                = (void (*)(T*, const T*, const T*,
                            const typename geom_type<T>::value_type*,
                            const int*, const std::uint8_t*))fn_addr;
            new (ex) dolfinx::fem::Expression<T, U>(
                coefficients, constants, std::span(X.data(), X.size()),
                {X.shape(0), X.shape(1)}, tabulate_expression_ptr, value_shape,
                argument_function_space);
          },
          nb::arg("coefficients"), nb::arg("constants"), nb::arg("X"),
          nb::arg("fn"), nb::arg("value_shape"),
          nb::arg("argument_function_space"))
      .def(
          "eval",
          [](const dolfinx::fem::Expression<T, U>& self,
             const dolfinx::mesh::Mesh<U>& mesh,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             nb::ndarray<T, nb::ndim<2>, nb::c_contig> values)
          {
            std::span<T> foo(values.data(), values.size());
            self.eval(mesh, std::span(cells.data(), cells.size()), foo,
                      {values.shape(0), values.shape(1)});
          },
          nb::arg("mesh"), nb::arg("active_cells"), nb::arg("values"))
      .def("X",
           [](const dolfinx::fem::Expression<T, U>& self)
           {
             auto [X, shape] = self.X();
             return dolfinx_wrappers::as_nbarray(std::move(X), shape.size(),
                                                 shape.data());
           })
      .def_prop_ro("dtype", [](const dolfinx::fem::Expression<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("value_size", &dolfinx::fem::Expression<T, U>::value_size)
      .def_prop_ro("value_shape", &dolfinx::fem::Expression<T, U>::value_shape);

  std::string pymethod_create_expression
      = std::string("create_expression_") + type;
  m.def(
      pymethod_create_expression.c_str(),
      [](const std::uintptr_t expression,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>
             argument_function_space)
      {
        const ufcx_expression* p
            = reinterpret_cast<const ufcx_expression*>(expression);
        return dolfinx::fem::create_expression<T, U>(
            *p, coefficients, constants, argument_function_space);
      },
      nb::arg("expression"), nb::arg("coefficients"), nb::arg("constants"),
      nb::arg("argument_function_space").none(),
      "Create Expression from a pointer to ufc_form.");
}

template <typename T>
void declare_form(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  // dolfinx::fem::Form
  std::string pyclass_name_form = std::string("Form_") + type;
  nb::class_<dolfinx::fem::Form<T, U>>(m, pyclass_name_form.c_str(),
                                       "Variational form object")
      .def(
          "__init__",
          [](dolfinx::fem::Form<T, U>* fp,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
             const std::map<dolfinx::fem::IntegralType,
                            std::vector<std::tuple<
                                int, std::uintptr_t,
                                nb::ndarray<const std::int32_t, nb::ndim<1>,
                                            nb::c_contig>>>>& integrals,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             bool needs_permutation_data,
             const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                            nb::ndarray<const std::int32_t, nb::ndim<1>,
                                        nb::c_contig>>& entity_maps,
             std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
          {
            std::map<dolfinx::fem::IntegralType,
                     std::vector<dolfinx::fem::integral_data<T>>>
                _integrals;

            // Loop over kernel for each entity type
            for (auto& [type, kernels] : integrals)
            {
              for (auto& [id, ptr, e] : kernels)
              {
                auto kn_ptr
                    = (void (*)(T*, const T*, const T*,
                                const typename geom_type<T>::value_type*,
                                const int*, const std::uint8_t*))ptr;
                _integrals[type].emplace_back(
                    id, kn_ptr,
                    std::span<const std::int32_t>(e.data(), e.size()));
              }
            }
            
	    std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                     std::span<const int32_t>>
                _entity_maps;
            for (auto& [msh, map] : entity_maps)
              _entity_maps.emplace(msh, std::span(map.data(), map.size()));
#ifdef HAS_CUDA_TOOLKIT
            std::map<dolfinx::fem::IntegralType,
                      std::vector<std::tuple<
                          int,
                          std::function<void(int*, const char***, const char***,
                                             const char**, const char**)>>>> _cuda_integrals;
	    new (fp) dolfinx::fem::Form<T, U>(
                spaces, std::move(_integrals), _cuda_integrals, coefficients, constants,
                needs_permutation_data, _entity_maps, mesh);
#else
	    new (fp) dolfinx::fem::Form<T, U>(
                spaces, std::move(_integrals), coefficients, constants,
                needs_permutation_data, _entity_maps, mesh);
#endif
          },
          nb::arg("spaces"), nb::arg("integrals"), nb::arg("coefficients"),
          nb::arg("constants"), nb::arg("need_permutation_data"),
          nb::arg("entity_maps"), nb::arg("mesh").none())
      .def(
          "__init__",
          [](dolfinx::fem::Form<T, U>* fp, std::uintptr_t form,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             const std::map<
                 dolfinx::fem::IntegralType,
                 std::vector<std::pair<
                     std::int32_t, nb::ndarray<const std::int32_t, nb::ndim<1>,
                                               nb::c_contig>>>>& subdomains,
             const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                            nb::ndarray<const std::int32_t, nb::ndim<1>,
                                        nb::c_contig>>& entity_maps,
             std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
          {
            std::map<dolfinx::fem::IntegralType,
                     std::vector<std::pair<std::int32_t,
                                           std::span<const std::int32_t>>>>
                sd;
            for (auto& [itg, data] : subdomains)
            {
              std::vector<
                  std::pair<std::int32_t, std::span<const std::int32_t>>>
                  x;
              for (auto& [id, e] : data)
                x.emplace_back(id, std::span(e.data(), e.size()));
              sd.insert({itg, std::move(x)});
            }

            std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                     std::span<const int32_t>>
                _entity_maps;
            for (auto& [msh, map] : entity_maps)
              _entity_maps.emplace(msh, std::span(map.data(), map.size()));
            ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
            new (fp)
                dolfinx::fem::Form<T, U>(dolfinx::fem::create_form_factory<T>(
                    *p, spaces, coefficients, constants, sd, _entity_maps,
                    mesh));
          },
          nb::arg("form"), nb::arg("spaces"), nb::arg("coefficients"),
          nb::arg("constants"), nb::arg("subdomains"), nb::arg("entity_maps"),
          nb::arg("mesh").none(), "Create a Form from a pointer to a ufcx_form")
      .def_prop_ro("dtype", [](const dolfinx::fem::Form<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("coefficients", &dolfinx::fem::Form<T, U>::coefficients)
      .def_prop_ro("rank", &dolfinx::fem::Form<T, U>::rank)
      .def_prop_ro("mesh", &dolfinx::fem::Form<T, U>::mesh)
      .def_prop_ro("function_spaces",
                   &dolfinx::fem::Form<T, U>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<T, U>::integral_ids)
      .def_prop_ro("integral_types", &dolfinx::fem::Form<T, U>::integral_types)
      .def_prop_ro("needs_facet_permutations",
                   &dolfinx::fem::Form<T, U>::needs_facet_permutations)
      .def(
          "domains",
          [](const dolfinx::fem::Form<T, U>& self,
             dolfinx::fem::IntegralType type, int i)
          {
            std::span<const std::int32_t> _d = self.domain(type, i);
            switch (type)
            {
            case dolfinx::fem::IntegralType::cell:
              return nb::ndarray<const std::int32_t, nb::numpy>(_d.data(),
                                                                {_d.size()});
            case dolfinx::fem::IntegralType::exterior_facet:
            {
              return nb::ndarray<const std::int32_t, nb::numpy>(
                  _d.data(), {_d.size() / 2, 2});
            }
            case dolfinx::fem::IntegralType::interior_facet:
            {
              return nb::ndarray<const std::int32_t, nb::numpy>(
                  _d.data(), {_d.size() / 4, 2, 2});
            }
            default:
              throw ::std::runtime_error("Integral type unsupported.");
            }
          },
          nb::rv_policy::reference_internal, nb::arg("type"), nb::arg("i"));

  // Form
  std::string pymethod_create_form = std::string("create_form_") + type;
  m.def(
      pymethod_create_form.c_str(),
      [](std::uintptr_t form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::map<
             dolfinx::fem::IntegralType,
             std::vector<std::pair<
                 std::int32_t, nb::ndarray<const std::int32_t, nb::c_contig>>>>&
             subdomains,
         std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
      {
        std::map<
            dolfinx::fem::IntegralType,
            std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
            sd;
        for (auto& [itg, data] : subdomains)
        {
          std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>> x;
          for (auto& [id, idx] : data)
            x.emplace_back(id, std::span(idx.data(), idx.size()));
          sd.insert({itg, std::move(x)});
        }

        ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
        return dolfinx::fem::create_form_factory<T>(*p, spaces, coefficients,
                                                    constants, sd, {}, mesh);
      },
      nb::arg("form"), nb::arg("spaces"), nb::arg("coefficients"),
      nb::arg("constants"), nb::arg("subdomains"), nb::arg("mesh"),
      "Create Form from a pointer to ufcx_form.");

  m.def("create_sparsity_pattern",
        &dolfinx::fem ::create_sparsity_pattern<T, U>, nb::arg("a"),
        "Create a sparsity pattern.");
}


#ifdef HAS_CUDA_TOOLKIT
// declare templated cuda-related objects
// we keep everything CUDA-related here, because it is purely focused on assembly
template <typename T>
void declare_cuda_templated_objects(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string formclass_name = std::string("CUDAForm_") + type;
  nb::class_<dolfinx::fem::CUDAForm<T,U>>(m, formclass_name.c_str(), "Form on GPU")
      .def(
          "__init__",
           [](dolfinx::fem::CUDAForm<T,U>* cf, const dolfinx::CUDA::Context& cuda_context,
              dolfinx::fem::Form<T,U>& form)
             {
               new (cf) dolfinx::fem::CUDAForm<T,U>(
                 cuda_context,
                 &form
               );
             }, nb::arg("context"), nb::arg("form"))
      .def(
          "compile",
          [](dolfinx::fem::CUDAForm<T,U>& cf, const dolfinx::CUDA::Context& cuda_context,
             int32_t max_threads_per_block, int32_t min_blocks_per_multiprocessor)
             {
               cf.compile(cuda_context, max_threads_per_block,
                          min_blocks_per_multiprocessor, dolfinx::fem::assembly_kernel_type::ASSEMBLY_KERNEL_GLOBAL);
             }, nb::arg("context"), nb::arg("max_threads_per_block"), nb::arg("min_blocks_per_multiprocessor"))
      .def_prop_ro("compiled", &dolfinx::fem::CUDAForm<T,U>::compiled)
      .def("to_device", &dolfinx::fem::CUDAForm<T,U>::to_device);

  std::string pyclass_name = std::string("CUDAFormIntegral_") + type;
  nb::class_<dolfinx::fem::CUDAFormIntegral<T,U>>(m, pyclass_name.c_str(),
                                                  "Form Integral on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDAFormIntegral<T,U>* ci, const dolfinx::CUDA::Context& cuda_context,
             const dolfinx::fem::Form<T,U>& form,
             dolfinx::fem::IntegralType integral_type, int i, int32_t max_threads_per_block,
             int32_t min_blocks_per_multiprocessor, int32_t num_vertices_per_cell,
             int32_t num_coordinates_per_vertex, int32_t num_dofs_per_cell0,
             int32_t num_dofs_per_cell1, enum dolfinx::fem::assembly_kernel_type assembly_kernel_type,
             const char* cudasrcdir)
             {
               bool debug=true, verbose=false;
               CUjit_target target = dolfinx::CUDA::get_cujit_target(cuda_context);
               new (ci) dolfinx::fem::CUDAFormIntegral<T,U>(
                   cuda_context, target, form, integral_type, i, max_threads_per_block,
                   min_blocks_per_multiprocessor, num_vertices_per_cell, num_coordinates_per_vertex,
                   num_dofs_per_cell0, num_dofs_per_cell1, assembly_kernel_type, debug, cudasrcdir, verbose);
             },
          nb::arg("context"), nb::arg("form"), nb::arg("integral_type"), nb::arg("i"), nb::arg("max_threads"),
          nb::arg("min_blocks"), nb::arg("num_verts"), nb::arg("num_coords"),
          nb::arg("num_dofs_per_cell0"), nb::arg("num_dofs_per_cell1"),
          nb::arg("kernel_type"), nb::arg("tmpdir"));

  pyclass_name = std::string("CUDADirichletBC_") + type;
  nb::class_<dolfinx::fem::CUDADirichletBC<T,U>>(m, pyclass_name.c_str(),
                                                 "Dirichlet BC on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDADirichletBC<T,U>* bc, const dolfinx::CUDA::Context& cuda_context,
             const dolfinx::fem::FunctionSpace<T>& V,
             const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs)
             {
               new (bc) dolfinx::fem::CUDADirichletBC<T,U>(
                   cuda_context, V, bcs);
             },
          nb::arg("context"), nb::arg("V"), nb::arg("bcs"))
      .def("update", &dolfinx::fem::CUDADirichletBC<T,U>::update, nb::arg("bcs"));

  pyclass_name = std::string("CUDAFormConstants_") + type;
  nb::class_<dolfinx::fem::CUDAFormConstants<T>>(m, pyclass_name.c_str(),
                                                 "Form Constants on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDAFormConstants<T>* fc, const dolfinx::CUDA::Context& cuda_context,
             const dolfinx::fem::Form<T>* form)
            {
              new (fc) dolfinx::fem::CUDAFormConstants<T>(cuda_context, form);
            }, nb::arg("context"), nb::arg("form")
      );

  pyclass_name = std::string("CUDAFormCoefficients_") + type;
  nb::class_<dolfinx::fem::CUDAFormCoefficients<T,U>>(m, pyclass_name.c_str(),
                                                      "Form Coefficients on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDAFormCoefficients<T,U>* fc, const dolfinx::CUDA::Context& cuda_context,
             dolfinx::fem::Form<T,U>* form)
            {
              new (fc) dolfinx::fem::CUDAFormCoefficients<T,U>(cuda_context, form);
            }, nb::arg("context"), nb::arg("form")
      );

  std::string pyclass_cumesh_name = std::string("CUDAMesh_") + type;
  nb::class_<dolfinx::mesh::CUDAMesh<T>>(m, pyclass_cumesh_name.c_str(),
                                         "Mesh object on GPU")
      .def(
          "__init__",
          [](dolfinx::mesh::CUDAMesh<T>* cumesh, const dolfinx::CUDA::Context& cuda_context,
             const dolfinx::mesh::Mesh<T>& mesh) {
            new (cumesh) dolfinx::mesh::CUDAMesh<T>(cuda_context, mesh);
          },
          nb::arg("context"), nb::arg("mesh"));
}

// Declare the nontemplated CUDA wrappers
void declare_cuda_objects(nb::module_& m)
{
  import_petsc4py();

  nb::class_<dolfinx::CUDA::Context>(m, "CUDAContext", "CUDA Context")
      .def("__init__", [](dolfinx::CUDA::Context* c) { new (c) dolfinx::CUDA::Context();});

  nb::class_<dolfinx::la::CUDAMatrix>(m, "CUDAMatrix", "Matrix object on GPU")
      .def(
          "__init__",
          [](dolfinx::la::CUDAMatrix* cumat, const dolfinx::CUDA::Context& cuda_context, Mat A) {
            new (cumat) dolfinx::la::CUDAMatrix(cuda_context, A, false, false);
          }, nb::arg("context"), nb::arg("A"))
      .def("debug_dump", &dolfinx::la::CUDAMatrix::debug_dump)
      .def("to_host",
          [](dolfinx::la::CUDAMatrix& cumat, const dolfinx::CUDA::Context& cuda_context)
          {
            //cumat.copy_matrix_values_to_host(cuda_context);
            cumat.apply(MAT_FINAL_ASSEMBLY);
          }, nb::arg("cuda_context"), "Copy matrix values to host and finalize assembly.")
      .def_prop_ro("mat",
          [](dolfinx::la::CUDAMatrix& cumat) {
            Mat A = cumat.mat();
            PyObject* obj = PyPetscMat_New(A);
            PetscObjectDereference((PetscObject)A);
            return nb::borrow(obj);
          });

  nb::class_<dolfinx::la::CUDAVector>(m, "CUDAVector", "Vector object on GPU")
      .def(
          "__init__",
          [](dolfinx::la::CUDAVector* cuvec, const dolfinx::CUDA::Context& cuda_context, Vec x) {
            new (cuvec) dolfinx::la::CUDAVector(cuda_context, x, false, false);
          }, nb::arg("context"), nb::arg("x"))
      .def("to_host", &dolfinx::la::CUDAVector::copy_vector_values_to_host)
      .def_prop_ro("vector",
          [](dolfinx::la::CUDAVector& cuvec) {
            Vec b = cuvec.vector();
            PyObject* obj = PyPetscVec_New(b);
            PetscObjectDereference((PetscObject)b);
            return nb::borrow(obj);
          });

  nb::class_<dolfinx::fem::CUDADofMap>(m, "CUDADofMap", "DofMap object on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDADofMap* cudofmap, const dolfinx::CUDA::Context& cuda_context, const dolfinx::fem::DofMap& dofmap) {
            new (cudofmap) dolfinx::fem::CUDADofMap(cuda_context, dofmap);
          }, nb::arg("context"), nb::arg("dofmap"));

  auto assembler_class = nb::class_<dolfinx::fem::CUDAAssembler>(m, "CUDAAssembler", "Assembler object")
      .def(
          "__init__",
          [](dolfinx::fem::CUDAAssembler* assembler, const dolfinx::CUDA::Context& cuda_context,
             const char* cudasrcdir) {
            bool debug = true, verbose = false;
            CUjit_target target = dolfinx::CUDA::get_cujit_target(cuda_context);
            new (assembler) dolfinx::fem::CUDAAssembler(cuda_context, target, debug, cudasrcdir, verbose);
          }, nb::arg("context"), nb::arg("cudasrcdir"));
  
}

// Declare some functions that 
// simplify the process of CUDA assembly in Python
template <typename T, typename U>
void declare_cuda_funcs(nb::module_& m)
{

  m.def("copy_function_to_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::Function<T, U>& f)
        {
          f.x()->to_device(cuda_context);
        },
        nb::arg("context"), nb::arg("f"), "Copy function data to GPU"); 

  m.def("copy_function_space_to_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::FunctionSpace<T>& V)
        {
          V.create_cuda_dofmap(cuda_context);
        },
        nb::arg("context"), nb::arg("V"), "Copy function space dofmap to GPU");

  m.def("pack_coefficients",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
          dolfinx::fem::CUDAForm<T,U>& cuda_form)
        {
          assembler.pack_coefficients(cuda_context, cuda_form.coefficients());
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("cuda_form"), "Pack form coefficients on device.");

  m.def("pack_coefficients",
       [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
          dolfinx::fem::CUDAForm<T,U>& cuda_form, std::vector<std::shared_ptr<dolfinx::fem::Function<T,U>>>& coefficients)
       {
         if (!coefficients.size()) {
           // nothing to do
           return;
         }

         assembler.pack_coefficients(cuda_context, cuda_form.coefficients(), coefficients);
       },
       nb::arg("context"), nb::arg("assembler"), nb::arg("cuda_form"), nb::arg("coefficients"),
       "Pack a given subset of form coefficients on device");

  m.def("assemble_matrix_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::fem::CUDAForm<T,U>& cuda_form, dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAMatrix& cuda_A, dolfinx::fem::CUDADirichletBC<T,U>& cuda_bc0,
           dolfinx::fem::CUDADirichletBC<T,U>& cuda_bc1) {
          // Extract constant and coefficient data
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap0 =
            cuda_form.dofmap(0);
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap1 =
            cuda_form.dofmap(1);

          // not needed for global assembly kernel
          /*assembler.compute_lookup_tables(
            cuda_context, *cuda_dofmap0, *cuda_dofmap1,
            cuda_bc0, cuda_bc1, cuda_a_form_integrals, cuda_A, false);*/
          assembler.zero_matrix_entries(cuda_context, cuda_A);
          assembler.assemble_matrix(
            cuda_context, cuda_mesh, *cuda_dofmap0, *cuda_dofmap1,
            cuda_bc0, cuda_bc1, cuda_form.integrals(),
            cuda_form.constants(), cuda_form.coefficients(),
            cuda_A, false);
          assembler.add_diagonal(cuda_context, cuda_A, cuda_bc0);
          // TODO determine if this copy is actually needed. . .
          // This unfortunately may be the case with PETSc matrices
          cuda_A.copy_matrix_values_to_host(cuda_context);
          //cuda_A.apply(MAT_FINAL_ASSEMBLY);
 
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"),
        nb::arg("A"), nb::arg("bcs0"), nb::arg("bcs1"), "Assemble matrix on GPU."
  );

  m.def("assemble_vector_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::fem::CUDAForm<T,U>& cuda_form,
           dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAVector& cuda_b)
         {
          
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap0 =
            cuda_form.dofmap(0);
          
          assembler.zero_vector_entries(cuda_context, cuda_b);
          assembler.assemble_vector(
             cuda_context, cuda_mesh, *cuda_dofmap0,
             cuda_form.integrals(), cuda_form.constants(),
             cuda_form.coefficients(), cuda_b, false);
          
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"), nb::arg("b"),
        "Assemble vector on GPU."
  );

  m.def("apply_lifting_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           std::vector<std::shared_ptr<dolfinx::fem::CUDAForm<T,U>>>& cuda_form,
           dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAVector& cuda_b,
           std::vector<std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>>>& bcs,
           std::vector<std::shared_ptr<dolfinx::la::Vector<T>>>& cuda_x0,
           float scale)
        {
          bool missing_x0 = (cuda_x0.size() == 0);
          if (bcs.size() != cuda_form.size()) throw std::runtime_error("Number of bc lists must match number of forms!");
          if (!missing_x0 && (cuda_x0.size() != cuda_form.size())) 
            throw std::runtime_error("Number of x0 vectors must match number of forms!");

          for (size_t i = 0; i < cuda_form.size(); i++) {
            auto form = cuda_form[i];
            std::shared_ptr<dolfinx::la::Vector<T>> x0 = (missing_x0) ? nullptr : cuda_x0[i]; 
            assembler.lift_bc(
              cuda_context, cuda_mesh, *form->dofmap(0), *form->dofmap(1),
              form->integrals(), form->constants(), form->coefficients(),
              *bcs[i], x0, scale, cuda_b, false
            );
          }

        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("x0"), nb::arg("scale"),
        "Apply lifting on GPU" 
  );

  m.def("set_bc_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::la::CUDAVector& cuda_b,
           std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>> bc0,
           std::shared_ptr<dolfinx::la::Vector<T>> cuda_x0,
           float scale)
        {
          assembler.set_bc(cuda_context, *bc0, cuda_x0, scale, cuda_b);
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("x0"), nb::arg("scale"),
        "Set boundary conditions on GPU"
  );

  m.def("set_bc_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::la::CUDAVector& cuda_b,
           std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>> bc0,
           float scale)
        {
          std::shared_ptr<dolfinx::la::Vector<T>> x0 = nullptr;
          assembler.set_bc(cuda_context, *bc0, x0, scale, cuda_b);
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("scale"),
        "Set boundary conditions on GPU"
  );
 
}

#endif


template <typename T>
void declare_cmap(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("CoordinateElement_") + type;
  nb::class_<dolfinx::fem::CoordinateElement<T>>(m, pyclass_name.c_str(),
                                                 "Coordinate map element")
      .def(nb::init<dolfinx::mesh::CellType, int>(), nb::arg("celltype"),
           nb::arg("degree"))
      .def(nb::init<std::shared_ptr<const basix::FiniteElement<T>>>(),
           nb::arg("element"))
      .def(
          "__init__",
          [](dolfinx::fem::CoordinateElement<T>* cm, dolfinx::mesh::CellType ct,
             int d, int var)
          {
            new (cm) dolfinx::fem::CoordinateElement<T>(
                ct, d, static_cast<basix::element::lagrange_variant>(var));
          },
          nb::arg("celltype"), nb::arg("degree"), nb::arg("variant"))
      .def_prop_ro("dtype", [](const dolfinx::fem::CoordinateElement<T>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def("create_dof_layout",
           &dolfinx::fem::CoordinateElement<T>::create_dof_layout)
      .def_prop_ro("degree", &dolfinx::fem::CoordinateElement<T>::degree)
      .def_prop_ro("dim", &dolfinx::fem::CoordinateElement<T>::dim)
      .def_prop_ro("variant", [](const dolfinx::fem::CoordinateElement<T>& self)
                   { return static_cast<int>(self.variant()); })
      .def(
          "push_forward",
          [](const dolfinx::fem::CoordinateElement<T>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> X,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> cell_x)
          {
            using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

            std::array<std::size_t, 2> Xshape{X.shape(0), X.shape(1)};
            std::array<std::size_t, 4> phi_shape
                = self.tabulate_shape(0, X.shape(0));
            std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(),
                                             1, std::multiplies{}));
            cmdspan4_t phi_full(phi_b.data(), phi_shape);
            self.tabulate(0, std::span(X.data(), X.size()), Xshape, phi_b);
            auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::
                MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
                    phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

            std::array<std::size_t, 2> shape = {X.shape(0), cell_x.shape(1)};
            std::vector<T> xb(shape[0] * shape[1]);
            self.push_forward(
                mdspan2_t(xb.data(), shape),
                cmdspan2_t(cell_x.data(), cell_x.shape(0), cell_x.shape(1)),
                phi);

            return dolfinx_wrappers::as_nbarray(std::move(xb),
                                                {X.shape(0), cell_x.shape(1)});
          },
          nb::arg("X"), nb::arg("cell_geometry"))
      .def(
          "pull_back",
          [](const dolfinx::fem::CoordinateElement<T>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> x,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> cell_geometry)
          {
            std::size_t num_points = x.shape(0);
            std::size_t gdim = x.shape(1);
            std::size_t tdim = dolfinx::mesh::cell_dim(self.cell_shape());

            using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

            std::vector<T> Xb(num_points * tdim);
            mdspan2_t X(Xb.data(), num_points, tdim);
            cmdspan2_t _x(x.data(), x.shape(0), x.shape(1));
            cmdspan2_t g(cell_geometry.data(), cell_geometry.shape(0),
                         cell_geometry.shape(1));

            if (self.is_affine())
            {
              std::vector<T> J_b(gdim * tdim);
              mdspan2_t J(J_b.data(), gdim, tdim);
              std::vector<T> K_b(tdim * gdim);
              mdspan2_t K(K_b.data(), tdim, gdim);

              std::array<std::size_t, 4> phi_shape = self.tabulate_shape(1, 1);
              std::vector<T> phi_b(std::reduce(
                  phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
              cmdspan4_t phi(phi_b.data(), phi_shape);

              self.tabulate(1, std::vector<T>(tdim), {1, tdim}, phi_b);
              auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::
                  MDSPAN_IMPL_PROPOSED_NAMESPACE::submdspan(
                      phi, std::pair(1, tdim + 1), 0,
                      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

              self.compute_jacobian(dphi, g, J);
              self.compute_jacobian_inverse(J, K);
              std::array<T, 3> x0 = {0, 0, 0};
              for (std::size_t i = 0; i < g.extent(1); ++i)
                x0[i] += g(0, i);
              self.pull_back_affine(X, K, x0, _x);
            }
            else
              self.pull_back_nonaffine(X, _x, g);

            return dolfinx_wrappers::as_nbarray(std::move(Xb),
                                                {num_points, tdim});
          },
          nb::arg("x"), nb::arg("cell_geometry"));
}

template <typename T>
void declare_real_functions(nb::module_& m)
{
  m.def(
      "create_dofmap",
      [](const dolfinx_wrappers::MPICommWrapper comm, std::uintptr_t dofmap,
         dolfinx::mesh::Topology& topology,
         const dolfinx::fem::FiniteElement<T>& element)
      {
        ufcx_dofmap* p = reinterpret_cast<ufcx_dofmap*>(dofmap);
        assert(p);
        dolfinx::fem::ElementDofLayout layout
            = dolfinx::fem::create_element_dof_layout(*p, topology.cell_type());

        std::function<void(std::span<std::int32_t>, std::uint32_t)>
            unpermute_dofs = nullptr;
        if (element.needs_dof_permutations())
          unpermute_dofs = element.get_dof_permutation_function(true, true);
        return dolfinx::fem::create_dofmap(comm.get(), layout, topology,
                                           unpermute_dofs, nullptr);
      },
      nb::arg("comm"), nb::arg("dofmap"), nb::arg("topology"),
      nb::arg("element"),
      "Create DofMap object from a pointer to ufcx_dofmap.");

  m.def(
      "locate_dofs_topological",
      [](const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>>& V,
         int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         bool remote)
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_topological(
                *V[0].get()->mesh()->topology_mutable(),
                {*V[0].get()->dofmap(), *V[1].get()->dofmap()}, dim,
                std::span(entities.data(), entities.size()), remote);
        return std::array<nb::ndarray<std::int32_t, nb::numpy>, 2>(
            {dolfinx_wrappers::as_nbarray(std::move(dofs[0])),
             dolfinx_wrappers::as_nbarray(std::move(dofs[1]))});
      },
      nb::arg("V"), nb::arg("dim"), nb::arg("entities"),
      nb::arg("remote") = true);
  m.def(
      "locate_dofs_topological",
      [](const dolfinx::fem::FunctionSpace<T>& V, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         bool remote)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::fem::locate_dofs_topological(
                *V.mesh()->topology_mutable(), *V.dofmap(), dim,
                std::span(entities.data(), entities.size()), remote));
      },
      nb::arg("V"), nb::arg("dim"), nb::arg("entities"),
      nb::arg("remote") = true);
  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>>& V,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");

        auto _marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)});
          auto marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_geometrical<T>({*V[0], *V[1]}, _marker);
        return std::array<nb::ndarray<std::int32_t, nb::numpy>, 2>(
            {dolfinx_wrappers::as_nbarray(std::move(dofs[0])),
             dolfinx_wrappers::as_nbarray(std::move(dofs[1]))});
      },
      nb::arg("V"), nb::arg("marker"));
  m.def(
      "locate_dofs_geometrical",
      [](const dolfinx::fem::FunctionSpace<T>& V,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        auto _marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)});
          auto marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return dolfinx_wrappers::as_nbarray(
            dolfinx::fem::locate_dofs_geometrical(V, _marker));
      },
      nb::arg("V"), nb::arg("marker"));

  m.def(
      "interpolation_coords",
      [](const dolfinx::fem::FiniteElement<T>& e,
         const dolfinx::mesh::Geometry<T>& geometry,
         nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig> cells)
      {
        std::vector<T> x = dolfinx::fem::interpolation_coords(
            e, geometry, std::span(cells.data(), cells.size()));
        return dolfinx_wrappers::as_nbarray(std::move(x), {3, x.size() / 3});
      },
      nb::arg("element"), nb::arg("V"), nb::arg("cells"));

  m.def(
      "create_nonmatching_meshes_interpolation_data",
      [](const dolfinx::mesh::Mesh<T>& mesh0,
         const dolfinx::fem::FiniteElement<T>& element0,
         const dolfinx::mesh::Mesh<T>& mesh1, T padding)
      {
        int tdim = mesh0.topology()->dim();
        auto cell_map = mesh0.topology()->index_map(tdim);
        assert(cell_map);
        std::int32_t num_cells
            = cell_map->size_local() + cell_map->num_ghosts();
        std::vector<std::int32_t> cells(num_cells, 0);
        std::iota(cells.begin(), cells.end(), 0);
        auto [src_owner, dest_owner, dest_points, dest_cells]
            = dolfinx::fem::create_nonmatching_meshes_interpolation_data(
                mesh0.geometry(), element0, mesh1,
                std::span(cells.data(), cells.size()), padding);
        return std::tuple(dolfinx_wrappers::as_nbarray(std::move(src_owner)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_owner)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_points)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_cells)));
      },
      nb::arg("mesh0"), nb::arg("element0"), nb::arg("mesh1"),
      nb::arg("padding"));
  m.def(
      "create_nonmatching_meshes_interpolation_data",
      [](const dolfinx::mesh::Geometry<T>& geometry0,
         const dolfinx::fem::FiniteElement<T>& element0,
         const dolfinx::mesh::Mesh<T>& mesh1,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         T padding)
      {
        auto [src_owner, dest_owner, dest_points, dest_cells]
            = dolfinx::fem::create_nonmatching_meshes_interpolation_data(
                geometry0, element0, mesh1,
                std::span(cells.data(), cells.size()), padding);
        return std::tuple(dolfinx_wrappers::as_nbarray(std::move(src_owner)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_owner)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_points)),
                          dolfinx_wrappers::as_nbarray(std::move(dest_cells)));
      },
      nb::arg("geometry0"), nb::arg("element0"), nb::arg("mesh1"),
      nb::arg("cells"), nb ::arg("padding"));
}

} // namespace

namespace dolfinx_wrappers
{

void fem(nb::module_& m)
{
  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_form<float>(m, "float32");
  declare_form<double>(m, "float64");
  declare_form<std::complex<float>>(m, "complex64");
  declare_form<std::complex<double>>(m, "complex128");

  // fem::CoordinateElement
  declare_cmap<float>(m, "float32");
  declare_cmap<double>(m, "float64");

  m.def(
      "create_element_dof_layout",
      [](std::uintptr_t dofmap, const dolfinx::mesh::CellType cell_type,
         const std::vector<int>& parent_map)
      {
        ufcx_dofmap* p = reinterpret_cast<ufcx_dofmap*>(dofmap);
        return dolfinx::fem::create_element_dof_layout(*p, cell_type,
                                                       parent_map);
      },
      nb::arg("dofmap"), nb::arg("cell_type"), nb::arg("parent_map"),
      "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "build_dofmap",
      [](MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& layout)
      {
        assert(topology.entity_types(topology.dim()).size() == 1);
        auto [map, bs, dofmap] = dolfinx::fem::build_dofmap_data(
            comm.get(), topology, {layout},
            [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
            { return dolfinx::graph::reorder_gps(g); });
        return std::tuple(std::move(map), bs, std::move(dofmap));
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("layout"),
      "Build a dofmap on a mesh.");
  m.def(
      "transpose_dofmap",
      [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
         int num_cells)
      {
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const std::int32_t,
            MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
            _dofmap(dofmap.data(), dofmap.shape(0), dofmap.shape(1));
        return dolfinx::fem::transpose_dofmap(_dofmap, num_cells);
      },
      "Build the index to (cell, local index) map from a dofmap ((cell, local "
      "index) -> index).");
  m.def(
      "compute_integration_domains",
      [](dolfinx::fem::IntegralType type,
         const dolfinx::mesh::MeshTags<int>& meshtags)
      {
        return dolfinx::fem::compute_integration_domains(
            type, *meshtags.topology(), meshtags.indices(), meshtags.dim(),
            meshtags.values());
      },
      nb::arg("integral_type"), nb::arg("meshtags"));

  // dolfinx::fem::ElementDofLayout
  nb::class_<dolfinx::fem::ElementDofLayout>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(nb::init<int, const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<int>&,
                    const std::vector<dolfinx::fem::ElementDofLayout>&>(),
           nb::arg("block_size"), nb::arg("endity_dofs"),
           nb::arg("entity_closure_dofs"), nb::arg("parent_map"),
           nb::arg("sub_layouts"))
      .def_prop_ro("num_dofs", &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs,
           nb::arg("dim"))
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs,
           nb::arg("dim"))
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs,
           nb::arg("dim"), nb::arg("entity_index"))
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs, nb::arg("dim"),
           nb::arg("entity_index"))
      .def_prop_ro("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  nb::class_<dolfinx::fem::DofMap>(m, "DofMap", "DofMap object")
      .def(
          "__init__",
          [](dolfinx::fem::DofMap* self,
             const dolfinx::fem::ElementDofLayout& element,
             std::shared_ptr<const dolfinx::common::IndexMap> index_map,
             int index_map_bs,
             const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs)
          {
            new (self) dolfinx::fem::DofMap(element, index_map, index_map_bs,
                                            dofmap.array(), bs);
          },
          nb::arg("element_dof_layout"), nb::arg("index_map"),
          nb::arg("index_map_bs"), nb::arg("dofmap"), nb::arg("bs"))
      .def_ro("index_map", &dolfinx::fem::DofMap::index_map)
      .def_prop_ro("index_map_bs", &dolfinx::fem::DofMap::index_map_bs)
      .def_prop_ro("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def(
          "cell_dofs",
          [](const dolfinx::fem::DofMap& self, int cell)
          {
            std::span<const std::int32_t> dofs = self.cell_dofs(cell);
            return nb::ndarray<const std::int32_t, nb::numpy>(dofs.data(),
                                                              {dofs.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("cell"))
      .def_prop_ro("bs", &dolfinx::fem::DofMap::bs)
      .def(
          "map",
          [](const dolfinx::fem::DofMap& self)
          {
            auto dofs = self.map();
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data_handle(), {dofs.extent(0), dofs.extent(1)});
          },
          nb::rv_policy::reference_internal);

  nb::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell, "cell integral")
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet,
             "exterior facet integral")
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet,
             "exterior facet integral")
      .value("vertex", dolfinx::fem::IntegralType::vertex, "vertex integral");

  declare_function_space<float>(m, "float32");
  declare_function_space<double>(m, "float64");

  declare_real_functions<float>(m);
  declare_real_functions<double>(m);
#ifdef HAS_CUDA_TOOLKIT

  declare_cuda_templated_objects<float>(m, "float32");
  declare_cuda_templated_objects<double>(m, "float64");
  declare_cuda_objects(m);
  //declare_cuda_funcs<float, float>(m);
  declare_cuda_funcs<double, double>(m);
#endif
}
} // namespace dolfinx_wrappers
