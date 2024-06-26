// # Poisson equation (C++)
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Create and apply Dirichlet boundary conditions
// * Define Expressions
// * Define a FunctionSpace
//
// The solution for $u$ in this demo will look as follows:
//
// .. image:: ../poisson_u.png
//     :scale: 75 %
//
//
// Equation and problem definition
// -------------------------------
//
// The Poisson equation is the canonical elliptic partial differential
// equation.  For a domain $\Omega \subset \mathbb{R}^n$ with
// boundary $\partial \Omega = \Gamma_{D} \cup \Gamma_{N}$, the
// Poisson equation with particular boundary conditions reads:
//
// \begin{align*}
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
// \end{align*}
//
// Here, $f$ and $g$ are input data and $n$ denotes the
// outward directed boundary normal. The most standard variational form
// of Poisson equation reads: find $u \in V$ such that
//
// $$
//    a(u, v) = L(v) \quad \forall \ v \in V,
// $$
// where $V$ is a suitable function space and
//
// \begin{align*}
//    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
//    L(v)    &= \int_{\Omega} f v \, {\rm d} x
//    + \int_{\Gamma_{N}} g v \, {\rm d} s.
// \end{align*}
//
// The expression $a(u, v)$ is the bilinear form and $L(v)$
// is the linear form. It is assumed that all functions in $V$
// satisfy the Dirichlet boundary conditions ($u = 0 \ {\rm on} \
// \Gamma_{D}$).
//
// In this demo, we shall consider the following definitions of the input
// functions, the domain, and the boundaries:
//
// * $\Omega = [0,1] \times [0,1]$ (a unit square)
// * $\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}$
// (Dirichlet boundary)
// * $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$
// (Neumann boundary)
// * $g = \sin(5x)$ (normal derivative)
// * $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$ (source term)
//
//
// Implementation
// --------------
//
// The implementation is split in two files: a form file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: {download}`demo_poisson/main.cpp`,
// {download}`demo_poisson/poisson.py` and {download}`demo_poisson/CMakeLists.txt`.
//
//
// ### UFL form file
//
// The UFL file is implemented in {download}`demo_poisson/poisson.py`.
// ````{admonition} UFL form implemented in python
// :class: dropdown
// ![ufl-code]
// ````
//
// ### C++ program
//
// The main solver is implemented in the {download}`demo_poisson/main.cpp` file.
//
// At the top we include the DOLFINx header file and the generated header
// file "Poisson.h" containing the variational forms for the Poisson
// equation.  For convenience we also include the DOLFINx namespace.
//

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <dolfinx/common/CUDA.h>
#include <dolfinx/fem/CUDAAssembler.h>
#include <dolfinx/fem/CUDADirichletBC.h>
#include <dolfinx/fem/CUDADofMap.h>
#include <dolfinx/fem/CUDAFormIntegral.h>
#include <dolfinx/fem/CUDAFormConstants.h>
#include <dolfinx/fem/CUDAFormCoefficients.h>
#include <dolfinx/la/CUDAMatrix.h>
#include <dolfinx/la/CUDAVector.h>
#include <dolfinx/mesh/CUDAMesh.h>
#include <petscdevicetypes.h> 
#include <petscdevice.h> 
#endif

#include <utility>
#include <vector>
#include <iostream>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

// Then follows the definition of the coefficient functions (for
// $f$ and $g$), which are derived from the
// {cpp:class}`Expression` class in DOLFINx
//

// Inside the ``main`` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the {cpp:class}`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified in
// the form file) defined relative to this mesh, we do as follows
//

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#ifdef HAS_CUDA_TOOLKIT  
  std::cout << "Initializing petsc device." << std::endl;
  PetscDeviceInitialize(PETSC_DEVICE_CUDA);
  std::cout << "Initializing CUDA driver API" << std::endl;
  CUresult cuda_err = cuInit(0);
  if (cuda_err != CUDA_SUCCESS) {
    const char * cuda_err_description;
    cuGetErrorString(cuda_err, &cuda_err_description);
    std::cout << "cuInit() failed with " << cuda_err_description;
    return cuda_err;
  }
#endif

  {
#if defined(HAS_CUDA_TOOLKIT)
    dolfinx::CUDA::Context cuda_context;

#endif
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

//  Next, we define the variational formulation by initializing the
//  bilinear and linear forms ($a$, $L$) using the previously
//  defined {cpp:class}`FunctionSpace` ``V``.  Then we can create the
//  source and boundary flux term ($f$, $g$) and attach these
//  to the linear form.
// 

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
#if !defined(HAS_CUDA_TOOLKIT)
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);
#else
    std::cout << "Initializing f and g." << std::endl;
    auto f = std::make_shared<fem::Function<T>>(cuda_context, V);
    auto g = std::make_shared<fem::Function<T>>(cuda_context, V);
#endif    
    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

//  Now, the Dirichlet boundary condition ($u = 0$) can be created
//  using the class {cpp:class}`DirichletBC`. A {cpp:class}`DirichletBC`
//  takes two arguments: the value of the boundary condition,
//  and the part of the boundary on which the condition applies.
//  In our example, the value of the boundary condition (0.0) can
//  represented using a {cpp:class}`Function`, and the Dirichlet boundary
//  is defined by the indices of degrees of freedom to which the boundary
//  condition applies.
//  The definition of the Dirichlet boundary condition then looks
//  as follows:
// 

    // Define boundary condition

    auto facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
              marker[p] = true;
          }
          return marker;
        });
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

#if defined(HAS_CUDA_TOOLKIT)
    std::cout << "copying f to device" << std::endl;
    f->cuda_vector(cuda_context).copy_vector_values_to_device(cuda_context);
#endif

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

#if defined(HAS_CUDA_TOOLKIT)
    g->cuda_vector(cuda_context).copy_vector_values_to_device(cuda_context);
#endif

//  Now, we have specified the variational forms and can consider the
//  solution of the variational problem. First, we need to define a
//  {cpp:class}`Function` ``u`` to store the solution. (Upon
//  initialization, it is simply set to the zero function.) Next, we can
//  call the ``solve`` function with the arguments ``a == L``, ``u`` and
//  ``bc`` as follows:
//  

    auto u = std::make_shared<fem::Function<T>>(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix_with_fixed_pattern(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());
#if defined(HAS_CUDA_TOOLKIT)

    // for now hardcode target to be RTX 5000
    CUjit_target cujit_target = CU_TARGET_COMPUTE_75;
    dolfinx::fem::CUDAAssembler assembler(
    cuda_context, cujit_target, true, "poisson_demo_cudasrcdir", false);
    dolfinx::mesh::CUDAMesh<U> cuda_mesh(cuda_context, *mesh);
    V->create_cuda_dofmap(cuda_context);
    std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap0 =
    a->function_spaces()[0]->cuda_dofmap();
    std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap1 =
    a->function_spaces()[1]->cuda_dofmap();
    std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>> bcs;
    bcs.emplace_back(bc);

    dolfinx::fem::CUDADirichletBC<T,U> cuda_bc0(
    cuda_context, *a->function_spaces()[0], bcs);
    dolfinx::fem::CUDADirichletBC<T,U> cuda_bc1(
    cuda_context, *a->function_spaces()[1], bcs);
    int max_threads_per_block = 1024;
    int min_blocks_per_sm = 1;

    std::map<dolfinx::fem::IntegralType,
           std::vector<dolfinx::fem::CUDAFormIntegral<T,U>>>
    cuda_a_form_integrals = cuda_form_integrals(
      cuda_context, cujit_target, *a, dolfinx::fem::ASSEMBLY_KERNEL_LOOKUP_TABLE,
      max_threads_per_block, min_blocks_per_sm, true, NULL, true);
    dolfinx::fem::CUDAFormConstants<T> cuda_a_form_constants(
    cuda_context, a.get());
    dolfinx::fem::CUDAFormCoefficients<T,U> cuda_a_form_coefficients(
    cuda_context, a.get());
    assembler.pack_coefficients(
      cuda_context, cuda_a_form_coefficients, false);
    dolfinx::la::CUDAMatrix cuda_A(cuda_context, A.mat(), false, false);
    assembler.compute_lookup_tables(
    cuda_context, *cuda_dofmap0, *cuda_dofmap1,
    cuda_bc0, cuda_bc1, cuda_a_form_integrals, cuda_A, false);
    assembler.zero_matrix_entries(cuda_context, cuda_A);
    assembler.assemble_matrix(
    cuda_context, cuda_mesh, *cuda_dofmap0, *cuda_dofmap1,
    cuda_bc0, cuda_bc1, cuda_a_form_integrals,
    cuda_a_form_constants, cuda_a_form_coefficients,
    cuda_A, false);
    assembler.add_diagonal(cuda_context, cuda_A, cuda_bc0);
    cuda_A.copy_matrix_values_to_host(cuda_context);
    cuda_A.apply(MAT_FINAL_ASSEMBLY);
    std::cout << "Asssembled stiffness matrix." << std::endl;
    //MatView(A.mat(), PETSC_VIEWER_STDOUT_WORLD);
    std::int32_t n;
    const std::int32_t* row_ptr = nullptr;
    const std::int32_t* column_indices = nullptr;
    PetscInt shift = 0;
    PetscBool symmetric = PETSC_FALSE;
    PetscBool inodecompressed = PETSC_FALSE;
    PetscBool status = PETSC_FALSE;
    MatGetRowIJ(
    A.mat(), shift, symmetric, inodecompressed,
    &n, &row_ptr, &column_indices, &status);
    PetscScalar* values = nullptr;
    auto ierr = MatSeqAIJGetArray(A.mat(), &values);
    int col_sum = 0;
    float val_sum = 0.0;
    if (ierr != 0) std::cout << "Cannot get info from mat" << std::endl;
    else {
      int nnz = (n==0) ? n : row_ptr[n];
      std::cout << "returned nonzeros " << nnz << std::endl;
      for (int j=0; j<nnz; j++) {
        //val_sum += values[j];
        col_sum += column_indices[j];
      }
    }
    std::cout << "Val sum: " << val_sum << " col sum " << col_sum << std::endl;

#else
    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
#endif

#if defined(HAS_CUDA_TOOLKIT)
    std::map<dolfinx::fem::IntegralType,
           std::vector<dolfinx::fem::CUDAFormIntegral<T,U>>>
    cuda_L_form_integrals_ = cuda_form_integrals(
      cuda_context, cujit_target, *L, dolfinx::fem::ASSEMBLY_KERNEL_GLOBAL,
      max_threads_per_block, min_blocks_per_sm, false, NULL, false);
    dolfinx::fem::CUDAFormConstants<T> cuda_L_form_constants(
    cuda_context, L.get());
    dolfinx::fem::CUDAFormCoefficients<T,U> cuda_L_form_coefficients(
    cuda_context, L.get());
    assembler.pack_coefficients(
      cuda_context, cuda_L_form_coefficients, false);
    la::petsc::Vector x0(*a->function_spaces()[1]->dofmap()->index_map,
		    a->function_spaces()[1]->dofmap()->index_map_bs());
    dolfinx::la::CUDAVector cuda_x0(cuda_context, x0.vec());
    dolfinx::la::CUDAVector cuda_b(cuda_context, la::petsc::create_vector_wrap(b));
    assembler.zero_vector_entries(cuda_context, cuda_x0);
    assembler.zero_vector_entries(cuda_context, cuda_b);
    cuda_b.copy_vector_values_to_host(cuda_context);
    //cuda_b.update_ghosts();
    assembler.assemble_vector(
        cuda_context, cuda_mesh, *cuda_dofmap0, cuda_bc0,
        cuda_L_form_integrals_, cuda_L_form_constants,
        cuda_L_form_coefficients, cuda_b, false);
    const std::vector<const dolfinx::fem::CUDADofMap *> dofmap1 = {cuda_dofmap1.get()};
    const std::vector<const std::map<dolfinx::fem::IntegralType, std::vector<dolfinx::fem::CUDAFormIntegral<T,U>>>*> 
	    cuda_a_form_integrals_list = {&cuda_a_form_integrals};
    const std::vector<const dolfinx::fem::CUDAFormConstants<T>*> cuda_a_form_constants_list = {&cuda_a_form_constants};
    const std::vector<const dolfinx::fem::CUDAFormCoefficients<T,U>*> cuda_a_form_coefficients_list = {&cuda_a_form_coefficients};
    const std::vector<const dolfinx::fem::CUDADirichletBC<T,U>*> cuda_bc1_list = {&cuda_bc1};
    const std::vector<const dolfinx::la::CUDAVector*> cuda_x0_list = {&cuda_x0};

    assembler.apply_lifting(
        cuda_context, cuda_mesh, *cuda_dofmap0,
        dofmap1, cuda_a_form_integrals_list, cuda_a_form_constants_list,
        cuda_a_form_coefficients_list,
        cuda_bc1_list, cuda_x0_list, 1.0, cuda_b, false);
    //cuda_b.apply_ghosts();
    assembler.set_bc(cuda_context, cuda_bc0, cuda_x0, 1.0, cuda_b);
    cuda_b.copy_vector_values_to_host(cuda_context);
    std::cout << "Assembly complete." << std::endl;
    
#else
    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});
#endif

    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    // Update ghost values before output
    u->x()->scatter_fwd();

//  The function ``u`` will be modified during the call to solve. A
//  {cpp:class}`Function` can be saved to a file. Here, we output the
//  solution to a ``VTK`` file (specified using the suffix ``.pvd``) for
//  visualisation in an external program such as Paraview.
//

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u}, 0.0);

#ifdef HAS_ADIOS2
    // Save solution in VTX format
    io::VTXWriter<U> vtx(MPI_COMM_WORLD, "u.bp", {u}, "bp4");
    vtx.write(0);
#endif
  }

  PetscFinalize();

  return 0;
}
