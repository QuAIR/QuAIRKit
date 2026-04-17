// Copyright (c) 2026 QuAIR team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cmath>
#include <complex>
#include <vector>

#include "bindings.h"
#include "common.h"

namespace py = pybind11;
using torch::indexing::Slice;

namespace quairkit_cpp {

namespace {

at::ScalarType complex_dtype_for(const torch::Tensor &t) {
  const auto st = t.scalar_type();
  if (st == at::kComplexFloat || st == at::kComplexDouble) {
    return st;
  }
  if (st == at::kFloat) {
    return at::kComplexFloat;
  }
  if (st == at::kDouble) {
    return at::kComplexDouble;
  }
  return at::kComplexDouble;
}

torch::Tensor eye(int64_t dim, at::ScalarType dtype = at::kComplexDouble) {
  return at::eye(dim, torch::TensorOptions().dtype(dtype));
}

torch::Tensor h(at::ScalarType dtype = at::kComplexDouble) {
  const double e = std::sqrt(2.0) / 2.0;
  auto out = at::empty({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, e);
  out.index_put_({0, 1}, e);
  out.index_put_({1, 0}, e);
  out.index_put_({1, 1}, -e);
  return out;
}

torch::Tensor s(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, c10::complex<double>(0.0, 1.0));
  return out;
}

torch::Tensor sdg(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, c10::complex<double>(0.0, -1.0));
  return out;
}

torch::Tensor t(at::ScalarType dtype = at::kComplexDouble) {
  const double r = std::sqrt(2.0) / 2.0;
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, c10::complex<double>(r, r));
  return out;
}

torch::Tensor tdg(at::ScalarType dtype = at::kComplexDouble) {
  const double r = std::sqrt(2.0) / 2.0;
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, c10::complex<double>(r, -r));
  return out;
}

torch::Tensor x(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 1}, 1.0);
  out.index_put_({1, 0}, 1.0);
  return out;
}

torch::Tensor y(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 1}, c10::complex<double>(0.0, -1.0));
  out.index_put_({1, 0}, c10::complex<double>(0.0, 1.0));
  return out;
}

torch::Tensor z(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({2, 2}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, -1.0);
  return out;
}

torch::Tensor cnot(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({4, 4}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, 1.0);
  out.index_put_({2, 3}, 1.0);
  out.index_put_({3, 2}, 1.0);
  return out;
}

torch::Tensor cy(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({4, 4}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, 1.0);
  out.index_put_({2, 3}, c10::complex<double>(0.0, -1.0));
  out.index_put_({3, 2}, c10::complex<double>(0.0, 1.0));
  return out;
}

torch::Tensor cz(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({4, 4}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 1}, 1.0);
  out.index_put_({2, 2}, 1.0);
  out.index_put_({3, 3}, -1.0);
  return out;
}

torch::Tensor swap(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::zeros({4, 4}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, 1.0);
  out.index_put_({1, 2}, 1.0);
  out.index_put_({2, 1}, 1.0);
  out.index_put_({3, 3}, 1.0);
  return out;
}

torch::Tensor ms(at::ScalarType dtype = at::kComplexDouble) {
  const double v1 = std::sqrt(2.0) / 2.0;
  const c10::complex<double> v2(0.0, 1.0 / std::sqrt(2.0));
  auto out = at::zeros({4, 4}, torch::TensorOptions().dtype(dtype));
  out.index_put_({0, 0}, v1);
  out.index_put_({0, 3}, v2);
  out.index_put_({1, 1}, v1);
  out.index_put_({1, 2}, v2);
  out.index_put_({2, 1}, v2);
  out.index_put_({2, 2}, v1);
  out.index_put_({3, 0}, v2);
  out.index_put_({3, 3}, v1);
  return out;
}

torch::Tensor cswap(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::eye(8, torch::TensorOptions().dtype(dtype));
  out.index_put_({5, 5}, 0.0);
  out.index_put_({6, 6}, 0.0);
  out.index_put_({5, 6}, 1.0);
  out.index_put_({6, 5}, 1.0);
  return out;
}

torch::Tensor toffoli(at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::eye(8, torch::TensorOptions().dtype(dtype));
  out.index_put_({6, 6}, 0.0);
  out.index_put_({7, 7}, 0.0);
  out.index_put_({6, 7}, 1.0);
  out.index_put_({7, 6}, 1.0);
  return out;
}

torch::Tensor param_generator(const torch::Tensor &theta,
                              const torch::Tensor &generator) {
  auto t = theta;
  auto g = generator;
  if (t.dim() == 1) {
    t = t.view({1, -1});
  }
  const auto num_param = g.size(0);
  auto theta4 = t.view({t.size(0), num_param, 1, 1});
  auto hamiltonian = (theta4 * g).sum(1);
  return at::matrix_exp(c10::complex<double>(0.0, 1.0) * hamiltonian);
}

torch::Tensor p(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto o1 = at::ones_like(t);
  auto phase = at::cos(t) + c10::complex<double>(0.0, 1.0) * at::sin(t);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, phase};
  return at::cat(pieces, -1).view({-1, 2, 2});
}

torch::Tensor rx(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto c = at::cos(t / 2);
  auto s = -c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {c, s, s, c};
  return at::cat(pieces, -1).view({-1, 2, 2});
}

torch::Tensor ry(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto c = at::cos(t / 2);
  auto s_ = at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {c, -s_, s_, c};
  return at::cat(pieces, -1).view({-1, 2, 2});
}

torch::Tensor rz(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto e0 = at::exp(-c10::complex<double>(0.0, 1.0) * t / 2);
  auto e1 = at::exp(c10::complex<double>(0.0, 1.0) * t / 2);
  std::vector<torch::Tensor> pieces = {e0, z0, z0, e1};
  return at::cat(pieces, -1).view({-1, 2, 2});
}

torch::Tensor u3(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 3, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto th = t.index({Slice(), 0});
  auto ph = t.index({Slice(), 1});
  auto la = t.index({Slice(), 2});
  auto c = at::cos(th / 2);
  auto s_ = at::sin(th / 2);
  auto e_la = at::exp(c10::complex<double>(0.0, 1.0) * la);
  auto e_ph = at::exp(c10::complex<double>(0.0, 1.0) * ph);
  auto e_phla = at::exp(c10::complex<double>(0.0, 1.0) * (ph + la));
  std::vector<torch::Tensor> pieces = {c, -e_la * s_, e_ph * s_, e_phla * c};
  return at::cat(pieces, -1).view({-1, 2, 2});
}

torch::Tensor cp(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto o1 = at::ones_like(t);
  auto phase = at::cos(t) + c10::complex<double>(0.0, 1.0) * at::sin(t);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, z0, z0, o1, z0, z0,
                                       z0, z0, o1, z0, z0, z0, z0, phase};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor crx(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto o1 = at::ones_like(t);
  auto c = at::cos(t / 2);
  auto s_ = -c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, z0, z0, o1, z0, z0,
                                       z0, z0, c,  s_, z0, z0, s_, c};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor cry(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto o1 = at::ones_like(t);
  auto c = at::cos(t / 2);
  auto s_ = at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, z0, z0, o1, z0, z0,
                                       z0, z0, c,  -s_, z0, z0, s_, c};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor crz(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto o1 = at::ones_like(t);
  auto a = at::cos(t / 2) - c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  auto b = at::cos(t / 2) + c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, z0, z0, o1, z0, z0,
                                       z0, z0, a,  z0, z0, z0, z0, b};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor cu(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 4, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto last = t.index({Slice(), -1});
  auto z0 = at::zeros_like(last);
  auto o1 = at::ones_like(last);
  auto th = t.index({Slice(), 0});
  auto ph = t.index({Slice(), 1});
  auto la = t.index({Slice(), 2});
  auto de = t.index({Slice(), 3});

  auto entry22 = at::cos(th / 2) *
                 (at::cos(de) + c10::complex<double>(0.0, 1.0) * at::sin(de));
  auto entry23 =
      -at::sin(th / 2) *
      (at::cos(la + de) + c10::complex<double>(0.0, 1.0) * at::sin(la + de));
  auto entry32 =
      at::sin(th / 2) *
      (at::cos(ph + de) + c10::complex<double>(0.0, 1.0) * at::sin(ph + de));
  auto entry33 =
      at::cos(th / 2) *
      (at::cos(ph + la + de) +
       c10::complex<double>(0.0, 1.0) * at::sin(ph + la + de));

  std::vector<torch::Tensor> pieces = {o1, z0, z0, z0, z0, o1, z0, z0,
                                       z0, z0, entry22, entry23, z0, z0, entry32, entry33};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor rxx(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto c = at::cos(t / 2);
  auto s_ = -c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {c, z0, z0, s_, z0, c, s_, z0,
                                       z0, s_, c, z0, s_, z0, z0, c};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor ryy(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto p1 = at::cos(t / 2);
  auto p2 = -c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  auto p3 = c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {p1, z0, z0, p3, z0, p1, p2, z0,
                                       z0, p2, p1, z0, p3, z0, z0, p1};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor rzz(const torch::Tensor &theta) {
  auto t_ = theta.view({-1, 1});
  const auto cdtype = complex_dtype_for(t_);
  auto t = t_.to(cdtype);
  auto z0 = at::zeros_like(t);
  auto p1 = at::cos(t / 2) - c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  auto p2 = at::cos(t / 2) + c10::complex<double>(0.0, 1.0) * at::sin(t / 2);
  std::vector<torch::Tensor> pieces = {p1, z0, z0, z0, z0, p2, z0, z0,
                                       z0, z0, p2, z0, z0, z0, z0, p1};
  return at::cat(pieces, -1).view({-1, 4, 4});
}

torch::Tensor phase(int64_t dim, at::ScalarType dtype = at::kComplexDouble) {
  constexpr double pi = 3.14159265358979323846;
  const auto two_pi = 2.0 * pi;
  const c10::complex<double> w =
      std::exp(c10::complex<double>(0.0, two_pi / static_cast<double>(dim)));
  auto out = at::zeros({dim, dim}, torch::TensorOptions().dtype(dtype));
  for (int64_t i = 0; i < dim; ++i) {
    auto val = std::pow(w, static_cast<double>(i));
    out.index_put_({i, i}, val);
  }
  return out;
}

torch::Tensor shift(int64_t dim, at::ScalarType dtype = at::kComplexDouble) {
  auto out = at::eye(dim, torch::TensorOptions().dtype(dtype));
  return at::roll(out, {1}, {0});
}

torch::Tensor qft(int64_t dim, at::ScalarType dtype = at::kComplexDouble) {
  constexpr double pi = 3.14159265358979323846;
  const auto two_pi_over_dim = (2.0 * pi) / static_cast<double>(dim);
  const auto inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(dim));

  auto out = at::empty({dim, dim}, torch::TensorOptions().dtype(dtype));
  for (int64_t i = 0; i < dim; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      const auto phase = two_pi_over_dim * static_cast<double>((i * j) % dim);
      const auto value = std::polar(inv_sqrt_dim, phase);
      out.index_put_({i, j}, c10::complex<double>(value.real(), value.imag()));
    }
  }
  return out;
}

torch::Tensor permutation(const std::vector<int64_t> &perm,
                          const std::vector<int64_t> &system_dim) {
  const auto num_system = static_cast<int64_t>(perm.size());
  const auto dimension = product_int64(system_dim);

  std::vector<int64_t> shape;
  shape.reserve(2 * system_dim.size());
  for (auto d : system_dim)
    shape.push_back(d);
  for (auto d : system_dim)
    shape.push_back(d);
  auto mat = at::eye(dimension, torch::TensorOptions().dtype(at::kFloat)).view(shape);

  std::vector<int64_t> idx;
  idx.reserve(2 * num_system);
  for (auto p_ : perm)
    idx.push_back(p_);
  for (int64_t i = 0; i < num_system; ++i)
    idx.push_back(num_system + i);
  return mat.permute(idx).reshape({dimension, dimension});
}

}

void bind_matrix(py::module_ &m) {
  auto matrix_mod = m.def_submodule("matrix", "Built-in gate matrices.");
  matrix_mod.def(
      "eye", &eye,
      R"doc(
Create an identity matrix.

Args:
    dim: Matrix dimension.
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (dim, dim).
)doc",
      py::arg("dim"), py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "h", &h,
      R"doc(
Return the Hadamard gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "s", &s,
      R"doc(
Return the S gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "sdg", &sdg,
      R"doc(
Return the S† (S-dagger) gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "t", &t,
      R"doc(
Return the T gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "tdg", &tdg,
      R"doc(
Return the T† (T-dagger) gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "x", &x,
      R"doc(
Return the Pauli-X gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "y", &y,
      R"doc(
Return the Pauli-Y gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "z", &z,
      R"doc(
Return the Pauli-Z gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (2, 2).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "cnot", &cnot,
      R"doc(
Return the CNOT gate matrix (control qubit first, target qubit second).

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (4, 4).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "cy", &cy,
      R"doc(
Return the controlled-Y gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (4, 4).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "cz", &cz,
      R"doc(
Return the controlled-Z gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (4, 4).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "swap", &swap,
      R"doc(
Return the SWAP gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (4, 4).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "ms", &ms,
      R"doc(
Return the Mølmer–Sørensen (MS) gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (4, 4).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "cswap", &cswap,
      R"doc(
Return the controlled-SWAP (Fredkin) gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (8, 8).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "toffoli", &toffoli,
      R"doc(
Return the Toffoli (CCNOT) gate matrix.

Args:
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (8, 8).
)doc",
      py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "p", &p,
      R"doc(
Construct phase gate matrices P(theta).

Args:
    theta: Phase angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 2, 2).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "rx", &rx,
      R"doc(
Construct rotation-X gate matrices Rx(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 2, 2).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "ry", &ry,
      R"doc(
Construct rotation-Y gate matrices Ry(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 2, 2).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "rz", &rz,
      R"doc(
Construct rotation-Z gate matrices Rz(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 2, 2).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "u3", &u3,
      R"doc(
Construct U3 gate matrices.

Args:
    theta: Parameters with shape (..., 3). The last dimension corresponds to (theta, phi, lambda).

Returns:
    A tensor of shape (batch, 2, 2).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "cp", &cp,
      R"doc(
Construct controlled-phase gate matrices CP(theta).

Args:
    theta: Phase angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "crx", &crx,
      R"doc(
Construct controlled-Rx gate matrices.

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "cry", &cry,
      R"doc(
Construct controlled-Ry gate matrices.

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "crz", &crz,
      R"doc(
Construct controlled-Rz gate matrices.

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "cu", &cu,
      R"doc(
Construct controlled-U gate matrices.

Args:
    theta: Parameters with shape (..., 4). The last dimension corresponds to (theta, phi, lambda, delta).

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "rxx", &rxx,
      R"doc(
Construct two-qubit XX rotation matrices Rxx(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "ryy", &ryy,
      R"doc(
Construct two-qubit YY rotation matrices Ryy(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "rzz", &rzz,
      R"doc(
Construct two-qubit ZZ rotation matrices Rzz(theta).

Args:
    theta: Rotation angles. Any shape; will be flattened to a batch.

Returns:
    A tensor of shape (batch, 4, 4).
)doc",
      py::arg("theta"));

  matrix_mod.def(
      "phase", &phase,
      R"doc(
Create the generalized phase gate (a diagonal matrix of roots of unity).

Args:
    dim: Local dimension.
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (dim, dim).
)doc",
      py::arg("dim"), py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "shift", &shift,
      R"doc(
Create the generalized shift gate (cyclic permutation matrix).

Args:
    dim: Local dimension.
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (dim, dim).
)doc",
      py::arg("dim"), py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "qft", &qft,
      R"doc(
Create the quantum Fourier transform matrix.

Args:
    dim: Matrix dimension.
    dtype: Output dtype. Defaults to complex128.

Returns:
    A tensor of shape (dim, dim).
)doc",
      py::arg("dim"), py::arg("dtype") = at::kComplexDouble);

  matrix_mod.def(
      "permutation", &permutation,
      R"doc(
Construct a permutation matrix that reorders subsystems.

Args:
    perm: Permutation of subsystem axes.
    system_dim: Dimensions of all subsystems.

Returns:
    A permutation matrix of shape (dim, dim), where dim = product(system_dim).
)doc",
      py::arg("perm"), py::arg("system_dim"));

  matrix_mod.def(
      "param_generator", &param_generator, py::arg("theta"), py::arg("generator"),
      R"doc(
Generate a unitary from parameters and generators: exp(i * sum_k theta_k * G_k).

Args:
    theta: Tensor with shape [B, num_param] or [num_param].
    generator: Tensor with shape [num_param, d, d].

Returns:
    Tensor with shape [B, d, d] or [d, d].
)doc");
}

}