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

#include <tuple>
#include <vector>

#include "bindings.h"
#include "common.h"

namespace py = pybind11;
using torch::indexing::Slice;

namespace quairkit_cpp {

namespace {

torch::Tensor dagger(const torch::Tensor &mat) {
  return mat.conj().transpose(-2, -1).contiguous();
}

torch::Tensor trace(const torch::Tensor &mat, int64_t axis1 = -2,
                    int64_t axis2 = -1) {
  axis1 = normalize_dim(axis1, mat.dim());
  axis2 = normalize_dim(axis2, mat.dim());
  auto diag = mat.diagonal( 0, axis1, axis2);
  return diag.sum(-1);
}

torch::Tensor kron(const torch::Tensor &a, const torch::Tensor &b) {
  const auto out_rows = a.size(-2) * b.size(-2);
  const auto out_cols = a.size(-1) * b.size(-1);

  auto out = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4);
  std::vector<int64_t> shape;
  shape.reserve(out.dim());
  for (int64_t i = 0; i < out.dim() - 4; ++i) {
    shape.push_back(out.size(i));
  }
  shape.push_back(out_rows);
  shape.push_back(out_cols);
  return out.reshape(shape);
}

torch::Tensor nkron(const std::vector<torch::Tensor> &tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("nkron requires at least one tensor.");
  }
  torch::Tensor result = tensors[0];
  for (size_t i = 1; i < tensors.size(); ++i) {
    const auto &b = tensors[i];
    const auto out_rows = result.size(-2) * b.size(-2);
    const auto out_cols = result.size(-1) * b.size(-1);
    auto out = result.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4);
    std::vector<int64_t> shape;
    shape.reserve(out.dim());
    for (int64_t j = 0; j < out.dim() - 4; ++j) {
      shape.push_back(out.size(j));
    }
    shape.push_back(out_rows);
    shape.push_back(out_cols);
    result = out.reshape(shape);
  }
  return result;
}

torch::Tensor trace_1(const torch::Tensor &mat, int64_t dim1) {
  const auto total_dim = mat.size(-1);
  const auto dim2 = total_dim / dim1;
  std::vector<int64_t> batch_dims;
  batch_dims.reserve(std::max<int64_t>(0, mat.dim() - 2));
  for (int64_t i = 0; i < mat.dim() - 2; ++i) {
    batch_dims.push_back(mat.size(i));
  }

  auto reshaped = mat.view({-1, dim1, dim2, dim1, dim2});
  auto traced = trace(reshaped, -2, -4);

  std::vector<int64_t> out_shape = batch_dims;
  out_shape.push_back(dim2);
  out_shape.push_back(dim2);
  return traced.reshape(out_shape);
}

torch::Tensor transpose_1(const torch::Tensor &mat, int64_t dim1) {
  const auto total_dim = mat.size(-1);
  const auto dim2 = total_dim / dim1;
  std::vector<int64_t> batch_dims;
  batch_dims.reserve(std::max<int64_t>(0, mat.dim() - 2));
  for (int64_t i = 0; i < mat.dim() - 2; ++i) {
    batch_dims.push_back(mat.size(i));
  }

  auto reshaped = mat.view({-1, dim1, dim2, dim1, dim2});
  auto permuted = reshaped.permute({0, 3, 2, 1, 4});

  std::vector<int64_t> out_shape = batch_dims;
  out_shape.push_back(total_dim);
  out_shape.push_back(total_dim);
  return permuted.reshape(out_shape);
}

torch::Tensor sqrtm(const torch::Tensor &a) {
  auto eig = at::linalg_eigh(a);
  auto w = std::get<0>(eig);
  auto v = std::get<1>(eig);
  w = at::clamp(w, 1e-20);
  auto sqrt_w = at::sqrt(w);
  return at::matmul(v * sqrt_w.unsqueeze(-2), dagger(v));
}

torch::Tensor permute_sv(const torch::Tensor &state,
                         const std::vector<int64_t> &perm,
                         const std::vector<int64_t> &system_dim) {
  const auto total_dim = product_int64(system_dim);
  auto opts = state.options().dtype(torch::kLong);
  auto base_idx = at::arange(total_dim, opts).view(system_dim).permute(perm);
  base_idx = base_idx.reshape({1, -1}).expand(state.sizes());
  return state.gather(1, base_idx);
}

torch::Tensor permute_dm(const torch::Tensor &state,
                         const std::vector<int64_t> &perm,
                         const std::vector<int64_t> &system_dim) {
  const auto total_dim = product_int64(system_dim);
  auto opts = state.options().dtype(torch::kLong);
  auto base_idx =
      at::arange(total_dim, opts).view(system_dim).permute(perm).contiguous();

  auto out = state.gather(1, base_idx.view({1, -1, 1}).expand(state.sizes()));
  out = out.gather(2, base_idx.view({1, 1, -1}).expand(state.sizes()));
  return out;
}

torch::Tensor permute_systems(const torch::Tensor &mat,
                              const std::vector<int64_t> &perm_list,
                              const std::vector<int64_t> &dim_list) {
  bool is_identity = true;
  for (int64_t i = 0; i < static_cast<int64_t>(perm_list.size()); ++i) {
    if (perm_list[static_cast<size_t>(i)] != i) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) {
    return mat;
  }

  auto original_shape = mat.sizes().vec();
  const auto dim = mat.size(-1);
  auto reshaped = mat.view({-1, dim, dim});
  auto out = permute_dm(reshaped, perm_list, dim_list);
  return out.view(original_shape).contiguous();
}

torch::Tensor get_swap_indices(const int64_t pos1, const int64_t pos2,
                               const std::vector<std::vector<int64_t>> &system_indices,
                               const std::vector<int64_t> &system_dim,
                               c10::optional<torch::Device> device_opt) {
  const auto total_dim = product_int64(system_dim);
  const auto n = static_cast<int64_t>(system_dim.size());

  std::vector<int64_t> weights(static_cast<size_t>(n), 1);
  for (int64_t i = n - 2; i >= 0; --i) {
    weights[static_cast<size_t>(i)] =
        weights[static_cast<size_t>(i + 1)] * system_dim[static_cast<size_t>(i + 1)];
  }

  auto opts = torch::TensorOptions().dtype(torch::kLong);
  if (device_opt.has_value()) {
    opts = opts.device(*device_opt);
  }
  auto indices = at::arange(total_dim, opts);

  for (auto it = system_indices.rbegin(); it != system_indices.rend(); ++it) {
    const auto &pair = *it;
    if (pair.size() != 2) {
      throw std::runtime_error("system_indices elements must have size 2.");
    }
    const auto i = pair[0];
    const auto j = pair[1];
    const auto dim_i = system_dim.at(static_cast<size_t>(i));
    const auto dim_j = system_dim.at(static_cast<size_t>(j));
    const auto weight_i = weights.at(static_cast<size_t>(i));
    const auto weight_j = weights.at(static_cast<size_t>(j));

    auto state_i = (indices / weight_i) % dim_i;
    auto state_j = (indices / weight_j) % dim_j;
    auto joint_state = state_i * dim_j + state_j;
    auto remainder = indices - state_i * weight_i - state_j * weight_j;

    auto new_joint_state = at::where(
        joint_state == pos1, pos2,
        at::where(joint_state == pos2, pos1, joint_state));
    auto new_state_i = new_joint_state / dim_j;
    auto new_state_j = new_joint_state % dim_j;

    indices = remainder + new_state_i * weight_i + new_state_j * weight_j;
  }

  return indices;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
vector_to_prod_sum(const torch::Tensor &vec,
                   const std::vector<int64_t> &system_dim, double tol) {
  if (tol < 0.0) {
    throw std::runtime_error("tol must be non-negative.");
  }
  if (system_dim.empty()) {
    throw std::runtime_error("system_dim must be non-empty.");
  }
  for (auto d : system_dim) {
    if (d <= 0) {
      throw std::runtime_error("All entries in system_dim must be positive.");
    }
  }

  auto data = vec;
  bool squeeze_batch = false;
  if (data.dim() >= 2 && data.size(-1) == 1) {
    data = data.squeeze(-1);
  }
  if (data.dim() == 1) {
    data = data.unsqueeze(0);
    squeeze_batch = true;
  }
  if (data.dim() < 2) {
    throw std::runtime_error("vec must be at least 1D after normalization.");
  }

  const auto total_dim = product_int64(system_dim);
  if (data.size(-1) != total_dim) {
    throw std::runtime_error("Input vector trailing dimension does not match system_dim.");
  }

  auto batch_shape = data.sizes().vec();
  batch_shape.pop_back();

  std::vector<torch::Tensor> factors;
  std::vector<torch::Tensor> coeffs;

  std::vector<int64_t> remaining_dims = system_dim;
  auto current_shape = batch_shape;
  current_shape.push_back(1);
  current_shape.push_back(total_dim);
  auto current = data.reshape(current_shape);

  int64_t left_rank = 1;
  while (remaining_dims.size() > 1) {
    const auto local_dim = remaining_dims.front();
    const auto right_dim = product_int64(
        std::vector<int64_t>(remaining_dims.begin() + 1, remaining_dims.end()));

    auto mat_shape = batch_shape;
    mat_shape.push_back(left_rank * local_dim);
    mat_shape.push_back(right_dim);
    auto mat = current.reshape(mat_shape);

    auto svd_out = at::linalg_svd(mat, false);
    auto u = std::get<0>(svd_out);
    auto s = std::get<1>(svd_out);
    auto vh = std::get<2>(svd_out);

    auto keep = (s > tol).sum(-1);
    int64_t rank = 1;
    if (keep.numel() > 0) {
      rank = std::max<int64_t>(1, keep.max().item<int64_t>());
    }

    u = u.narrow(-1, 0, rank);
    s = s.narrow(-1, 0, rank);
    vh = vh.narrow(-2, 0, rank);

    auto factor_shape = batch_shape;
    factor_shape.push_back(left_rank);
    factor_shape.push_back(local_dim);
    factor_shape.push_back(rank);
    factors.push_back(u.reshape(factor_shape));
    coeffs.push_back(s);

    current = vh;
    left_rank = rank;
    remaining_dims.erase(remaining_dims.begin());
  }

  auto last_shape = batch_shape;
  last_shape.push_back(left_rank);
  last_shape.push_back(remaining_dims.front());
  last_shape.push_back(1);
  factors.push_back(current.reshape(last_shape));

  if (squeeze_batch) {
    for (auto &f : factors) {
      f = f.squeeze(0);
    }
    for (auto &c : coeffs) {
      c = c.squeeze(0);
    }
  }
  return {factors, coeffs};
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
matrix_to_prod_sum(const torch::Tensor &matrix,
                   const std::vector<int64_t> &system_dim, double tol) {
  if (tol < 0.0) {
    throw std::runtime_error("tol must be non-negative.");
  }
  if (matrix.dim() < 2) {
    throw std::runtime_error("matrix must be at least 2D.");
  }
  if (system_dim.empty()) {
    throw std::runtime_error("system_dim must be non-empty.");
  }
  for (auto d : system_dim) {
    if (d <= 0) {
      throw std::runtime_error("All entries in system_dim must be positive.");
    }
  }

  const auto total_dim = product_int64(system_dim);
  if (matrix.size(-2) != total_dim || matrix.size(-1) != total_dim) {
    throw std::runtime_error("Input matrix trailing dims do not match system_dim.");
  }

  auto data = matrix;
  bool squeeze_batch = false;
  if (data.dim() == 2) {
    data = data.unsqueeze(0);
    squeeze_batch = true;
  }

  auto batch_shape = data.sizes().vec();
  batch_shape.pop_back();
  batch_shape.pop_back();
  const auto batch_ndim = static_cast<int64_t>(batch_shape.size());
  const auto num_sys = static_cast<int64_t>(system_dim.size());

  auto reshaped_shape = batch_shape;
  reshaped_shape.insert(reshaped_shape.end(), system_dim.begin(), system_dim.end());
  reshaped_shape.insert(reshaped_shape.end(), system_dim.begin(), system_dim.end());
  auto reshaped = data.reshape(reshaped_shape);

  std::vector<int64_t> perm;
  perm.reserve(static_cast<size_t>(batch_ndim + 2 * num_sys));
  for (int64_t i = 0; i < batch_ndim; ++i) {
    perm.push_back(i);
  }
  for (int64_t i = 0; i < num_sys; ++i) {
    perm.push_back(batch_ndim + i);
    perm.push_back(batch_ndim + num_sys + i);
  }
  auto interleaved = reshaped.permute(perm).contiguous();

  auto vec_shape = batch_shape;
  vec_shape.push_back(total_dim * total_dim);
  auto vec = interleaved.reshape(vec_shape);

  std::vector<int64_t> square_dims;
  square_dims.reserve(system_dim.size());
  for (auto d : system_dim) {
    square_dims.push_back(d * d);
  }

  auto out = vector_to_prod_sum(vec, square_dims, tol);
  auto factors = std::get<0>(out);
  auto coeffs = std::get<1>(out);

  std::vector<torch::Tensor> prod_sum_factors;
  prod_sum_factors.reserve(factors.size());
  for (size_t i = 0; i < factors.size(); ++i) {
    auto factor = factors[i];
    const auto local_dim = system_dim[i];
    const auto left_rank = factor.size(-3);
    const auto right_rank = factor.size(-1);

    auto shape = factor.sizes().vec();
    shape.pop_back();
    shape.pop_back();
    shape.pop_back();
    shape.push_back(left_rank);
    shape.push_back(local_dim);
    shape.push_back(local_dim);
    shape.push_back(right_rank);
    prod_sum_factors.push_back(factor.reshape(shape));
  }

  if (squeeze_batch) {
    for (auto &f : prod_sum_factors) {
      f = f.squeeze(0);
    }
    for (auto &c : coeffs) {
      c = c.squeeze(0);
    }
  }

  return {prod_sum_factors, coeffs};
}

}

void bind_linalg(py::module_ &m) {
  auto linalg_mod = m.def_submodule("linalg", "Linear algebra utilities.");

  linalg_mod.def(
      "dagger", &dagger,
      R"doc(
Compute the conjugate transpose (Hermitian adjoint) of a tensor.

Args:
    mat: Input tensor with at least 2 dimensions.

Returns:
    The conjugate-transposed tensor, contiguous in memory.
)doc",
      py::arg("mat"));

  linalg_mod.def(
      "trace", &trace,
      R"doc(
Compute the trace along two dimensions of a (batched) tensor.

Args:
    mat: Input tensor.
    axis1: First dimension for the trace. Defaults to -2.
    axis2: Second dimension for the trace. Defaults to -1.

Returns:
    The trace of mat along axis1 and axis2.
)doc",
      py::arg("mat"), py::arg("axis1") = -2, py::arg("axis2") = -1);

  linalg_mod.def(
      "kron", &kron,
      R"doc(
Compute the (batched) Kronecker product of two matrices.

Args:
    a: A tensor of shape (..., m, n).
    b: A tensor of shape (..., p, q).

Returns:
    A tensor of shape (..., m*p, n*q).
)doc",
      py::arg("a"), py::arg("b"));

  linalg_mod.def(
      "nkron", &nkron,
      R"doc(
Compute the (batched) Kronecker product of multiple matrices.

Args:
    tensors: A list of tensors, each of shape (..., m_i, n_i).

Returns:
    A tensor of shape (..., prod(m_i), prod(n_i)).
)doc",
      py::arg("tensors"));

  linalg_mod.def(
      "trace_1", &trace_1,
      R"doc(
Trace out the first subsystem of a density matrix.

Args:
    mat: Density matrix tensor of shape (..., d, d) where d = dim1 * dim2.
    dim1: Dimension of the first subsystem.

Returns:
    The reduced density matrix with the first subsystem traced out, shape (..., dim2, dim2).
)doc",
      py::arg("mat"), py::arg("dim1"));

  linalg_mod.def(
      "transpose_1", &transpose_1,
      R"doc(
Transpose the first subsystem of a density matrix.

Args:
    mat: Density matrix tensor of shape (..., d, d) where d = dim1 * dim2.
    dim1: Dimension of the first subsystem.

Returns:
    The transformed density matrix of shape (..., d, d).
)doc",
      py::arg("mat"), py::arg("dim1"));

  linalg_mod.def(
      "sqrtm", &sqrtm,
      R"doc(
Matrix square root for (batched) Hermitian matrices via eigen-decomposition.

Args:
    a: Hermitian matrix tensor of shape (..., d, d).

Returns:
    The matrix square root of a with the same shape as a.

Notes:
    This function clamps eigenvalues to a small positive value to improve numerical stability.
)doc",
      py::arg("a"));

  linalg_mod.def(
      "permute_sv", &permute_sv,
      R"doc(
Permute subsystems of a state vector using gather-based indexing.

Args:
    state: State vector tensor of shape (batch, dim).
    perm: Permutation of subsystem axes.
    system_dim: Dimensions of all subsystems. The product equals dim.

Returns:
    The permuted state vector with the same shape as state.
)doc",
      py::arg("state"), py::arg("perm"), py::arg("system_dim"));

  linalg_mod.def(
      "permute_dm", &permute_dm,
      R"doc(
Permute subsystems of a density matrix using gather-based indexing.

Args:
    state: Density matrix tensor of shape (batch, dim, dim).
    perm: Permutation of subsystem axes.
    system_dim: Dimensions of all subsystems. The product equals dim.

Returns:
    The permuted density matrix with the same shape as state.
)doc",
      py::arg("state"), py::arg("perm"), py::arg("system_dim"));

  linalg_mod.def(
      "permute_systems", &permute_systems,
      R"doc(
Permute subsystems of a (batched) density matrix.

Args:
    mat: Input tensor of shape (..., dim, dim).
    perm_list: Permutation of subsystem axes.
    dim_list: Dimensions of all subsystems. The product equals dim.

Returns:
    The permuted tensor with the same shape as mat.
)doc",
      py::arg("mat"), py::arg("perm_list"), py::arg("dim_list"));

  linalg_mod.def(
      "get_swap_indices", &get_swap_indices, py::arg("pos1"), py::arg("pos2"),
      py::arg("system_indices"), py::arg("system_dim"),
      py::arg("device") = c10::nullopt,
      R"doc(
Compute swapped indices for a sequence of pairwise swaps on a product space.

Args:
    pos1: First position to swap.
    pos2: Second position to swap.
    system_indices: A list of swap pairs, e.g., [[0,1],[2,3]].
    system_dim: Dimensions of subsystems.
    device: Optional torch.device for the output tensor.

Returns:
    A 1-D LongTensor of length prod(system_dim) mapping original indices to swapped ones.
)doc");

  linalg_mod.def(
      "vector_to_prod_sum", &vector_to_prod_sum, py::arg("vec"),
      py::arg("system_dim"),
      py::arg("tol"),
      R"doc(
Factorize a (batched) vector into subgroup-level product-sum factors.

Args:
    vec: Input vector with shape [..., prod(system_dim)] or [..., prod(system_dim), 1].
    system_dim: Local dimensions for each subgroup.
    tol: Singular-value truncation threshold (absolute).

Returns:
    A tuple (factors, coeffs):
    - factors[i] has shape [..., r_{i-1}, d_i, r_i]
    - coeffs[i] has shape [..., r_i]
)doc");

  linalg_mod.def(
      "matrix_to_prod_sum", &matrix_to_prod_sum, py::arg("matrix"),
      py::arg("system_dim"), py::arg("tol"),
      R"doc(
Factorize a (batched) matrix into subgroup-level product-sum factors.

Args:
    matrix: Input matrix with shape [..., prod(system_dim), prod(system_dim)].
    system_dim: Local dimensions for each subgroup.
    tol: Singular-value truncation threshold (absolute).

Returns:
    A tuple (factors, coeffs):
    - factors[i] has shape [..., r_{i-1}, d_i, d_i, r_i]
    - coeffs[i] has shape [..., r_i]
)doc");
}

}