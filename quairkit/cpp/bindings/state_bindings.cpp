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

#include <algorithm>
#include <tuple>
#include <vector>

#include "bindings.h"

#include "../state/state_base/probability_data.h"
#include "../state/state_base/common.h"
#include "../state/pure_state/pure_state.h"
#include "../state/mixed_state/mixed_state.h"
#include "../state/product_state/product_state.h"

namespace py = pybind11;

namespace quairkit_cpp {

namespace {

torch::Tensor pure_evolve(const torch::Tensor &data, const torch::Tensor &unitary,
                          int64_t dim, int64_t applied_dim, int64_t prob_dim,
                          bool on_batch) {
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, applied_dim, dim / applied_dim});
  return at::matmul(u, d).view({-1, dim});
}

torch::Tensor pure_evolve_keep_dim(const torch::Tensor &data,
                                   const torch::Tensor &unitary, int64_t dim,
                                   int64_t applied_dim, int64_t prob_dim,
                                   bool on_batch) {
  const auto num_unitary = (unitary.dim() >= 3) ? unitary.size(-3) : 1;
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, num_unitary, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, num_unitary, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, 1, applied_dim, dim / applied_dim});
  return at::matmul(u, d).view({-1, dim});
}

torch::Tensor mixed_evolve(const torch::Tensor &data, const torch::Tensor &unitary,
                           int64_t dim, int64_t applied_dim, int64_t prob_dim,
                           bool on_batch) {
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, applied_dim, (dim * dim) / applied_dim});
  d = at::matmul(u, d);
  d = d.view({-1, prob_dim, dim, applied_dim, dim / applied_dim});
  auto u2 = u.unsqueeze(-3).conj();
  return at::matmul(u2, d).view({-1, dim, dim});
}

torch::Tensor mixed_evolve_keep_dim(const torch::Tensor &data,
                                    const torch::Tensor &unitary, int64_t dim,
                                    int64_t applied_dim, int64_t prob_dim,
                                    bool on_batch) {
  const auto num_unitary = (unitary.dim() >= 3) ? unitary.size(-3) : 1;
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, num_unitary, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, num_unitary, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, 1, applied_dim, (dim * dim) / applied_dim});
  d = at::matmul(u, d);

  d = d.view({-1, prob_dim, num_unitary, dim, applied_dim, dim / applied_dim});
  auto u2 = u.unsqueeze(-3).conj();
  d = at::matmul(u2, d);
  return d.view({-1, dim, dim});
}

std::tuple<torch::Tensor, torch::Tensor>
pure_measure_collapse(const torch::Tensor &data, int64_t dim) {
  auto d = data.view({-1, dim, 1});
  auto prob = at::real(at::matmul(d.transpose(-2, -1).conj(), d));
  prob = prob.view({-1, 1, 1});
  auto mask = at::abs(prob) >= 1e-10;
  auto denom = at::sqrt(at::where(mask, prob, at::ones_like(prob)));
  auto collapsed = d / denom;
  collapsed = collapsed * mask.to(collapsed.scalar_type());
  collapsed = at::where(at::isnan(collapsed), at::zeros_like(collapsed),
                        collapsed);
  return {prob, collapsed.view({-1, dim})};
}

std::tuple<torch::Tensor, torch::Tensor>
mixed_measure_collapse(const torch::Tensor &data, int64_t dim) {
  auto diag = data.diagonal( 0, -2, -1);
  auto prob = at::real(diag.sum(-1)).view({-1, 1, 1});
  auto mask = at::abs(prob) >= 1e-10;
  auto denom = at::where(mask, prob, at::ones_like(prob));
  auto collapsed = data / denom;
  collapsed = collapsed * mask.to(collapsed.scalar_type());
  collapsed = at::where(at::isnan(collapsed), at::zeros_like(collapsed),
                        collapsed);
  return {prob, collapsed.view({-1, dim, dim})};
}

torch::Tensor pure_expec_val(const torch::Tensor &data, const torch::Tensor &obs,
                             int64_t dim, int64_t applied_dim) {
  const auto num_obs = obs.size(-3);
  auto state = data.view({-1, 1, applied_dim, dim / applied_dim});
  auto measured_state = at::matmul(obs, state).view({-1, num_obs, dim});
  auto out = at::sum(data.unsqueeze(-2).conj() * measured_state, -1);
  return out;
}

torch::Tensor mixed_expec_val(const torch::Tensor &data, const torch::Tensor &obs,
                              int64_t dim, int64_t applied_dim) {
  auto state = data.view({-1, 1, applied_dim, (dim * dim) / applied_dim});
  auto tmp = at::matmul(obs, state).view({-1, dim, dim});
  auto tr = tmp.diagonal(0, -2, -1).sum(-1);
  return tr;
}

torch::Tensor mixed_kraus_transform(const torch::Tensor &data,
                                    const torch::Tensor &kraus, int64_t dim,
                                    int64_t applied_dim, int64_t prob_dim,
                                    bool on_batch) {
  auto evolved =
      mixed_evolve_keep_dim(data, kraus, dim, applied_dim, prob_dim, on_batch);
  const auto rank = kraus.size(-3);
  return evolved.view({-1, rank, dim, dim}).sum(-3);
}

torch::Tensor ptrace_1(const torch::Tensor &xy, const torch::Tensor &x) {
  const auto dim_x = x.size(-1);
  const auto dim_xy = xy.size(-1);
  const auto dim_y = dim_xy / dim_x;

  auto xy_sizes = xy.sizes().vec();
  if (xy_sizes.size() < 2) {
    throw std::runtime_error("xy must have at least 2 dimensions.");
  }
  std::vector<int64_t> batch_dims(xy_sizes.begin(), xy_sizes.end() - 2);
  std::vector<int64_t> xy_view_shape = batch_dims;
  xy_view_shape.push_back(-1);
  xy_view_shape.push_back(dim_x);
  xy_view_shape.push_back(dim_y);

  auto xy_view = xy.view(xy_view_shape);
  auto x_view = x.view({-1, dim_x, 1});
  auto y = (xy_view * x_view.conj()).sum(-2);
  return y.squeeze(-2);
}

std::string create_state_type(const torch::Tensor &data) {
  auto sizes = data.sizes();
  const auto ndim = static_cast<int64_t>(sizes.size());
  if (ndim == 1) {
    return "pure";
  }
  if (ndim == 2) {
    return (sizes[0] == sizes[1]) ? "mixed" : "pure";
  }
  if (sizes[ndim - 1] == 1) {
    return "pure";
  }
  if (sizes[ndim - 1] == sizes[ndim - 2]) {
    return "mixed";
  }
  throw std::runtime_error("Input data shape does not match default backend.");
}

torch::Tensor make_controlled_unitary(const torch::Tensor &unitary,
                                      int64_t ctrl_dim, int64_t index) {
  auto opts = unitary.options().dtype(unitary.scalar_type()).device(unitary.device());
  auto proj = torch::zeros({ctrl_dim, ctrl_dim}, opts);
  proj.index_put_({index, index}, 1);
  auto eye_c = torch::eye(ctrl_dim, opts);
  auto eye_u = torch::eye(unitary.size(-1), opts);

  auto u1 = at::kron(proj, unitary);
  auto u2 = at::kron(eye_c - proj, eye_u);
  return u1 + u2;
}

static torch::Tensor trace_two_dims(const torch::Tensor &x, int64_t dim1,
                                   int64_t dim2) {
  auto d1 = dim1 < 0 ? dim1 + x.dim() : dim1;
  auto d2 = dim2 < 0 ? dim2 + x.dim() : dim2;
  auto diag = x.diagonal( 0, d1, d2);
  return diag.sum(-1);
}

torch::Tensor mixed_choi_transform(const torch::Tensor &data,
                                  const torch::Tensor &choi, int64_t dim_refer,
                                  int64_t dim_in, int64_t dim_out,
                                  int64_t prob_dim, bool on_batch) {
  const auto dim_in_total = dim_refer * dim_in;
  auto shape = std::vector<int64_t>{-1, prob_dim};

  auto d = data.view({-1, dim_in_total, dim_in_total});
  d = d.view({shape[0], shape[1], dim_refer, dim_in, dim_refer, dim_in});

  std::vector<int64_t> choi_shape;
  if (on_batch) {
    choi_shape = {-1, 1, dim_in, dim_in * (dim_out * dim_out)};
  } else {
    choi_shape = {1, -1, dim_in, dim_in * (dim_out * dim_out)};
  }
  auto c = choi.view(choi_shape);

  d = d.transpose(-1, -3)
          .reshape({shape[0], shape[1], (dim_refer * dim_refer) * dim_in, dim_in});
  d = at::matmul(d, c);

  d = d.view({shape[0], shape[1], dim_in, dim_refer, dim_out, dim_in, dim_out});
  d = d.transpose(-3, -4);
  d = trace_two_dims(d, -2, -5);

  return d.view({-1, dim_refer * dim_out, dim_refer * dim_out});
}

torch::Tensor mixed_trace_1(const torch::Tensor &data, int64_t trace_dim) {
  const auto dim = data.size(-1);
  const auto rest = dim / trace_dim;
  auto d = data.view({-1, trace_dim, rest, trace_dim, rest});
  d = trace_two_dims(d, 1, 3);
  return d.view({-1, rest, rest});
}

torch::Tensor mixed_reset_1(const torch::Tensor &data,
                            const torch::Tensor &replace_dm,
                            int64_t trace_dim) {
  auto traced = mixed_trace_1(data, trace_dim);
  return at::kron(traced, replace_dm);
}

ProductDenseState py_obj_to_dense_state(const py::object &obj) {
  if (py::isinstance<PureState>(obj)) {
    return obj.cast<PureState>();
  }
  if (py::isinstance<MixedState>(obj)) {
    return obj.cast<MixedState>();
  }
  throw std::runtime_error(
      "Expected a default backend dense block (PureState or MixedState).");
}

py::object dense_state_to_py_obj(const ProductDenseState &state) {
  return std::visit([](const auto &st) -> py::object { return py::cast(st); }, state);
}

std::vector<ProductDenseState>
py_list_to_dense_blocks(const std::vector<py::object> &blocks) {
  std::vector<ProductDenseState> out;
  out.reserve(blocks.size());
  for (const auto &obj : blocks) {
    out.push_back(py_obj_to_dense_state(obj));
  }
  return out;
}

torch::Tensor extract_prod_sum_branch_local(const torch::Tensor &factor,
                                            const std::vector<int64_t> &branch,
                                            int64_t group_idx, int64_t num_groups,
                                            bool is_matrix) {
  if (is_matrix) {
    if (num_groups == 1) {
      return factor.squeeze(-4).squeeze(-1);
    }
    if (group_idx == 0) {
      return factor.squeeze(-4).select(-1, branch[0]);
    }
    if (group_idx == num_groups - 1) {
      return factor.squeeze(-1).select(-3, branch[static_cast<size_t>(group_idx - 1)]);
    }
    return factor.select(-4, branch[static_cast<size_t>(group_idx - 1)])
        .select(-1, branch[static_cast<size_t>(group_idx)]);
  }

  if (num_groups == 1) {
    return factor.squeeze(-3).squeeze(-1);
  }
  if (group_idx == 0) {
    return factor.squeeze(-3).select(-1, branch[0]);
  }
  if (group_idx == num_groups - 1) {
    return factor.squeeze(-1).select(-2, branch[static_cast<size_t>(group_idx - 1)]);
  }
  return factor.select(-3, branch[static_cast<size_t>(group_idx - 1)])
      .select(-1, branch[static_cast<size_t>(group_idx)]);
}

struct FoldProdSumResult {
  std::vector<torch::Tensor> local_tensors;
  c10::optional<torch::Tensor> shared_coeff;
  std::vector<int64_t> batch_shape;
};

void enumerate_branches(const std::vector<int64_t> &sizes, size_t depth,
                        std::vector<int64_t> &current,
                        std::vector<std::vector<int64_t>> &all) {
  if (depth == sizes.size()) {
    all.push_back(current);
    return;
  }
  for (int64_t i = 0; i < sizes[depth]; ++i) {
    current.push_back(i);
    enumerate_branches(sizes, depth + 1, current, all);
    current.pop_back();
  }
}

FoldProdSumResult fold_prod_sum_chain_to_shared_branch(
    const std::vector<torch::Tensor> &factors, const std::vector<torch::Tensor> &coeffs,
    bool is_matrix) {
  if (factors.empty()) {
    throw std::runtime_error("factors must be non-empty.");
  }

  auto shape = factors[0].sizes().vec();
  const auto trim = is_matrix ? 4 : 3;
  std::vector<int64_t> batch_shape(shape.begin(), shape.end() - trim);
  const auto num_groups = static_cast<int64_t>(factors.size());
  if (coeffs.empty()) {
    std::vector<torch::Tensor> locals;
    locals.reserve(factors.size());
    for (size_t i = 0; i < factors.size(); ++i) {
      locals.push_back(extract_prod_sum_branch_local(
          factors[i], {}, static_cast<int64_t>(i), num_groups, is_matrix));
    }
    return {locals, c10::nullopt, batch_shape};
  }

  std::vector<int64_t> branch_sizes;
  branch_sizes.reserve(coeffs.size());
  for (const auto &coeff : coeffs) {
    branch_sizes.push_back(coeff.size(-1));
  }
  std::vector<std::vector<int64_t>> branches;
  std::vector<int64_t> current;
  enumerate_branches(branch_sizes, 0, current, branches);

  auto coeff_base =
      batch_shape.empty() ? factors[0].new_ones({})
                          : torch::ones(batch_shape, factors[0].options());

  std::vector<torch::Tensor> shared_coeff;
  shared_coeff.reserve(branches.size());
  std::vector<std::vector<torch::Tensor>> local_by_group(
      static_cast<size_t>(num_groups));

  for (const auto &branch : branches) {
    auto branch_coeff = coeff_base.clone();
    for (size_t i = 0; i < coeffs.size(); ++i) {
      branch_coeff =
          branch_coeff * coeffs[i].select(-1, branch[static_cast<size_t>(i)]);
    }
    shared_coeff.push_back(branch_coeff);
    for (int64_t g = 0; g < num_groups; ++g) {
      local_by_group[static_cast<size_t>(g)].push_back(
          extract_prod_sum_branch_local(factors[static_cast<size_t>(g)], branch, g,
                                        num_groups, is_matrix));
    }
  }

  const auto stack_dim = static_cast<int64_t>(batch_shape.size());
  std::vector<torch::Tensor> local_tensors;
  local_tensors.reserve(local_by_group.size());
  for (auto &items : local_by_group) {
    local_tensors.push_back(torch::stack(items, stack_dim));
  }
  return {local_tensors, torch::stack(shared_coeff, stack_dim), batch_shape};
}

torch::Tensor first_nonzero_phase(const torch::Tensor &local_tensor, double tol) {
  auto flat = local_tensor.reshape({-1});
  auto nz = torch::nonzero(torch::abs(flat) > tol);
  if (nz.numel() == 0) {
    return torch::ones({}, local_tensor.options());
  }
  auto value = flat.index({nz[0].item<int64_t>()});
  return value / torch::abs(value);
}

std::pair<torch::Tensor, std::vector<torch::Tensor>> canonicalize_shared_branch(
    const std::vector<torch::Tensor> &branch_locals, const torch::Tensor &branch_coeff,
    double tol) {
  auto coeff = branch_coeff;
  std::vector<torch::Tensor> canonical_locals;
  canonical_locals.reserve(branch_locals.size());
  for (auto local : branch_locals) {
    auto norm = torch::linalg_norm(local.reshape({-1}));
    if (std::abs(norm.item<double>()) <= tol) {
      std::vector<torch::Tensor> zeros;
      zeros.reserve(branch_locals.size());
      for (const auto &item : branch_locals) {
        zeros.push_back(torch::zeros_like(item));
      }
      return {coeff.new_zeros({}), zeros};
    }
    local = local / norm;
    coeff = coeff * norm;
    auto phase = first_nonzero_phase(local, tol);
    local = local / phase;
    coeff = coeff * phase;
    canonical_locals.push_back(local);
  }
  return {coeff, canonical_locals};
}

std::pair<std::vector<torch::Tensor>, c10::optional<torch::Tensor>>
compress_shared_prod_sum_branches(const std::vector<torch::Tensor> &local_tensors_in,
                                  c10::optional<torch::Tensor> coeff_tensor_opt,
                                  double tol) {
  if (!coeff_tensor_opt.has_value()) {
    return {local_tensors_in, coeff_tensor_opt};
  }
  auto local_tensors = local_tensors_in;
  auto coeff_tensor = *coeff_tensor_opt;

  const auto branch_axis = coeff_tensor.dim() - 1;
  auto flat_coeff = coeff_tensor.reshape({-1, coeff_tensor.size(-1)});
  auto keep_mask = torch::amax(torch::abs(flat_coeff), 0) > tol;
  if (!torch::any(keep_mask).item<bool>()) {
    keep_mask.index_put_({0}, true);
  }
  if (!torch::all(keep_mask).item<bool>()) {
    auto keep_idx = torch::nonzero(keep_mask).flatten();
    coeff_tensor = coeff_tensor.index_select(branch_axis, keep_idx);
    for (auto &local : local_tensors) {
      local = local.index_select(branch_axis, keep_idx);
    }
  }

  if (coeff_tensor.dim() != 1) {
    return {local_tensors, coeff_tensor};
  }

  const auto merge_tol =
      std::max(tol, coeff_tensor.scalar_type() == torch::kComplexDouble ? 1e-12 : 1e-6);
  std::vector<std::pair<torch::Tensor, std::vector<torch::Tensor>>> merged;
  const auto num_branches = coeff_tensor.size(-1);
  for (int64_t i = 0; i < num_branches; ++i) {
    auto branch_coeff = coeff_tensor.index({i});
    if (torch::abs(branch_coeff).item<double>() <= merge_tol) {
      continue;
    }
    std::vector<torch::Tensor> branch_locals;
    branch_locals.reserve(local_tensors.size());
    for (const auto &local : local_tensors) {
      branch_locals.push_back(local.index({i}));
    }
    auto canonical = canonicalize_shared_branch(branch_locals, branch_coeff, merge_tol);
    branch_coeff = canonical.first;
    branch_locals = canonical.second;
    if (torch::abs(branch_coeff).item<double>() <= merge_tol) {
      continue;
    }

    bool matched = false;
    for (auto &existing : merged) {
      bool all_match = true;
      for (size_t g = 0; g < branch_locals.size(); ++g) {
        if (!torch::allclose(branch_locals[g], existing.second[g], merge_tol, merge_tol)) {
          all_match = false;
          break;
        }
      }
      if (all_match) {
        existing.first = existing.first + branch_coeff;
        matched = true;
        break;
      }
    }
    if (!matched) {
      merged.push_back({branch_coeff, branch_locals});
    }
  }

  std::vector<std::pair<torch::Tensor, std::vector<torch::Tensor>>> filtered;
  for (auto &item : merged) {
    if (torch::abs(item.first).item<double>() > merge_tol) {
      filtered.push_back(item);
    }
  }
  if (filtered.empty()) {
    return {local_tensors, coeff_tensor};
  }

  std::vector<torch::Tensor> coeff_out;
  coeff_out.reserve(filtered.size());
  for (const auto &item : filtered) {
    coeff_out.push_back(item.first);
  }
  coeff_tensor = torch::stack(coeff_out, 0);

  std::vector<torch::Tensor> local_out;
  local_out.reserve(local_tensors.size());
  for (size_t g = 0; g < local_tensors.size(); ++g) {
    std::vector<torch::Tensor> group_items;
    group_items.reserve(filtered.size());
    for (const auto &item : filtered) {
      group_items.push_back(item.second[g]);
    }
    local_out.push_back(torch::stack(group_items, 0));
  }
  return {local_out, coeff_tensor};
}

ProductState build_product_from_prod_sum_factors(
    const std::vector<torch::Tensor> &factors, const std::vector<torch::Tensor> &coeffs,
    const std::vector<std::vector<int64_t>> &subgroup_indices,
    const std::vector<std::vector<int64_t>> &subgroup_system_dims,
    const std::vector<torch::Tensor> &base_prob, bool keep_dim, bool is_matrix,
    double compress_tol) {
  auto folded = fold_prod_sum_chain_to_shared_branch(factors, coeffs, is_matrix);
  auto compressed = compress_shared_prod_sum_branches(
      folded.local_tensors, folded.shared_coeff, compress_tol);
  auto local_tensors = compressed.first;
  auto shared_coeff = compressed.second;

  auto target_batch_shape = folded.batch_shape;
  if (shared_coeff.has_value()) {
    target_batch_shape.push_back(shared_coeff->size(-1));
  }

  std::vector<ProductDenseState> blocks;
  blocks.reserve(local_tensors.size());
  for (size_t i = 0; i < local_tensors.size(); ++i) {
    const auto local_dim = product_int64(subgroup_system_dims[i]);
    if (is_matrix) {
      auto data = local_tensors[i].reshape({-1, local_dim, local_dim});
      MixedState block(data, subgroup_system_dims[i], {}, {});
      if (!target_batch_shape.empty()) {
        block.set_batch_dim(target_batch_shape);
      }
      blocks.push_back(block);
    } else {
      auto data = local_tensors[i].reshape({-1, local_dim, 1});
      PureState block(data, subgroup_system_dims[i], {}, {});
      if (!target_batch_shape.empty()) {
        block.set_batch_dim(target_batch_shape);
      }
      blocks.push_back(block);
    }
  }

  std::vector<torch::Tensor> probability = base_prob;
  std::vector<std::string> roles(base_prob.size(), ProductState::kRoleClassical);
  if (shared_coeff.has_value()) {
    probability.push_back(*shared_coeff);
    roles.push_back(ProductState::kRoleProdSum);
  }
  return ProductState(blocks, subgroup_indices, probability, roles, keep_dim);
}

}

void bind_state(py::module_ &m) {
  auto state_mod = m.def_submodule("state", "Default backend state kernels.");

  py::class_<ProbabilityDataImpl>(state_mod, "ProbabilityData",
                                  "Probability history container used by simulators.")
      .def(py::init<>(), R"doc(
Create an empty probability history.
)doc")
      .def(py::init<const std::vector<torch::Tensor> &>(), py::arg("probs"),
           R"doc(
Create a probability history from an initial list of distributions.

Args:
    probs: A list of probability tensors, each ending with an outcome dimension.
)doc")
      .def("__len__", &ProbabilityDataImpl::size)
      .def("__bool__", [](const ProbabilityDataImpl &self) {
        return !self.empty();
      })
      .def_property_readonly(
          "list",
          [](const ProbabilityDataImpl &self) { return self.list(); },
          R"doc(
Return the list of probability tensors.
)doc")
      .def("clone_list", &ProbabilityDataImpl::clone_list, R"doc(
Clone each probability tensor and return as a list.
)doc")
      .def_property_readonly("shape", &ProbabilityDataImpl::shape, R"doc(
Return the per-variable outcome dimensions.
)doc")
      .def_property_readonly("non_prob_dim", &ProbabilityDataImpl::non_prob_dim,
                             R"doc(
Return the non-probability leading dimensions (broadcast batch dims).
)doc")
      .def_property_readonly("product_dim", &ProbabilityDataImpl::product_dim,
                             R"doc(
Return the product of all probability outcome dimensions.
)doc")
      .def("clear", &ProbabilityDataImpl::clear, R"doc(
Clear all stored probability distributions.
)doc")
      .def("prepare_new", &ProbabilityDataImpl::prepare_new, py::arg("prob"),
           py::arg("dtype") = c10::nullopt, py::arg("device") = c10::nullopt,
           py::arg("real_only") = false,
           R"doc(
Canonicalize a fresh probability tensor to shape [1]*num_prev + [-1].

Args:
    prob: The input probability tensor.
    dtype: Optional dtype cast.
    device: Optional device cast.
    real_only: If True, keep only the real part.

Returns:
    The reshaped (and optionally cast) probability tensor.
)doc")
      .def("append", &ProbabilityDataImpl::append, py::arg("prob"),
           py::arg("normalize") = false, py::arg("dtype") = c10::nullopt,
           py::arg("device") = c10::nullopt, py::arg("real_only") = false,
           R"doc(
Append a probability tensor into the history.

Args:
    prob: The tensor to append.
    normalize: If True, reshape to [1]*num_prev + [-1] before appending.
    dtype: Optional dtype cast.
    device: Optional device cast.
    real_only: If True, keep only the real part.
)doc")
      .def("joint", &ProbabilityDataImpl::joint, py::arg("prob_idx"),
           R"doc(
Compute the joint distribution across the selected indices (with broadcasting).

Args:
    prob_idx: Indices of probability variables to multiply.

Returns:
    The joint distribution tensor.
)doc")
      .def("clone", &ProbabilityDataImpl::clone, R"doc(
Return a deep copy of this probability history.
)doc");

  py::class_<PureState>(state_mod, "PureState", "Default backend pure state.")
      .def(py::init<const torch::Tensor &, const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<torch::Tensor> &>(),
           py::arg("data"), py::arg("sys_dim"), py::arg("system_seq") = std::vector<int64_t>{},
           py::arg("probability") = std::vector<torch::Tensor>{},
           R"doc(
Create a pure state (state vector) for the default backend.

Args:
    data: State tensor. Accepts shapes compatible with the default backend rules.
    sys_dim: Dimensions of subsystems.
    system_seq: Optional current subsystem order.
    probability: Optional probability distributions attached to the state.
)doc")
      .def("data", &PureState::data, R"doc(
Return the internal flattened data tensor (shape: [-1, dim]).
)doc")
      .def("set_data", &PureState::set_data, py::arg("data"))
      .def_property_readonly("system_dim", &PureState::system_dim)
      .def_property_readonly("system_seq", &PureState::system_seq)
      .def_property_readonly("batch_dim", &PureState::batch_dim)
      .def("set_batch_dim", &PureState::set_batch_dim, py::arg("batch_dim"))
      .def_property_readonly("dim", &PureState::dim)
      .def("set_system_dim", &PureState::set_system_dim, py::arg("sys_dim"))
      .def_property_readonly(
          "prob", [](PureState &self) -> ProbabilityDataImpl & { return self.prob(); },
          py::return_value_policy::reference_internal)
      .def("reset_sequence", &PureState::reset_sequence)
      .def("set_system_seq_metadata", &PureState::set_system_seq_metadata,
           py::arg("target_seq"))
      .def("set_system_seq", &PureState::set_system_seq, py::arg("target_seq"))
      .def("ket", &PureState::ket)
      .def("density_matrix", &PureState::density_matrix)
      .def("clone", &PureState::clone)
      .def("index_select", &PureState::index_select, py::arg("new_indices"))
      .def("to", &PureState::to, py::arg("dtype") = c10::nullopt,
           py::arg("device") = c10::nullopt)
      .def("add_probability", &PureState::add_probability, py::arg("prob"))
      .def("prob_select", &PureState::prob_select, py::arg("outcome_idx"),
           py::arg("prob_idx") = -1)
      .def("evolve", &PureState::evolve, py::arg("unitary"), py::arg("sys_idx"),
           py::arg("on_batch") = true)
      .def("evolve_many", &PureState::evolve_many, py::arg("unitary"),
           py::arg("sys_idx_list"), py::arg("on_batch") = true)
      .def("evolve_many_batched", &PureState::evolve_many_batched,
           py::arg("unitary_groups"), py::arg("sys_idx_groups"),
           py::arg("on_batch") = true)
      .def("evolve_keep_dim", &PureState::evolve_keep_dim, py::arg("unitary"),
           py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("evolve_ctrl", &PureState::evolve_ctrl, py::arg("unitary"),
           py::arg("index"), py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("expectation_value", &PureState::expectation_value, py::arg("obs"),
           py::arg("sys_idx"))
      .def("expectation_value_product", &PureState::expectation_value_product,
           py::arg("obs_list"), py::arg("sys_idx_list"))
      .def("expec_val_pauli_terms", &PureState::expec_val_pauli_terms,
           py::arg("pauli_words_r"), py::arg("sites"))
      .def("measure", &PureState::measure, py::arg("measure_op"),
           py::arg("sys_idx"))
      .def("measure_many", &PureState::measure_many, py::arg("measure_op"),
           py::arg("sys_idx_list"))
      .def("measure_by_state", &PureState::measure_by_state,
           py::arg("measure_basis_ket"), py::arg("sys_idx"), py::arg("keep_rest"))
      .def("measure_by_state_product", &PureState::measure_by_state_product,
           py::arg("list_kets"), py::arg("list_sys_idx"), py::arg("keep_rest"));

  py::class_<MixedState>(state_mod, "MixedState", "Default backend mixed state.")
      .def(py::init<const torch::Tensor &, const std::vector<int64_t> &,
                    const std::vector<int64_t> &,
                    const std::vector<torch::Tensor> &>(),
           py::arg("data"), py::arg("sys_dim"), py::arg("system_seq") = std::vector<int64_t>{},
           py::arg("probability") = std::vector<torch::Tensor>{},
           R"doc(
Create a mixed state (density matrix) for the default backend.
)doc")
      .def("data", &MixedState::data, R"doc(
Return the internal flattened data tensor (shape: [-1, dim, dim]).
)doc")
      .def("set_data", &MixedState::set_data, py::arg("data"))
      .def_property_readonly("system_dim", &MixedState::system_dim)
      .def_property_readonly("system_seq", &MixedState::system_seq)
      .def_property_readonly("batch_dim", &MixedState::batch_dim)
      .def("set_batch_dim", &MixedState::set_batch_dim, py::arg("batch_dim"))
      .def_property_readonly("dim", &MixedState::dim)
      .def("set_system_dim", &MixedState::set_system_dim, py::arg("sys_dim"))
      .def_property_readonly(
          "prob", [](MixedState &self) -> ProbabilityDataImpl & { return self.prob(); },
          py::return_value_policy::reference_internal)
      .def("reset_sequence", &MixedState::reset_sequence)
      .def("set_system_seq_metadata", &MixedState::set_system_seq_metadata,
           py::arg("target_seq"))
      .def("set_system_seq", &MixedState::set_system_seq, py::arg("target_seq"))
      .def("density_matrix", &MixedState::density_matrix)
      .def("clone", &MixedState::clone)
      .def("index_select", &MixedState::index_select, py::arg("new_indices"))
      .def("to", &MixedState::to, py::arg("dtype") = c10::nullopt,
           py::arg("device") = c10::nullopt)
      .def("add_probability", &MixedState::add_probability, py::arg("prob"))
      .def("prob_select", &MixedState::prob_select, py::arg("outcome_idx"),
           py::arg("prob_idx") = -1)
      .def("evolve", &MixedState::evolve, py::arg("unitary"), py::arg("sys_idx"),
           py::arg("on_batch") = true)
      .def("evolve_many", &MixedState::evolve_many, py::arg("unitary"),
           py::arg("sys_idx_list"), py::arg("on_batch") = true)
      .def("evolve_many_batched", &MixedState::evolve_many_batched,
           py::arg("unitary_groups"), py::arg("sys_idx_groups"),
           py::arg("on_batch") = true)
      .def("evolve_keep_dim", &MixedState::evolve_keep_dim, py::arg("unitary"),
           py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("evolve_ctrl", &MixedState::evolve_ctrl, py::arg("unitary"),
           py::arg("index"), py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("expectation_value", &MixedState::expectation_value, py::arg("obs"),
           py::arg("sys_idx"))
      .def("expectation_value_product", &MixedState::expectation_value_product,
           py::arg("obs_list"), py::arg("sys_idx_list"))
      .def("expec_val_pauli_terms", &MixedState::expec_val_pauli_terms,
           py::arg("pauli_words_r"), py::arg("sites"))
      .def("expec_state", &MixedState::expec_state, py::arg("prob_idx"))
      .def("measure", &MixedState::measure, py::arg("measure_op"),
           py::arg("sys_idx"))
      .def("measure_many", &MixedState::measure_many, py::arg("measure_op"),
           py::arg("sys_idx_list"))
      .def("measure_by_state", &MixedState::measure_by_state,
           py::arg("measure_basis_ket"), py::arg("sys_idx"), py::arg("keep_rest"))
      .def("measure_by_state_product", &MixedState::measure_by_state_product,
           py::arg("list_kets"), py::arg("list_sys_idx"), py::arg("keep_rest"))
      .def("transform_kraus", &MixedState::transform_kraus, py::arg("list_kraus"),
           py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("transform_many_kraus", &MixedState::transform_many_kraus,
           py::arg("list_kraus"), py::arg("sys_idx_list"),
           py::arg("on_batch") = true)
      .def("transform_choi", &MixedState::transform_choi, py::arg("choi"),
           py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("transform_many_choi", &MixedState::transform_many_choi,
           py::arg("choi"), py::arg("sys_idx_list"), py::arg("on_batch") = true)
      .def("trace", &MixedState::trace, py::arg("trace_idx"))
      .def("reset", &MixedState::reset, py::arg("reset_idx"), py::arg("replace_dm"))
      .def("transpose", &MixedState::transpose, py::arg("transpose_idx"));

  py::class_<ProductState>(
      state_mod, "ProductState",
      "Default backend product-state container composed of dense blocks.")
      .def(py::init([](const std::vector<py::object> &blocks,
                       const std::vector<std::vector<int64_t>> &block_indices,
                       const std::vector<torch::Tensor> &probability,
                       const std::vector<std::string> &roles, bool keep_dim) {
             return ProductState(py_list_to_dense_blocks(blocks), block_indices,
                                 probability, roles, keep_dim);
           }),
           py::arg("blocks"), py::arg("block_indices"),
           py::arg("probability") = std::vector<torch::Tensor>{},
           py::arg("roles") = std::vector<std::string>{},
           py::arg("keep_dim") = false)
      .def("clone", &ProductState::clone)
      .def_property_readonly("backend", &ProductState::backend)
      .def_property_readonly("num_systems", &ProductState::num_systems)
      .def_property_readonly("system_dim", &ProductState::system_dim)
      .def_property_readonly("batch_dim", &ProductState::batch_dim)
      .def_property_readonly("roles", &ProductState::roles)
      .def_property_readonly("prob_list", &ProductState::prob_list)
      .def_property_readonly("classical_indices", &ProductState::classical_indices)
      .def_property_readonly("prod_sum_indices", &ProductState::prod_sum_indices)
      .def_property_readonly("num_blocks", &ProductState::num_blocks)
      .def_property(
          "keep_dim", &ProductState::keep_dim, &ProductState::set_keep_dim)
      .def("block_layout", &ProductState::block_layout)
      .def("export_blocks",
           [](const ProductState &self, bool clone_state) {
             std::vector<std::tuple<py::object, std::vector<int64_t>>> out;
             for (const auto &item : self.export_blocks(clone_state)) {
               out.emplace_back(dense_state_to_py_obj(item.state),
                                item.global_indices);
             }
             return out;
           },
           py::arg("clone_state") = true)
      .def("probability", &ProductState::probability)
      .def("ket", &ProductState::ket)
      .def("density_matrix", &ProductState::density_matrix)
      .def("merged_dense",
           [](const ProductState &self) {
             return dense_state_to_py_obj(self.merged_dense());
           })
      .def("prob_select_dense",
           [](const ProductState &self, const torch::Tensor &outcome_idx,
              int64_t prob_idx) {
             return dense_state_to_py_obj(
                 self.prob_select_dense(outcome_idx, prob_idx));
           },
           py::arg("outcome_idx"), py::arg("prob_idx") = -1)
      .def("expec_state_dense",
           [](const ProductState &self,
              const std::vector<int64_t> &classical_prob_idx) {
             return dense_state_to_py_obj(
                 self.expec_state_dense(classical_prob_idx));
           },
           py::arg("classical_prob_idx"))
      .def("export_block_dense",
           [](const ProductState &self) {
             return dense_state_to_py_obj(self.export_block_dense());
           })
      .def("expec_val", &ProductState::expec_val, py::arg("obs"),
           py::arg("sys_idx"))
      .def("expec_val_product", &ProductState::expec_val_product,
           py::arg("obs_list"), py::arg("sys_idx_list"))
      .def("expec_val_pauli_terms", &ProductState::expec_val_pauli_terms,
           py::arg("pauli_words_r"), py::arg("sites"))
      .def("to", &ProductState::to, py::arg("dtype") = c10::nullopt,
           py::arg("device") = c10::nullopt)
      .def("normalize", &ProductState::normalize)
      .def("add_probability", &ProductState::add_probability, py::arg("prob"))
      .def("evolve", &ProductState::evolve, py::arg("unitary"),
           py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("evolve_many", &ProductState::evolve_many, py::arg("unitary"),
           py::arg("sys_idx_list"), py::arg("on_batch") = true)
      .def("evolve_many_batched", &ProductState::evolve_many_batched,
           py::arg("unitary_groups"), py::arg("sys_idx_groups"),
           py::arg("on_batch") = true)
      .def("evolve_keep_dim", &ProductState::evolve_keep_dim,
           py::arg("unitary"), py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("evolve_ctrl", &ProductState::evolve_ctrl, py::arg("unitary"),
           py::arg("index"), py::arg("sys_idx"), py::arg("on_batch") = true)
      .def("transform", &ProductState::transform, py::arg("op"),
           py::arg("sys_idx"), py::arg("repr_type"),
           py::arg("on_batch") = true)
      .def("transform_many", &ProductState::transform_many, py::arg("op"),
           py::arg("sys_idx_list"), py::arg("repr_type"),
           py::arg("on_batch") = true)
      .def("measure",
           [](const ProductState &self, const torch::Tensor &measure_op,
              const std::vector<int64_t> &sys_idx) {
             auto out = self.measure(measure_op, sys_idx);
             return py::make_tuple(std::get<0>(out), std::get<1>(out));
           },
           py::arg("measure_op"), py::arg("sys_idx"))
      .def("measure_many",
           [](const ProductState &self, const torch::Tensor &measure_op,
              const std::vector<std::vector<int64_t>> &sys_idx_list) {
             auto out = self.measure_many(measure_op, sys_idx_list);
             return py::make_tuple(std::get<0>(out), std::get<1>(out));
           },
           py::arg("measure_op"), py::arg("sys_idx_list"))
      .def("measure_by_state",
           [](const ProductState &self, const py::object &measure_basis,
              const std::vector<int64_t> &sys_idx, bool keep_rest) {
             if (py::isinstance<PureState>(measure_basis)) {
               auto out = self.measure_by_state(
                   measure_basis.cast<PureState>(), sys_idx, keep_rest);
               return py::make_tuple(std::get<0>(out), std::get<1>(out));
             }
             if (py::isinstance<ProductState>(measure_basis)) {
               auto out = self.measure_by_product_state(
                   measure_basis.cast<ProductState>(), sys_idx, keep_rest);
               return py::make_tuple(std::get<0>(out), std::get<1>(out));
             }
             throw std::runtime_error(
                 "measure_basis must be PureState or ProductState.");
           },
           py::arg("measure_basis"), py::arg("sys_idx"), py::arg("keep_rest"))
      .def("trace", &ProductState::trace, py::arg("trace_idx"))
      .def("reset",
           [](const ProductState &self, const std::vector<int64_t> &reset_idx,
              const py::object &replace_state) {
             if (py::isinstance<ProductState>(replace_state)) {
               return self.reset_with_product(
                   reset_idx, replace_state.cast<ProductState>());
             }
             return self.reset(reset_idx, py_obj_to_dense_state(replace_state));
           },
           py::arg("reset_idx"), py::arg("replace_state"))
      .def("transpose", &ProductState::transpose, py::arg("transpose_idx"))
      .def("permute", &ProductState::permute, py::arg("target_seq"))
      .def("kron_dense",
           [](const ProductState &self, const py::object &other) {
             return self.kron_dense(py_obj_to_dense_state(other));
           },
           py::arg("other"))
      .def("kron_product", &ProductState::kron_product, py::arg("other"));

  state_mod.def(
      "create_product_state",
      [](const std::vector<py::object> &blocks,
         const std::vector<std::vector<int64_t>> &block_indices,
         const std::vector<torch::Tensor> &probability,
         const std::vector<std::string> &roles, bool keep_dim) {
        return ProductState(py_list_to_dense_blocks(blocks), block_indices,
                            probability, roles, keep_dim);
      },
      py::arg("blocks"), py::arg("block_indices"),
      py::arg("probability") = std::vector<torch::Tensor>{},
      py::arg("roles") = std::vector<std::string>{},
      py::arg("keep_dim") = false,
      R"doc(
Factory function to create a ProductState.

Args:
    blocks: Dense component states (PureState / MixedState).
    block_indices: Global subsystem indices for each block.
    probability: Optional shared probability history at container level.
    roles: Optional per-probability role labels.
    keep_dim: Keep-dim flag propagated to higher-level wrappers.
)doc");

  state_mod.def(
      "build_product_from_vector_prod_sum",
      [](const std::vector<torch::Tensor> &factors,
         const std::vector<torch::Tensor> &coeffs,
         const std::vector<std::vector<int64_t>> &subgroup_indices,
         const std::vector<std::vector<int64_t>> &subgroup_system_dims,
         const std::vector<torch::Tensor> &base_probability, bool keep_dim,
         double compress_tol) {
        return build_product_from_prod_sum_factors(
            factors, coeffs, subgroup_indices, subgroup_system_dims,
            base_probability, keep_dim, false, compress_tol);
      },
      py::arg("factors"), py::arg("coeffs"), py::arg("subgroup_indices"),
      py::arg("subgroup_system_dims"), py::arg("base_probability"),
      py::arg("keep_dim"), py::arg("compress_tol"));

  state_mod.def(
      "build_product_from_matrix_prod_sum",
      [](const std::vector<torch::Tensor> &factors,
         const std::vector<torch::Tensor> &coeffs,
         const std::vector<std::vector<int64_t>> &subgroup_indices,
         const std::vector<std::vector<int64_t>> &subgroup_system_dims,
         const std::vector<torch::Tensor> &base_probability, bool keep_dim,
         double compress_tol) {
        return build_product_from_prod_sum_factors(
            factors, coeffs, subgroup_indices, subgroup_system_dims,
            base_probability, keep_dim, true, compress_tol);
      },
      py::arg("factors"), py::arg("coeffs"), py::arg("subgroup_indices"),
      py::arg("subgroup_system_dims"), py::arg("base_probability"),
      py::arg("keep_dim"), py::arg("compress_tol"));

  state_mod.def(
      "create_state_type", &create_state_type, py::arg("data"),
      R"doc(
Infer whether a tensor should be interpreted as a pure state vector or a density matrix
under the default backend rules.

Args:
    data: Input tensor representing a state.

Returns:
    Either "pure" or "mixed".
)doc");

  state_mod.def(
      "create_state",
      [](const torch::Tensor &data, const std::vector<int64_t> &sys_dim,
         const std::vector<int64_t> &system_seq,
         const std::vector<torch::Tensor> &probability) -> py::object {
        const auto kind = create_state_type(data);
        if (kind == "pure") {
          return py::cast(PureState(data, sys_dim, system_seq, probability));
        }
        return py::cast(MixedState(data, sys_dim, system_seq, probability));
      },
      py::arg("data"), py::arg("sys_dim"),
      py::arg("system_seq") = std::vector<int64_t>{},
      py::arg("probability") = std::vector<torch::Tensor>{},
      R"doc(
Factory function to create a default-backend state instance.

Args:
    data: Input tensor representing a state.
    sys_dim: Dimensions of subsystems.
    system_seq: Optional current subsystem order.
    probability: Optional probability distributions attached to the state.

Returns:
    A `PureState` or `MixedState` instance.
)doc");

  state_mod.def(
      "make_controlled_unitary", &make_controlled_unitary, py::arg("unitary"),
      py::arg("ctrl_dim"), py::arg("index"),
      R"doc(
Build a controlled-unitary acting on a combined control+target space.

This returns Uc = kron(P, U) + kron(I - P, I), where P projects onto the control
computational basis state |index>.

Args:
    unitary: Target unitary of shape (applied_dim, applied_dim).
    ctrl_dim: Dimension of the control subsystem(s).
    index: Basis index in [0, ctrl_dim).

Returns:
    Controlled unitary of shape (ctrl_dim * applied_dim, ctrl_dim * applied_dim).
)doc");

  state_mod.def(
      "mixed_choi_transform", &mixed_choi_transform, py::arg("data"),
      py::arg("choi"), py::arg("dim_refer"), py::arg("dim_in"),
      py::arg("dim_out"), py::arg("prob_dim"), py::arg("on_batch") = true,
      R"doc(
Apply a Choi representation channel to a density matrix.

Args:
    data: Density matrix tensor of shape (-1, dim, dim), where dim = dim_refer * dim_in.
    choi: Choi operator tensor of shape (..., dim_out^2, dim_in^2).
    dim_refer: Dimension of the reference (untouched) subsystem(s).
    dim_in: Dimension of the input subsystem(s).
    dim_out: Dimension of the output subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch.

Returns:
    The transformed density matrix tensor of shape (-1, dim_refer*dim_out, dim_refer*dim_out).
)doc");

  state_mod.def(
      "mixed_trace_1", &mixed_trace_1, py::arg("data"), py::arg("trace_dim"),
      R"doc(
Partial trace over the leading subsystem of dimension trace_dim.

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    trace_dim: Dimension of the subsystem to trace out (assumed to be leading).

Returns:
    Reduced density matrix tensor of shape (-1, dim/trace_dim, dim/trace_dim).
)doc");

  state_mod.def(
      "mixed_reset_1", &mixed_reset_1, py::arg("data"), py::arg("replace_dm"),
      py::arg("trace_dim"),
      R"doc(
Reset the leading subsystem by tracing it out and tensoring in a replacement state.

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    replace_dm: Replacement density matrix tensor of shape (dim_replace, dim_replace).
    trace_dim: Dimension of the subsystem to reset (assumed to be leading).

Returns:
    The reset density matrix tensor.
)doc");

  state_mod.def(
      "pure_evolve", &pure_evolve,
      R"doc(
Evolve a (batched) pure state vector by left-multiplying a unitary on selected subsystems.

Args:
    data: State vector tensor of shape (-1, dim).
    unitary: Unitary tensor of shape (..., applied_dim, applied_dim).
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the applied subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch. Defaults to True.

Returns:
    The evolved state vector of shape (-1, dim).
)doc",
      py::arg("data"), py::arg("unitary"), py::arg("dim"), py::arg("applied_dim"),
      py::arg("prob_dim"), py::arg("on_batch") = true);

  state_mod.def(
      "pure_evolve_keep_dim", &pure_evolve_keep_dim,
      R"doc(
Evolve a pure state vector while keeping an extra operator-outcome dimension.

This is used by measurement and Kraus-style transformations where the operator carries
an additional leading dimension (e.g., number of outcomes or number of Kraus ops).

Args:
    data: State vector tensor of shape (-1, dim).
    unitary: Operator tensor of shape (..., r, applied_dim, applied_dim), where r is the kept dimension.
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the applied subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch. Defaults to True.

Returns:
    The evolved state vector of shape (-1, dim), with the kept dimension folded into the leading axis.
)doc",
      py::arg("data"), py::arg("unitary"), py::arg("dim"), py::arg("applied_dim"),
      py::arg("prob_dim"), py::arg("on_batch") = true);

  state_mod.def(
      "mixed_evolve", &mixed_evolve,
      R"doc(
Evolve a (batched) density matrix by applying U * rho * U^H on selected subsystems.

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    unitary: Unitary tensor of shape (..., applied_dim, applied_dim).
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the applied subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch. Defaults to True.

Returns:
    The evolved density matrix of shape (-1, dim, dim).
)doc",
      py::arg("data"), py::arg("unitary"), py::arg("dim"), py::arg("applied_dim"),
      py::arg("prob_dim"), py::arg("on_batch") = true);

  state_mod.def(
      "mixed_evolve_keep_dim", &mixed_evolve_keep_dim,
      R"doc(
Evolve a density matrix while keeping an extra operator-outcome dimension.

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    unitary: Operator tensor of shape (..., r, applied_dim, applied_dim), where r is the kept dimension.
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the applied subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch. Defaults to True.

Returns:
    The evolved density matrix of shape (-1, dim, dim), with the kept dimension folded into the leading axis.
)doc",
      py::arg("data"), py::arg("unitary"), py::arg("dim"), py::arg("applied_dim"),
      py::arg("prob_dim"), py::arg("on_batch") = true);

  state_mod.def(
      "pure_measure_collapse", &pure_measure_collapse,
      R"doc(
Compute measurement probabilities and collapsed pure states from a pre-evolved keep-dim state.

Args:
    data: State vector tensor of shape (-1, dim) that already includes the measurement-outcome dimension in its leading axis.
    dim: Full Hilbert space dimension of the state.

Returns:
    A tuple (prob, collapsed) where:
      - prob has shape (-1, 1, 1) (flattened over batches/outcomes)
      - collapsed has shape (-1, dim)
)doc",
      py::arg("data"), py::arg("dim"));

  state_mod.def(
      "mixed_measure_collapse", &mixed_measure_collapse,
      R"doc(
Compute measurement probabilities and collapsed density matrices from a pre-evolved keep-dim state.

Args:
    data: Density matrix tensor of shape (-1, dim, dim) that already includes the measurement-outcome dimension in its leading axis.
    dim: Full Hilbert space dimension of the state.

Returns:
    A tuple (prob, collapsed) where:
      - prob has shape (-1, 1, 1) (flattened over batches/outcomes)
      - collapsed has shape (-1, dim, dim)
)doc",
      py::arg("data"), py::arg("dim"));

  state_mod.def(
      "pure_expec_val", &pure_expec_val,
      R"doc(
Compute expectation values of observables for a pure state (internal kernel).

Args:
    data: State vector tensor of shape (-1, dim).
    obs: Observable tensor of shape (num_obs, applied_dim, applied_dim).
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the measured subsystem(s).

Returns:
    A tensor of shape (-1, num_obs), flattened over batch/prob dimensions.
)doc",
      py::arg("data"), py::arg("obs"), py::arg("dim"), py::arg("applied_dim"));

  state_mod.def(
      "mixed_expec_val", &mixed_expec_val,
      R"doc(
Compute expectation values of observables for a mixed state (internal kernel).

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    obs: Observable tensor of shape (num_obs, applied_dim, applied_dim).
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the measured subsystem(s).

Returns:
    A tensor of shape (-1,), flattened over batch/prob/num_obs dimensions. The caller reshapes it.
)doc",
      py::arg("data"), py::arg("obs"), py::arg("dim"), py::arg("applied_dim"));

  state_mod.def(
      "mixed_kraus_transform", &mixed_kraus_transform,
      R"doc(
Apply a Kraus operator list to a density matrix (internal kernel).

Args:
    data: Density matrix tensor of shape (-1, dim, dim).
    kraus: Kraus operators tensor of shape (..., r, applied_dim, applied_dim).
    dim: Full Hilbert space dimension of the state.
    applied_dim: Dimension of the applied subsystem(s).
    prob_dim: Product dimension of attached probability variables.
    on_batch: Whether the operator batch dimension aligns with the state batch. Defaults to True.

Returns:
    The transformed density matrix tensor of shape (-1, dim, dim).
)doc",
      py::arg("data"), py::arg("kraus"), py::arg("dim"), py::arg("applied_dim"),
      py::arg("prob_dim"), py::arg("on_batch") = true);

  state_mod.def(
      "ptrace_1", &ptrace_1,
      R"doc(
Trace out the first subsystem of a product state vector xy = x ⊗ y.

This is an internal kernel used by the default backend for product-state partial trace.

Args:
    xy: Tensor with shape (..., n, dim_x * dim_y).
    x: Tensor with shape (n, dim_x). Each row is a (possibly batched) unit vector.

Returns:
    A tensor containing y with shape (..., n, dim_y).
)doc",
      py::arg("xy"), py::arg("x"));
}

}