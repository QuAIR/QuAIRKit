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

#include "mixed_state.h"

#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include "../state_base/common.h"
#include "../state_base/kernels.h"

namespace quairkit_cpp {

namespace {

std::vector<int64_t> arange_vec(int64_t n) {
  std::vector<int64_t> v(static_cast<size_t>(n));
  std::iota(v.begin(), v.end(), 0);
  return v;
}

std::vector<int64_t> perm_of_list(const std::vector<int64_t> &src,
                                  const std::vector<int64_t> &dst) {
  std::vector<int64_t> perm;
  perm.reserve(dst.size());
  for (auto v : dst) {
    auto it = std::find(src.begin(), src.end(), v);
    if (it == src.end()) {
      throw std::runtime_error("Invalid permutation target.");
    }
    perm.push_back(static_cast<int64_t>(std::distance(src.begin(), it)));
  }
  return perm;
}

torch::Tensor permute_dm_impl(const torch::Tensor &state,
                              const std::vector<int64_t> &perm,
                              const std::vector<int64_t> &system_dim) {
  const auto total_dim = product_int64(system_dim);
  const auto n = static_cast<int64_t>(system_dim.size());

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(2 * n + 1));
  shape.push_back(state.size(0));
  shape.insert(shape.end(), system_dim.begin(), system_dim.end());
  shape.insert(shape.end(), system_dim.begin(), system_dim.end());

  std::vector<int64_t> axes;
  axes.reserve(static_cast<size_t>(2 * n + 1));
  axes.push_back(0);
  for (auto p : perm) {
    axes.push_back(1 + p);
  }
  for (auto p : perm) {
    axes.push_back(1 + n + p);
  }

  return state.reshape(shape)
      .permute(axes)
      .contiguous()
      .view({state.size(0), total_dim, total_dim});
}

bool is_supported_pauli_char(char p) {
  const auto u = static_cast<char>(
      std::toupper(static_cast<unsigned char>(p)));
  return u == 'I' || u == 'X' || u == 'Y' || u == 'Z';
}

torch::Tensor pauli_matrix_from_char(char p, const torch::TensorOptions &options) {
  const auto u = static_cast<char>(
      std::toupper(static_cast<unsigned char>(p)));
  auto mat = torch::zeros({2, 2}, options);
  if (u == 'I') {
    mat.index_put_({0, 0}, 1.0);
    mat.index_put_({1, 1}, 1.0);
    return mat;
  }
  if (u == 'X') {
    mat.index_put_({0, 1}, 1.0);
    mat.index_put_({1, 0}, 1.0);
    return mat;
  }
  if (u == 'Y') {
    mat.index_put_({0, 1}, c10::complex<double>(0.0, -1.0));
    mat.index_put_({1, 0}, c10::complex<double>(0.0, 1.0));
    return mat;
  }
  if (u == 'Z') {
    mat.index_put_({0, 0}, 1.0);
    mat.index_put_({1, 1}, -1.0);
    return mat;
  }
  throw std::runtime_error("Unsupported Pauli operator character.");
}

torch::Tensor kron_two_small(const torch::Tensor &left,
                             const torch::Tensor &right) {
  auto kron =
      left.unsqueeze(-1).unsqueeze(-3) * right.unsqueeze(-2).unsqueeze(-4);
  return kron.contiguous().view(
      {left.size(0) * right.size(0), left.size(1) * right.size(1)});
}

bool can_use_small_support_fast_path(const std::string &word,
                                     const std::vector<int64_t> &site) {
  if (word.size() != site.size()) {
    return false;
  }
  if (site.empty() || site.size() > 2) {
    return false;
  }
  return std::all_of(word.begin(), word.end(), is_supported_pauli_char);
}

torch::Tensor build_local_pauli_obs(const std::string &word,
                                    const torch::TensorOptions &options) {
  if (word.empty()) {
    throw std::runtime_error("Pauli word cannot be empty.");
  }
  auto op = pauli_matrix_from_char(word[0], options);
  for (size_t i = 1; i < word.size(); ++i) {
    op = kron_two_small(op, pauli_matrix_from_char(word[i], options));
  }
  return op;
}

}

MixedState::MixedState(const torch::Tensor &data,
                       const std::vector<int64_t> &sys_dim,
                       const std::vector<int64_t> &system_seq,
                       const std::vector<torch::Tensor> &probability)
    : sys_dim_(sys_dim), system_seq_(system_seq), prob_(probability) {
  if (sys_dim_.empty()) {
    throw std::runtime_error("sys_dim must be non-empty.");
  }
  if (system_seq_.empty()) {
    system_seq_ = arange_vec(static_cast<int64_t>(sys_dim_.size()));
  }

  const auto d = dim();
  const auto num_prob = static_cast<int64_t>(prob_.size());
  infer_batch_dim_from_input_shape(data, num_prob);
  data_ = data.reshape({-1, d, d});
}

void MixedState::infer_batch_dim_from_input_shape(const torch::Tensor &input_data,
                                                  int64_t num_prob) {
  auto sizes = input_data.sizes().vec();
  if (sizes.size() <= 2) {
    batch_dim_.clear();
    return;
  }
  const auto non_batch_len = 2 + num_prob;
  const auto prefix_len =
      std::max<int64_t>(0, static_cast<int64_t>(sizes.size()) - non_batch_len);
  batch_dim_.assign(sizes.begin(), sizes.begin() + prefix_len);
}

int64_t MixedState::dim() const { return product_int64(sys_dim_); }

void MixedState::reset_sequence() { set_system_seq(arange_vec(static_cast<int64_t>(sys_dim_.size()))); }

void MixedState::set_system_dim(const std::vector<int64_t> &sys_dim) {
  sys_dim_ = sys_dim;
  const auto n = static_cast<int64_t>(sys_dim_.size());
  if (static_cast<int64_t>(system_seq_.size()) != n) {
    system_seq_ = arange_vec(n);
  }
}

void MixedState::set_system_seq(const std::vector<int64_t> &target_seq) {
  if (target_seq == system_seq_) {
    return;
  }

  const auto perm_map = perm_of_list(system_seq_, target_seq);
  std::vector<int64_t> current_dim;
  current_dim.reserve(system_seq_.size());
  for (auto x : system_seq_) {
    current_dim.push_back(sys_dim_.at(static_cast<size_t>(x)));
  }

  data_ = permute_dm_impl(data_, perm_map, current_dim).contiguous();
  system_seq_ = target_seq;
}

void MixedState::set_system_seq_metadata(const std::vector<int64_t> &target_seq) {
  const auto n = static_cast<int64_t>(sys_dim_.size());
  if (static_cast<int64_t>(target_seq.size()) != n) {
    throw std::runtime_error("system_seq length mismatch.");
  }
  std::vector<bool> seen(static_cast<size_t>(n), false);
  for (auto v : target_seq) {
    if (v < 0 || v >= n) {
      throw std::runtime_error("system_seq contains out-of-range index.");
    }
    auto idx = static_cast<size_t>(v);
    if (seen[idx]) {
      throw std::runtime_error("system_seq is not a permutation (duplicate).");
    }
    seen[idx] = true;
  }
  system_seq_ = target_seq;
}

torch::Tensor MixedState::density_matrix() const {
  const auto d = dim();
  std::vector<int64_t> shape = batch_dim_;
  auto prob_shape = prob_.shape();
  shape.insert(shape.end(), prob_shape.begin(), prob_shape.end());
  shape.push_back(d);
  shape.push_back(d);
  return data_.reshape(shape).clone();
}

torch::Tensor MixedState::ket() const {
  throw std::runtime_error(
      "Mixed state does not support state vector representation.");
}

MixedState MixedState::clone() const {
  MixedState out = *this;
  out.data_ = data_.clone();
  out.prob_ = prob_.clone();
  return out;
}

void MixedState::to(c10::optional<torch::Dtype> dtype,
                    c10::optional<torch::Device> device) {
  if (!dtype.has_value() && !device.has_value()) {
    return;
  }
  auto opts = data_.options();
  if (dtype.has_value()) {
    opts = opts.dtype(*dtype);
  }
  if (device.has_value()) {
    opts = opts.device(*device);
  }
  data_ = data_.to(opts);
}

void MixedState::index_select(const torch::Tensor &new_indices) {
  set_system_seq(arange_vec(static_cast<int64_t>(sys_dim_.size())));
  data_ = data_.index_select(-2, new_indices).index_select(-1, new_indices);
}

void MixedState::add_probability(const torch::Tensor &prob) {
  auto shaped = prob_.prepare_new(prob, c10::nullopt,
                                 c10::nullopt, true);

  const auto d = dim();
  std::vector<int64_t> view_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  view_shape.insert(view_shape.end(), prob_shape.begin(), prob_shape.end());
  view_shape.push_back(1);
  view_shape.push_back(d);
  view_shape.push_back(d);

  auto data_view = data_.view(view_shape);
  auto reps = std::vector<int64_t>(view_shape.size(), 1);
  reps[static_cast<size_t>(view_shape.size() - 3)] = shaped.size(-1);
  data_view = data_view.repeat(reps);
  data_ = data_view.view({-1, d, d});

  prob_.append(shaped, false, c10::nullopt, c10::nullopt,
               false);
}

MixedState MixedState::prob_select(const torch::Tensor &outcome_idx,
                                   int64_t prob_idx) {
  const auto num_prob = static_cast<int64_t>(prob_.size());
  if (prob_idx > 0) {
    prob_idx -= num_prob;
  }

  std::vector<torch::Tensor> new_prob;
  new_prob.reserve(static_cast<size_t>(num_prob));
  for (int64_t i = 0; i < num_prob; ++i) {
    const auto &p = prob_.list().at(static_cast<size_t>(i));
    if (num_prob + prob_idx > i) {
      new_prob.push_back(p.clone());
    } else {
      auto selected_p = p.index_select( prob_idx, outcome_idx);
      if (selected_p.dim() > 1) {
        selected_p = selected_p.squeeze(prob_idx);
      }
      new_prob.push_back(selected_p);
    }
  }

  const auto data_idx = prob_idx - 2;
  auto dm = density_matrix();
  auto selected = dm.index_select( data_idx, outcome_idx).squeeze(data_idx);
  return MixedState(selected, sys_dim_, system_seq_, new_prob);
}

void MixedState::evolve(const torch::Tensor &unitary,
                        const std::vector<int64_t> &sys_idx, bool on_batch) {
  if (on_batch && batch_dim_.empty()) {
    auto u_sizes = unitary.sizes().vec();
    if (u_sizes.size() >= 2) {
      batch_dim_.assign(u_sizes.begin(), u_sizes.end() - 2);
    }
  }

  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      target.push_back(x);
    }
  }
  set_system_seq(target);

  const auto applied_dim = unitary.size(-1);
  const auto prob_dim = prob_.product_dim();
  data_ = quairkit_cpp::mixed_evolve(data_, unitary, dim(), applied_dim, prob_dim,
                                   on_batch);
}

namespace {
torch::Tensor kron_repeat(const torch::Tensor &u, int64_t count) {
  auto out = u;
  for (int64_t i = 1; i < count; ++i) {
    out = at::kron(out, u);
  }
  return out.contiguous();
}

int64_t max_fusable_count(int64_t local_dim, int64_t max_hilbert = 64) {
  if (local_dim <= 1) {
    return 1;
  }
  int64_t k = 1;
  int64_t acc = local_dim;
  while (acc * local_dim <= max_hilbert) {
    acc *= local_dim;
    ++k;
  }
  return k;
}

torch::Tensor kraus_kron_repeat(const torch::Tensor &list_kraus, int64_t count) {
  if (count <= 1) {
    return list_kraus;
  }
  const auto rank = list_kraus.size(-3);
  std::vector<torch::Tensor> factors;
  factors.reserve(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    factors.push_back(list_kraus.select(-3, i));
  }

  std::vector<torch::Tensor> out = factors;
  for (int64_t c = 1; c < count; ++c) {
    std::vector<torch::Tensor> next;
    next.reserve(out.size() * factors.size());
    for (const auto &left : out) {
      for (const auto &right : factors) {
        next.push_back(at::kron(left, right));
      }
    }
    out = std::move(next);
  }
  return torch::stack(out, 0).contiguous();
}

std::vector<std::vector<int64_t>>
build_disjoint_single_groups(const std::vector<int64_t> &flat, int64_t chunk) {
  std::vector<std::vector<int64_t>> groups;
  if (chunk <= 1) {
    groups.reserve(flat.size());
    for (auto idx : flat) {
      groups.push_back({idx});
    }
    return groups;
  }

  size_t i = 0;
  while (i < flat.size()) {
    std::vector<int64_t> group;
    group.reserve(static_cast<size_t>(chunk));
    std::set<int64_t> used;
    while (i < flat.size() &&
           static_cast<int64_t>(group.size()) < chunk) {
      const auto idx = flat[i];
      if (used.find(idx) != used.end()) {
        break;
      }
      group.push_back(idx);
      used.insert(idx);
      ++i;
    }
    if (group.empty()) {
      group.push_back(flat[i]);
      ++i;
    }
    groups.push_back(std::move(group));
  }
  return groups;
}
}

void MixedState::evolve_many(const torch::Tensor &unitary,
                            const std::vector<std::vector<int64_t>> &sys_idx_list,
                            bool on_batch) {
  if (sys_idx_list.empty()) {
    return;
  }

  const bool all_single =
      std::all_of(sys_idx_list.begin(), sys_idx_list.end(),
                  [](const std::vector<int64_t> &v) { return v.size() == 1; });
  const bool single_matrix = unitary.dim() == 2;

  if (!all_single || !single_matrix) {
    for (const auto &idx : sys_idx_list) {
      evolve(unitary, idx, on_batch);
    }
    return;
  }

  const auto local_dim = unitary.size(-1);
  const auto chunk = max_fusable_count(local_dim, 64);

  std::vector<int64_t> flat;
  flat.reserve(sys_idx_list.size());
  for (const auto &idx : sys_idx_list) {
    flat.push_back(idx[0]);
  }

  for (size_t i = 0; i < flat.size(); i += static_cast<size_t>(chunk)) {
    const auto len =
        std::min<int64_t>(static_cast<int64_t>(flat.size() - i), chunk);
    std::vector<int64_t> slice(flat.begin() + static_cast<int64_t>(i),
                               flat.begin() + static_cast<int64_t>(i + len));
    auto u_fused = (len == 1) ? unitary : kron_repeat(unitary, len);
    evolve(u_fused, slice, on_batch);
  }
}

void MixedState::evolve_many_batched(
    const std::vector<torch::Tensor> &unitary_groups,
    const std::vector<std::vector<std::vector<int64_t>>> &sys_idx_groups,
    bool on_batch) {
  if (unitary_groups.size() != sys_idx_groups.size()) {
    throw std::runtime_error("evolve_many_batched: unitary_groups and sys_idx_groups size mismatch.");
  }
  for (size_t g = 0; g < unitary_groups.size(); ++g) {
    const auto &unitary = unitary_groups[g];
    const auto &idx_list = sys_idx_groups[g];
    if (idx_list.empty()) {
      continue;
    }
    if (unitary.dim() == 2) {
      for (const auto &idx : idx_list) {
        evolve(unitary, idx, on_batch);
      }
      continue;
    }
    if (unitary.dim() < 3) {
      throw std::runtime_error("evolve_many_batched: unitary must be 2D or batched with leading dim.");
    }
    if (static_cast<int64_t>(idx_list.size()) != unitary.size(0)) {
      throw std::runtime_error("evolve_many_batched: leading dim of unitary must match sys_idx_list length.");
    }
    for (size_t i = 0; i < idx_list.size(); ++i) {
      auto u_i = unitary.select(0, static_cast<int64_t>(i));
      evolve(u_i, idx_list[i], on_batch);
    }
  }
}

void MixedState::evolve_keep_dim(const torch::Tensor &unitary,
                                 const std::vector<int64_t> &sys_idx,
                                 bool on_batch) {
  auto u_batch = unitary.sizes().vec();
  if (u_batch.size() < 2) {
    throw std::runtime_error("unitary must have at least 2 dimensions.");
  }
  u_batch.resize(u_batch.size() - 2);

  if (on_batch) {
    std::vector<int64_t> left;
    if (!batch_dim_.empty()) {
      left = batch_dim_;
    } else if (u_batch.size() >= 1) {
      left.assign(u_batch.begin(), u_batch.end() - 1);
    }
    std::vector<int64_t> out = left;
    if (!u_batch.empty()) {
      out.push_back(u_batch.back());
    }
    batch_dim_ = out;
  }

  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      target.push_back(x);
    }
  }
  set_system_seq(target);

  const auto applied_dim = unitary.size(-1);
  const auto prob_dim = prob_.product_dim();
  data_ = quairkit_cpp::mixed_evolve_keep_dim(data_, unitary, dim(), applied_dim,
                                            prob_dim, on_batch);
}

void MixedState::evolve_ctrl(const torch::Tensor &unitary, int64_t index,
                             const std::vector<std::vector<int64_t>> &sys_idx,
                             bool on_batch) {
  if (sys_idx.empty()) {
    throw std::runtime_error("sys_idx must be non-empty.");
  }
  const auto &ctrl_idx = sys_idx[0];
  int64_t ctrl_dim = 1;
  for (auto i : ctrl_idx) {
    ctrl_dim *= sys_dim_.at(static_cast<size_t>(i));
  }

  std::vector<int64_t> flat;
  for (auto i : ctrl_idx) {
    flat.push_back(i);
  }
  for (size_t k = 1; k < sys_idx.size(); ++k) {
    if (sys_idx[k].size() != 1) {
      throw std::runtime_error("Applied sys_idx must be scalar indices.");
    }
    flat.push_back(sys_idx[k][0]);
  }
  for (auto x : system_seq_) {
    if (std::find(flat.begin(), flat.end(), x) == flat.end()) {
      flat.push_back(x);
    }
  }
  set_system_seq(flat);

  if (on_batch && batch_dim_.empty()) {
    auto u_sizes = unitary.sizes().vec();
    if (u_sizes.size() >= 2) {
      batch_dim_.assign(u_sizes.begin(), u_sizes.end() - 2);
    }
  }

  auto u_ctrl = quairkit_cpp::make_controlled_unitary(unitary, ctrl_dim, index);
  const auto applied_dim = u_ctrl.size(-1);
  const auto prob_dim = prob_.product_dim();
  data_ = quairkit_cpp::mixed_evolve(data_, u_ctrl, dim(), applied_dim, prob_dim,
                                   on_batch);
}

torch::Tensor
MixedState::expectation_value(const torch::Tensor &obs,
                              const std::vector<int64_t> &sys_idx) {
  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      target.push_back(x);
    }
  }
  set_system_seq(target);

  const auto applied_dim = obs.size(-1);
  auto out = quairkit_cpp::mixed_expec_val(data_, obs, dim(), applied_dim);
  const auto num_obs = obs.size(-3);

  std::vector<int64_t> out_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  out_shape.insert(out_shape.end(), prob_shape.begin(), prob_shape.end());
  out_shape.push_back(num_obs);
  return out.view(out_shape);
}

namespace {
int64_t product_range(const std::vector<int64_t> &dims, size_t begin,
                      size_t end) {
  int64_t out = 1;
  for (size_t i = begin; i < end; ++i) {
    out *= dims.at(i);
  }
  return out;
}

torch::Tensor expec_product_recursive(
    const std::vector<torch::Tensor> &obs_list,
    const std::vector<int64_t> &group_sizes, size_t idx,
    const std::vector<int64_t> &dims_remaining, const torch::Tensor &rho) {
  if (idx >= obs_list.size()) {
    return rho.diagonal( 0, -2, -1).sum(-1);
  }

  const auto len = static_cast<size_t>(group_sizes.at(idx));

  const auto d_act = product_range(dims_remaining, 0, len);
  const auto rest_dim = rho.size(-1) / d_act;

  std::vector<int64_t> rest_dims(dims_remaining.begin() + static_cast<int64_t>(len),
                                 dims_remaining.end());

  auto rho_view = rho.view(
      {rho.size(0), d_act, rest_dim, d_act, rest_dim});

  auto result = torch::zeros({rho.size(0)}, rho.options());
  const auto &O = obs_list[idx];

  for (int64_t i = 0; i < d_act; ++i) {
    for (int64_t j = 0; j < d_act; ++j) {
      auto block =
          rho_view.select(1, j).select(2, i);
      auto sub =
          expec_product_recursive(obs_list, group_sizes, idx + 1, rest_dims,
                                  block);
      result += O.index({i, j}) * sub;
    }
  }
  return result;
}
}

torch::Tensor
MixedState::expectation_value_product(const std::vector<torch::Tensor> &obs_list,
                                      const std::vector<std::vector<int64_t>>
                                          &sys_idx_list) {
  if (obs_list.empty()) {
    return torch::ones({data_.size(0)}, data_.options());
  }

  std::vector<int64_t> flat_sys_idx;
  flat_sys_idx.reserve(sys_idx_list.size());
  for (const auto &group : sys_idx_list) {
    flat_sys_idx.insert(flat_sys_idx.end(), group.begin(), group.end());
  }

  std::vector<int64_t> target_seq = flat_sys_idx;
  for (auto x : system_seq_) {
    if (std::find(flat_sys_idx.begin(), flat_sys_idx.end(), x) ==
        flat_sys_idx.end()) {
      target_seq.push_back(x);
    }
  }
  set_system_seq(target_seq);

  std::vector<int64_t> dims_current;
  dims_current.reserve(system_seq_.size());
  for (auto x : system_seq_) {
    dims_current.push_back(sys_dim_.at(static_cast<size_t>(x)));
  }

  std::vector<int64_t> group_sizes;
  group_sizes.reserve(obs_list.size());
  for (const auto &group : sys_idx_list) {
    group_sizes.push_back(static_cast<int64_t>(group.size()));
  }

  return expec_product_recursive(obs_list, group_sizes, 0, dims_current, data_);
}

torch::Tensor MixedState::expec_val_pauli_terms(
    const std::vector<std::string> &pauli_words_r,
    const std::vector<std::vector<int64_t>> &sites) {
  if (pauli_words_r.size() != sites.size()) {
    throw std::runtime_error(
        "pauli_words_r and sites must have the same length.");
  }
  if (pauli_words_r.empty()) {
    return torch::empty({0, data_.size(0)}, data_.options());
  }

  std::vector<torch::Tensor> results;
  results.reserve(pauli_words_r.size());
  const auto options = data_.options();

  size_t i = 0;
  while (i < pauli_words_r.size()) {
    const auto &word_i = pauli_words_r[i];
    const auto &site_i = sites[i];

    if (can_use_small_support_fast_path(word_i, site_i)) {
      size_t j = i + 1;
      while (j < pauli_words_r.size() &&
             can_use_small_support_fast_path(pauli_words_r[j], sites[j]) &&
             sites[j] == site_i) {
        ++j;
      }

      std::vector<torch::Tensor> stacked_obs;
      stacked_obs.reserve(j - i);
      for (size_t k = i; k < j; ++k) {
        stacked_obs.push_back(build_local_pauli_obs(pauli_words_r[k], options));
      }

      auto expec_group =
          expectation_value(torch::stack(stacked_obs, 0), site_i);
      auto group_rows = expec_group.reshape(
          {-1, static_cast<int64_t>(j - i)}).transpose(0, 1).contiguous();
      for (int64_t r = 0; r < group_rows.size(0); ++r) {
        results.push_back(group_rows.select(0, r));
      }
      i = j;
      continue;
    }

    if (word_i.size() != site_i.size()) {
      throw std::runtime_error(
          "Each pauli word length must match its site count.");
    }
    std::vector<torch::Tensor> obs_list;
    std::vector<std::vector<int64_t>> sys_idx_list;
    obs_list.reserve(word_i.size());
    sys_idx_list.reserve(word_i.size());
    for (size_t pos = 0; pos < word_i.size(); ++pos) {
      obs_list.push_back(pauli_matrix_from_char(word_i[pos], options));
      sys_idx_list.push_back({site_i[pos]});
    }
    results.push_back(expectation_value_product(obs_list, sys_idx_list));
    ++i;
  }

  return torch::stack(results, 0);
}

MixedState MixedState::expec_state(const std::vector<int64_t> &prob_idx) const {
  const auto d = dim();
  const auto num_prob = static_cast<int64_t>(prob_.size());
  if (num_prob == 0) {
    return clone();
  }

  auto joint = prob_.joint(prob_idx);

  std::vector<int64_t> states_shape;
  states_shape.push_back(-1);
  auto prob_shape = prob_.shape();
  states_shape.insert(states_shape.end(), prob_shape.begin(), prob_shape.end());
  states_shape.push_back(d * d);
  auto states = data_.view(states_shape);

  const auto last_p = prob_.list().back();
  const auto expand_ones = last_p.dim() - joint.dim() + 1;
  auto joint_shape = joint.sizes().vec();
  joint_shape.insert(joint_shape.end(), expand_ones, 1);
  joint = joint.view(joint_shape);

  auto prob_state = joint * states;

  std::vector<int64_t> sum_idx;
  sum_idx.reserve(prob_idx.size());
  for (auto idx : prob_idx) {
    sum_idx.push_back(idx + 1);
  }
  auto expectation = prob_state.sum(sum_idx);

  std::vector<torch::Tensor> new_prob;
  if (static_cast<int64_t>(prob_idx.size()) != num_prob) {
    for (int64_t i = 0; i < num_prob; ++i) {
      if (std::find(prob_idx.begin(), prob_idx.end(), i) == prob_idx.end()) {
        new_prob.push_back(prob_.list().at(static_cast<size_t>(i)).clone());
      }
    }

    const auto batch_prob_len = static_cast<int64_t>(last_p.dim()) - num_prob;
    std::vector<int64_t> tail_dims;
    if (!new_prob.empty()) {
      auto shp = new_prob.back().sizes().vec();
      if (static_cast<int64_t>(shp.size()) > batch_prob_len) {
        tail_dims.assign(shp.begin() + batch_prob_len, shp.end());
      }
    }

    std::vector<int64_t> view_shape = batch_dim_;
    view_shape.insert(view_shape.end(), tail_dims.begin(), tail_dims.end());
    view_shape.push_back(d);
    view_shape.push_back(d);
    expectation = expectation.view(view_shape);
    return MixedState(expectation, sys_dim_, system_seq_, new_prob);
  }

  std::vector<int64_t> view_shape = batch_dim_;
  view_shape.push_back(d);
  view_shape.push_back(d);
  expectation = expectation.view(view_shape);
  return MixedState(expectation, sys_dim_, system_seq_, new_prob);
}

std::tuple<torch::Tensor, MixedState>
MixedState::measure(const torch::Tensor &measure_op,
                    const std::vector<int64_t> &sys_idx) {
  auto out = clone();
  out.evolve_keep_dim(measure_op, sys_idx, true);

  auto prob_and = quairkit_cpp::mixed_measure_collapse(out.data_, out.dim());
  auto prob = std::get<0>(prob_and);
  auto collapsed = std::get<1>(prob_and);

  int64_t num_outcomes = 1;
  if (!out.batch_dim_.empty()) {
    num_outcomes = out.batch_dim_.back();
  } else {
    num_outcomes = measure_op.size(0);
  }
  std::vector<int64_t> base_batch = out.batch_dim_;
  if (!base_batch.empty()) {
    base_batch.pop_back();
  }

  auto prob_shape = prob_.shape();
  std::vector<int64_t> target_shape = base_batch;
  target_shape.insert(target_shape.end(), prob_shape.begin(), prob_shape.end());
  target_shape.push_back(num_outcomes);
  prob = prob.view(target_shape);

  const auto d = out.dim();
  out.data_ = collapsed.view({-1, d, d});
  if (!out.batch_dim_.empty()) {
    out.batch_dim_.pop_back();
  }
  out.prob_.append(prob, false, c10::nullopt, c10::nullopt,
                   false);
  return {prob, out};
}

std::tuple<torch::Tensor, MixedState>
MixedState::measure_many(const torch::Tensor &measure_op,
                         const std::vector<std::vector<int64_t>> &sys_idx_list) {
  const auto num_measure = static_cast<size_t>(measure_op.size(0));
  if (num_measure != sys_idx_list.size()) {
    throw std::runtime_error(
        "measure_many: measure_op.size(0) must equal sys_idx_list.size(), got " +
        std::to_string(num_measure) + " and " +
        std::to_string(sys_idx_list.size()));
  }

  auto out = clone();
  const auto prob_start = out.prob_.size();

  for (size_t i = 0; i < num_measure; ++i) {
    auto op_i = measure_op.select(0, static_cast<int64_t>(i));
    auto result = out.measure(op_i, sys_idx_list[i]);
    out = std::move(std::get<1>(result));
  }

  std::vector<int64_t> new_idx;
  const auto prob_end = out.prob_.size();
  new_idx.reserve(static_cast<size_t>(prob_end - prob_start));
  for (auto j = prob_start; j < prob_end; ++j) {
    new_idx.push_back(j);
  }
  auto joint = out.prob_.joint(new_idx);

  return {joint, out};
}

std::tuple<torch::Tensor, std::optional<MixedState>>
MixedState::measure_by_state(const torch::Tensor &measure_basis_ket,
                             const std::vector<int64_t> &sys_idx,
                             bool keep_rest) {

  auto out = clone();

  auto k = measure_basis_ket;
  if (k.dim() == 2) {
    k = k.unsqueeze(0);
  }
  int64_t num_outcomes = 1;
  for (int64_t i = 0; i < k.dim() - 2; ++i) {
    num_outcomes *= k.size(i);
  }

  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      target.push_back(x);
    }
  }
  out.set_system_seq(target);

  const auto applied_dim = k.size(-2);
  k = k.contiguous().view({num_outcomes, applied_dim, 1});
  const auto total_dim = out.dim();
  const auto rest_dim = total_dim / applied_dim;

  auto rho =
      out.data_.contiguous().reshape({-1, applied_dim, rest_dim, applied_dim, rest_dim});
  auto k_flat = k.contiguous().reshape({num_outcomes, applied_dim});

  const auto B_r = rho.size(0);
  auto k_conj_flat = k_flat.conj();
  auto rho_perm = rho.permute({1, 0, 2, 3, 4}).contiguous();
  auto rho_flat = rho_perm.reshape(
      {applied_dim, B_r * rest_dim * applied_dim * rest_dim});
  auto temp_mat = at::matmul(k_conj_flat, rho_flat);
  auto temp =
      temp_mat.reshape({num_outcomes, B_r, rest_dim, applied_dim, rest_dim});

  auto temp_perm = temp.permute({0, 1, 2, 4, 3}).contiguous();
  auto temp_flat = temp_perm.reshape(
      {num_outcomes, B_r * rest_dim * rest_dim, applied_dim});
  auto k_mat = k_flat.unsqueeze(-1);
  auto rho_rest_flat =
      at::bmm(temp_flat, k_mat).squeeze(-1);
  auto rho_rest = rho_rest_flat
                      .reshape({num_outcomes, B_r, rest_dim, rest_dim})
                      .permute({1, 0, 2, 3})
                      .contiguous();

  auto prob =
      at::real(rho_rest.diagonal( 0, -2, -1).sum(-1));

  std::vector<int64_t> data_batch = out.batch_dim_;
  std::vector<int64_t> prior_prob = out.prob().shape();
  std::vector<int64_t> base_batch = data_batch;
  base_batch.insert(base_batch.end(), prior_prob.begin(), prior_prob.end());

  prob = prob.contiguous();
  std::vector<int64_t> target_shape = base_batch;
  target_shape.push_back(num_outcomes);
  prob = prob.reshape(target_shape);

  if (!keep_rest) {
    return {prob, std::nullopt};
  }

  auto prob_flat =
      at::real(rho_rest.diagonal( 0, -2, -1).sum(-1));
  auto mask = at::abs(prob_flat) >= 1e-10;
  auto denom = at::where(mask, prob_flat, at::ones_like(prob_flat));
  auto rho_rest_norm = rho_rest / denom.unsqueeze(-1).unsqueeze(-1);
  rho_rest_norm = rho_rest_norm * mask.unsqueeze(-1).unsqueeze(-1);
  rho_rest_norm =
      at::where(at::isnan(rho_rest_norm), at::zeros_like(rho_rest_norm),
                rho_rest_norm);

  std::vector<int64_t> rest_sys_dim;
  rest_sys_dim.reserve(sys_dim_.size() - sys_idx.size());
  std::vector<int64_t> rest_orig_seq;
  rest_orig_seq.reserve(sys_dim_.size() - sys_idx.size());
  std::set<int64_t> measured_set(sys_idx.begin(), sys_idx.end());
  for (size_t i = 0; i < sys_dim_.size(); ++i) {
    auto orig_idx = system_seq_[i];
    if (measured_set.find(orig_idx) == measured_set.end()) {
      rest_sys_dim.push_back(sys_dim_[orig_idx]);
      rest_orig_seq.push_back(orig_idx);
    }
  }

  if (rest_sys_dim.empty()) {
    rest_sys_dim.push_back(1);
    rest_orig_seq.push_back(0);
  }

  rho_rest_norm = rho_rest_norm.contiguous().view({-1, rest_dim, rest_dim});

  auto rest_orig_sorted = rest_orig_seq;
  std::sort(rest_orig_sorted.begin(), rest_orig_sorted.end());
  if (rest_orig_sorted != rest_orig_seq) {
    const auto perm_map = perm_of_list(rest_orig_seq, rest_orig_sorted);
    rho_rest_norm = permute_dm_impl(rho_rest_norm, perm_map, rest_sys_dim).contiguous();
    std::vector<int64_t> canonical_dim;
    canonical_dim.reserve(rest_orig_sorted.size());
    for (auto idx : rest_orig_sorted) {
      canonical_dim.push_back(sys_dim_.at(static_cast<size_t>(idx)));
    }
    rest_sys_dim = std::move(canonical_dim);
  }

  std::vector<int64_t> rest_sys_seq =
      arange_vec(static_cast<int64_t>(rest_sys_dim.size()));

  MixedState rest_state(rho_rest_norm, rest_sys_dim, rest_sys_seq, {});
  rest_state.set_batch_dim(data_batch);
  
  for (const auto &p : out.prob().list()) {
    rest_state.prob().append(p, false, c10::nullopt, c10::nullopt,
                             false);
  }
  rest_state.prob().append(prob, false, c10::nullopt, c10::nullopt,
                           false);

  return {prob, std::optional<MixedState>(std::move(rest_state))};
}

std::tuple<torch::Tensor, std::optional<MixedState>>
MixedState::measure_by_state_product(
    const std::vector<torch::Tensor> &list_kets,
    const std::vector<std::vector<int64_t>> &list_sys_idx,
    bool keep_rest) {

  if (list_kets.empty()) {
    throw std::runtime_error(
        "measure_by_state_product: list_kets cannot be empty");
  }
  if (list_kets.size() != list_sys_idx.size()) {
    throw std::runtime_error(
        "measure_by_state_product: list_kets and list_sys_idx must have the same length");
  }

  std::vector<int64_t> merged_sys_idx;
  merged_sys_idx.reserve(sys_dim_.size());
  for (const auto &idx_group : list_sys_idx) {
    merged_sys_idx.insert(merged_sys_idx.end(), idx_group.begin(),
                          idx_group.end());
  }
  if (merged_sys_idx.empty()) {
    throw std::runtime_error(
        "measure_by_state_product: list_sys_idx cannot be empty");
  }
  {
    std::set<int64_t> seen;
    for (auto x : merged_sys_idx) {
      if (seen.find(x) != seen.end()) {
        throw std::runtime_error(
            "measure_by_state_product: sys_idx contains duplicates");
      }
      seen.insert(x);
    }
  }

  auto out = clone();

  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : merged_sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(merged_sys_idx.begin(), merged_sys_idx.end(), x) ==
        merged_sys_idx.end()) {
      target.push_back(x);
    }
  }
  out.set_system_seq(target);

  struct KetInfo {
    torch::Tensor k_flat;
    int64_t a = 0;
    int64_t o = 0;
  };
  std::vector<KetInfo> infos;
  infos.reserve(list_kets.size());

  int64_t num_outcomes = 1;
  int64_t applied_dim_total = 1;
  for (const auto &ket_in : list_kets) {
    auto k = ket_in;
    if (k.dim() == 2) {
      k = k.unsqueeze(0);
    }
    if (k.dim() < 3) {
      throw std::runtime_error(
          "measure_by_state_product: ket must have shape [..., dim, 1]");
    }
    const auto a = k.size(-2);
    int64_t o = 1;
    for (int64_t i = 0; i < k.dim() - 2; ++i) {
      o *= k.size(i);
    }
    auto k_flat = k.contiguous().view({o, a, 1}).reshape({o, a});

    infos.push_back(KetInfo{std::move(k_flat), a, o});
    applied_dim_total *= a;
    num_outcomes = std::max<int64_t>(num_outcomes, o);
  }

  for (auto &info : infos) {
    if (info.o == num_outcomes) {
      continue;
    }
    if (info.o == 1) {
      info.k_flat = info.k_flat.expand({num_outcomes, info.a}).contiguous();
      info.o = num_outcomes;
      continue;
    }
    throw std::runtime_error(
        "measure_by_state_product: kets have incompatible num_outcomes (must be equal or 1)");
  }

  const auto total_dim = out.dim();
  if (applied_dim_total <= 0 || total_dim % applied_dim_total != 0) {
    throw std::runtime_error(
        "measure_by_state_product: applied_dim_total does not divide total_dim");
  }
  const auto rest_dim = total_dim / applied_dim_total;

  auto rho0 =
      out.data_.contiguous().view({-1, applied_dim_total, rest_dim, applied_dim_total, rest_dim});
  const auto B = rho0.size(0);

  int64_t applied_rem = applied_dim_total;
  torch::Tensor cur;
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto a = infos[i].a;
    if (applied_rem % a != 0) {
      throw std::runtime_error(
          "measure_by_state_product: internal applied_dim factorization mismatch");
    }
    const auto next_applied_rem = applied_rem / a;
    const auto AR = next_applied_rem * rest_dim;
    auto k_conj = infos[i].k_flat.conj();
    auto k_plain = infos[i].k_flat;

    if (i == 0) {
      auto v = rho0.view({B, a, AR, a, AR});

      auto v_perm = v.permute({1, 0, 2, 3, 4}).contiguous();
      auto v_flat = v_perm.reshape({a, B * AR * a * AR});
      auto temp_mat = at::matmul(k_conj, v_flat);
      auto temp =
          temp_mat.reshape({num_outcomes, B, AR, a, AR});

      auto temp_perm =
          temp.permute({0, 1, 2, 4, 3}).contiguous();
      auto temp_flat =
          temp_perm.reshape({num_outcomes, B * AR * AR, a});
      auto k_mat = k_plain.unsqueeze(-1);
      auto cur_flat =
          at::bmm(temp_flat, k_mat).squeeze(-1);
      cur = cur_flat.reshape({num_outcomes, B, AR, AR})
                .permute({1, 0, 2, 3})
                .contiguous();
    } else {
      auto v = cur.view({B, num_outcomes, a, AR, a, AR});

      auto v_perm =
          v.permute({1, 0, 3, 4, 5, 2}).contiguous();
      auto v_flat = v_perm.reshape(
          {num_outcomes, B * AR * a * AR, a});
      auto k_conj_mat = k_conj.unsqueeze(-1);
      auto temp_flat =
          at::bmm(v_flat, k_conj_mat).squeeze(-1);
      auto temp =
          temp_flat.reshape({num_outcomes, B, AR, a, AR});

      auto temp_perm =
          temp.permute({0, 1, 2, 4, 3}).contiguous();
      auto temp_flat2 =
          temp_perm.reshape({num_outcomes, B * AR * AR, a});
      auto k_plain_mat = k_plain.unsqueeze(-1);
      auto cur_flat =
          at::bmm(temp_flat2, k_plain_mat).squeeze(-1);
      cur = cur_flat.reshape({num_outcomes, B, AR, AR})
                .permute({1, 0, 2, 3})
                .contiguous();
    }

    applied_rem = next_applied_rem;
  }

  auto rho_rest = cur.contiguous().view({B, num_outcomes, rest_dim, rest_dim});

  auto prob =
      at::real(rho_rest.diagonal( 0, -2, -1).sum(-1));

  std::vector<int64_t> data_batch = out.batch_dim_;
  std::vector<int64_t> prior_prob = out.prob().shape();
  std::vector<int64_t> base_batch = data_batch;
  base_batch.insert(base_batch.end(), prior_prob.begin(), prior_prob.end());

  prob = prob.contiguous();
  std::vector<int64_t> target_shape = base_batch;
  target_shape.push_back(num_outcomes);
  prob = prob.reshape(target_shape);

  if (!keep_rest) {
    return {prob, std::nullopt};
  }

  auto prob_flat =
      at::real(rho_rest.diagonal( 0, -2, -1).sum(-1));
  auto mask = at::abs(prob_flat) >= 1e-10;
  auto denom = at::where(mask, prob_flat, at::ones_like(prob_flat));
  auto rho_rest_norm = rho_rest / denom.unsqueeze(-1).unsqueeze(-1);
  rho_rest_norm = rho_rest_norm * mask.unsqueeze(-1).unsqueeze(-1);
  rho_rest_norm =
      at::where(at::isnan(rho_rest_norm), at::zeros_like(rho_rest_norm),
                rho_rest_norm);

  std::vector<int64_t> rest_sys_dim;
  rest_sys_dim.reserve(sys_dim_.size() - merged_sys_idx.size());
  std::vector<int64_t> rest_orig_seq;
  rest_orig_seq.reserve(sys_dim_.size() - merged_sys_idx.size());
  std::set<int64_t> measured_set(merged_sys_idx.begin(), merged_sys_idx.end());
  for (size_t i = 0; i < sys_dim_.size(); ++i) {
    auto orig_idx = system_seq_[i];
    if (measured_set.find(orig_idx) == measured_set.end()) {
      rest_sys_dim.push_back(sys_dim_[orig_idx]);
      rest_orig_seq.push_back(orig_idx);
    }
  }

  if (rest_sys_dim.empty()) {
    rest_sys_dim.push_back(1);
    rest_orig_seq.push_back(0);
  }

  rho_rest_norm = rho_rest_norm.contiguous().view({-1, rest_dim, rest_dim});

  auto rest_orig_sorted = rest_orig_seq;
  std::sort(rest_orig_sorted.begin(), rest_orig_sorted.end());
  if (rest_orig_sorted != rest_orig_seq) {
    const auto perm_map = perm_of_list(rest_orig_seq, rest_orig_sorted);
    rho_rest_norm =
        permute_dm_impl(rho_rest_norm, perm_map, rest_sys_dim).contiguous();
    std::vector<int64_t> canonical_dim;
    canonical_dim.reserve(rest_orig_sorted.size());
    for (auto idx : rest_orig_sorted) {
      canonical_dim.push_back(sys_dim_.at(static_cast<size_t>(idx)));
    }
    rest_sys_dim = std::move(canonical_dim);
  }

  std::vector<int64_t> rest_sys_seq =
      arange_vec(static_cast<int64_t>(rest_sys_dim.size()));

  MixedState rest_state(rho_rest_norm, rest_sys_dim, rest_sys_seq, {});
  rest_state.set_batch_dim(data_batch);

  for (const auto &p : out.prob().list()) {
    rest_state.prob().append(p, false, c10::nullopt, c10::nullopt,
                             false);
  }
  rest_state.prob().append(prob, false, c10::nullopt, c10::nullopt,
                           false);

  return {prob, std::optional<MixedState>(std::move(rest_state))};
}

void MixedState::transform_kraus(const torch::Tensor &list_kraus,
                                 const std::vector<int64_t> &sys_idx,
                                 bool on_batch) {
  if (on_batch) {
    auto unitary_batch_dim = list_kraus.sizes().vec();
    if (unitary_batch_dim.size() >= 3) {
      unitary_batch_dim.resize(unitary_batch_dim.size() - 2);
      auto op_batch_dim = unitary_batch_dim;
      if (!op_batch_dim.empty()) {
        op_batch_dim.pop_back();
      }
      if (batch_dim_.empty() && !op_batch_dim.empty()) {
        batch_dim_ = op_batch_dim;
      }
    }
  }

  std::vector<int64_t> target;
  target.reserve(system_seq_.size());
  for (auto x : sys_idx) {
    target.push_back(x);
  }
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      target.push_back(x);
    }
  }
  set_system_seq(target);

  const auto applied_dim = list_kraus.size(-1);
  const auto prob_dim = prob_.product_dim();
  data_ = quairkit_cpp::mixed_kraus_transform(data_, list_kraus, dim(), applied_dim,
                                            prob_dim, on_batch);
}

void MixedState::transform_many_kraus(
    const torch::Tensor &list_kraus,
    const std::vector<std::vector<int64_t>> &sys_idx_list,
    bool on_batch) {
  if (sys_idx_list.empty()) {
    return;
  }

  const bool all_single =
      std::all_of(sys_idx_list.begin(), sys_idx_list.end(),
                  [](const std::vector<int64_t> &v) { return v.size() == 1; });
  const bool unbatched_kraus = list_kraus.dim() == 3;
  if (!all_single || !unbatched_kraus) {
    for (const auto &idx : sys_idx_list) {
      transform_kraus(list_kraus, idx, on_batch);
    }
    return;
  }

  std::vector<int64_t> flat;
  flat.reserve(sys_idx_list.size());
  for (const auto &idx : sys_idx_list) {
    flat.push_back(idx[0]);
  }

  const auto local_dim = list_kraus.size(-1);
  if (local_dim <= 0) {
    for (const auto &idx : sys_idx_list) {
      transform_kraus(list_kraus, idx, on_batch);
    }
    return;
  }
  const bool local_dim_match = std::all_of(
      flat.begin(), flat.end(), [&](int64_t q) {
        return sys_dim_.at(static_cast<size_t>(q)) == local_dim;
      });
  if (!local_dim_match) {
    for (const auto &idx : sys_idx_list) {
      transform_kraus(list_kraus, idx, on_batch);
    }
    return;
  }

  const auto rank = list_kraus.size(-3);
  const auto chunk_dim = max_fusable_count(local_dim, 64);
  const auto chunk_rank =
      (rank <= 1) ? chunk_dim : max_fusable_count(rank, 64);
  const auto chunk = std::max<int64_t>(1, std::min(chunk_dim, chunk_rank));

  const auto groups = build_disjoint_single_groups(flat, chunk);
  for (const auto &group : groups) {
    const auto len = static_cast<int64_t>(group.size());
    auto fused = (len == 1) ? list_kraus : kraus_kron_repeat(list_kraus, len);
    std::vector<int64_t> seq(group.rbegin(), group.rend());
    transform_kraus(fused, seq, on_batch);
  }
  return;
}

void MixedState::transform_many_choi(
    const torch::Tensor &choi,
    const std::vector<std::vector<int64_t>> &sys_idx_list,
    bool on_batch) {
  for (const auto &idx : sys_idx_list) {
    transform_choi(choi, idx, on_batch);
  }
}

void MixedState::transform_choi(const torch::Tensor &choi,
                                const std::vector<int64_t> &sys_idx,
                                bool on_batch) {
  if (on_batch) {
    auto choi_batch_dim = choi.sizes().vec();
    if (choi_batch_dim.size() >= 2) {
      choi_batch_dim.resize(choi_batch_dim.size() - 2);
      if (!choi_batch_dim.empty() && batch_dim_.empty()) {
        batch_dim_ = choi_batch_dim;
      }
    }
  }

  std::vector<int64_t> refer_sys_idx;
  refer_sys_idx.reserve(system_seq_.size());
  for (auto x : system_seq_) {
    if (std::find(sys_idx.begin(), sys_idx.end(), x) == sys_idx.end()) {
      refer_sys_idx.push_back(x);
    }
  }

  int64_t dim_refer = 1;
  for (auto x : refer_sys_idx) {
    dim_refer *= sys_dim_.at(static_cast<size_t>(x));
  }
  int64_t dim_in = 1;
  for (auto x : sys_idx) {
    dim_in *= sys_dim_.at(static_cast<size_t>(x));
  }
  const auto dim_out = choi.size(-1) / dim_in;

  std::vector<int64_t> target = refer_sys_idx;
  target.insert(target.end(), sys_idx.begin(), sys_idx.end());
  set_system_seq(target);

  const auto prob_dim = prob_.product_dim();
  data_ = quairkit_cpp::mixed_choi_transform(data_, choi, dim_refer, dim_in, dim_out,
                                            prob_dim, on_batch);
}

MixedState MixedState::trace(const std::vector<int64_t> &trace_idx) {
  std::vector<int64_t> remain_seq;
  std::vector<int64_t> remain_dim;
  for (auto x : system_seq_) {
    if (std::find(trace_idx.begin(), trace_idx.end(), x) == trace_idx.end()) {
      remain_seq.push_back(x);
      remain_dim.push_back(sys_dim_.at(static_cast<size_t>(x)));
    }
  }

  int64_t trace_dim = 1;
  for (auto x : trace_idx) {
    trace_dim *= sys_dim_.at(static_cast<size_t>(x));
  }

  std::vector<int64_t> target = trace_idx;
  target.insert(target.end(), remain_seq.begin(), remain_seq.end());
  set_system_seq(target);

  auto reduced = quairkit_cpp::mixed_trace_1(data_, trace_dim);
  const auto rest_dim = dim() / trace_dim;
  std::vector<int64_t> out_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  out_shape.insert(out_shape.end(), prob_shape.begin(), prob_shape.end());
  out_shape.push_back(rest_dim);
  out_shape.push_back(rest_dim);
  reduced = reduced.view(out_shape);

  std::vector<int64_t> sorted_remain = remain_seq;
  std::sort(sorted_remain.begin(), sorted_remain.end());
  std::vector<int64_t> mapped;
  mapped.reserve(remain_seq.size());
  for (auto v : remain_seq) {
    auto it = std::find(sorted_remain.begin(), sorted_remain.end(), v);
    mapped.push_back(static_cast<int64_t>(std::distance(sorted_remain.begin(), it)));
  }

  return MixedState(reduced, remain_dim, mapped, prob_.clone_list());
}

MixedState MixedState::reset(const std::vector<int64_t> &reset_idx,
                             const torch::Tensor &replace_dm) {
  std::vector<int64_t> remain_seq;
  for (auto x : system_seq_) {
    if (std::find(reset_idx.begin(), reset_idx.end(), x) == reset_idx.end()) {
      remain_seq.push_back(x);
    }
  }
  std::sort(remain_seq.begin(), remain_seq.end());

  int64_t trace_dim = 1;
  for (auto x : reset_idx) {
    trace_dim *= sys_dim_.at(static_cast<size_t>(x));
  }

  std::vector<int64_t> target = reset_idx;
  target.insert(target.end(), remain_seq.begin(), remain_seq.end());
  set_system_seq(target);

  auto out = quairkit_cpp::mixed_reset_1(data_, replace_dm, trace_dim);
  const auto d = dim();
  std::vector<int64_t> out_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  out_shape.insert(out_shape.end(), prob_shape.begin(), prob_shape.end());
  out_shape.push_back(d);
  out_shape.push_back(d);
  out = out.view(out_shape);
  return MixedState(out, sys_dim_, system_seq_, prob_.clone_list());
}

MixedState MixedState::transpose(const std::vector<int64_t> &transpose_idx) {
  std::vector<int64_t> target = transpose_idx;
  for (auto x : system_seq_) {
    if (std::find(transpose_idx.begin(), transpose_idx.end(), x) ==
        transpose_idx.end()) {
      target.push_back(x);
    }
  }
  set_system_seq(target);

  int64_t transpose_dim = 1;
  for (auto x : transpose_idx) {
    transpose_dim *= sys_dim_.at(static_cast<size_t>(x));
  }

  auto total_dim = dim();
  auto dim2 = total_dim / transpose_dim;
  auto reshaped = data_.reshape({-1, transpose_dim, dim2, transpose_dim, dim2});
  auto permuted = reshaped.permute({0, 3, 2, 1, 4});
  auto out = permuted.reshape({-1, total_dim, total_dim});
  return MixedState(out, sys_dim_, system_seq_, prob_.clone_list());
}

}