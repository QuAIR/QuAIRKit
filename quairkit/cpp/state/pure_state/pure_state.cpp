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

#include "pure_state.h"

#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <set>
#include <stdexcept>

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

torch::Tensor permute_sv_impl(const torch::Tensor &state,
                              const std::vector<int64_t> &perm,
                              const std::vector<int64_t> &system_dim) {
  const auto total_dim = product_int64(system_dim);
  std::vector<int64_t> shape;
  shape.reserve(system_dim.size() + 1);
  shape.push_back(state.size(0));
  shape.insert(shape.end(), system_dim.begin(), system_dim.end());

  std::vector<int64_t> axes;
  axes.reserve(perm.size() + 1);
  axes.push_back(0);
  for (auto p : perm) {
    axes.push_back(1 + p);
  }

  return state.reshape(shape)
      .permute(axes)
      .contiguous()
      .view({state.size(0), total_dim});
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

PureState::PureState(const torch::Tensor &data,
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
  data_ = data.reshape({-1, d});
}

void PureState::infer_batch_dim_from_input_shape(const torch::Tensor &input_data,
                                                 int64_t num_prob) {
  auto sizes = input_data.sizes().vec();
  if (sizes.empty()) {
    batch_dim_.clear();
    return;
  }

  if (static_cast<int64_t>(sizes.size()) == 1) {
    batch_dim_.clear();
    return;
  }

  const auto non_batch_len = 2 + num_prob;
  const auto prefix_len =
      std::max<int64_t>(0, static_cast<int64_t>(sizes.size()) - non_batch_len);
  batch_dim_.assign(sizes.begin(), sizes.begin() + prefix_len);
}

int64_t PureState::dim() const { return product_int64(sys_dim_); }

void PureState::reset_sequence() { set_system_seq(arange_vec(num_systems())); }

void PureState::set_system_dim(const std::vector<int64_t> &sys_dim) {
  sys_dim_ = sys_dim;
  const auto n = static_cast<int64_t>(sys_dim_.size());
  if (static_cast<int64_t>(system_seq_.size()) != n) {
    system_seq_ = arange_vec(n);
  }
}

int64_t PureState::num_systems() const {
  return static_cast<int64_t>(sys_dim_.size());
}

void PureState::set_system_seq(const std::vector<int64_t> &target_seq) {
  if (target_seq == system_seq_) {
    return;
  }

  const auto perm_map = perm_of_list(system_seq_, target_seq);
  std::vector<int64_t> current_dim;
  current_dim.reserve(system_seq_.size());
  for (auto x : system_seq_) {
    current_dim.push_back(sys_dim_.at(static_cast<size_t>(x)));
  }

  data_ = permute_sv_impl(data_, perm_map, current_dim).contiguous();
  system_seq_ = target_seq;
}

void PureState::set_system_seq_metadata(const std::vector<int64_t> &target_seq) {
  const auto n = num_systems();
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

torch::Tensor PureState::ket() const {
  const auto d = dim();
  std::vector<int64_t> shape = batch_dim_;
  auto prob_shape = prob_.shape();
  shape.insert(shape.end(), prob_shape.begin(), prob_shape.end());
  shape.push_back(d);
  shape.push_back(1);
  return data_.reshape(shape).clone();
}

torch::Tensor PureState::density_matrix() const {
  auto k = ket();
  return at::matmul(k, k.mH());
}

PureState PureState::clone() const {
  PureState out = *this;
  out.data_ = data_.clone();
  out.prob_ = prob_.clone();
  return out;
}

void PureState::to(c10::optional<torch::Dtype> dtype,
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

void PureState::index_select(const torch::Tensor &new_indices) {
  set_system_seq(arange_vec(num_systems()));
  data_ = data_.index_select( -1, new_indices);
}

void PureState::add_probability(const torch::Tensor &prob) {
  auto shaped = prob_.prepare_new(prob, c10::nullopt,
                                 c10::nullopt, false);

  const auto d = dim();
  std::vector<int64_t> view_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  view_shape.insert(view_shape.end(), prob_shape.begin(), prob_shape.end());
  view_shape.push_back(1);
  view_shape.push_back(d);

  auto data_view = data_.view(view_shape);
  auto reps = std::vector<int64_t>(view_shape.size(), 1);
  reps[static_cast<size_t>(view_shape.size() - 2)] = shaped.size(-1);
  data_view = data_view.repeat(reps);
  data_ = data_view.view({-1, d});

  prob_.append(shaped, false, c10::nullopt, c10::nullopt,
               false);
}

PureState PureState::prob_select(const torch::Tensor &outcome_idx,
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
  auto k = ket();
  auto selected = k.index_select( data_idx, outcome_idx).squeeze(data_idx);
  return PureState(selected, sys_dim_, system_seq_, new_prob);
}

void PureState::evolve(const torch::Tensor &unitary,
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
  data_ = quairkit_cpp::pure_evolve(data_, unitary, dim(), applied_dim, prob_dim,
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
  int64_t k = 1;
  int64_t acc = local_dim;
  while (acc * local_dim <= max_hilbert) {
    acc *= local_dim;
    ++k;
  }
  return k;
}
}

void PureState::evolve_many(const torch::Tensor &unitary,
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

void PureState::evolve_many_batched(
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

void PureState::evolve_keep_dim(const torch::Tensor &unitary,
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
  data_ = quairkit_cpp::pure_evolve_keep_dim(data_, unitary, dim(), applied_dim,
                                            prob_dim, on_batch);
}

void PureState::evolve_ctrl(const torch::Tensor &unitary, int64_t index,
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
  flat.reserve(system_seq_.size());
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
  data_ = quairkit_cpp::pure_evolve(data_, u_ctrl, dim(), applied_dim, prob_dim,
                                   on_batch);
}

torch::Tensor PureState::expectation_value(const torch::Tensor &obs,
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
  auto out = quairkit_cpp::pure_expec_val(data_, obs, dim(), applied_dim);
  const auto num_obs = obs.size(-3);

  std::vector<int64_t> out_shape = batch_dim_;
  auto prob_shape = prob_.shape();
  out_shape.insert(out_shape.end(), prob_shape.begin(), prob_shape.end());
  out_shape.push_back(num_obs);
  return out.view(out_shape);
}

namespace {
torch::Tensor apply_observable_contiguous(const torch::Tensor &psi,
                                          const torch::Tensor &op,
                                          const std::vector<int64_t> &dims,
                                          int64_t start, int64_t len) {
  int64_t prefix = 1;
  for (int64_t i = 0; i < start; ++i) {
    prefix *= dims.at(static_cast<size_t>(i));
  }

  int64_t d_act = 1;
  for (int64_t i = 0; i < len; ++i) {
    d_act *= dims.at(static_cast<size_t>(start + i));
  }

  int64_t suffix = 1;
  for (size_t i = static_cast<size_t>(start + len); i < dims.size(); ++i) {
    suffix *= dims.at(i);
  }

  const auto total_dim = prefix * d_act * suffix;
  auto psi_view = psi.view({-1, total_dim});

  auto reshaped = psi_view.view({-1, prefix, d_act, suffix});
  auto permuted = reshaped.permute({0, 1, 3, 2}).contiguous();
  auto flat = permuted.view({-1, d_act});

  auto applied = at::matmul(flat, op.transpose(-2, -1));

  auto restored =
      applied.view({-1, prefix, suffix, d_act}).permute({0, 1, 3, 2});
  return restored.contiguous().view({-1, total_dim});
}
}

torch::Tensor
PureState::expectation_value_product(const std::vector<torch::Tensor> &obs_list,
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

  auto phi = data_.clone();

  int64_t start = 0;
  for (size_t k = 0; k < obs_list.size(); ++k) {
    const auto &idx_group = sys_idx_list[k];
    const auto &op = obs_list[k];
    const auto len = static_cast<int64_t>(idx_group.size());

    if (len == 0) {
      continue;
    }

    phi = apply_observable_contiguous(phi, op, dims_current, start, len);
    start += len;
  }

  auto result = at::sum(data_.conj() * phi, -1);
  return result.reshape({-1});
}

torch::Tensor PureState::expec_val_pauli_terms(
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

std::tuple<torch::Tensor, PureState>
PureState::measure(const torch::Tensor &measure_op,
                   const std::vector<int64_t> &sys_idx) {
  auto out = clone();
  out.evolve_keep_dim(measure_op, sys_idx, true);

  auto prob_and = quairkit_cpp::pure_measure_collapse(out.data_, out.dim());
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

  out.data_ = collapsed.view({-1, out.dim()});
  if (!out.batch_dim_.empty()) {
    out.batch_dim_.pop_back();
  }
  out.prob_.append(prob, false, c10::nullopt, c10::nullopt,
                   false);
  return {prob, out};
}

std::tuple<torch::Tensor, PureState>
PureState::measure_many(const torch::Tensor &measure_op,
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

std::tuple<torch::Tensor, std::optional<PureState>>
PureState::measure_by_state(const torch::Tensor &measure_basis_ket,
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

  auto psi = out.data_.view({-1, applied_dim, rest_dim});
  auto bra = k.conj().transpose(-2, -1);
  auto amp = at::matmul(bra, psi.unsqueeze(1)).squeeze(-2);

  auto prob = at::real(at::sum(at::abs(amp) * at::abs(amp), -1));

  std::vector<int64_t> data_batch = out.batch_dim_;
  std::vector<int64_t> prior_prob = out.prob().shape();
  std::vector<int64_t> base_batch = data_batch;
  base_batch.insert(base_batch.end(), prior_prob.begin(), prior_prob.end());

  prob = prob.contiguous();
  std::vector<int64_t> target_shape = base_batch;
  target_shape.push_back(num_outcomes);
  prob = prob.view(target_shape);

  if (!keep_rest) {
    return {prob, std::nullopt};
  }

  auto prob_flat = at::real(at::sum(at::abs(amp) * at::abs(amp), -1));
  auto mask = at::abs(prob_flat) >= 1e-10;
  auto denom = at::sqrt(at::where(mask, prob_flat, at::ones_like(prob_flat)));
  auto denom_view = denom.unsqueeze(-1);
  auto rest_collapsed = amp / denom_view;
  rest_collapsed = rest_collapsed * mask.unsqueeze(-1);
  rest_collapsed =
      at::where(at::isnan(rest_collapsed), at::zeros_like(rest_collapsed),
                rest_collapsed);

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

  rest_collapsed = rest_collapsed.contiguous().view({-1, rest_dim});

  auto rest_orig_sorted = rest_orig_seq;
  std::sort(rest_orig_sorted.begin(), rest_orig_sorted.end());
  if (rest_orig_sorted != rest_orig_seq) {
    const auto perm_map = perm_of_list(rest_orig_seq, rest_orig_sorted);
    rest_collapsed = permute_sv_impl(rest_collapsed, perm_map, rest_sys_dim).contiguous();
    std::vector<int64_t> canonical_dim;
    canonical_dim.reserve(rest_orig_sorted.size());
    for (auto idx : rest_orig_sorted) {
      canonical_dim.push_back(sys_dim_.at(static_cast<size_t>(idx)));
    }
    rest_sys_dim = std::move(canonical_dim);
  }

  std::vector<int64_t> rest_sys_seq =
      arange_vec(static_cast<int64_t>(rest_sys_dim.size()));

  PureState rest_state(rest_collapsed, rest_sys_dim, rest_sys_seq, {});
  rest_state.set_batch_dim(data_batch);
  
  for (const auto &p : out.prob().list()) {
    rest_state.prob().append(p, false, c10::nullopt, c10::nullopt,
                             false);
  }
  rest_state.prob().append(prob, false, c10::nullopt, c10::nullopt,
                           false);

  return {prob, std::optional<PureState>(std::move(rest_state))};
}

std::tuple<torch::Tensor, std::optional<PureState>>
PureState::measure_by_state_product(
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

  auto psi = out.data_.view({-1, applied_dim_total, rest_dim});
  const auto B = psi.size(0);

  int64_t applied_rem = applied_dim_total;
  torch::Tensor cur;
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto a = infos[i].a;
    if (applied_rem % a != 0) {
      throw std::runtime_error(
          "measure_by_state_product: internal applied_dim factorization mismatch");
    }
    const auto next_applied_rem = applied_rem / a;
    const auto rem = next_applied_rem * rest_dim;
    auto k_conj = infos[i].k_flat.conj();

    if (i == 0) {
      auto cur_view = psi.view({B, a, rem});
      cur = at::einsum("oa,bar->bor", {k_conj, cur_view});
    } else {
      auto cur_view = cur.view({B, num_outcomes, a, rem});
      cur = at::einsum("oa,boar->bor", {k_conj, cur_view});
    }

    applied_rem = next_applied_rem;
  }

  auto amp = cur.contiguous().view({B, num_outcomes, rest_dim});

  auto prob = at::real(at::sum(at::abs(amp) * at::abs(amp), -1));

  std::vector<int64_t> data_batch = out.batch_dim_;
  std::vector<int64_t> prior_prob = out.prob().shape();
  std::vector<int64_t> base_batch = data_batch;
  base_batch.insert(base_batch.end(), prior_prob.begin(), prior_prob.end());

  prob = prob.contiguous();
  std::vector<int64_t> target_shape = base_batch;
  target_shape.push_back(num_outcomes);
  prob = prob.view(target_shape);

  if (!keep_rest) {
    return {prob, std::nullopt};
  }

  auto prob_flat = at::real(at::sum(at::abs(amp) * at::abs(amp), -1));
  auto mask = at::abs(prob_flat) >= 1e-10;
  auto denom = at::sqrt(at::where(mask, prob_flat, at::ones_like(prob_flat)));
  auto denom_view = denom.unsqueeze(-1);
  auto rest_collapsed = amp / denom_view;
  rest_collapsed = rest_collapsed * mask.unsqueeze(-1);
  rest_collapsed =
      at::where(at::isnan(rest_collapsed), at::zeros_like(rest_collapsed),
                rest_collapsed);

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

  rest_collapsed = rest_collapsed.contiguous().view({-1, rest_dim});

  auto rest_orig_sorted = rest_orig_seq;
  std::sort(rest_orig_sorted.begin(), rest_orig_sorted.end());
  if (rest_orig_sorted != rest_orig_seq) {
    const auto perm_map = perm_of_list(rest_orig_seq, rest_orig_sorted);
    rest_collapsed =
        permute_sv_impl(rest_collapsed, perm_map, rest_sys_dim).contiguous();
    std::vector<int64_t> canonical_dim;
    canonical_dim.reserve(rest_orig_sorted.size());
    for (auto idx : rest_orig_sorted) {
      canonical_dim.push_back(sys_dim_.at(static_cast<size_t>(idx)));
    }
    rest_sys_dim = std::move(canonical_dim);
  }

  std::vector<int64_t> rest_sys_seq =
      arange_vec(static_cast<int64_t>(rest_sys_dim.size()));

  PureState rest_state(rest_collapsed, rest_sys_dim, rest_sys_seq, {});
  rest_state.set_batch_dim(data_batch);

  for (const auto &p : out.prob().list()) {
    rest_state.prob().append(p, false, c10::nullopt, c10::nullopt,
                             false);
  }
  rest_state.prob().append(prob, false, c10::nullopt, c10::nullopt,
                           false);

  return {prob, std::optional<PureState>(std::move(rest_state))};
}

}