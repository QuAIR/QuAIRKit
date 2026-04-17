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

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../state_base/probability_data.h"

namespace quairkit_cpp {

class PureState {
public:
  PureState(const torch::Tensor &data, const std::vector<int64_t> &sys_dim,
            const std::vector<int64_t> &system_seq = {},
            const std::vector<torch::Tensor> &probability = {});

  void evolve(const torch::Tensor &unitary, const std::vector<int64_t> &sys_idx,
              bool on_batch = true);
  void evolve_many(const torch::Tensor &unitary,
                   const std::vector<std::vector<int64_t>> &sys_idx_list,
                   bool on_batch = true);
  void evolve_many_batched(
      const std::vector<torch::Tensor> &unitary_groups,
      const std::vector<std::vector<std::vector<int64_t>>> &sys_idx_groups,
      bool on_batch = true);
  void evolve_keep_dim(const torch::Tensor &unitary,
                       const std::vector<int64_t> &sys_idx,
                       bool on_batch = true);
  void evolve_ctrl(const torch::Tensor &unitary, int64_t index,
                   const std::vector<std::vector<int64_t>> &sys_idx,
                   bool on_batch = true);

  torch::Tensor expectation_value(const torch::Tensor &obs,
                                  const std::vector<int64_t> &sys_idx);
  torch::Tensor expectation_value_product(
      const std::vector<torch::Tensor> &obs_list,
      const std::vector<std::vector<int64_t>> &sys_idx_list);
  torch::Tensor expec_val_pauli_terms(
      const std::vector<std::string> &pauli_words_r,
      const std::vector<std::vector<int64_t>> &sites);

  std::tuple<torch::Tensor, PureState>
  measure(const torch::Tensor &measure_op, const std::vector<int64_t> &sys_idx);
  std::tuple<torch::Tensor, PureState>
  measure_many(const torch::Tensor &measure_op,
               const std::vector<std::vector<int64_t>> &sys_idx_list);
  std::tuple<torch::Tensor, std::optional<PureState>>
  measure_by_state(const torch::Tensor &measure_basis_ket,
                   const std::vector<int64_t> &sys_idx, bool keep_rest);
  std::tuple<torch::Tensor, std::optional<PureState>>
  measure_by_state_product(const std::vector<torch::Tensor> &list_kets,
                           const std::vector<std::vector<int64_t>> &list_sys_idx,
                           bool keep_rest);

  void add_probability(const torch::Tensor &prob);
  PureState prob_select(const torch::Tensor &outcome_idx, int64_t prob_idx = -1);

  void set_system_seq(const std::vector<int64_t> &target_seq);
  void reset_sequence();
  void set_system_seq_metadata(const std::vector<int64_t> &target_seq);

  torch::Tensor data() const { return data_; }
  void set_data(const torch::Tensor &data) { data_ = data; }
  std::vector<int64_t> system_dim() const { return sys_dim_; }
  std::vector<int64_t> system_seq() const { return system_seq_; }
  std::vector<int64_t> batch_dim() const { return batch_dim_; }
  void set_batch_dim(const std::vector<int64_t> &batch_dim) { batch_dim_ = batch_dim; }
  int64_t dim() const;

  ProbabilityDataImpl &prob() { return prob_; }
  const ProbabilityDataImpl &prob() const { return prob_; }

  torch::Tensor ket() const;
  torch::Tensor density_matrix() const;

  PureState clone() const;
  void index_select(const torch::Tensor &new_indices);
  void to(c10::optional<torch::Dtype> dtype, c10::optional<torch::Device> device);

  void set_system_dim(const std::vector<int64_t> &sys_dim);

private:
  void infer_batch_dim_from_input_shape(const torch::Tensor &input_data,
                                       int64_t num_prob);

  torch::Tensor data_;
  std::vector<int64_t> sys_dim_;
  std::vector<int64_t> system_seq_;
  std::vector<int64_t> batch_dim_;
  ProbabilityDataImpl prob_;

  int64_t num_systems() const;
};

}