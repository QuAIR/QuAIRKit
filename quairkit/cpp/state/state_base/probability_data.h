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

#include <algorithm>
#include <cstdint>
#include <vector>

namespace quairkit_cpp {

class ProbabilityDataImpl {
public:
  ProbabilityDataImpl() = default;
  explicit ProbabilityDataImpl(const std::vector<torch::Tensor> &probs) {
    for (const auto &p : probs) {
      list_.push_back(p);
      dims_.push_back(static_cast<int64_t>(p.size(-1)));
    }
  }

  int64_t size() const { return static_cast<int64_t>(list_.size()); }
  bool empty() const { return list_.empty(); }

  const std::vector<torch::Tensor> &list() const { return list_; }
  std::vector<torch::Tensor> clone_list() const {
    std::vector<torch::Tensor> out;
    out.reserve(list_.size());
    for (const auto &p : list_) {
      out.push_back(p.clone());
    }
    return out;
  }

  std::vector<int64_t> shape() const { return dims_; }

  std::vector<int64_t> non_prob_dim() const {
    const auto num_prob = static_cast<int64_t>(list_.size());
    if (num_prob == 0) {
      return {};
    }
    auto sizes = list_.back().sizes().vec();
    const auto keep = static_cast<int64_t>(sizes.size()) - num_prob;
    if (keep <= 0) {
      return {};
    }
    sizes.resize(static_cast<size_t>(keep));
    return sizes;
  }

  int64_t product_dim() const {
    int64_t p = 1;
    for (auto d : dims_) {
      p *= d;
    }
    return p;
  }

  void clear() {
    list_.clear();
    dims_.clear();
  }

  torch::Tensor prepare_new(const torch::Tensor &prob,
                            c10::optional<torch::Dtype> dtype,
                            c10::optional<torch::Device> device,
                            bool real_only) const {
    const auto num_prev = static_cast<int64_t>(list_.size());
    std::vector<int64_t> view_shape(static_cast<size_t>(num_prev), 1);
    view_shape.push_back(-1);

    auto p = prob.view(view_shape);
    if (dtype.has_value() || device.has_value()) {
      auto opts = p.options();
      if (dtype.has_value()) {
        opts = opts.dtype(*dtype);
      }
      if (device.has_value()) {
        opts = opts.device(*device);
      }
      p = p.to(opts);
    }
    if (real_only) {
      p = at::real(p);
    }
    return p;
  }

  void append(const torch::Tensor &prob, bool normalize,
              c10::optional<torch::Dtype> dtype,
              c10::optional<torch::Device> device, bool real_only) {
    auto p = normalize ? prepare_new(prob, dtype, device, real_only) : prob;

    if (!normalize && (dtype.has_value() || device.has_value())) {
      auto opts = p.options();
      if (dtype.has_value()) {
        opts = opts.dtype(*dtype);
      }
      if (device.has_value()) {
        opts = opts.device(*device);
      }
      p = p.to(opts);
    }
    if (!normalize && real_only) {
      p = at::real(p);
    }

    list_.push_back(p);
    dims_.push_back(static_cast<int64_t>(p.size(-1)));
  }

  torch::Tensor joint(const std::vector<int64_t> &prob_idx) const {
    if (list_.empty()) {
      return torch::tensor(1.0);
    }

    auto dtype = list_[0].scalar_type();
    auto device = list_[0].device();
    auto result =
        torch::tensor(1.0, torch::TensorOptions().dtype(dtype).device(device));

    std::vector<int64_t> idx = prob_idx;
    std::sort(idx.begin(), idx.end());
    for (auto i : idx) {
      const auto &p = list_.at(static_cast<size_t>(i));
      if (p.dim() > result.dim()) {
        auto shape = result.sizes().vec();
        shape.insert(shape.end(), p.dim() - result.dim(), 1);
        result = result.view(shape);
      }
      result = result * p;
    }
    return result;
  }

  ProbabilityDataImpl clone() const { return ProbabilityDataImpl(clone_list()); }

private:
  std::vector<torch::Tensor> list_;
  std::vector<int64_t> dims_;
};

}