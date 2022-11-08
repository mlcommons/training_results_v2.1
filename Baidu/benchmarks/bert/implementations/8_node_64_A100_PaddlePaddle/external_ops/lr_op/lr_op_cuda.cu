// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "common_headers.h"  // NOLINT

// x_data[0]: current step which is numbered from 0.
// Note: when computing, we should use x_data[0] + 1.
// y_data[0]: the lr var of this step
__global__ void compute_lr_fwd_kernel(const int64_t* x_data,
                                      float* y_data,
                                      float base_lr,
                                      float end_lr,
                                      float degree,
                                      int64_t start_warmup_step,
                                      int64_t warmup_step,
                                      int64_t max_step) {
  int64_t step = x_data[0] + 1;
  int64_t offset_step = (start_warmup_step == 0 ? 1 : 0);
  int64_t mod_step = step - offset_step - start_warmup_step;

  double y;
  if (mod_step < warmup_step) {
    auto p = mod_step / (warmup_step + 1e-6);
    y = base_lr * p;
  } else {
    auto p = (step - offset_step) / static_cast<double>(max_step);
    p = (p >= 1 ? 0 : (::pow(1 - p, degree)));
    y = (base_lr - end_lr) * p + end_lr;
  }

  y_data[0] = static_cast<float>(y);
  // y_data[0] = base_lr * (static_cast<float>(max_step - x_data[0]) /
  // max_step);
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_lr, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  auto* out = ctx.Output<f::Tensor>("Out");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();

  float base_lr = ctx.Attr<float>("base_lr");
  float end_lr = ctx.Attr<float>("end_lr");
  float degree = ctx.Attr<float>("degree");

  int64_t start_warmup_step = ctx.Attr<int64_t>("start_warmup_step");
  int64_t warmup_step = ctx.Attr<int64_t>("warmup_step");
  int64_t max_step = ctx.Attr<int64_t>("max_step");

  const auto& x_dims = x->dims();
  if (x_dims.size() != 1 || x_dims[0] != 1) {
    PD_THROW("The shape of input x must be [1].");
  }
  auto out_dims = x_dims;
  out->Resize(out_dims);

  const int64_t* x_data = x->data<int64_t>();
  float* out_data = out->mutable_data<float>(x->place());

  compute_lr_fwd_kernel<<<1, 1, 0, stream>>>(x_data,
                                             out_data,
                                             base_lr,
                                             end_lr,
                                             degree,
                                             start_warmup_step,
                                             warmup_step,
                                             max_step);
}
