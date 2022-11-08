// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "common_headers.h"    // NOLINT
#include "fused_dense_cuda.h"  // NOLINT

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y = ctx.Input<f::Tensor>("Y");
  const auto* bias = ctx.Input<f::Tensor>("Bias");
  auto* out = ctx.Output<f::Tensor>("Out");
  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr(transx) must be False currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y_dims = y->dims();
  int y_k = y_dims[0];
  int y_n = y_dims[1];
  if (transy) {
    y_k = y_dims[1];
    y_n = y_dims[0];
  }
  if (x_k != y_k) {
    PD_THROW("The reudce dim of A and B in matmul is not equal.");
  }

  auto out_dims = x_dims;
  out_dims[x_dims.size() - 1] = y_n;
  out->Resize(out_dims);

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_bias_cuda_forward_impl", ([&] {
        linear_bias_cuda_forward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            y->data<data_t>(),
            bias->data<data_t>(),
            transx,
            transy,
            x_k,
            x_m,
            y_n,
            out->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense_grad, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y = ctx.Input<f::Tensor>("Y");
  const auto* grad_out = ctx.Input<f::Tensor>(f::GradVarName("Out"));
  auto* grad_x = ctx.Output<f::Tensor>(f::GradVarName("X"));
  auto* grad_y = ctx.Output<f::Tensor>(f::GradVarName("Y"));
  auto* grad_bias = ctx.Output<f::Tensor>(f::GradVarName("Bias"));

  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  bool use_addto = ctx.Attr<bool>("use_addto");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr(transx) must be False currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y_dims = y->dims();
  int y_k = y_dims[0];
  int y_n = y_dims[1];
  if (transy) {
    y_k = y_dims[1];
    y_n = y_dims[0];
  }
  if (x_k != y_k) {
    PD_THROW("The reudce dim of A and B in matmul is not equal.");
  }

  grad_x->Resize(x_dims);
  grad_y->Resize(y_dims);
  grad_bias->Resize({y_n});

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  PD_THROW(
      "fused_dense_cuda_backward is not supported on cuda_version < 11000");
#endif

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_bias_cuda_backward_impl", ([&] {
        linear_bias_cuda_backward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            y->data<data_t>(),
            grad_out->data<data_t>(),
            transx,
            transy,
            use_addto,
            x_k,
            x_m,
            y_n,
            grad_y->mutable_data<data_t>(place),
            grad_bias->mutable_data<data_t>(place),
            grad_x->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}
