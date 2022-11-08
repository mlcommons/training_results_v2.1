// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
// Copyright Apex Library Authros. All Rights Reserved.
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

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense_gelu_dense, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y1 = ctx.Input<f::Tensor>("Y1");
  const auto* y2 = ctx.Input<f::Tensor>("Y2");
  const auto* bias1 = ctx.Input<f::Tensor>("Bias1");
  const auto* bias2 = ctx.Input<f::Tensor>("Bias2");
  auto* out1 = ctx.Output<f::Tensor>("Out1");
  auto* out2 = ctx.Output<f::Tensor>("Out2");
  auto* gelu_input = ctx.Output<f::Tensor>("Gelu_Input");
  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr(transx) must be False currently.");
  }
  if (!transy) {
    PD_THROW("Attr(transy) must be True currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y1_dims = y1->dims();
  const auto& y2_dims = y2->dims();

  int y1_k = y1_dims[0];
  int y1_n = y1_dims[1];
  int y2_k = y2_dims[0];
  int y2_n = y2_dims[1];
  if (transy) {
    y1_k = y1_dims[1];
    y1_n = y1_dims[0];
    y2_k = y2_dims[1];
    y2_n = y2_dims[0];
  }
  if (x_k != y1_k) {
    PD_THROW("The reudce dim of the first in matmul is not equal.");
  }
  if (y1_n != y2_k) {
    PD_THROW("The reudce dim of the second in matmul is not equal.");
  }

  auto out1_dims = x_dims;
  auto out2_dims = x_dims;
  out1_dims[x_dims.size() - 1] = y1_n;
  out2_dims[x_dims.size() - 1] = y2_n;

  out1->Resize(out1_dims);
  out2->Resize(out2_dims);
  gelu_input->Resize(out1_dims);

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_gelu_linear_cuda_forward_impl", ([&] {
        linear_gelu_linear_cuda_forward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            y1->data<data_t>(),
            y2->data<data_t>(),
            bias1->data<data_t>(),
            bias2->data<data_t>(),
            transx,
            transy,
            x_k,   // in_feature (1024)
            x_m,   // batch_size
            y1_n,  // hidden_feature (4096)
            y2_n,  // out_feature (1024)
            out1->mutable_data<data_t>(place),
            gelu_input->mutable_data<data_t>(place),
            out2->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}

__PD_DEFINE_RAW_OP_KERNEL_FUNC(custom_fused_dense_gelu_dense_grad, ctx) {
  namespace f = paddle::framework;
  const auto* x = ctx.Input<f::Tensor>("X");
  const auto* y1 = ctx.Input<f::Tensor>("Y1");
  const auto* y2 = ctx.Input<f::Tensor>("Y2");
  const auto* output1 = ctx.Input<f::Tensor>("Out1");
  const auto* gelu_input = ctx.Input<f::Tensor>("Gelu_Input");
  const auto* grad_out2 = ctx.Input<f::Tensor>(f::GradVarName("Out2"));

  auto* grad_x = ctx.Output<f::Tensor>(f::GradVarName("X"));
  auto* grad_y1 = ctx.Output<f::Tensor>(f::GradVarName("Y1"));
  auto* grad_y2 = ctx.Output<f::Tensor>(f::GradVarName("Y2"));
  auto* grad_bias1 = ctx.Output<f::Tensor>(f::GradVarName("Bias1"));
  auto* grad_bias2 = ctx.Output<f::Tensor>(f::GradVarName("Bias2"));

  bool transx = ctx.Attr<bool>("transx");
  bool transy = ctx.Attr<bool>("transy");
  auto& dev_ctx = ctx.cuda_device_context();
  auto place = dev_ctx.GetPlace();

  if (transx) {
    PD_THROW("Attr (transx) must be False currently.");
  }
  if (!transy) {
    PD_THROW("Attr (transy) must be True currently.");
  }

  const auto& x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_k = x_dims[x_dims.size() - 1];

  const auto& y1_dims = y1->dims();
  const auto& y2_dims = y2->dims();
  int y1_k = y1_dims[0];
  int y1_n = y1_dims[1];
  int y2_k = y2_dims[0];
  int y2_n = y2_dims[1];
  if (transy) {
    y1_k = y1_dims[1];
    y1_n = y1_dims[0];
    y2_k = y2_dims[1];
    y2_n = y2_dims[0];
  }
  if (x_k != y1_k) {
    PD_THROW("The reudce dim of the first matmul is not equal.");
  }
  if (y1_n != y2_k) {
    PD_THROW("The reudce dim of the second matmul is not equal.");
  }

  grad_x->Resize(x_dims);
  grad_y1->Resize(y1_dims);
  grad_y2->Resize(y2_dims);
  grad_bias1->Resize({y1_n});
  grad_bias2->Resize({y2_n});

  f::Tensor grad_out1;
  grad_out1.Resize({output1->dims()});

  f::Tensor lt_workspace;
  lt_workspace.Resize({kWorkspaceSize});

#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  PD_THROW(
      "fused_dense_gelu_dense_cuda_backward is "
      "not supported on cuda_version < 11000");
#endif

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x->dtype(), "linear_gelu_linear_cuda_backward_impl", ([&] {
        linear_gelu_linear_cuda_backward_impl<data_t>(
            dev_ctx,
            x->data<data_t>(),
            gelu_input->data<data_t>(),
            output1->data<data_t>(),
            y1->data<data_t>(),
            y2->data<data_t>(),
            grad_out2->data<data_t>(),
            transx,
            transy,
            x_k,   // in_features
            x_m,   // batch_size
            y1_n,  // hidden_features
            y2_n,  // out_features
            grad_out1.mutable_data<data_t>(place),
            grad_x->mutable_data<data_t>(place),
            grad_y1->mutable_data<data_t>(place),
            grad_y2->mutable_data<data_t>(place),
            grad_bias1->mutable_data<data_t>(place),
            grad_bias2->mutable_data<data_t>(place),
            lt_workspace.mutable_data<data_t>(place));
      }));
}
