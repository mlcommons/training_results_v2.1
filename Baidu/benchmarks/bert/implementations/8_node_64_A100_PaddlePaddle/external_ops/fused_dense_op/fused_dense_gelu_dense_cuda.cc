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

#include <vector>
#include "paddle/extension.h"

// @x: [x, in_feature] or [xx, xx, in_feature]
// @y1: [inter_feature, in_feature]
// @out1: [x, inter_feature] or [xx, xx, inter_feature]
// @y2: [in_feature, inter_feature]
// @out2: [x, in_feature] or [xx, xx, in_feature]
// Now only support transx=false, transy=true.
// formula: x*y1->out1, out1*y2=>out2
std::vector<std::vector<int64_t>> FusedDenseGeluDenseInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y1_shape,
    const std::vector<int64_t>& y2_shape,
    const std::vector<int64_t>& bias1_shape,
    const std::vector<int64_t>& bias2_shape,
    const bool& transx,
    const bool& transy) {
  int x_size = x_shape.size();
  int x_m = 1;
  for (int i = 0; i < (x_size - 1); i++) {
    x_m *= x_shape[i];
  }
  int x_k = x_shape[x_size - 1];

  int y1_k = y1_shape[0];
  int y1_n = y1_shape[1];
  int y2_k = y2_shape[0];
  int y2_n = y2_shape[1];
  if (transy) {
    y1_k = y1_shape[1];
    y1_n = y1_shape[0];
    y2_k = y2_shape[1];
    y2_n = y2_shape[0];
  }

  if (x_k != y1_k) {
    PD_THROW("The reudce dim of the first matmul is not equal.");
  }
  if (y1_n != y2_k) {
    PD_THROW("The reudce dim of the second matmul is not equal.");
  }

  if (transx) {
    PD_THROW("Only support cases: transx is False, transy are True.");
  }
  if (!transy) {
    PD_THROW("Only support cases: transx is False, transy are True.");
  }

  std::vector<int64_t> out1_shape(x_shape);
  std::vector<int64_t> out2_shape(x_shape);
  out1_shape[x_size - 1] = y1_n;
  out2_shape[x_size - 1] = y2_n;
  return {out1_shape, out2_shape, out1_shape};
}

std::vector<paddle::DataType> FusedDenseGeluDenseInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType y1_dtype,
    paddle::DataType y2_dtype,
    paddle::DataType bias1_dtype,
    paddle::DataType bias2_dtype) {
  return {x_dtype, x_dtype, x_dtype};
}

std::vector<std::vector<int64_t>> FusedDenseGeluDenseGradInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y1_shape,
    const std::vector<int64_t>& y2_shape,
    const std::vector<int64_t>& out1_shape,
    const std::vector<int64_t>& gelu_input_shape,
    const std::vector<int64_t>& grad_out2_shape,
    const bool& transx,
    const bool& transy) {
  
  int y1_n = transy ? y1_shape[0] : y1_shape[1];
  int y2_n = transy ? y2_shape[0] : y2_shape[1];
  std::vector<int64_t> bias1_shape = {y1_n};
  std::vector<int64_t> bias2_shape = {y2_n};
  return {x_shape, y1_shape, y2_shape, bias1_shape, bias2_shape};
}

PD_BUILD_OP(custom_fused_dense_gelu_dense)
    .Inputs({"X", "Y1", "Y2", "Bias1", "Bias2"})
    .Outputs({"Out1", "Out2", "Gelu_Input"})
    .Attrs({"transx: bool", "transy: bool"})
    .SetInferShapeFn(PD_INFER_SHAPE(FusedDenseGeluDenseInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedDenseGeluDenseInferDtype));

PD_BUILD_GRAD_OP(custom_fused_dense_gelu_dense)
    .Inputs({"X", "Y1", "Y2", "Out1", "Gelu_Input", paddle::Grad("Out2")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y1"), paddle::Grad("Y2"), paddle::Grad("Bias1"), paddle::Grad("Bias2")})
    .Attrs({"transx: bool", "transy: bool"})
    .SetInferShapeFn(PD_INFER_SHAPE(FusedDenseGeluDenseGradInferShape));
