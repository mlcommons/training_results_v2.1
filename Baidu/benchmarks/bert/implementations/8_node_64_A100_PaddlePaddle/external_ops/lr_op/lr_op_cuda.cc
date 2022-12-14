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

#include <vector>
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> LrInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

// input: [1], int64
// output: [1], float
std::vector<paddle::DataType> LrInferDtype(paddle::DataType x_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(custom_lr)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"base_lr: float",
            "end_lr: float",
            "degree: float",
            "warmup_step: int64_t",
            "start_warmup_step: int64_t",
            "max_step: int64_t"})
    .SetInferShapeFn(PD_INFER_SHAPE(LrInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LrInferDtype));
