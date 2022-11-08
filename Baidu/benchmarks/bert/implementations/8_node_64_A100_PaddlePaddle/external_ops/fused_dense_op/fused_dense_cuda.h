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

#ifndef FUSED_DENSE_CUDA_H_  // NOLINT
#define FUSED_DENSE_CUDA_H_  // NOLINT

#define CUBLAS_VERSION 13000

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
// includes cublaslt
#include <cublasLt.h>
#endif

#include "paddle/extension.h"
#include "paddle/fluid/framework/custom_raw_op_kernel_func.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

#define CHECK_CUBLAS_ERR(error_code)                 \
  do {                                               \
    if (error_code != CUBLAS_STATUS_SUCCESS) {       \
      PD_THROW("cublas error code is ", error_code); \
    }                                                \
  } while (0)

// todo: allocate 4MB. (the following code looks like 4MB * sizeof(T)?)
constexpr auto kWorkspaceSize = (1 << 22);

// FP64 Wrapper around cublas GEMMEx
// TODO(limin): in fact, alpha and beta are double type.
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const double* A,
                         int lda,
                         const double* B,
                         int ldb,
                         const float* beta,
                         double* C,
                         int ldc);

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const float* A,
                         int lda,
                         const float* B,
                         int ldb,
                         const float* beta,
                         float* C,
                         int ldc);

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(cublasHandle_t handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const paddle::float16* A,
                         int lda,
                         const paddle::float16* B,
                         int ldb,
                         const float* beta,
                         paddle::float16* C,
                         int ldc);

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
// float16 and float32
template <typename T>
cublasStatus_t cublaslt_matmul_desc_init(
    cublasLtMatmulDescOpaque_t* operationDesc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status =
      cublasLtMatmulDescInit(operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  return status;
}

// float64
template <>
cublasStatus_t cublaslt_matmul_desc_init<double>(
    cublasLtMatmulDescOpaque_t* operationDesc);

// float16
template <typename T>
cublasStatus_t set_cublaslt_matrix_layout_init(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_16F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_16F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_16F, m, n, ldc);

CLEANUP:
  return status;
}

template <>
cublasStatus_t set_cublaslt_matrix_layout_init<float>(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc);

template <>
cublasStatus_t set_cublaslt_matrix_layout_init<double>(
    cublasLtMatrixLayoutOpaque_t* Adesc,
    cublasLtMatrixLayoutOpaque_t* Bdesc,
    cublasLtMatrixLayoutOpaque_t* Cdesc,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc);
#endif

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
// fused_gemm_bias and fused_gemm_bias_gelu
template <typename T>
int gemm_bias_lt(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float* alpha,
                 const T* A,
                 int lda,
                 const T* B,
                 int ldb,
                 const float* beta,
                 T* C,
                 int ldc,
                 void* workspace,
                 size_t workspaceSize,
                 cudaStream_t stream,
                 bool use_bias,
                 const void* bias,
                 bool fuse_gelu = false,
                 const void* gelu_in = nullptr) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  if (fuse_gelu) {
    if (gelu_in == nullptr) {
      PD_THROW("when fuse_gelu, the gelu_in pointer can't be nullptr");
    }
    epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
  }

  status = cublaslt_matmul_desc_init<T>(&operationDesc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (fuse_gelu) {
    status = cublasLtMatmulDescSetAttribute(
        &operationDesc,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        &gelu_in,
        sizeof(gelu_in));
    status =
        cublasLtMatmulDescSetAttribute(&operationDesc,
                                       CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                       &ldc,
                                       sizeof(ldc));
  }

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    if (fuse_gelu) {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue,
                                          sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = set_cublaslt_matrix_layout_init<T>(
      &Adesc, &Bdesc, &Cdesc, transa, transb, m, n, k, lda, ldb, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
      &preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                          &operationDesc,
                                          &Adesc,
                                          &Bdesc,
                                          &Cdesc,
                                          &Cdesc,
                                          &preference,
                                          1,
                                          &heuristicResult,
                                          &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }

  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          // &heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  PADDLE_ENFORCE_GPU_SUCCESS(status);

  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float* alpha,
                      const phi::dtype::float16* A,
                      int lda,
                      const phi::dtype::float16* B,
                      int ldb,
                      const float* beta,
                      phi::dtype::float16* C,
                      int64_t ldc,
                      void* workspace,
                      size_t workspaceSize,
                      cudaStream_t stream,
                      bool use_bias,
                      const void* gelu_in,
                      const void* bias);

int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float* alpha,
                      const double* A,
                      int lda,
                      const double* B,
                      int ldb,
                      const float* beta,
                      double* C,
                      int ldc,
                      void* workspace,
                      size_t workspaceSize,
                      cudaStream_t stream,
                      bool use_bias,
                      const void* gelu_in,
                      const void* bias);

int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float* alpha,
                      const float* A,
                      int lda,
                      const float* B,
                      int ldb,
                      const float* beta,
                      float* C,
                      int64_t ldc,
                      void* workspace,
                      size_t workspaceSize,
                      cudaStream_t stream,
                      bool use_bias,
                      const void* gelu_in,
                      const void* bias);

#endif

template <typename T>
static int linear_bias_cuda_forward_impl(const phi::GPUContext& dev_ctx,
                                         const T* input_data,
                                         const T* weight_data,
                                         const T* bias_data,
                                         bool transx,
                                         bool transy,
                                         int in_features,
                                         int batch_size,
                                         int out_features,
                                         T* output_data,
                                         void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  // nt
  cublasOperation_t transpose_x = CUBLAS_OP_T;
  cublasOperation_t transpose_y = CUBLAS_OP_N;
  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    status = gemm_bias_lt((cublasLtHandle_t)handle,
                          transpose_x,
                          transpose_y,
                          out_features,
                          batch_size,
                          in_features,
                          &alpha,
                          weight_data,
                          in_features,
                          input_data,
                          in_features,
                          &beta_zero,
                          output_data,
                          out_features,
                          lt_workspace,
                          kWorkspaceSize,
                          stream,
                          true,
                          bias_data);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
#if 0
        output.copy_(bias);
        status = gemm_bias(
            handle,
            transpose_x,
            transpose_y,
            out_features,
            batch_size,
            in_features,
            &alpha,
            weight,
            in_features,
            input_data,
            in_features,
            &beta_one,
            output_data,
            out_features);
#endif
    }
  } else {
    // nn
    transpose_x = CUBLAS_OP_N;
    transpose_y = CUBLAS_OP_N;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    status = gemm_bias_lt((cublasLtHandle_t)handle,
                          transpose_x,
                          transpose_y,
                          out_features,
                          batch_size,
                          in_features,
                          &alpha,
                          weight_data,
                          out_features,
                          input_data,
                          in_features,
                          &beta_zero,
                          output_data,
                          out_features,
                          lt_workspace,
                          kWorkspaceSize,
                          stream,
                          true,
                          bias_data);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }
  }
  return status;
}

// dweight+dbias and dweight+dgelu+dbias
template <typename T>
int gemm_bgradb_lt(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha, /* host pointer */
                   const T* A,
                   int lda,
                   const T* B,
                   int ldb,
                   const float* beta, /* host pointer */
                   T* C,
                   int ldc,
                   void* workspace,
                   size_t workspaceSize,
                   cudaStream_t stream,
                   bool use_bias,
                   const void* bgrad,
                   cublasLtEpilogue_t epilogue,
                   bool fuse_gelu = false,
                   const void* gelu_in = nullptr) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  if (fuse_gelu) {
    if (gelu_in == nullptr) {
      PD_THROW("whe fuse_gelu, the gelu_in pointer can't be nullptr");
    }
    epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
  }

  status = cublaslt_matmul_desc_init<T>(&operationDesc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(
      &operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bgrad,
                                            sizeof(bgrad));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  }
  if (fuse_gelu) {
    status = cublasLtMatmulDescSetAttribute(
        &operationDesc,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        &gelu_in,
        sizeof(gelu_in));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status =
        cublasLtMatmulDescSetAttribute(&operationDesc,
                                       CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                       &ldc,
                                       sizeof(ldc));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue,
                                          sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = set_cublaslt_matrix_layout_init<T>(
      &Adesc, &Bdesc, &Cdesc, transa, transb, m, n, k, lda, ldb, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
      &preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                          &operationDesc,
                                          &Adesc,
                                          &Bdesc,
                                          &Cdesc,
                                          &Cdesc,
                                          &preference,
                                          1,
                                          &heuristicResult,
                                          &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          // &heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const phi::dtype::float16* A,
                         int lda,
                         const phi::dtype::float16* B,
                         int ldb,
                         const float* beta,
                         phi::dtype::float16* C,
                         int64_t ldc,
                         void* workspace,
                         size_t workspaceSize,
                         cudaStream_t stream,
                         const void* gelu_in,
                         const void* bgrad);

int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const double* A,
                         int lda,
                         const double* B,
                         int ldb,
                         const float* beta,
                         double* C,
                         int ldc,
                         void* workspace,
                         size_t workspaceSize,
                         cudaStream_t stream,
                         const void* gelu_in,
                         const void* bgrad);

int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float* alpha,
                         const float* A,
                         int lda,
                         const float* B,
                         int ldb,
                         const float* beta,
                         float* C,
                         int64_t ldc,
                         void* workspace,
                         size_t workspaceSize,
                         cudaStream_t stream,
                         const void* gelu_in,
                         const void* bgrad);

template <typename T>
int linear_bias_cuda_backward_impl(const phi::GPUContext& dev_ctx,
                                   const T* input,
                                   const T* weight,
                                   const T* d_output,
                                   bool transx,
                                   bool transy,
                                   bool use_addto,
                                   int in_features,
                                   int batch_size,
                                   int out_features,
                                   T* d_weight,
                                   T* d_bias,
                                   T* d_input,
                                   void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;
    status = gemm_bgradb_lt((cublasLtHandle_t)handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            in_features,
                            out_features,
                            batch_size,
                            &alpha,
                            input,
                            in_features,
                            d_output,
                            out_features,
                            &beta_zero,
                            d_weight,
                            in_features,
                            lt_workspace,
                            kWorkspaceSize,
                            stream,
                            true,
                            static_cast<const void*>(d_bias),
                            epilogue);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
#if 0
      status = gemm_bias(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          in_features,
          out_features,
          batch_size,
          &alpha,
          input,
          in_features,
          d_output,
          out_features,
          &beta_zero,
          d_weight,
          in_features);
#endif
    }
  } else {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADA;
    status = gemm_bgradb_lt((cublasLtHandle_t)handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            out_features,
                            in_features,
                            batch_size,
                            &alpha,
                            d_output,
                            out_features,
                            input,
                            in_features,
                            &beta_zero,
                            d_weight,
                            out_features,
                            lt_workspace,
                            kWorkspaceSize,
                            stream,
                            true,
                            static_cast<const void*>(d_bias),
                            epilogue);
#endif
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }
  }

  cublasOperation_t transpose_x = CUBLAS_OP_N;
  cublasOperation_t transpose_y = CUBLAS_OP_N;
  const float beta_dinput = (use_addto ? beta_one : beta_zero);
  if (transy) {
    status = gemm_bias(handle,
                       transpose_x,
                       transpose_y,
                       in_features,
                       batch_size,
                       out_features,
                       &alpha,
                       weight,
                       in_features,
                       d_output,
                       out_features,
                       &beta_dinput,
                       d_input,
                       in_features);
  } else {
    transpose_x = CUBLAS_OP_T;
    transpose_y = CUBLAS_OP_N;
    status = gemm_bias(handle,
                       transpose_x,
                       transpose_y,
                       in_features,
                       batch_size,
                       out_features,
                       &alpha,
                       weight,
                       out_features,
                       d_output,
                       out_features,
                       &beta_dinput,
                       d_input,
                       in_features);
  }
  return status;
}

template <typename T>
static int linear_gelu_linear_cuda_forward_impl(const phi::GPUContext& dev_ctx,
                                                const T* input,
                                                const T* weight1,
                                                const T* weight2,
                                                const T* bias1,
                                                const T* bias2,
                                                bool transx,
                                                bool transy,
                                                int in_features,
                                                int batch_size,
                                                int hidden_features,
                                                int out_features,
                                                T* output1,
                                                T* gelu_input,
                                                T* output2,
                                                void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    // weight1 * input = y
    // gelu(y + bias1) = output1
    // call gemm_bias_gelu_lt
    status = gemm_bias_gelu_lt((cublasLtHandle_t)handle,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               hidden_features,
                               batch_size,
                               in_features,
                               &alpha,
                               weight1,
                               in_features,
                               input,
                               in_features,
                               &beta_zero,
                               output1,
                               hidden_features,
                               lt_workspace,
                               kWorkspaceSize,
                               stream,
                               true,  // fuse_bias
                               static_cast<const void*>(gelu_input),
                               static_cast<const void*>(bias1));
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias_gelu_lt failed with error code ", status);
    }

    // weight2 * output1 + bias2 = output2
    status = gemm_bias_lt((cublasLtHandle_t)handle,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          out_features,
                          batch_size,
                          hidden_features,
                          &alpha,
                          weight2,
                          hidden_features,
                          output1,
                          hidden_features,
                          &beta_zero,
                          output2,
                          out_features,
                          lt_workspace,
                          kWorkspaceSize,
                          stream,
                          true,
                          static_cast<const void*>(bias2));
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias_lt failed with error code ", status);
    }
    return status;
#else
    return 1;
#endif
  } else {
    PD_THROW("Only support attribute transy=True now. ");
  }
  return status;
}

template <typename T>
static int linear_gelu_linear_cuda_backward_impl(const phi::GPUContext& dev_ctx,
                                                 const T* input,
                                                 const T* gelu_in,
                                                 const T* output1,
                                                 const T* weight1,
                                                 const T* weight2,
                                                 const T* d_output2,
                                                 bool transx,
                                                 bool transy,
                                                 int in_features,
                                                 int batch_size,
                                                 int hidden_features,
                                                 int out_features,
                                                 T* d_output1,
                                                 T* d_input,
                                                 T* d_weight1,
                                                 T* d_weight2,
                                                 T* d_bias1,
                                                 T* d_bias2,
                                                 void* lt_workspace) {
  auto handle = dev_ctx.cublas_handle();
  auto stream = dev_ctx.stream();

  const float alpha = 1.0;
  const float beta_zero = 0.0;
  const float beta_one = 1.0;
  int status = 1;

  if (transy) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    // wgrad for second gemm, fuse gemm, dbias
    // get d_weight2 = output1 * d_output2
    // get d_bias2
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;
    status = gemm_bgradb_lt((cublasLtHandle_t)handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            hidden_features,
                            out_features,
                            batch_size,
                            &alpha, /* host pointer */
                            output1,
                            hidden_features,
                            d_output2,
                            out_features,
                            &beta_zero, /* host pointer */
                            d_weight2,
                            hidden_features,
                            lt_workspace,
                            kWorkspaceSize,
                            stream,
                            true,
                            static_cast<const void*>(d_bias2),
                            epilogue);
    if (status != 0) {
      PD_THROW("cublaslt gemm_bgradb_lt failed with error code ", status);
    }

    // dgrad for second GEMM, fuse gemm, dgelu, dbias
    // get dgelu
    // get d_output1 = weight2 * d_output2
    // get d_bias1
    epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    status = gemm_dgelu_bgradb_lt((cublasLtHandle_t)handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_features,
                                  batch_size,
                                  out_features,
                                  &alpha,
                                  weight2,
                                  hidden_features,
                                  d_output2,
                                  out_features,
                                  &beta_zero,
                                  d_output1,
                                  hidden_features,
                                  lt_workspace,
                                  kWorkspaceSize,
                                  stream,
                                  static_cast<const void*>(gelu_in),
                                  static_cast<const void*>(d_bias1));
    if (status != 0) {
      PD_THROW(
          "cublaslt gemm_dgelu_bgradb_lt failed "
          "with error code ",
          status);
    }

    // wgrad for the first GEMM
    // d_weight1 = input * d_output1
    status = gemm_bias(handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       in_features,
                       hidden_features,
                       batch_size,
                       &alpha,
                       input,
                       in_features,
                       d_output1,
                       hidden_features,
                       &beta_zero,
                       d_weight1,
                       in_features);
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }

    // dgrad for the first GEMM
    // d_input = weight1 * d_output1
    status = gemm_bias(handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       in_features,
                       batch_size,
                       hidden_features,
                       &alpha,
                       weight1,
                       in_features,
                       d_output1,
                       hidden_features,
                       &beta_zero,
                       d_input,
                       in_features);
    if (status != 0) {
      PD_THROW("cublaslt gemm_bias failed with error code ", status);
    }
#endif
    return status;
  } else {
    PD_THROW("Only support attribute transy=True now. ");
  }
}
#endif  // FUSED_DENSE_CUDA_H_ // NOLINT
