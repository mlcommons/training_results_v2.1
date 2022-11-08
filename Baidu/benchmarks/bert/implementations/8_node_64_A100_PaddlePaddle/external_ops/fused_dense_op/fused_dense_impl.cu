// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fused_dense_cuda.h"

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
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_64F,
                      lda,
                      B,
                      CUDA_R_64F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_64F,
                      ldc,
                      CUDA_R_64F,
                      CUBLAS_GEMM_DEFAULT);
}

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
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_32F,
                      lda,
                      B,
                      CUDA_R_32F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_32F,
                      ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
}

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
                         int ldc) {
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      CUDA_R_16F,
                      lda,
                      B,
                      CUDA_R_16F,
                      ldb,
                      beta,
                      C,
                      CUDA_R_16F,
                      ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600

// float64
template <>
cublasStatus_t cublaslt_matmul_desc_init<double>(
    cublasLtMatmulDescOpaque_t* operationDesc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status =
      cublasLtMatmulDescInit(operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
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
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_32F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_32F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_32F, m, n, ldc);
CLEANUP:
  return status;
}

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
    int ldc) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasLtMatrixLayoutInit(Adesc,
                                    CUDA_R_64F,
                                    transa == CUBLAS_OP_N ? m : k,
                                    transa == CUBLAS_OP_N ? k : m,
                                    lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Bdesc,
                                    CUDA_R_64F,
                                    transb == CUBLAS_OP_N ? k : n,
                                    transb == CUBLAS_OP_N ? n : k,
                                    ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(Cdesc, CUDA_R_64F, m, n, ldc);

CLEANUP:
  return status;
}


int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const phi::dtype::float16* A,
    int lda,
    const phi::dtype::float16* B,
    int ldb,
    const float *beta,
    phi::dtype::float16* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* gelu_in,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX;

  status = cublasLtMatmulDescInit(&operationDesc,
                                  CUBLAS_COMPUTE_32F,
                                  CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_TRANSA,
                  &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_TRANSB,
                  &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                  &gelu_in, sizeof(gelu_in));
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                  &ldc, sizeof(ldc));

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc,
                    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                  &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k,
                  transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n,
                  transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

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
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}


int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    const float *beta,
    double* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void *gelu_in,
    const void* bias) {
  return 1;
}


int gemm_bias_gelu_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* gelu_in,
    const void* bias) {
  return 1;
}

int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const phi::dtype::float16* A,
    int lda,
    const phi::dtype::float16* B,
    int ldb,
    const float *beta,
    phi::dtype::float16* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

  status = cublasLtMatmulDescInit(&operationDesc,
                                  CUBLAS_COMPUTE_32F,
                                  CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_TRANSA,
                  &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_TRANSB,
                  &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                  &bgrad, sizeof(bgrad));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                  &gelu_in, sizeof(gelu_in));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }
  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                  &ldc, sizeof(ldc));

  status = cublasLtMatmulDescSetAttribute(&operationDesc,
                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                  &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k,
                  transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n,
                  transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

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
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    const float *beta,
    double *C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
    return 1;
}

int gemm_dgelu_bgradb_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    const void *gelu_in,
    const void *bgrad) {
    return 1;
}

#endif

