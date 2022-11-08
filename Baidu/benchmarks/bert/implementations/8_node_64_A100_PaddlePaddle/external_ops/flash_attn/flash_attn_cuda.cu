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
#include "flash_attn.h"      // NOLINT

#define CHECK_FLASH_ATTN_ERROR()                          \
  do {                                                    \
    const auto* __err_msg = flash_attn_error();           \
    if (__err_msg != nullptr) {                           \
      PD_THROW("flash_attn error code is %s", __err_msg); \
    }                                                     \
  } while (0)

static int cur_dev_id = -1;

static int GetDeviceId() {
  if (cur_dev_id < 0) {
    cur_dev_id = paddle::platform::GetCurrentDeviceId();
  }
  return cur_dev_id;
}

static const phi::GPUContext& GetCurrentCUDADeviceContext() {
  auto dev_id = GetDeviceId();
  paddle::platform::CUDAPlace place(dev_id);
  return *paddle::platform::DeviceContextPool::Instance().GetByPlace(place);
}

static std::string CuSeqLenToStr(const int* cu_seqlen,
                                 int bs,
                                 cudaStream_t stream) {
  std::vector<int> cpu_len(bs + 1);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(cpu_len.data(),
                      cu_seqlen,
                      cpu_len.size() * sizeof(cpu_len[0]),
                      cudaMemcpyDeviceToHost,
                      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  std::stringstream ss;
  ss << cpu_len[0];
  for (int i = 1; i <= bs; ++i) {
    ss << ", " << cpu_len[i];
  }
  return std::string("[") + ss.str() + "]";
}

static void SeedIncFunc(uint64_t inc,
                        uint64_t* seed,
                        const int64_t** offset_ptr,
                        uint64_t* offset,
                        bool* is_device_rnd) {
  auto& dev_ctx = GetCurrentCUDADeviceContext();
  paddle::operators::GetSeedDataAndIncrement(
      dev_ctx, nullptr, false, 0, inc, seed, offset);
  *offset_ptr = nullptr;
  *is_device_rnd = false;
}

static constexpr bool kIsCausal = false;

static const int kMaxSupportedSeqLength = 512;

static const int MAX_GROUP_SIZE = 3;
static cudaStream_t all_streams[MAX_GROUP_SIZE];
static cudaEvent_t all_events[MAX_GROUP_SIZE + 1];

static void InitStreamEventsOnce(cudaStream_t stream) {
  static std::once_flag flag;
  std::call_once(flag, [stream] {
    all_streams[0] = stream;
    for (int i = 1; i < MAX_GROUP_SIZE; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamCreateWithFlags(&all_streams[i], cudaStreamNonBlocking));
    }

    for (int i = 0; i < MAX_GROUP_SIZE + 1; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&all_events[i], cudaEventDisableTiming));
    }
  });
}

static cudaStream_t GetStream(cudaStream_t stream, size_t i) {
  i %= MAX_GROUP_SIZE;
  return i == 0 ? stream : all_streams[i];
}

static cudaEvent_t GetStartEvent() { return all_events[MAX_GROUP_SIZE]; }

static cudaEvent_t GetEndEvent(size_t i) {
  return all_events[i % MAX_GROUP_SIZE];
}

static void PreRecordEvent(cudaStream_t stream, size_t n) {
  auto event = GetStartEvent();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, stream));
  for (size_t i = 1; i < n; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(GetStream(stream, i), event));
  }
}

static void PostRecordEvent(cudaStream_t stream, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    auto event = GetEndEvent(i);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, GetStream(stream, i)));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(stream, event));
  }
}

static int seq_len_round(int real_seq_len, int head_size) {
  int ret = flash_attn_seq_len(head_size, real_seq_len);
  if (ret < 0 || ret > kMaxSupportedSeqLength) {
    PD_THROW("Error of seq_len_round when use_fmha_mke_opt=1.");
  }
  return ret;
}

struct FlashAttnSeqGroup {
  FlashAttnSeqGroup() {}
  FlashAttnSeqGroup(
      int seq_offset, int token_offset, int batch_size, int total, int seq_len)
      : seq_offset(seq_offset),
        token_offset(token_offset),
        batch_size(batch_size),
        total(total),
        seq_len(seq_len) {}

  int seq_offset;
  int token_offset;
  int batch_size;
  int total;
  int seq_len;
};

static std::vector<FlashAttnSeqGroup> GetFlashAttnSeqGroup(
    const int* prefix_sum_seq_len,
    const int batch_size,
    const int head_size,
    const bool use_fmha_mke_opt) {
  if (!use_fmha_mke_opt) {
    int max_seq_len = 0;
    for (int i = 0; i < batch_size; ++i) {
      max_seq_len = std::max(max_seq_len,
                             prefix_sum_seq_len[i + 1] - prefix_sum_seq_len[i]);
    }
    return {FlashAttnSeqGroup(0,
                              0,
                              batch_size,
                              prefix_sum_seq_len[batch_size],
                              seq_len_round(max_seq_len, head_size))};
  }

  std::vector<FlashAttnSeqGroup> infos;
  infos.reserve(batch_size);
  int prev_max_seq_len =
      seq_len_round(prefix_sum_seq_len[1] - prefix_sum_seq_len[0], head_size);
  int prev_idx = 0;
  for (int i = 1; i < batch_size; ++i) {
    int cur_seq_len = seq_len_round(
        prefix_sum_seq_len[i + 1] - prefix_sum_seq_len[i], head_size);
    if (cur_seq_len != prev_max_seq_len) {
      infos.emplace_back(prev_idx,
                         prefix_sum_seq_len[prev_idx],
                         i - prev_idx,
                         prefix_sum_seq_len[i] - prefix_sum_seq_len[prev_idx],
                         prev_max_seq_len);
      prev_idx = i;
      prev_max_seq_len = cur_seq_len;
    }
  }

  infos.emplace_back(
      prev_idx,
      prefix_sum_seq_len[prev_idx],
      batch_size - prev_idx,
      prefix_sum_seq_len[batch_size] - prefix_sum_seq_len[prev_idx],
      prev_max_seq_len);
  return infos;
}

static std::unordered_map<const void*, phi::Generator::GeneratorState>
    rng_states;

static void SaveRandomState(const void* ptr) {
  auto dev_id = GetDeviceId();
  const auto& gen = paddle::framework::DefaultCUDAGenerator(dev_id);
  PADDLE_ENFORCE_EQ(rng_states.emplace(ptr, gen->GetState()).second, true);
}

static phi::Generator::GeneratorState RestoreRandomState(const void* ptr) {
  auto dev_id = GetDeviceId();
  const auto& gen = paddle::framework::DefaultCUDAGenerator(dev_id);
  auto iter = rng_states.find(ptr);
  PADDLE_ENFORCE_EQ(iter != rng_states.end(), true);
  auto old_state = gen->GetState();
  gen->SetState(iter->second);
  rng_states.erase(iter);
  return old_state;
}

static void RestoreRandomState(const phi::Generator::GeneratorState& state) {
  auto dev_id = GetDeviceId();
  const auto& gen = paddle::framework::DefaultCUDAGenerator(dev_id);
  gen->SetState(state);
}

std::vector<paddle::Tensor> flash_attn_cuda_forward(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& host_seqlen,
    bool is_test,
    float dropout_rate,
    bool zero_tensors,
    bool use_fmha_mke_opt) {
  if (qkv.type() != paddle::DataType::FLOAT16) {
    PD_THROW("FMHALib only supports float16 inputs.");
  }

  auto qkv_dims = qkv.shape();
  int total = qkv_dims[0];
  int num_heads = qkv_dims[2];
  int head_size = qkv_dims[3];

  auto cu_seqlen_dims = cu_seqlen.shape();
  int batch_size = cu_seqlen_dims[0] - 1;

  auto groups = GetFlashAttnSeqGroup(host_seqlen.data<int>(),
                                     batch_size,
                                     head_size,
                                     use_fmha_mke_opt && (!is_test));

  cudaStream_t stream = qkv.stream();
  InitStreamEventsOnce(stream);

  const std::vector<int64_t> ctx_out_shape = {total, num_heads, head_size};
  auto place = qkv.place();
  auto dtype = qkv.dtype();
  auto ctx_out = paddle::experimental::empty(ctx_out_shape, dtype, place);
  const std::vector<int64_t> s_out_shape = {
      batch_size, num_heads, kMaxSupportedSeqLength};
  auto s_out = paddle::experimental::empty(
      s_out_shape, paddle::DataType::FLOAT32, place);

  paddle::Tensor workspace_tensor;
  int8_t* workspace_ptr = nullptr;
  std::vector<uint64_t> workspace_size_offsets(groups.size() + 1);
  uint64_t total_workspace_size = 0;

  const auto* qkv_ptr = qkv.data<paddle::float16>();
  const auto* cu_seqlens_ptr = cu_seqlen.data<int>();
  auto* ctx_ptr = ctx_out.data<paddle::float16>();
  auto* softmax_lse_ptr = s_out.data<float>();

  auto ctx_stride = static_cast<int64_t>(num_heads) * head_size;
  auto qkv_stride = ctx_stride * 3;
  auto s_stride = static_cast<int64_t>(num_heads) * kMaxSupportedSeqLength;
  const float softmax_scale = 1.0f / std::sqrt(head_size);

  VLOG(10) << "total = " << total << " num_heads = " << num_heads
           << " head_size = " << head_size << " batch_size = " << batch_size;

  if (!is_test) SaveRandomState(qkv_ptr);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (i > 0 && total_workspace_size > 0) {
      workspace_tensor = paddle::experimental::empty(
          {static_cast<int64_t>(total_workspace_size)},
          paddle::DataType::INT8,
          place);
      workspace_ptr = workspace_tensor.data<int8_t>();
      PreRecordEvent(stream, groups.size());
    }

    uint64_t cur_workspace_size = 0;
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      const auto& group = groups[group_idx];
      flash_attn_fwd(
          qkv_ptr + group.token_offset * qkv_stride,
          cu_seqlens_ptr + group.seq_offset,
          group.total,
          group.batch_size,
          num_heads,
          head_size,
          group.seq_len,
          !is_test,
          zero_tensors,
          kIsCausal,
          dropout_rate,
          softmax_scale,
          SeedIncFunc,
          GetStream(stream, group_idx),
          i == 0 ? nullptr : ctx_ptr + group.token_offset * ctx_stride,
          softmax_lse_ptr + group.seq_offset * s_stride,
          nullptr,
          i == 0 ? nullptr : workspace_ptr + workspace_size_offsets[group_idx],
          &cur_workspace_size);
      if (i == 0) {
        VLOG(10) << "Group[" << group_idx << "] : "
                 << " total = " << group.total
                 << " batch_size = " << group.batch_size
                 << " seq_len = " << group.seq_len
                 << " token_offset = " << group.token_offset
                 << " seq_offset = " << group.seq_offset
                 << " cur_workspace_size = " << cur_workspace_size;
      }
      if (i == 0) {
        total_workspace_size += cur_workspace_size;
        workspace_size_offsets[group_idx + 1] = total_workspace_size;
      }
    }

    if (i > 0 && total_workspace_size > 0) {
      PostRecordEvent(stream, groups.size());
    }
  }

  CHECK_FLASH_ATTN_ERROR();
  return {ctx_out, s_out};
}

std::vector<paddle::Tensor> flash_attn_cuda_backward(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& host_seqlen,
    const paddle::Tensor& ctx_input,
    const paddle::Tensor& softmax_input,
    const paddle::Tensor& d_ctx_out,
    bool is_test,
    float dropout_rate,
    bool zero_tensors,
    bool use_fmha_mke_opt) {
  if (qkv.type() != paddle::DataType::FLOAT16) {
    PD_THROW("FMHALib only supports float16 inputs.");
  }

  auto qkv_dims = qkv.shape();
  int total = qkv_dims[0];
  int num_heads = qkv_dims[2];
  int head_size = qkv_dims[3];

  auto cu_seqlen_dims = cu_seqlen.shape();
  int batch_size = cu_seqlen_dims[0] - 1;

  auto groups = GetFlashAttnSeqGroup(host_seqlen.data<int>(),
                                     batch_size,
                                     head_size,
                                     use_fmha_mke_opt && (!is_test));
  cudaStream_t stream = qkv.stream();
  auto place = qkv.place();
  auto dtype = qkv.dtype();

  auto grad_qkv_out =
      paddle::experimental::empty(qkv.shape(), qkv.dtype(), place);
  auto* grad_qkv_out_data = grad_qkv_out.data<paddle::float16>();

  const auto* softmax_input_data = softmax_input.data<float>();
  const auto* ctx_input_data = ctx_input.data<paddle::float16>();
  const auto* d_ctx_out_data = d_ctx_out.data<paddle::float16>();
  const auto* qkv_data = qkv.data<paddle::float16>();
  const auto* cu_seqlen_data = cu_seqlen.data<int>();

  auto ctx_stride = static_cast<int64_t>(num_heads) * head_size;
  auto qkv_stride = ctx_stride * 3;
  auto s_stride = static_cast<int64_t>(num_heads) * kMaxSupportedSeqLength;
  const float softmax_scale = 1.0f / std::sqrt(head_size);

  paddle::Tensor workspace_tensor;
  int8_t* workspace_ptr = nullptr;
  std::vector<uint64_t> workspace_size_offsets(groups.size() + 1);
  uint64_t total_workspace_size = 0;

  auto old_state = RestoreRandomState(qkv_data);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (i > 0 && total_workspace_size > 0) {
      workspace_tensor = paddle::experimental::empty(
          {static_cast<int64_t>(total_workspace_size)},
          paddle::DataType::INT8,
          place);
      workspace_ptr = workspace_tensor.data<int8_t>();
      PreRecordEvent(stream, groups.size());
    }

    uint64_t cur_workspace_size = 0;
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      const auto& group = groups[group_idx];
      flash_attn_bwd(
          d_ctx_out_data + group.token_offset * ctx_stride,
          qkv_data + group.token_offset * qkv_stride,
          ctx_input_data + group.token_offset * ctx_stride,
          nullptr,
          softmax_input_data + group.seq_offset * s_stride,
          cu_seqlen_data + group.seq_offset,
          group.total,
          group.batch_size,
          num_heads,
          head_size,
          group.seq_len,
          zero_tensors,
          kIsCausal,
          dropout_rate,
          softmax_scale,
          SeedIncFunc,
          GetStream(stream, group_idx),
          i == 0 ? nullptr
                 : grad_qkv_out_data + group.token_offset * qkv_stride,
          i == 0 ? nullptr : workspace_ptr + workspace_size_offsets[group_idx],
          &cur_workspace_size);
      if (i == 0) {
        total_workspace_size += cur_workspace_size;
        workspace_size_offsets[group_idx + 1] = total_workspace_size;
      }
    }

    if (i > 0 && total_workspace_size > 0) {
      PostRecordEvent(stream, groups.size());
    }
  }

  CHECK_FLASH_ATTN_ERROR();
  RestoreRandomState(old_state);
  return {grad_qkv_out};
}
