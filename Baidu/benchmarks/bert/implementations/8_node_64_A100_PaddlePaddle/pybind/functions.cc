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

#include <chrono>
#include <cstdint>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace framework = paddle::framework;
namespace platform = paddle::platform;

PYBIND11_MAKE_OPAQUE(framework::LoDTensorArray);

constexpr int kInputIdsIdx = 0;
constexpr int kSegmentIdsIdx = 1;
constexpr int kInputMaskIdx = 2;
constexpr int kMaskedLmLabelsIdx = 3;
constexpr int kNextSentenceLabelsIdx = 4;
constexpr int kSeqLenIdx = 5;
constexpr int kPrefixSumSeqLenIdx = 6;
constexpr int kNonZerosIndicesIdx = 7;
constexpr int kMaskedLmIdsIdx = 8;
constexpr int kMaskedLmPositionIdx = 9;
constexpr int kNumValidIdx = 10;

constexpr int kNumTensors = 11;

template <typename T>
static std::string ListToString(const T *ptr, size_t n) {
  if (n == 0) return "[]";
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) ss << ", ";
    ss << ptr[i];
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::vector<std::vector<framework::LoDTensorArray>>
ProcessAllGatheredBERTInputsBase(const T *arr,
                                 size_t length,
                                 size_t num_samples,
                                 size_t max_seq_length,
                                 size_t batch_size,
                                 size_t trainer_id,
                                 size_t num_trainers,
                                 bool drop_last,
                                 bool load_balance) {
  using TensorT = framework::LoDTensor;

  const size_t nbatch = drop_last ? num_samples / batch_size
                                  : (num_samples + batch_size - 1) / batch_size;
  VLOG(10) << "num_samples = " << num_samples
           << " , max_seq_length = " << max_seq_length
           << " , batch_size = " << batch_size
           << " , trainer_id = " << trainer_id
           << " , num_trainers = " << num_trainers << " , nbatch = " << nbatch
           << " , length = " << length;

  std::unique_ptr<T[]> seq_indices(new T[batch_size * num_trainers]);

  const size_t numel = num_samples * max_seq_length;
  const size_t num_per_device = numel * 4 + num_samples * 2;
  PADDLE_ENFORCE_EQ(num_per_device * num_trainers, length);

  auto resize_and_alloc = [](TensorT *t, const framework::DDim &dim) -> T * {
    t->Resize(dim);
    return t->mutable_data<T>(platform::CPUPlace());
  };

  auto resize_and_alloc_int = [](TensorT *t,
                                 const framework::DDim &dim) -> int * {
    t->Resize(dim);
    return t->mutable_data<int>(platform::CPUPlace());
  };
  auto resize_and_alloc_float32 = [](TensorT *t,
                                     const framework::DDim &dim) -> float * {
    t->Resize(dim);
    return t->mutable_data<float>(platform::CPUPlace());
  };

  std::vector<std::vector<framework::LoDTensorArray>> gpu_cpu_tensors;
  std::vector<framework::LoDTensorArray> tensors(nbatch);
  std::vector<framework::LoDTensorArray> tensors_2(nbatch);
  for (size_t i = 0; i < nbatch; ++i) {
    const size_t cur_bs =
        std::min((i + 1) * batch_size, num_samples) - i * batch_size;
    VLOG(10) << "Mini batch " << i << " " << cur_bs;
    const size_t seq_length_offset =
        num_samples * max_seq_length * 4 + num_samples + i * batch_size;
    const size_t total_seq_length = cur_bs * num_trainers;

    std::iota(seq_indices.get(),
              seq_indices.get() + total_seq_length,
              static_cast<size_t>(0));

    std::sort(seq_indices.get(),
              seq_indices.get() + total_seq_length,
              [&](size_t idx1, size_t idx2) {
                size_t real_idx1 = (idx1 / cur_bs) * num_per_device +
                                   (idx1 % cur_bs) + seq_length_offset;
                size_t real_idx2 = (idx2 / cur_bs) * num_per_device +
                                   (idx2 % cur_bs) + seq_length_offset;
                return arr[real_idx1] > arr[real_idx2];
              });

    VLOG(10) << "Mini batch " << i << " " << cur_bs << " , seq_indices = "
             << ListToString(seq_indices.get(), cur_bs * num_trainers);

    tensors[i].resize(kNumTensors);
    tensors_2[i].resize(1);
    auto *input_ids = resize_and_alloc(
        &tensors[i][kInputIdsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *segment_ids = resize_and_alloc(
        &tensors[i][kSegmentIdsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *input_mask = resize_and_alloc(
        &tensors[i][kInputMaskIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *masked_lm_labels = resize_and_alloc(
        &tensors[i][kMaskedLmLabelsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *next_sentence_labels = resize_and_alloc(
        &tensors[i][kNextSentenceLabelsIdx], {static_cast<int64_t>(cur_bs)});

    auto *seq_len = resize_and_alloc_int(&tensors[i][kSeqLenIdx],
                                         {static_cast<int64_t>(cur_bs)});
    auto *prefix_sum_seq_len = resize_and_alloc_int(
        &tensors[i][kPrefixSumSeqLenIdx], {static_cast<int64_t>(cur_bs + 1)});

    auto *num_valid = resize_and_alloc_float32(&tensors[i][kNumValidIdx],
                                               {static_cast<int64_t>(1)});

    // cpu tensor
    auto *host_prefix_sum_seq_len = resize_and_alloc_int(
        &tensors_2[i][0], {static_cast<int64_t>(cur_bs + 1)});

    prefix_sum_seq_len[0] = 0;
    int sum_seq_len = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      const size_t cur_trainer_id = (j % 2 != 0 && load_balance)
                                        ? (num_trainers - 1 - trainer_id)
                                        : trainer_id;
      const size_t idx = seq_indices.get()[j * num_trainers + cur_trainer_id];
      const size_t dev_id = idx / cur_bs;
      const T *data = arr + dev_id * num_per_device;
      const size_t sample_id = idx % cur_bs + i * batch_size;
      std::memcpy(input_ids + j * max_seq_length,
                  data + sample_id * max_seq_length,
                  max_seq_length * sizeof(T));
      std::memcpy(segment_ids + j * max_seq_length,
                  data + numel + sample_id * max_seq_length,
                  max_seq_length * sizeof(T));
      std::memcpy(input_mask + j * max_seq_length,
                  data + 2 * numel + sample_id * max_seq_length,
                  max_seq_length * sizeof(T));
      std::memcpy(masked_lm_labels + j * max_seq_length,
                  data + 3 * numel + sample_id * max_seq_length,
                  max_seq_length * sizeof(T));
      next_sentence_labels[j] = data[4 * numel + sample_id];

      seq_len[j] = data[4 * numel + num_samples + sample_id];
      sum_seq_len += seq_len[j];
      if (j > 0) {
        prefix_sum_seq_len[j] = prefix_sum_seq_len[j - 1] + seq_len[j - 1];
      }
    }
    prefix_sum_seq_len[cur_bs] =
        prefix_sum_seq_len[cur_bs - 1] + seq_len[cur_bs - 1];

    std::memcpy(host_prefix_sum_seq_len,
                prefix_sum_seq_len,
                sizeof(int) * (cur_bs + 1));

    PADDLE_ENFORCE_LE(sum_seq_len, cur_bs * max_seq_length);

    auto *nonzeros_indices = resize_and_alloc_int(
        &tensors[i][kNonZerosIndicesIdx], {static_cast<int64_t>(sum_seq_len)});
    int cur_nonzero_ind = 0;
    int cur_num_valid = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      for (size_t k = 0; k < max_seq_length; ++k) {
        int ids = j * max_seq_length + k;
        if (input_mask[ids] != 0) {
          nonzeros_indices[cur_nonzero_ind++] = static_cast<int>(ids);
        }
        if (masked_lm_labels[ids] != 0) {
          cur_num_valid += 1;
        }
      }
    }
    PADDLE_ENFORCE_EQ(cur_nonzero_ind, sum_seq_len);

    *num_valid = static_cast<float>(cur_num_valid);
    auto *masked_lm_ids = resize_and_alloc_int(
        &tensors[i][kMaskedLmIdsIdx], {static_cast<int64_t>(cur_num_valid)});
    auto *masked_lm_positions =
        resize_and_alloc_int(&tensors[i][kMaskedLmPositionIdx],
                             {static_cast<int64_t>(cur_num_valid)});

    cur_num_valid = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      for (size_t k = 0; k < max_seq_length; ++k) {
        int ids = j * max_seq_length + k;
        if (masked_lm_labels[ids] != 0) {
          masked_lm_positions[cur_num_valid] = ids;
          masked_lm_ids[cur_num_valid] = masked_lm_labels[ids];
          cur_num_valid += 1;
        }
      }
    }
  }
  gpu_cpu_tensors.push_back(tensors);
  gpu_cpu_tensors.push_back(tensors_2);
  return gpu_cpu_tensors;
}

template <typename T>
std::vector<std::vector<framework::LoDTensorArray>>
ProcessAllGatheredBERTInputs(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    size_t num_samples,
    size_t max_seq_length,
    size_t batch_size,
    size_t trainer_id,
    size_t num_trainers,
    bool drop_last,
    bool load_balance) {
  using TensorT = framework::LoDTensor;

  int ndim = array.ndim();
  PADDLE_ENFORCE_EQ(ndim, 1);
  size_t length = array.shape()[0];
  const T *arr = array.data();
  py::gil_scoped_release guard;
  return ProcessAllGatheredBERTInputsBase<T>(arr,
                                             length,
                                             num_samples,
                                             max_seq_length,
                                             batch_size,
                                             trainer_id,
                                             num_trainers,
                                             drop_last,
                                             load_balance);
}

template <typename T>
std::vector<std::vector<framework::LoDTensorArray>> ProcessBERTEvalInputs(
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    size_t max_seq_length,
    size_t batch_size,
    bool need_sort) {
  using TensorT = framework::LoDTensor;

  PADDLE_ENFORCE_EQ(array.ndim(), 2);
  size_t num_samples = array.shape()[0];
  size_t one_sample_len = array.shape()[1];
  const T *arr = array.data();

  py::gil_scoped_release guard;

  std::unique_ptr<size_t[]> seq_indices;
  if (need_sort) {
    seq_indices.reset(new size_t[num_samples]);
    std::iota(seq_indices.get(),
              seq_indices.get() + num_samples,
              static_cast<size_t>(0));
    std::sort(seq_indices.get(),
              seq_indices.get() + num_samples,
              [arr, one_sample_len](size_t idx1, size_t idx2) {
                idx1 = (idx1 + 1) * one_sample_len - 1;
                idx2 = (idx2 + 1) * one_sample_len - 1;
                return arr[idx1] < arr[idx2];
              });
  }

  const size_t nbatch = (num_samples + batch_size - 1) / batch_size;

  auto resize_and_alloc = [](TensorT *t, const framework::DDim &dim) -> T * {
    t->Resize(dim);
    return t->mutable_data<T>(platform::CPUPlace());
  };
  auto resize_and_alloc_int = [](TensorT *t,
                                 const framework::DDim &dim) -> int * {
    t->Resize(dim);
    return t->mutable_data<int>(platform::CPUPlace());
  };
  auto resize_and_alloc_float32 = [](TensorT *t,
                                     const framework::DDim &dim) -> float * {
    t->Resize(dim);
    return t->mutable_data<float>(platform::CPUPlace());
  };

  VLOG(10) << "one_sample_len = " << one_sample_len;
  VLOG(10) << "num_samples = " << num_samples;
  VLOG(10) << "max_seq_length = " << max_seq_length;
  VLOG(10) << "batch_size = " << batch_size;
  VLOG(10) << "nbatch = " << nbatch;

  std::vector<std::vector<framework::LoDTensorArray>> gpu_cpu_tensors;
  std::vector<framework::LoDTensorArray> tensors(nbatch);
  std::vector<framework::LoDTensorArray> tensors_2(nbatch);
  for (size_t i = 0; i < nbatch; ++i) {
    const size_t cur_bs =
        std::min((i + 1) * batch_size, num_samples) - i * batch_size;
    VLOG(10) << "Mini batch " << i << " " << cur_bs;

    tensors[i].resize(kNumTensors);
    tensors_2[i].resize(1);
    auto *input_ids = resize_and_alloc(
        &tensors[i][kInputIdsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *segment_ids = resize_and_alloc(
        &tensors[i][kSegmentIdsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *input_mask = resize_and_alloc(
        &tensors[i][kInputMaskIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *masked_lm_labels = resize_and_alloc(
        &tensors[i][kMaskedLmLabelsIdx],
        {static_cast<int64_t>(cur_bs), static_cast<int64_t>(max_seq_length)});
    auto *next_sentence_labels = resize_and_alloc(
        &tensors[i][kNextSentenceLabelsIdx], {static_cast<int64_t>(cur_bs)});

    auto *seq_len = resize_and_alloc_int(&tensors[i][kSeqLenIdx],
                                         {static_cast<int64_t>(cur_bs)});
    auto *prefix_sum_seq_len = resize_and_alloc_int(
        &tensors[i][kPrefixSumSeqLenIdx], {static_cast<int64_t>(cur_bs + 1)});
    auto *num_valid = resize_and_alloc_float32(&tensors[i][kNumValidIdx],
                                               {static_cast<int64_t>(1)});

    // cpu tensor
    auto *host_prefix_sum_seq_len = resize_and_alloc_int(
        &tensors_2[i][0], {static_cast<int64_t>(cur_bs + 1)});

    prefix_sum_seq_len[0] = 0;
    int sum_seq_len = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      const T *data = arr;
      size_t sample_id = j + i * batch_size;
      if (need_sort) sample_id = seq_indices.get()[sample_id];
      std::memcpy(input_ids + j * max_seq_length,
                  data + sample_id * one_sample_len,
                  max_seq_length * sizeof(T));
      std::memcpy(segment_ids + j * max_seq_length,
                  data + sample_id * one_sample_len + max_seq_length,
                  max_seq_length * sizeof(T));
      std::memcpy(input_mask + j * max_seq_length,
                  data + sample_id * one_sample_len + 2 * max_seq_length,
                  max_seq_length * sizeof(T));
      std::memcpy(masked_lm_labels + j * max_seq_length,
                  data + sample_id * one_sample_len + 3 * max_seq_length,
                  max_seq_length * sizeof(T));
      next_sentence_labels[j] =
          data[sample_id * one_sample_len + 4 * max_seq_length];

      seq_len[j] = data[sample_id * one_sample_len + 4 * max_seq_length + 1];
      sum_seq_len += seq_len[j];
      if (j > 0) {
        prefix_sum_seq_len[j] = prefix_sum_seq_len[j - 1] + seq_len[j - 1];
      }
    }
    prefix_sum_seq_len[cur_bs] =
        prefix_sum_seq_len[cur_bs - 1] + seq_len[cur_bs - 1];

    std::memcpy(host_prefix_sum_seq_len,
                prefix_sum_seq_len,
                sizeof(int) * (cur_bs + 1));

    PADDLE_ENFORCE_LE(sum_seq_len, cur_bs * max_seq_length);

    auto *nonzeros_indices = resize_and_alloc_int(
        &tensors[i][kNonZerosIndicesIdx], {static_cast<int64_t>(sum_seq_len)});
    int cur_nonzero_ind = 0;
    int cur_num_valid = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      for (size_t k = 0; k < max_seq_length; ++k) {
        int ids = j * max_seq_length + k;
        if (input_mask[ids] != 0) {
          nonzeros_indices[cur_nonzero_ind++] = static_cast<int>(ids);
        }
        if (masked_lm_labels[ids] != 0) {
          cur_num_valid += 1;
        }
      }
    }
    PADDLE_ENFORCE_EQ(cur_nonzero_ind, sum_seq_len);

    *num_valid = cur_num_valid;
    auto *masked_lm_ids = resize_and_alloc_int(
        &tensors[i][kMaskedLmIdsIdx], {static_cast<int64_t>(cur_num_valid)});
    auto *masked_lm_positions =
        resize_and_alloc_int(&tensors[i][kMaskedLmPositionIdx],
                             {static_cast<int64_t>(cur_num_valid)});

    cur_num_valid = 0;
    for (size_t j = 0; j < cur_bs; ++j) {
      for (size_t k = 0; k < max_seq_length; ++k) {
        int ids = j * max_seq_length + k;
        if (masked_lm_labels[ids] != 0) {
          masked_lm_positions[cur_num_valid] = ids;
          masked_lm_ids[cur_num_valid] = masked_lm_labels[ids];
          cur_num_valid += 1;
        }
      }
    }
  }
  gpu_cpu_tensors.push_back(tensors);
  gpu_cpu_tensors.push_back(tensors_2);
  return gpu_cpu_tensors;
}

class Timer {
 public:
  using ClockType = std::chrono::high_resolution_clock;

  explicit Timer(bool enable = true) : enable_(enable) { reset(); }

  void reset() {
    if (enable_) {
      start_ = ClockType::now();
    }
  }

  double elapsed_time() {
    if (enable_) {
      auto end = ClockType::now();
      auto ms =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
              .count() /
          1.0e9;
      return ms;
    } else {
      return -1.0;
    }
  }

 private:
  std::chrono::time_point<ClockType> start_;
  const bool enable_;
};

class CommBuffer {
 public:
  CommBuffer(platform::CUDAPlace place,
             int ring_id,
             size_t nbytes,
             size_t trainer_id,
             size_t num_trainers) {
    nbytes_ = nbytes;
    num_trainers_ = num_trainers;
    trainer_id_ = trainer_id;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&gpu_ptr_, nbytes_ * num_trainers_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaHostAlloc(
        &cpu_ptr_, nbytes_ * num_trainers_, cudaHostAllocPortable));
    stream_ =
        platform::DeviceContextPool::Instance().GetByPlace(place)->stream();

    if (num_trainers > 1) {
      comm_ = platform::NCCLCommContext::Instance().Get(ring_id, place)->comm();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          gpu_ptr_, gpu_ptr_, 1, ncclInt8, ncclSum, comm_, stream_));
    } else {
      comm_ = nullptr;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream_));
    VLOG(10) << "nbytes = " << nbytes_ << " , trainer_id = " << trainer_id
             << " , num_trainers = " << num_trainers_;
  }

  ~CommBuffer() noexcept(false) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(gpu_ptr_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(cpu_ptr_));
  }

  template <typename T>
  std::vector<std::vector<framework::LoDTensorArray>> ExchangePadding(
      const std::vector<
          py::array_t<T, py::array::c_style | py::array::forcecast>> &array,
      size_t num_samples,
      size_t max_seq_length,
      size_t batch_size,
      bool drop_last,
      bool load_balance) {
    std::vector<std::pair<const void *, size_t>> input_metas;
    input_metas.reserve(array.size());
    for (const auto &arr : array) {
      const void *cur_cpu_ptr = arr.data();
      size_t cur_size = arr.size() * sizeof(T);
      input_metas.emplace_back(cur_cpu_ptr, cur_size);
    }

    py::gil_scoped_release guard;
    size_t total_nbytes = 0;
    auto gpu_ptr = gpu_ptr_ + trainer_id_ * nbytes_;

    constexpr auto kTimerLogLevel = 5;
    bool enable_timer = VLOG_IS_ON(kTimerLogLevel);
    Timer timer(enable_timer);
    timer.reset();
    for (size_t i = 0; i < input_metas.size(); ++i) {
      const void *cur_cpu_ptr = input_metas[i].first;
      size_t cur_size = input_metas[i].second;
      if (num_trainers_ > 1) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(gpu_ptr + total_nbytes,
                                                   cur_cpu_ptr,
                                                   cur_size,
                                                   cudaMemcpyHostToDevice,
                                                   stream_));
      } else {
        std::memcpy(cpu_ptr_ + total_nbytes, cur_cpu_ptr, cur_size);
      }
      total_nbytes += cur_size;
    }
    PADDLE_ENFORCE_EQ(total_nbytes, nbytes_);

    if (num_trainers_ > 1) {
      if (enable_timer) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream_));
        VLOG(kTimerLogLevel) << "H2D time: " << timer.elapsed_time();
      }

      timer.reset();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
          gpu_ptr, gpu_ptr_, nbytes_, ncclInt8, comm_, stream_));
      if (enable_timer) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream_));
        VLOG(kTimerLogLevel) << "AllGather time: " << timer.elapsed_time();
      }

      timer.reset();
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(cpu_ptr_,
                                                 gpu_ptr_,
                                                 nbytes_ * num_trainers_,
                                                 cudaMemcpyDeviceToHost,
                                                 stream_));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream_));
      if (enable_timer) {
        VLOG(kTimerLogLevel) << "D2H time: " << timer.elapsed_time();
      }
    }

    timer.reset();
    auto ret = ProcessAllGatheredBERTInputsBase<T>(
        reinterpret_cast<const T *>(cpu_ptr_),
        total_nbytes / sizeof(T) * num_trainers_,
        num_samples,
        max_seq_length,
        batch_size,
        trainer_id_,
        num_trainers_,
        drop_last,
        load_balance);
    VLOG(kTimerLogLevel) << "Process data time: " << timer.elapsed_time();
    return ret;
  }

 private:
  uint8_t *gpu_ptr_;
  uint8_t *cpu_ptr_;
  ncclComm_t comm_;
  cudaStream_t stream_;
  size_t nbytes_;
  size_t trainer_id_;
  size_t num_trainers_;
};

PYBIND11_MODULE(MLPERF_EXTENSION_NAME, m) {
  m.def("process_allgathered_inputs", &ProcessAllGatheredBERTInputs<int16_t>);
  m.def("process_allgathered_inputs", &ProcessAllGatheredBERTInputs<int32_t>);
  m.def("process_allgathered_inputs", &ProcessAllGatheredBERTInputs<int64_t>);
  m.def("process_eval_inputs", &ProcessBERTEvalInputs<int16_t>);
  m.def("process_eval_inputs", &ProcessBERTEvalInputs<int32_t>);
  m.def("process_eval_inputs", &ProcessBERTEvalInputs<int64_t>);

  py::class_<CommBuffer>(m, "CommBuffer", "CommBuffer Class")
      .def(py::init<platform::CUDAPlace, int, size_t, size_t, size_t>())
      .def("exchange_padding", &CommBuffer::ExchangePadding<int16_t>)
      .def("exchange_padding", &CommBuffer::ExchangePadding<int32_t>)
      .def("exchange_padding", &CommBuffer::ExchangePadding<int64_t>);
}
