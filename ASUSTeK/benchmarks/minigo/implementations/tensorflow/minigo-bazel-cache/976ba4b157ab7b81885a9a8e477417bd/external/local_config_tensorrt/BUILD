# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [
        "tensorrt/include/tensorrt_config.h",
        
    ],
    include_prefix = "third_party/tensorrt",
    strip_include_prefix = "tensorrt/include",
)

cc_library(
    name = "tensorrt",
    srcs = [],
    copts = cuda_default_copts(),
    data = [],
    linkstatic = 1,
    deps = [
        ":tensorrt_headers",
        "@local_config_cuda//cuda",
    ],
)


