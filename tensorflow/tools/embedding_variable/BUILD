# Description:
#   Utilities that perform useful transformations on graphs

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cuda_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_cc_binary(
    name = "ev_ckpt_transformer",
    srcs = ["ckpt_format_transform.cc"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
    ],
)
