/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/python/ops/prefetch_runner.h"
%}

%rename("_TF_RegisterPrefetchRunner") TF_RegisterPrefetchRunner;

%include "tensorflow/python/ops/prefetch_runner.h"

%insert("python") %{
  def TF_RegisterPrefetchRunner(graph_key, runner_name, runner_options):
    opt_str = runner_options.SerializeToString()
    _TF_RegisterPrefetchRunner(graph_key, runner_name, opt_str)
%}
