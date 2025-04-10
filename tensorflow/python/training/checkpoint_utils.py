# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tools to work with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "load_checkpoint", "load_variable", "list_variables",
    "checkpoints_iterator", "init_from_checkpoint"
]


@tf_export("train.load_checkpoint")
def load_checkpoint(ckpt_dir_or_file):
  """Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

  If `ckpt_dir_or_file` resolves to a directory with multiple checkpoints,
  reader for the latest checkpoint is returned.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
      file.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: If `ckpt_dir_or_file` resolves to a directory with no
      checkpoints.
  """
  filename = _get_checkpoint_filename(ckpt_dir_or_file)
  if filename is None:
    raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % ckpt_dir_or_file)
  return pywrap_tensorflow.NewCheckpointReader(filename)


@tf_export("train.load_variable")
def load_variable(ckpt_dir_or_file, name):
  """Returns the tensor value of the given variable in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.

  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  """
  # TODO(b/29227106): Fix this in the right place and remove this.
  if name.endswith(":0"):
    name = name[:-2]
  reader = load_checkpoint(ckpt_dir_or_file)
  return reader.get_tensor(name)


@tf_export("train.list_variables")
def list_variables(ckpt_dir_or_file):
  """Returns list of all variables in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  """
  reader = load_checkpoint(ckpt_dir_or_file)
  variable_map = reader.get_variable_to_shape_map()
  names = sorted(variable_map.keys())
  result = []
  for name in names:
    result.append((name, variable_map[name]))
  return result


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
  logging.info("Waiting for new checkpoint at %s", checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    checkpoint_path = checkpoint_management.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info("Found new checkpoint at %s", checkpoint_path)
      return checkpoint_path


@tf_export("train.checkpoints_iterator")
def checkpoints_iterator(checkpoint_dir,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None):
  """Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  """
  checkpoint_path = None
  while True:
    new_checkpoint_path = wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_path, timeout=timeout)
    if new_checkpoint_path is None:
      if not timeout_fn:
        # timed out
        logging.info("Timed-out waiting for a checkpoint.")
        return
      if timeout_fn():
        # The timeout_fn indicated that we are truly done.
        return
      else:
        # The timeout_fn indicated that more checkpoints may come.
        continue
    start = time.time()
    checkpoint_path = new_checkpoint_path
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


@tf_export(v1=["train.init_from_checkpoint"])
def init_from_checkpoint(ckpt_dir_or_file, assignment_map, reset_version=False):
  """Replaces `tf.Variable` initializers so they load from a checkpoint file.

  Values are not loaded immediately, but when the initializer is run
  (typically by running a `tf.compat.v1.global_variables_initializer` op).

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching tensor
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with tensor 'scope_variable_name' from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with tensor 'scope_variable_name' from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Example:

  ```python

  # Say, '/tmp/model.ckpt' has the following tensors:
  #  -- name='old_scope_1/var1', shape=[20, 2]
  #  -- name='old_scope_1/var2', shape=[50, 4]
  #  -- name='old_scope_2/var3', shape=[100, 100]

  # Create new model's variables
  with tf.compat.v1.variable_scope('new_scope_1'):
    var1 = tf.compat.v1.get_variable('var1', shape=[20, 2],
                           initializer=tf.compat.v1.zeros_initializer())
  with tf.compat.v1.variable_scope('new_scope_2'):
    var2 = tf.compat.v1.get_variable('var2', shape=[50, 4],
                           initializer=tf.compat.v1.zeros_initializer())
    # Partition into 5 variables along the first axis.
    var3 = tf.compat.v1.get_variable(name='var3', shape=[100, 100],
                           initializer=tf.compat.v1.zeros_initializer(),
                           partitioner=lambda shape, dtype: [5, 1])

  # Initialize all variables in `new_scope_1` from `old_scope_1`.
  init_from_checkpoint('/tmp/model.ckpt', {'old_scope_1/': 'new_scope_1'})

  # Use names to specify which variables to initialize from checkpoint.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': 'new_scope_1/var1',
                        'old_scope_1/var2': 'new_scope_2/var2'})

  # Or use tf.Variable objects to identify what to initialize.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': var1,
                        'old_scope_1/var2': var2})

  # Initialize partitioned variables using variable's name
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': 'new_scope_2/var3'})

  # Or specify the list of tf.Variable objects.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': var3._get_variable_list()})

  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of the variables in the
      checkpoint and values are current variables or names of current variables
      (in default graph).

  Raises:
    ValueError: If missing variables in current graph, or if missing
      checkpoints or tensors in checkpoints.
  """
  init_from_checkpoint_fn = lambda _: _init_from_checkpoint(
      ckpt_dir_or_file, assignment_map, reset_version)
  if distribution_strategy_context.get_cross_replica_context():
    init_from_checkpoint_fn(None)
  else:
    distribution_strategy_context.get_replica_context().merge_call(
        init_from_checkpoint_fn)


def _init_from_checkpoint(ckpt_dir_or_file, assignment_map, reset_version=False):
  """See `init_from_checkpoint` for documentation."""
  ckpt_file = _get_checkpoint_filename(ckpt_dir_or_file)
  reader = load_checkpoint(ckpt_dir_or_file)
  variable_map = reader.get_variable_to_shape_map()
  for tensor_name_in_ckpt, current_var_or_name in sorted(
      six.iteritems(assignment_map)):
    var = None
    # Check if this is Variable object or list of Variable objects (in case of
    # partitioned variables).
    if _is_variable(current_var_or_name) or (
        isinstance(current_var_or_name, list)
        and all(_is_variable(v) for v in current_var_or_name)):
      var = current_var_or_name
    else:
      store_vars = vs._get_default_variable_store()._vars  # pylint:disable=protected-access
      # Check if this variable is in var_store.
      var = store_vars.get(current_var_or_name, None)
      # Also check if variable is partitioned as list.
      if var is None:
        var = _collect_partitioned_variable(current_var_or_name, store_vars)
    if var is not None:
      # If 1 to 1 mapping was provided, find variable in the checkpoint.
      if tensor_name_in_ckpt not in variable_map:
        raise ValueError("Tensor %s is not found in %s checkpoint %s" % (
            tensor_name_in_ckpt, ckpt_dir_or_file, variable_map
        ))
      if _is_variable(var):
        # Additional at-call-time checks.
        if not var.get_shape().is_compatible_with(
            variable_map[tensor_name_in_ckpt]):
          raise ValueError(
              "Shape of variable %s (%s) doesn't match with shape of "
              "tensor %s (%s) from checkpoint reader." % (
                  var.name, str(var.get_shape()),
                  tensor_name_in_ckpt, str(variable_map[tensor_name_in_ckpt])
              ))
        var_name = var.name
      else:
        var_name = ",".join([v.name for v in var])
      _set_variable_or_list_initializer(var, ckpt_file, tensor_name_in_ckpt)
      logging.debug("Initialize variable %s from checkpoint %s with %s",
                    var_name, ckpt_dir_or_file, tensor_name_in_ckpt)
    else:
      scopes = ""
      # TODO(vihanjain): Support list of 'current_var_or_name' here.
      if "/" in current_var_or_name:
        scopes = current_var_or_name[:current_var_or_name.rindex("/")]
      if not tensor_name_in_ckpt.endswith("/"):
        raise ValueError(
            "Assignment map with scope only name {} should map to scope only "
            "{}. Should be 'scope/': 'other_scope/'.".format(
                scopes, tensor_name_in_ckpt))
      # If scope to scope mapping was provided, find all variables in the scope
      # and create variable to variable mapping.
      scope_variables = set()
      for var_name in store_vars:
        var = store_vars.get(var_name, None)
        if not scopes or var_name.startswith(scopes + "/"):
          # Consume /part_ if partitioned variable.
          if "/part_" in var_name:
            if isinstance(var, kv_variable_ops.EmbeddingVariable):
              part_index = var_name.find("/part_")
              for i in range(part_index + 1, len(var_name)):
                if var_name[i] == "/":
                  break
              if i == len(var_name) - 1:
                part_str = var_name[part_index :]
              else:
                part_str = var_name[part_index : i]
              var_name =var_name.replace(part_str, "")
            else:
              var_name = var_name[:var_name.index("/part_")]
          scope_variables.add(var_name)
      for var_name in sorted(scope_variables):
        # Lookup name with specified prefix and suffix from current variable.
        # If tensor_name given is '/' (root), don't use it for full name.
        var = store_vars.get(var_name, None)
        full_tensor_name = var_name[len(scopes):]
        if current_var_or_name != "/":
          full_tensor_name = full_tensor_name[1:]
        if tensor_name_in_ckpt != "/":
          full_tensor_name = tensor_name_in_ckpt + full_tensor_name
        # Remove trailing '/', if any, in the full_tensor_name
        if full_tensor_name.endswith("/"):
          full_tensor_name = full_tensor_name[:-1]
        if full_tensor_name not in variable_map and var is not None and (
            not isinstance(var, kv_variable_ops.EmbeddingVariable)):
          raise ValueError(
              "Tensor %s (%s in %s) is not found in %s checkpoint" % (
                  full_tensor_name, var_name[len(scopes) + 1:],
                  tensor_name_in_ckpt, ckpt_dir_or_file
              ))
        if var is None:
          var = _collect_partitioned_variable(var_name, store_vars)
        _set_variable_or_list_initializer(var, ckpt_file, full_tensor_name, reset_version)
        logging.debug("Initialize variable %s from checkpoint %s with %s",
                      var_name, ckpt_dir_or_file, full_tensor_name)


def _get_checkpoint_filename(ckpt_dir_or_file):
  """Returns checkpoint filename given directory or specific checkpoint file."""
  if gfile.IsDirectory(ckpt_dir_or_file):
    return checkpoint_management.latest_checkpoint(ckpt_dir_or_file)
  return ckpt_dir_or_file


def _set_checkpoint_initializer(variable,
                                ckpt_file,
                                tensor_name,
                                slice_spec,
                                name="checkpoint_initializer",
                                reset_version=False):
  """Overrides given variable's initialization op.

  Sets variable initializer to assign op that initializes variable from tensor's
  value in the checkpoint.

  Args:
    variable: `tf.Variable` object.
    ckpt_file: string, full path of the checkpoint.
    tensor_name: Name of the tensor to load from the checkpoint.
    slice_spec: Slice specification for loading partitioned tensors.
    name: Name of the operation.
  """
  if isinstance(variable, kv_variable_ops.EmbeddingVariable):
    base_type = variable.dtype.base_dtype
    with ops.colocate_with(variable):
      if "/part_" in variable.op.name and "/part_" not in tensor_name:
        if variable.op.name.split('/')[-1].startswith('part_'):
          tensor_name = tensor_name + '/' + variable.op.name.split('/')[-1]
        else:
          part_index = variable.op.name.find("/part_")
          for i in range(part_index + 1, len(variable.op.name)):
            if variable.op.name[i] == "/":
              break
          if i == len(variable.op.name) - 1:
            part_str = variable.op.name[part_index :]
          else:
            part_str = variable.op.name[part_index : i]
          for i in range(len(tensor_name)-1, -1, -1):
            if tensor_name[i] == "/":
              break
          prev_str = tensor_name[:i]
          next_str = tensor_name[i:]
          tensor_name = prev_str + part_str + next_str
      is_partitioned_ev = variable._save_slice_info is not None
      partition_id = variable._save_slice_info.var_offset[0] if is_partitioned_ev else 0
      partition_num = variable._save_slice_info.full_shape[0] if is_partitioned_ev else 1
      restore_dependency = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLE_RESTORE_DEPENDENCY)[0]
      with ops.control_dependencies(restore_dependency[variable._primary_handle]):
        rank = variable.initial_value.get_shape().rank - 1
        restore_op = gen_kv_variable_ops.kv_resource_import_v3(
            ckpt_file,
            variable.handle,
            tensor_name,
            ops.convert_to_tensor(variable.invalid_key),
            shape=variable.initial_value.get_shape()[rank:],
            partition_id=partition_id,
            partition_num=partition_num,
            dtype=variable._dtype,
            reset_version=reset_version
        )
        variable._initializer_op = restore_op

  else:
    base_type = variable.dtype.base_dtype
    # Do not colocate with variable since RestoreV2 op only runs on CPU and
    # colocation will force variable (and other ops that colocate with variable)
    # to be on CPU as well. It is okay to place the variable's initializer op on
    # CPU since it will only be run once at the start.
    with ops.device(variable.device), ops.device("/cpu:0"):
      restore_op = io_ops.restore_v2(
          ckpt_file, [tensor_name], [slice_spec], [base_type], name=name)[0]

      names_to_saveables = saveable_object_util.op_list_to_dict([variable])
      saveable_objects = []
      for name, op in names_to_saveables.items():
        for s in saveable_object_util.saveable_objects_for_op(op, name):
          saveable_objects.append(s)

      assert len(saveable_objects) == 1  # Should be only one variable.
    init_op = saveable_objects[0].restore([restore_op], restored_shapes=None)

    # pylint:disable=protected-access
    variable._initializer_op = init_op
    restore_op.set_shape(variable.shape)
    variable._initial_value = restore_op
  # pylint:enable=protected-access


def _set_variable_or_list_initializer(variable_or_list, ckpt_file,
                                      tensor_name, reset_version=False):
  """Overrides initialization op of given variable or list of variables.

  Calls `_set_checkpoint_initializer` for each variable in the given list of
  variables.

  Args:
    variable_or_list: `tf.Variable` object or a list of `tf.Variable` objects.
    ckpt_file: string, full path of the checkpoint.
    tensor_name: Name of the tensor to load from the checkpoint.

  Raises:
    ValueError: if all objects in `variable_or_list` are not partitions of the
      same large variable.
  """
  if isinstance(variable_or_list, (list, tuple)):
    # A set of slices.
    slice_name = None
    for v in variable_or_list:
      slice_info = v._save_slice_info  # pylint:disable=protected-access
      if slice_name is None:
        slice_name = slice_info.full_name
      elif slice_name != slice_info.full_name:
        raise ValueError("Slices must all be from the same tensor: %s != %s" %
                         (slice_name, slice_info.full_name))
      _set_checkpoint_initializer(v, ckpt_file, tensor_name, slice_info.spec, reset_version=reset_version)
  else:
    _set_checkpoint_initializer(variable_or_list, ckpt_file, tensor_name, "", reset_version=reset_version)


def _is_variable(x):
  return (isinstance(x, variables.Variable) or
          resource_variable_ops.is_resource_variable(x))


def _collect_partitioned_variable(name, all_vars):
  """Returns list of `tf.Variable` that comprise the partitioned variable."""
  if name + "/part_0" in all_vars:
    var = []
    i = 0
    while name + "/part_%d" % i in all_vars:
      var.append(all_vars[name + "/part_%d" % i])
      i += 1
    return var
  else:
    if name == "/":
      st = 1
    else:
      st = 2
    for i in range(len(name) - st, -1, -1):
      if name[i] == "/":
        break
    prev_str = name[:i]
    next_str = name[i:]
    if prev_str + "/part_0" + next_str in all_vars:
      var = []
      i = 0
      while prev_str + "/part_%d" % i + next_str in all_vars:
        var.append(all_vars[prev_str + "/part_%d" % i + next_str])
        i += 1
      return var
  return None
