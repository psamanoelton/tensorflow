# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow summary operations for TensorBoard."""

import functools
import threading

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as cond_lib
from tensorflow.python.ops import gen_audio_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.util.tf_export import tf_export


# Sentinel used for LazyTensorCreator._tensor to indicate that a value is
# currently being computed, in order to fail hard on reentrancy.
_CALL_IN_PROGRESS_SENTINEL = object()


class LazyTensorCreator:
  """Lazy auto-converting wrapper for a callable that returns a `tf.Tensor`.

  This class wraps an arbitrary callable that returns a `Tensor` so that it
  will be automatically converted to a `Tensor` by any logic that calls
  `tf.convert_to_tensor()`. This also memoizes the callable so that it is
  called at most once.

  The intended use of this class is to defer the construction of a `Tensor`
  (e.g. to avoid unnecessary wasted computation, or ensure any new ops are
  created in a context only available later on in execution), while remaining
  compatible with APIs that expect to be given an already materialized value
  that can be converted to a `Tensor`.

  This class is thread-safe.
  """

  def __init__(self, tensor_callable):
    """Initializes a LazyTensorCreator object.

    Args:
      tensor_callable: A callable that returns a `tf.Tensor`.
    """
    if not callable(tensor_callable):
      raise ValueError("Not a callable: %r" % tensor_callable)
    self._tensor_callable = tensor_callable
    self._tensor = None
    self._tensor_lock = threading.RLock()
    _register_conversion_function_once()

  def __call__(self):
    if self._tensor is None or self._tensor is _CALL_IN_PROGRESS_SENTINEL:
      with self._tensor_lock:
        if self._tensor is _CALL_IN_PROGRESS_SENTINEL:
          raise RuntimeError(
              "Cannot use LazyTensorCreator with reentrant callable"
          )
        elif self._tensor is None:
          self._tensor = _CALL_IN_PROGRESS_SENTINEL
          self._tensor = self._tensor_callable()
    return self._tensor


def _lazy_tensor_creator_converter(value, dtype=None, name=None, as_ref=False):
  """Converts a LazyTensorCreator to a Tensor."""
  del name  # ignored
  if not isinstance(value, LazyTensorCreator):
    raise RuntimeError("Expected LazyTensorCreator, got %r" % value)
  if as_ref:
    raise RuntimeError("Cannot use LazyTensorCreator to create ref tensor")
  tensor = value()
  if dtype not in (None, tensor.dtype):
    raise RuntimeError(
        "Cannot convert LazyTensorCreator returning dtype %s to dtype %s"
        % (tensor.dtype, dtype)
    )
  return tensor


# Use module-level bit and lock to ensure that registration of the
# LazyTensorCreator conversion function happens only once.
_conversion_registered = False
_conversion_registered_lock = threading.Lock()


def _register_conversion_function_once():
  """Performs one-time registration of `_lazy_tensor_creator_converter`.

  This helper can be invoked multiple times but only registers the conversion
  function on the first invocation, making it suitable for calling when
  constructing a LazyTensorCreator.

  Deferring the registration is necessary because doing it at at module import
  time would trigger the lazy TensorFlow import to resolve, and that in turn
  would break the delicate `tf.summary` import cycle avoidance scheme.
  """
  global _conversion_registered
  if not _conversion_registered:
    with _conversion_registered_lock:
      if not _conversion_registered:
        _conversion_registered = True
        tensor_conversion_registry.register_tensor_conversion_function(
            base_type=LazyTensorCreator,
            conversion_func=_lazy_tensor_creator_converter,
            priority=0,
        )


def _create_summary_metadata(display_name, description, plugin_name):
  return summary_pb2.SummaryMetadata(
      display_name=display_name,
      summary_description=description,
      plugin_data=summary_pb2.SummaryMetadata.PluginData(
          plugin_name=plugin_name
      ),
  )


@tf_export("summary.audio", v1=[])
def audio(
    name,
    data,
    sample_rate,
    step=None,
    max_outputs=3,
    encoding=None,
    description=None,
):
  """Write an audio summary."""
  audio_ops = gen_audio_ops

  if encoding is None:
    encoding = "wav"
  if encoding != "wav":
    raise ValueError("Unknown encoding: %r" % encoding)
  summary_metadata = _create_summary_metadata(
      display_name=None,
      description=description,
      plugin_name="audio",
  )
  inputs = [data, sample_rate, max_outputs, step]
  with summary_ops_v2.summary_scope(name, "audio_summary", values=inputs) as (
      tag,
      _,
  ):

    @LazyTensorCreator
    def lazy_tensor():
      ops.convert_to_tensor(data).shape.with_rank(3)
      ops.convert_to_tensor(max_outputs).shape.with_rank(0)
      limited_audio = data[:max_outputs]
      encode_fn = functools.partial(
          audio_ops.encode_wav, sample_rate=sample_rate
      )
      encoded_audio = map_fn.map_fn(
          encode_fn,
          limited_audio,
          dtype=dtypes.string,
          name="encode_each_audio",
      )
      # Workaround for map_fn returning float dtype for an empty elems input.
      encoded_audio = cond_lib.cond(
          array_ops.shape(input=encoded_audio)[0] > 0,
          lambda: encoded_audio,
          lambda: constant_op.constant([], dtypes.string),
      )
      limited_labels = array_ops.tile(
          [""], array_ops.shape(input=limited_audio)[:1]
      )
      return array_ops.transpose(
          a=array_ops.pack([encoded_audio, limited_labels])
      )

    return summary_ops_v2.write(
        tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
    )


DEFAULT_HISTOGRAM_BUCKET_COUNT = 30


def _buckets(data, bucket_count=None):
  """Create a TensorFlow op to group data into histogram buckets."""
  if bucket_count is None:
    bucket_count = DEFAULT_HISTOGRAM_BUCKET_COUNT
  with ops.name_scope("buckets"):
    ops.convert_to_tensor(bucket_count).shape.with_rank(0)
    # Treat a negative bucket count as zero.
    bucket_count = math_ops.maximum(0, bucket_count)
    data = array_ops.reshape(data, shape=[-1])  # flatten
    data = math_ops.cast(data, dtypes.float64)
    data_size = array_ops.size(input=data)
    is_empty = math_ops.logical_or(
        math_ops.equal(data_size, 0), math_ops.less_equal(bucket_count, 0)
    )

    def when_empty():
      """When input data is empty or bucket_count is zero."""
      return array_ops.zeros((bucket_count, 3), dtype=dtypes.float64)

    def when_nonempty():
      min_ = math_ops.reduce_min(input_tensor=data)
      max_ = math_ops.reduce_max(input_tensor=data)
      range_ = max_ - min_
      has_single_value = math_ops.equal(range_, 0)

      def when_multiple_values():
        """When input data contains multiple values."""
        bucket_width = range_ / math_ops.cast(bucket_count, dtypes.float64)
        offsets = data - min_
        bucket_indices = math_ops.cast(
            math_ops.floor(offsets / bucket_width), dtype=dtypes.int32
        )
        clamped_indices = math_ops.minimum(bucket_indices, bucket_count - 1)
        one_hots = array_ops.one_hot(
            clamped_indices, depth=bucket_count, dtype=dtypes.float64
        )
        bucket_counts = math_ops.cast(
            math_ops.reduce_sum(input_tensor=one_hots, axis=0),
            dtype=dtypes.float64,
        )
        edges = math_ops.linspace(min_, max_, bucket_count + 1)
        edges = array_ops.concat([edges[:-1], [max_]], 0)
        left_edges = edges[:-1]
        right_edges = edges[1:]
        return array_ops.transpose(
            a=array_ops.pack([left_edges, right_edges, bucket_counts])
        )

      def when_single_value():
        """When input data contains a single unique value."""
        edges = array_ops.fill([bucket_count], max_)
        zeroes = array_ops.fill([bucket_count], 0)
        bucket_counts = math_ops.cast(
            array_ops.concat([zeroes[:-1], [data_size]], 0)[:bucket_count],
            dtype=dtypes.float64,
        )
        return array_ops.transpose(
            a=array_ops.pack([edges, edges, bucket_counts])
        )

      return cond_lib.cond(
          has_single_value, when_single_value, when_multiple_values
      )

    return cond_lib.cond(is_empty, when_empty, when_nonempty)


@tf_export("summary.histogram", v1=[])
def histogram(name, data, step=None, buckets=None, description=None):
  """Write a histogram summary."""
  data = array_ops.stop_gradient(data)
  summary_metadata = _create_summary_metadata(
      display_name=None, description=description, plugin_name="histograms"
  )
  with summary_ops_v2.summary_scope(
      name, "histogram_summary", values=[data, buckets, step]
  ) as (tag, _):

    def lazy_tensor():
      return _buckets(data, buckets)

    return summary_ops_v2.write(
        tag=tag,
        tensor=lazy_tensor,
        step=step,
        metadata=summary_metadata,
    )


@tf_export("summary.image", v1=[])
def image(name, data, step=None, max_outputs=3, description=None):
  """Write an image summary."""
  summary_metadata = _create_summary_metadata(
      display_name=None, description=description, plugin_name="images"
  )
  with summary_ops_v2.summary_scope(
      name, "image_summary", values=[data, max_outputs, step]
  ) as (tag, _):

    def lazy_tensor():
      ops.convert_to_tensor(data).shape.with_rank(4)
      ops.convert_to_tensor(max_outputs).shape.with_rank(0)
      images = image_ops.convert_image_dtype(data, dtypes.uint8, saturate=True)
      limited_images = images[:max_outputs]
      encoded_images = image_ops.encode_png(limited_images)
      image_shape = array_ops.shape(input=images)
      dimensions = array_ops.pack(
          [
              string_ops.as_string(image_shape[2]),
              string_ops.as_string(image_shape[1]),
          ],
          name="dimensions",
      )
      return array_ops.concat([dimensions, encoded_images], axis=0)

    return summary_ops_v2.write(
        tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
    )


@tf_export("summary.scalar", v1=[])
def scalar(name, data, step=None, description=None):
  """Write a scalar summary."""
  summary_metadata = _create_summary_metadata(
      display_name=None, description=description, plugin_name="scalars"
  )
  with summary_ops_v2.summary_scope(
      name, "scalar_summary", values=[data, step]
  ) as (tag, _):
    ops.convert_to_tensor(data).shape.with_rank(0)
    return summary_ops_v2.write(
        tag=tag,
        tensor=math_ops.cast(data, dtypes.float32),
        step=step,
        metadata=summary_metadata,
    )


@tf_export("summary.text", v1=[])
def text(name, data, step=None, description=None):
  r"""Write a text summary."""
  summary_metadata = _create_summary_metadata(
      display_name=None, description=description, plugin_name="text"
  )
  with summary_ops_v2.summary_scope(
      name, "text_summary", values=[data, step]
  ) as (tag, _):
    tensor = ops.convert_to_tensor(data)
    if tensor.dtype != dtypes.string:
      raise TypeError("Tensor must be of type string, got %s" % tensor.dtype)
    return summary_ops_v2.write(
        tag=tag, tensor=data, step=step, metadata=summary_metadata
    )
