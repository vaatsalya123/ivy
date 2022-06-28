"""Collection of tests for unified image-related functions."""

# global

from hypothesis import given, strategies as st

# local
import ivy
import random
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# stack_images
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=8),
    ratio=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=2),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="stack_images"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
)
def test_stack_images(
    shape,
    ratio,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and input_dtype == "float16":
        return
    images = [img for img in ivy.random_normal(shape=shape)]
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "stack_images",
        images=images,
        desired_aspect_ratio=ratio,
    )


# linear_resample
@given(
    shape=helpers.list_of_length(st.integers(min_value=2, max_value=8), 2),
    num_samples=st.integers(min_value=2, max_value=8),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="linear_resample"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
)
def test_linear_resample(
    shape,
    num_samples,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    x = ivy_np.random_normal(shape=shape, device=ivy.default_device())
    axis = random.randint(0, len(shape) - 1)
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "linear_resample",
        x=x,
        num_samples=num_samples,
        axis=axis,
    )


# bilinear_resample
@given(
    batch_shape=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4),
    h_w=helpers.list_of_length(st.integers(min_value=2, max_value=8), 2),
    n_dims=st.integers(min_value=1, max_value=8),
    n_samples=st.integers(min_value=1, max_value=8),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="bilinear_resample"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
)
def test_bilinear_resample(
    batch_shape,
    h_w,
    n_dims,
    n_samples,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    x = ivy.random_normal(shape=batch_shape + h_w + [n_dims])
    warp = ivy.random_uniform(shape=batch_shape + [n_samples, 2])
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "bilinear_resample",
        x=x,
        warp=warp,
    )


# gradient_image
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=4, max_size=8),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="gradient_image"),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_gradient_image(
    shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
    device,
):
    if fw == "torch" and input_dtype == "float16":
        return
    # ToDo: remove the random internal generation, which is causing flaky behaviour
    #  and hypothesis failures
    x = ivy_np.random_normal(shape=shape, device=device)
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "gradient_image",
        x=x,
    )


# float_img_to_uint8_img
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=3, max_size=8),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="float_img_to_uint8_img"),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_float_img_to_uint8_img(
    shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and input_dtype == "float16":
        return
    x = ivy.random_normal(shape=shape)
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "float_img_to_uint8_img",
        x=x,
    )


# uint8_img_to_float_img
@given(
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=3, max_size=8),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="uint8_img_to_float_img"),
    native_array=st.booleans(),
    container=st.booleans(),
)
def test_uint8_img_to_float_img(
    shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and input_dtype == "float16":
        return
    x = ivy.randint(0, 256, shape=shape + [4])
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "uint8_img_to_float_img",
        x=x,
    )


# random_crop
@given(
    shape=st.lists(st.integers(min_value=2, max_value=8), min_size=3, max_size=3),
    seed=st.integers(min_value=1, max_value=8),
    input_dtype=st.sampled_from(ivy.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    num_positional_args=helpers.num_positional_args(fn_name="random_crop"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
)
def test_random_crop(
    shape,
    seed,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    if fw == "torch" and input_dtype == "float16":
        return
    x = ivy.random_normal(shape=[3] + shape)
    crop_size = [random.randint(1, shape[-3] * 2), random.randint(1, shape[-2] * 2)]
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "random_crop",
        x=x,
        crop_size=crop_size,
        seed=seed,
    )
