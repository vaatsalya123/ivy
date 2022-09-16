# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Correlate
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        shared_dtype=True
    ),
    as_variable=helpers.array_bools(num_arrays=2),
    mode=st.sampled_from(["valid", "same", "full"]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.correlate"
    ),
    native_array=helpers.array_bools(num_arrays=2),)
def test_numpy_correlate(
    dtype_and_x,
    mode,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    (input_dtype_x, input_dtype_y), (x, y) = dtype_and_x
    input_dtype = [input_dtype_x, input_dtype_y]
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="correlate",
        a=np.asarray(x, dtype=input_dtype[0]),
        v=np.asarray(y, dtype=input_dtype[1]),
        mode=mode,
    )
