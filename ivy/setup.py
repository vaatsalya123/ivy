from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import sys
import os

# os.environ['LDSHARED'] = 'clang -shared' # ToDo add user compiler inputs instead

# Enable AddressSanitizer in Cython
get_directive_defaults()["linetrace"] = True
get_directive_defaults()["binding"] = True
# get_directive_defaults()['cdivision'] = True
# get_directive_defaults()['embedsignature'] = True
get_directive_defaults()["boundscheck"] = False
# get_directive_defaults()['wraparound'] = False
get_directive_defaults()["profile"] = False
get_directive_defaults()["infer_types"] = True
get_directive_defaults()["annotation_typing"] = True
get_directive_defaults()["fast_getattr"] = True
get_directive_defaults()["c_string_encoding"] = "ascii"
get_directive_defaults()["c_string_type"] = "char"


def get_modules():
    modules = []
    for filename in os.listdir():
        if filename.endswith(".pyx"):
            modules.append(Extension(filename[:-4], [filename]))
        elif filename.endswith(".so") and ".cpython" in filename:
            os.rename(filename, filename[: filename.find(".cpython")] + ".so")
    return modules


# Check if --no-docstrings command is present
no_docstrings = "--no-docstrings" in sys.argv
if no_docstrings:
    sys.argv.remove("--no-docstrings")

setup(
    ext_modules=cythonize(
        get_modules(),
        compiler_directives={"language_level": 3},
        annotate=not no_docstrings,
    ),
)

get_modules()
