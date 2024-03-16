"""
`pyproject.toml` isn't sufficient and `setup.py` is needed for two reasons:
  1. setuptools versions available for Python 3.6 don't support `pyproject.toml`, so the package metadata needs to be
     present in `setup.cfg` or `setup.py` when building the package using Python 3.6.
  2. Custom build commands are used for building the package, and there's no way to achieve that using `pyproject.toml`.

Defining the package metadata in two places is sure redundant, but causes no conflicts, because when using a setuptools
version that supports `pyproject.toml`, its metadata overrides `setup.py` metadata.
"""
import os
import ast
from pathlib import Path
import platform
from typing import Tuple, List
import subprocess

from setuptools import setup
from setuptools.dist import Distribution
from setuptools._distutils.sysconfig import customize_compiler
from setuptools._distutils.ccompiler import new_compiler
# Can't import distutils directly because it's deprecated and was removed in python 3.12.
from wheel.bdist_wheel import bdist_wheel


def is_apple_silicon_native() -> bool:
    """Returns True if Python is running on Apple Silicon natively."""
    # This only returns 'arm' when running natively on Apple Silicon, and not under Rosetta, which is desired behaviour.
    return platform.system() == 'Darwin' and platform.processor() == 'arm'


def is_clang() -> bool:
    """Returns whether the compiler being used is Clang."""
    # If clang is explicitly set as the default compiler.
    if os.environ.get('CC', None):
        return 'clang' in os.environ['CC']
    # When no compiler is explicitly set, `cc` is used, we check if it's clang by checking the output of `-v`.
    # Weirdly, the output of `cc -v` is redirected to stderr not stdout.
    return b'clang' in subprocess.run(['cc', '-v'], stderr=subprocess.PIPE).stderr.lower()


def get_compiler():
    """
    Returns an instance of CCompiler that can be used to compile C/C++ source code that uses the default compiler
    on each platform (MSVC on Windows, GCC on Linux, Clang on MacOS).
    The compiler type can be defined in `pyproject.toml` under the key `compiler`.
    """
    with open('pyproject.toml', 'r') as config_file:
        for line in config_file.readlines():
            line = "".join(line.split())
            idx = line.find('compiler=')
            if idx != -1:
                compiler_type = ast.literal_eval(line[idx + len('compiler='):])
    if compiler_type:
        os.environ['CC'] = compiler_type
    if platform.system() == 'Windows':
        if any(comp_id in compiler_type for comp_id in ['gcc', 'clang', 'mingw']):
            compiler_type = 'mingw32'
        compiler = new_compiler(compiler=compiler_type or None)
        compiler.shared_lib_extension = '.dll'  # Default dll extension for GCC and Clang on Windows is 'dll.a'.
    else:
        compiler = new_compiler()
        if compiler_type:
            customize_compiler(compiler)  # Need to call this function to use the `CC` defined above.
    return compiler


def get_compiler_flags(compiler) -> Tuple[List[str], List[str]]:
    """Return a tuple containing compiler and linker flags based on the compiler and platform."""
    if 'msvc' in type(compiler).__name__.lower():
        # /MT: Link the CRT statically, so we don't have to ship it alongside the package.
        compiler_flags = ['/openmp', '/Ox', '/MT']
        linker_flags = []
    elif platform.system() == 'Windows':  # For GCC (MinGW) or Clang on Windows.
        compiler_flags = ['-fopenmp', '-O3']  # Need to link statically because there's no automatic/easy way
        # to bundle openmp library with the package on Windows, unlike on Linux and macOS.
        linker_flags = ['-fopenmp', '-lucrt', '-static']
        compiler.dll_libraries = []
        # Empty this list because the code doesn't depend on msvcr and there's no need to include it.
    elif platform.system() == 'Darwin':
        libomp_dir_path = "/opt/homebrew/opt/libomp" if is_apple_silicon_native() else "/usr/local/opt/libomp"
        # /usr/local/opt/ is the installation path on Intel and Rosetta 2.
        compiler_flags = [f"-I{libomp_dir_path}/include", "-fopenmp", "-O3", "-fPIC"]
        linker_flags = [f"-L{libomp_dir_path}/lib", "-lomp"]
        if is_clang():
            compiler_flags.insert(1, '-Xclang')
    else:
        compiler_flags = ['-fopenmp', '-O3', '-fPIC']
        linker_flags = ['-fopenmp']
    return compiler_flags, linker_flags


class ExtModules(list):
    """Empty list that tests as truthful."""
    def __bool__(self) -> bool:
        return True


class BinaryDistribution(Distribution):
    """
    Custom Distribution which tells that it has C extension modules. This is needed because the package does have source
    files that it builds a shared library from, they just don't depend on the Python C API, and we use a custom build
    method to build the shared library. This way when creating a wheel, it's created like we have a distribution with
    C extension (a platform lib) which is what we want.
    """

    def __init__(self, *args, **kwargs):
        Distribution.__init__(self, *args, **kwargs)
        self.cmdclass['bdist_wheel'] = BuildSharedLib  # To call our custom build step when building the wheel.
        self.ext_modules = ExtModules()  # List that tests as truthful to trick the distribution to build a platlib.
        # See https://github.com/microsoft/debugpy/blob/main/setup.py#L36-L46 for more information about why it's needed

    def has_ext_modules(self) -> bool:
        """To trick the distribution to think it has C extension modules."""
        return True


class BuildSharedLib(bdist_wheel):
    """
    Custom build command for creating the built distribution (wheel) from the source distribution.
    See `get_tag()` and `run()` for more information about how it works.
    """

    def get_tag(self) -> Tuple[str, str, str]:
        """
        The build is only platform-dependent, so we only get the platform tag from the full tag (containing Python
        version and ABI version), and replace Python and ABI to 'py3' and 'none' respectively.
        We do that because the built shared library is independent of the Python API, it's not an extension module, and
        by doing that we'll only need to build the wheel once and not for every Python version.
        """
        python, abi, platform_tag = bdist_wheel.get_tag(self)
        return 'py3', 'none', platform_tag

    def run(self) -> None:
        """
        Custom `run` function to compile source files into a shared library from,
        place it in the build directory (lib/), and include it in the built distribution (wheel).
        """
        # Create an instance of CCompiler (msvc, gcc, cygwin, etc)
        compiler = get_compiler()
        compiler_flags, linker_flags = get_compiler_flags(compiler)

        # Prepare paths.
        shared_lib_filename = 'c_layers' + compiler.shared_lib_extension  # The name with extension.
        sources_paths = [os.path.join('xrnn', source_filename) for source_filename in ['layers_f.c', 'layers_d.c']]
        build_dir = os.path.join('xrnn', 'lib')  # Where the built library would reside.
        shared_lib_relative_path = os.path.join('lib', shared_lib_filename)  # This is relative from the package source.

        # Need to manually create the output directory for the shared library or a link error might occur.
        if not os.path.exists('lib'):
            os.mkdir('lib')

        # Compile the source into object files.
        objects = compiler.compile(sources_paths, extra_postargs=compiler_flags)

        # Create a dynamic/shared library from the object files named `shared_lib_filename` and place at `build_dir`.
        compiler.link_shared_object(objects, shared_lib_filename, build_dir, extra_postargs=linker_flags)

        # Add the library to the package data, so it would be included in the package's directory.
        self.distribution.package_data['xrnn'].append(shared_lib_relative_path)

        bdist_wheel.run(self)


# To read the README.md file and use it to populate `long_description`. Basically what readme="README.md" do.
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="xrnn",
    version="1.1.1",
    author="Yazan Sharaya",  # This is done in one line in `pyproject.toml`: authors = [{name = "", email = ""}]
    author_email="yazan.sharaya.yes@gmail.com",
    description="Light weight fast machine learning framework.",
    long_description=long_description,  # These two lines are equivalent to: readme = "README.md" in `pyproject.toml`.
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[  # Same as dependencies in `pyproject.toml`.
        "numpy>=1.17",
        "typing-extensions; python_version<'3.8'",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: C",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development"
    ],

    extras_require={  # Same as [project.optional-dependencies] in `pyproject.toml`.
        "test": [
            "pytest"
        ],
        "dev": [
            "build"
        ]
    },

    project_urls={  # Same as [project.urls]; repository = "" in `pyproject.toml`.
        "Repository": "https://github.com/Yazan-Sharaya/xrnn",
    },

    license_files=("LICENSE", ),  # Same as [tool.setuptools]; license-files = ["LICENSE"] in `pyproject.toml`.
    packages=["xrnn"],  # Same as [tool.setuptools]; packages = ["xrnn"] in `pyproject.toml`.

    package_data={"xrnn": ["*.c"]},  # Same as [tool.setuptools.package-data] in `pyproject.toml`.
    include_package_data=True,  # This line isn't present in `pyproject.toml` because it's implicitly set to True there.

    distclass=BinaryDistribution,  # A custom distribution class that is needed because of the custom build process.
)
