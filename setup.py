"""
There are two reasons for having a `setup.py` file and not just `pyproject.toml`.
 1. When using a setuptools version <=61.0.0, because versions before that don't support `pyproject.toml`. Version 61.0.0
    (which is the minimum version that supports reading metadat from `pyproject.toml`) isn't available for Python <3.7, and
    since this package supports python 3.6, I didn't want the build process to require Python > 3.6 to work.
 2. To build platform specific builds, because this package relies on compiled C code (which is platform specific),
    without the need to manually rename the built distribution (the `.whl` file) after the build process.

Notes
-----
* If support for python 3.6 is ever dropped, this `setup.py` will still be needed but stripped down to only
  `setup(distclass=BinaryDistribution)`. Until `setuptools` adds support for building binary distributions using
  `pyproject.toml`, `setup.py` will be needed.
* The structure of the arguments supplied to `setup` matches the structure of `pyproject.toml`.
* When using setuptools >=61.0.0 and subsequently python >= 3.7, The metadata in `pyproject.toml` override the
  metadata in `setup.py`, so for example if `version="2.0.0"` in `pyproject.toml` and `version="1.5.0" in `setup.py`,
  `version="2.0.0" is the one making it to the build.
"""
from setuptools.dist import Distribution
from setuptools import setup
from pathlib import Path


# To read the README.md file and use it to populate `long_description`. Basically what readme="README.md" do.
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


class BinaryDistribution(Distribution):
    """Passed to `distclass` argument in `setup` to build a platform specific wheel, since the compiled C source
    code is platform specific, so this prevents creating a universal wheel with the tag <any.none>."""
    @staticmethod
    def has_ext_modules(_=None):
        return True


setup(
    name="xrnn",
    version="1.0.1",
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
        # "Operating System :: MacOS",
        # "Operating System :: POSIX",
        # "Operating System :: Unix",
        "Programming Language :: C",
        "Programming Language :: C++",
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

    licens_files=("LICENSE", ),  # Same as [tool.setuptools]; license-files = ["LICENSE"] in `pyproject.toml`.
    packages=["xrnn"],  # Same as [tool.setuptools]; packages = ["xrnn"] in `pyproject.toml`.

    # Include the pre-compiled extension (shared/dynamic library)
    package_data={"xrnn": ["lib/*"]},  # Same as [tool.setuptools.package-data] in `pyproject.toml`.

    distclass=BinaryDistribution
)
