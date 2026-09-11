# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

def get_python_minor_version():
    """Get the current Python minor version."""
    return sys.version_info.minor

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Check if PREBUILT_SO environment variable is set
        prebuilt_so = os.environ.get("PREBUILT_SO")

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        if prebuilt_so and os.path.exists(prebuilt_so):
            # Use prebuilt .so file
            print(f"Using prebuilt .so from: {prebuilt_so}\n")
            if not extdir.exists():
                extdir.mkdir(parents=True)

            # Use the filename from the prebuilt .so to preserve the correct architecture tag
            prebuilt_filename = os.path.basename(prebuilt_so)
            target_path = extdir / prebuilt_filename
            shutil.copy2(prebuilt_so, target_path)
            return

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Get Python minor version
        python_minor_version = get_python_minor_version()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_MINOR_VERSION={python_minor_version}"
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed to build for SBSA)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

setup(
    name='pyservicemaker',
    version='0.0.1',
    packages=['pyservicemaker'],
    package_data={'pyservicemaker': ['_pydeepstream*.so']},
    description='DeepStream Python Service Maker',
    cmdclass={'build_ext': CMakeBuild},
    ext_modules=[CMakeExtension('pyservicemaker._pydeepstream')],
    python_requires=">=3.10"
)