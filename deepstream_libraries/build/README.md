# DeepStream Libraries - Build

## Overview
DeepStream Libraries is a meta-package that bundles NVIDIA's accelerated computer vision and video processing libraries for Python developers. This package provides a unified installation experience for CVCUDA, NvImageCodec, and PyNvVideoCodec.

## Package Versions

### Current Release (v1.4)
The following package versions are included in DeepStream Libraries v1.4:

| Package | Version | Description |
|---------|---------|-------------|
| `cvcuda-cu13` | 0.16.0 | CUDA-accelerated Computer Vision library |
| `PyNvVideoCodec` | 2.1 | Python bindings for NVIDIA Video Codec SDK |
| `nvidia-nvimgcodec-cu13` | 0.8.0.22 | NVIDIA Image Codec library for accelerated image I/O |
| `numpy` | >=1.26.0,<2.0 | NumPy library for numerical computing |

### System Requirements
- **Python**: >=3.12,<3.15
- **CUDA**: 13.2
- **TRT**: 10.16.0.72
- **Platform**: Ubuntu 24.04

## Build Instructions
1. From the repository root, navigate to the build directory:
   ```bash
   cd deepstream_libraries/build
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Build the wheel package using the provided build script:
   ```bash
   bash build_package.sh
   ```

4. To select an output directory, pass it as the first argument:
   ```bash
   bash build_package.sh /path/to/wheel-output
   ```
   To select a package version, pass it as the second argument; the default is `1.4`:
   ```bash
   bash build_package.sh /path/to/wheel-output 1.5
   ```
   You can also run the build from the repository root:
   ```bash
   bash build/build.sh --deepstream-libraries-wheel=/path/to/wheel-output
   bash build/build.sh --deepstream-libraries-wheel-version=1.5
   ```

5. The build script will:
   - Create the output directory if it does not exist; it defaults to the repository's `artifacts/` directory
   - Rename the wheel to include the `cp312-cp312-linux_x86_64` platform tag while preserving the selected package version
   - Temporarily package `deepstream_libraries/README.md`, then remove the copied package README
   - Accept a different platform tag through `WHEEL_TAG`, for example: `WHEEL_TAG=cp313-cp313-linux_x86_64 bash build_package.sh`
   - Print the wheel's absolute output path when the build completes

## Documentation
For more information, visit the [official DeepStream Libraries documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html).