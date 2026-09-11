# DeepStream Libraries
DeepStream Libraries provide [CVCUDA](https://github.com/CVCUDA/CV-CUDA), [NvImageCodec](https://github.com/NVIDIA/nvImageCodec), and [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/) modules as Python APIs to easily integrate into custom frameworks.
Developers can build complete Python applications with fully accelerated components leveraging intuitive Python APIs.
Most of the DeepStream Libraries building blocks and their Python APIs are available today as standalone packages. DeepStream Libraries provide a way for Python developers to install these packages with a single installer.
All these packages are built against the same CUDA version and validated with the specified driver version. Reference applications are provided to demonstrate the usage of Python APIs.


## System Requirements

- **Operating System:**
  - [Ubuntu 24.04](https://releases.ubuntu.com/noble/)

- **Python:**
  - Python >=3.12,<3.15 (Should be pre-installed with Ubuntu 24.04)

- **CUDA:**
  - [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-13-2-0-download-archive)

- **NVIDIA Driver:**
  - [NVIDIA Driver 595.58.03](https://developer.nvidia.com/datacenter-driver-595-downloads)

- **TensorRT:**
  - [TensorRT 10.16.0.72](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/installing-tensorrt/installing.html)

## DeepStream Libraries Repository Setup

Follow these steps to set up your environment for running [sample applications](https://github.com/NVIDIA/DeepStream/tree/main/deepstream_libraries):

### 1. Clone Repository
```bash
git clone https://github.com/NVIDIA/DeepStream.git
cd deepstream/deepstream_libraries
```

### 2. Install System Dependencies
```bash
sudo sh scripts/install_sys_pkgs.sh
```

### 3. Download Sample Data
```bash
sh scripts/download_data.sh
```

### 4. Setup Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv deepstream_libraries_env

# Activate virtual environment
source deepstream_libraries_env/bin/activate

# Verify activation
which python3  # Should point to virtual environment
```
**Note:** Activate the virtual environment for Python dependencies and wheel installation, and in each new terminal session.

### 5. Install Python Dependencies
```bash
sh scripts/install_python_pkgs.sh
```

## DeepStream Libraries Installation
1. Download the DeepStream Libraries wheel file from the `artifacts/` directory in the [DeepStream repository release assets](https://github.com/NVIDIA/DeepStream/releases#release-v9.1.0).
    - Wheel file: `artifacts/deepstream_libraries-1.4-cp312-cp312-linux_x86_64.whl`

    Note: You can also download it from the [DeepStream v9.1 release assets](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0).

2. Install DeepStream Libraries package.
    ```bash
    pip3 install deepstream_libraries-1.4-cp312-cp312-linux_x86_64.whl
    ```

## Getting Started with DeepStream Libraries APIs
We can use DeepStream Libraries API's to create an application.

Consider the below reference example:
- Read an image from the given file path using NvImageCodec
- Resize the image with specified dimensions and Cubic interpolation method using CVCUDA
- Align output dimensions to ensure compatibility with nvImageCodec
- Save the resized image using NvImageCodec

```python
# Import necessary libraries
import cvcuda
from nvidia import nvimgcodec

# Create Decoder
decoder = nvimgcodec.Decoder()

# Read image with nvImageCodec (using CodeStream API)
with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()
code_stream = nvimgcodec.CodeStream(image_bytes)
inputImage = decoder.decode(code_stream)  # Returns single Image

# Pass it to cvcuda using as_tensor
nvcvInputTensor = cvcuda.as_tensor(inputImage, "HWC")

# Align output dimensions to 32-byte boundaries for nvImageCodec compatibility
output_width, output_height, alignment = 320, 240 , 32
aligned_width, aligned_height = ((output_width + alignment - 1) // alignment) * alignment , ((output_height + alignment - 1) // alignment) * alignment

# Resize with cvcuda using aligned dimensions
cvcuda_stream = cvcuda.Stream()
with cvcuda_stream:
    nvcvResizeTensor = cvcuda.resize(nvcvInputTensor,(aligned_height, aligned_width, 3), cvcuda.Interp.CUBIC)

# Write with nvImageCodec
encoder = nvimgcodec.Encoder()
output_image_path = "output.jpg"
encoder.write(output_image_path, nvimgcodec.as_image(nvcvResizeTensor.cuda(), cuda_stream = cvcuda_stream.handle))
```

## Sample Applications
| Application               | Description                                                                                                                                                      |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classification    | A CUDA-accelerated image and video classification pipeline integrating PyTorch or TensorRT for efficient processing on NVIDIA GPUs                               |
| Object-Detection  | GPU accelerated Object detection using CV-CUDA library with TensorFlow or TensorRT                                                 |
| Segmentation      | GPU accelerated Semantic segmentation by utilizing the CV-CUDA library with PyTorch or TensorRT                                               |
| Resize-Image      | A sample app that decodes, resizes, and encodes images using the CVCUDA and NvImageCodec Python API's                                                            |
| Decode-Video      | Decodes encoded bitstreams using PyNvVideoCodec decode APIs                                                               |
| Encode-Video      | Encodes a raw YUV file using PyNvVideoCodec encode APIs                                                                |
| Transcode-Video   | Transcodes the video files using PyNvVideoCodec API's                                                                        |

## Additional References and Applications
For more references and application please refer to the below link:
- [CVCUDA](https://github.com/CVCUDA/CV-CUDA/releases/tag/v0.16.0)
- [NvImageCodec](https://github.com/NVIDIA/nvImageCodec/releases/tag/v0.8.0)
- [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/2.1/)
- [Deepstream Libraries](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Libraries.html)

## Notes
- **VPF (VideoProcessingFramework)**: The main VPF package is deprecated. PyNvVideoCodec (installed via wheel) is the modern replacement. PytorchNvCodec extension is available for PyTorch support. VPF installation failures are non-fatal as PyNvVideoCodec provides the required functionality.
- **nvcv Compatibility**: A compatibility module (`common/nvcv.py`) is provided for legacy code that uses the `nvcv` API. This module maps `nvcv` calls to equivalent `cvcuda` functionality.
- **Archived Releases**: The old DeepStream Libraries releases are archived at [DeepStream Libraries releases](https://github.com/NVIDIA-AI-IOT/deepstream_libraries/releases).

## License

`deepstream_libraries-1.4-cp312-cp312-linux_x86_64.whl` is governed by the
[Software License Agreement for NVIDIA Software Development Kits](https://developer.download.nvidia.com/assets/Deepstream/DeepStream_6.0/NVIDIA_DeepStream_SDK_EULA_6.0.pdf). Refer to the
`LicenseAgreement` included within all GitHub Release Assets packages.

Use of DeepStream Libraries is also governed by the license terms of its
third-party components:
- [CVCUDA License](https://github.com/CVCUDA/CV-CUDA/blob/main/LICENSE.md)
- [NvImageCodec License](https://github.com/NVIDIA/nvImageCodec/blob/main/LICENSE.txt)
- [PyNvVideoCodec License](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec/files)
