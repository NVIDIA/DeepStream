# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Import necessary libraries
import argparse
import cvcuda
from nvidia import nvimgcodec


def resize_image(input_image_path, output_width, output_height):
    """
    Resizes an image to the specified width and height.

    This function reads an image from the given file path, resizes it to the specified dimensions using a specified interpolation method, and saves the resized image back to the same file path. It is designed to work with images of any size and aspect ratio, and it aims to maintain the original aspect ratio by default, unless explicitly overridden.

    Parameters:
    - input_image_path (str): The file path of the image to be resized. The path must be a valid path to an image file.
    - output_width (int): The desired width of the resized image in pixels. Must be a positive integer.
    - output_height (int): The desired height of the resized image in pixels. Must be a positive integer.

    Returns:
    - None. The function saves the resized image to the same path as the input image.

    Example:
    >>> resize_image("path/to/image.jpg", 320, 240)
    This will resize the image at "path/to/image.jpg" to 320 pixels wide and 240 pixels tall, and save the resized image to the same path.
    """
    # Create Decoder
    decoder = nvimgcodec.Decoder()

    # Read image with nvImageCodec
    # New API: Create CodeStream from file bytes, then decode
    with open(input_image_path, "rb") as f:
        image_bytes = f.read()
    code_stream = nvimgcodec.CodeStream(image_bytes)
    inputImage = decoder.decode(code_stream)  # Single CodeStream returns single Image
    print(f"Reading image with nvImageCodec -> {input_image_path} ({inputImage.width}x{inputImage.height})")

    # Pass it to cvcuda using as_tensor
    nvcvInputTensor = cvcuda.as_tensor(inputImage, "HWC")

    # Resize with cvcuda
    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        nvcvResizeTensor = cvcuda.resize(nvcvInputTensor, (output_width, output_height, 3), cvcuda.Interp.CUBIC)
        nvcvResizeTensor.cuda().__cuda_array_interface__
    print("Resizing image with cvcuda")

    # Write with nvImageCodec
    encoder = nvimgcodec.Encoder()
    output_image_path = "output.jpg"
    encoder.write(output_image_path, nvimgcodec.as_image(nvcvResizeTensor.cuda(), cuda_stream = cvcuda_stream.handle))
    print(f"Writing image with nvImageCodec -> {output_image_path} ({output_width}x{output_height})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image using cvcuda and nvImageCodec.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("output_width", type=int, help="Width of the output image.", default=320, nargs='?')
    parser.add_argument("output_height", type=int, help="Height of the output image.", default=320, nargs='?')

    args = parser.parse_args()

    resize_image(args.input_image, args.output_width, args.output_height)