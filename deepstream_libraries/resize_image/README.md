# Resize-Image

## Overview

This is a command-line interface (CLI) application that allows users to resize images to specified dimensions. It leverages the power of `cvcuda` and `nvimgcodec` libraries to efficiently process images on systems with NVIDIA GPUs. The application is designed to be simple to use, requiring only the input image path and the desired output dimensions.

## Usage

To use Image Resize App, users must provide the path to the input image and can optionally specify the desired dimensions for the output image. The application can be run directly from the command line, providing a user-friendly experience.

## Output

The application will log operations at each step. After successful execution, it saves the resulting resized image in the current directory.

## Command Line Arguments
- `--input_image`: Path to the input image file.
- `--output_width`: Width of the output image. Default is 320.
- `--output_height`: Height of the output image. Default is 320.

## Example
```bash
python3 resize.py ../assets/images/tabby_tiger_cat.jpg 320 320
```