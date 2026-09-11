# Decode-Video

## Overview

This application is a command-line interface (CLI) tool designed for decoding video files to raw NV12 format using NVIDIA GPUs. It leverages the PyCuda library and PyNvVIdeoCodec SDK to perform efficient video decoding. The tool is capable of handling videos encoded in various formats by utilizing the GPU's hardware-accelerated decoding capabilities. This is particularly useful for developers, researchers, and enthusiasts working with video processing and analysis who require access to raw video frames for further processing.

## Usage

The application is executed from the command line, where you can specify the GPU ID, the path to the encoded video file, the path where the decoded raw NV12 video file will be saved, and whether the decoder output surface is in device memory or host memory.

## Output 

During execution, the application extensively logs various details including configuration settings, warnings, assumptions made during processing, and timing information for both session initialization and deinitialization. After successful execution, it saves the decoded output video in the current directory.

## Command Line Arguments
- `--gpu_id`: The ID of the GPU to use for decoding.
- `--encoded_file_path`: The file path of the encoded video.
- `--raw_file_path`: The file path where the decoded raw NV12 video will be saved.
- `--use_device_memory`: Specify 1 Decoder output surface is in device memory else 0 for host memory.

## Example
```bash
python3 decode.py --gpu_id 0 --encoded_file_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --raw_file_path output.yuv --use_device_memory 1
```

Please verify the supported decoder for the GPU device specified at the following [link](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new).