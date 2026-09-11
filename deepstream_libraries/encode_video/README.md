# Encode-Video

## Overview

This application is a command-line interface (CLI) tool designed for encoding raw video files into different formats using NVIDIA GPUs. It utilizes the PyNvVideoCodec library to perform efficient video encoding. The tool is capable of encoding a sequence of raw video frames (surface formats -NV12, YUV 420, YUV 444, ARGB, ABGR) into video bitstream (codecs - H.264, HEVC, and AV1), leveraging the GPU's hardware-accelerated encoding capabilities. This is particularly useful for developers, content creators, and researchers who need to encode raw video data for various applications.

## Usage

To use the video encoding tool, you need to specify the GPU ID, the path to the raw video file, the path where the encoded video file will be saved, the resolution of the raw video frames, the format of the raw video, and the encoding configuration parameters. The application is executed from the command line, providing an efficient way to encode videos on systems equipped with NVIDIA GPUs.

## Output 

During execution, the application logs  key actions related to context management and stream creation. After successful execution, it saves the encoded output video in the current directory.

## Command Line Arguments
- `--gpu_id`: The ID of the GPU to use for encoding. Default is 0.
- `--raw_file_path`: The file path of the raw video.
- `--encoded_file_path`: The file path where the encoded video will be saved.
- `--size`: widthxheight of raw frame.
- `--format`: Format of input file.
- `--codec`: The codec to use for encoding (e.g., h264, hevc, av1).
- `--config_file`: path of json config file. Default is None.

## Example
```bash
python3 encode.py --gpu_id 0 --raw_file_path ../decode_video/output.yuv  --encoded_file_path output.h264 --size 1920x1080 --format nv12 --codec h264 --config_file ../assets/configs/encode_config.json
```
Note: Above command uses the output of decode_video app as input, so user need to run the decode video 1st to run this encode video example.

Please verify the supported encoders for the GPU device specified at the following [link](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new).