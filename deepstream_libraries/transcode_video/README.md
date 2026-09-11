# Transcode-Video

## Overview

This application is a command-line interface (CLI) tool designed for transcoding video files using NVIDIA GPUs. It leverages the PyCuda library and PyNvVIdeoCodec SDK to perform efficient video transcoding. The tool supports various input formats and can transcode videos to different output formats, resolutions, and bitrates, making use of the GPU's hardware-accelerated decoding and encoding capabilities This is particularly useful for developers, content creators, and researchers who need to convert videos into formats suitable for different devices or for further video processing tasks.

## Usage

To use the transcode tool, you need to specify the GPU ID, the path to the input video file, the path where the transcoded video file will be saved, and the encoding configuration parameters. The application is executed from the command line, providing an efficient way to transcode videos on systems equipped with NVIDIA GPUs.
The provided code snippet demonstrates the transcoding process, where the input video is decoded and then re-encoded according to the specified encoding configuration. The transcoding process involves decoding the video to raw frames and then encoding these frames into the target format. The tool prints out the width and height of the video for reference and keeps track of the number of frames processed. The transcoded video is then saved to the specified output path.

## Output 

During execution, the application extensively logs various details including assumptions made during processing, warnings, and timing information for both session initialization and deinitialization, along with details such as the width and height of the video frames and the total frame count. After successful execution, it saves the transcoded output video in the current directory.

## Command Line Arguments
- `--gpu_id`: The ID of the GPU to use for transcoding. Default is 0.
- `--in_file_path`: The file path of the input video.
- `--out_file_path`: The file path where the transcoded video will be saved.
- `--codec`: The codec to use for encoding (e.g., h264, hevc, av1).
- `--preset`: The encoding preset to use (e.g., P1, P2, P3, P4, P5, P6, P7). Default is None.
- `--config_file`: path of json config file. Various encoding properties can be set via this config file like FPS, Bitrate. Default is None.

## Example
```bash
python3 transcode.py --gpu_id 0 --in_file_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --out_file_path output.h264 --codec "h264" --preset "P1" --config_file ../assets/configs/encode_config.json
```

Please verify the supported decoders and encoders for the GPU device specified at the following [link](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new).