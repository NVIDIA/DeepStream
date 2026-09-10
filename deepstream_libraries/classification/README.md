# Classification

## Overview

This application demonstrates a complete pipeline for classifying images or video frames using CUDA-accelerated operations on NVIDIA GPUs. It showcases the integration of CUDA with high-level frameworks like PyTorch or TensorRT for performing image classification tasks. The sample is designed to handle both image and video inputs, applying a series of processing stages including decoding, pre-processing, inference, and post-processing.

## Usage

The application is executed from the command line, where you can specify the input path, output directory, batch size, target image dimensions, device ID, and the backend framework for inference (PyTorch or TensorRT).

## Output

The application  logs comprehensive data throughout every operational phase. Upon successful execution, it outputs the top 5 inference results on the console.

## Command Line Arguments
- `--input_path`:  Path to the input file (image or video) or directory containing images. Default is ../assets/images/tabby_tiger_cat.jpg.
- `--output_dir`: Directory where the classification results will be saved. Default is /tmp.
- `--batch_size`: Number of images or frames to process in a batch. Default is 4.
- `--target_img_height`: Height of the images after resizing. Default is 224.
- `--target_img_width`: Width of the images after resizing. Default is 224.
- `--device_id`: ID of the CUDA device to use for processing. Default is 0.
- `--backend`: Backend framework to use for inference (pytorch or tensorrt). Default is tensorrt.
- `--log_level`: Logging level (e.g., INFO, DEBUG). Default is info.

## Examples of Classification
- Run classification on a single image with batch size 1 with TensorRT backend
    ```bash
    python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1 --target_img_height 224 --target_img_width 224 --device_id 0 --backend tensorrt
    ```
- Run classification on folder containing images with Pytorch backend
    ```bash
    python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 2  --backend pytorch
    ```
- Run classification on a video file with TensorRT backend
    ```bash
    python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4 --output_dir ./output --batch_size 4 --backend tensorrt
    ```