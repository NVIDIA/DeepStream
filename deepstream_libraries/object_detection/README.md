# Object-Detection

## Overview

This application is a command-line interface (CLI) tool designed for object detection using NVIDIA GPUs with CUDA acceleration. It utilizes the CV-CUDA library along with high-level machine learning frameworks such as TensorFlow or TensorRT to perform object detection on images or video frames. The sample demonstrates a complete pipeline that includes decoding, pre-processing, inference, post-processing, and encoding stages.

## Usage

The application is executed from the command line, where you can specify the input path, output directory, batch size, target image dimensions, device ID, backend framework for inference, confidence threshold, and Intersection over Union (IoU) threshold.

## Output

The application  logs comprehensive data throughout every operational phase. Upon successful execution, it stores the resulting image or video in the designated output directory (output_dir). The saved output includes detected objects, which are blurred while being enclosed within bounding boxes.

## Command Line Arguments
- `--input_path`: Path to the input file (image or video) or directory containing images. Default is ../assets/images/peoplenet.jpg'.
- `--output_dir`:  Directory where the detection results and output images or video will be saved. Default is /tmp.
- `--batch_size`: Number of images or frames to process in a batch. Default is 4.
- `--target_img_height`: Height of the images after resizing. Default is 544.
- `--target_img_width`: Width of the images after resizing. Default is 960.
- `--device_id`: ID of the CUDA device to use for processing. Default is 0.
- `--backend`: Backend framework to use for inference (tensorflow or tensorrt). Default is tensorrt.
- `--confidence_threshold`: Confidence threshold for filtering out detected bounding boxes. Default is 0.4.
- `--iou_threshold`: IoU threshold for Non-Maximum Suppression (NMS). Default is 0.2.
- `--log_level`: Logging level (e.g., INFO, DEBUG). Default is info.


## Examples of Object-Detection
- Run object detection on a single image with batch size 1 with TensorRT backend
    ```bash
    python3 main.py --input_path ../assets/images/peoplenet.jpg --output_dir ./output --target_img_height 544 --target_img_width 960 --device_id 0 --backend tensorrt --confidence_threshold 0.9 --iou_threshold 0.2
    ```
- Run object detection  on folder containing images with TensorRT backend
    ```bash
    python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 3  --backend tensorrt
    ```
- Run object detection  on a video file with TensorFlow backend
    ```bash
    python3 main.py --input_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --output_dir ./output --batch_size 2 --backend tensorflow
    ```

Note: To use the tensorflow backend in a MultiGPU device we need to export CUDA_VISIBLE_DEVICES='0'.