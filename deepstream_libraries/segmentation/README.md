# Semantic Segmentation

## Overview

This application is a command-line interface (CLI) tool designed for semantic segmentation using NVIDIA GPUs with CUDA acceleration. It utilizes the CV-CUDA library along with high-level machine learning frameworks such as PyTorch or TensorRT to perform semantic segmentation on images or video frames. The sample demonstrates a complete pipeline that includes decoding, pre-processing, inference, post-processing, and encoding stages.

## Usage

The application is executed from the command line, where you can specify the input path, output directory, class name for visualization, batch size, target image dimensions, device ID, and backend framework for inference.

## Output

The application logs comprehensive data throughout every operational phase. Upon successful execution, it stores the resulting image or video in the designated output directory (output_dir). The saved output includes blurred segmented objects.

## Command Line Arguments
- `--input_path`: Path to the input file (image or video) or directory containing images. Default is ../assets/images/Weimaraner.jpg.
- `--output_dir`: Directory where the segmentation results and output images or video will be saved. Default is /tmp.
- `--class_name`: The class name to visualize the results for. Default is __background__.
- `--batch_size`: Number of images or frames to process in a batch. Default is 4.
- `--target_img_height`: Height of the images after resizing. Default is 224.
- `--target_img_width`: Width of the images after resizing. Default is 224.
- `--device_id`: ID of the CUDA device to use for processing. Default is 0.
- `--backend`: Backend framework to use for inference (pytorch or tensorrt). Default is tensorrt.
- `--log_level`: Logging level (e.g., INFO, DEBUG). Default is info.

## Examples of Segmentation without Triton
Run segmentation app for different data modalities
- Run segmentation on a single image
  ```bash
  python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1
  ```

- Run segmentation on folder containing images with pytorch backend
  ```bash
  python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 2 --backend pytorch
  ```

- Run segmentation on a video file with tensorrt backend
  ```bash
  python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  --output_dir ./output --batch_size 4 --backend tensorrt
  ```

- Run benchmark on segmentation app

  To benchmark this run, we can use the benchmark.py in the following way. It should launch 1 process, ignore 1 batch from front and end as warmup batches, save per process and overall numbers as JSON files in /tmp directory. To understand more about performance benchmarking in CV-CUDA, please refer to [Performance Benchmarking README](https://gitlab-master.nvidia.com/cv/cvcuda/-/blob/main/samples/scripts/README.md)
  ```
  python3 ../scripts/benchmark.py -np 1 -w 1 -o ./output main.py -b 4 -i ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4 
    ```

## Examples of Segmentation with Triton
### Triton Server instructions

Triton has different public [Docker images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver): `-py3-sdk` for Triton client libraries, `-py3` for Triton server libraries with TensorRT, ONNX, Pytorch, TensorFlow, `-pyt-python-py3` for Triton server libraries with PyTorch and Python backend only.

1. Launch the triton server
      ```bash
      docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti --gpus '"device=0"' -v <deepstream_libraries_local_mount_path>:/deepstream_libraries -w /deepstream_libraries nvcr.io/nvidia/tritonserver:25.03-py3
      ```

2. DeepStream Libraries Installation [steps](../README.md#deepstream_libraries-installation)

3. DeepStream Libraries Repository Setup [steps](../README.md#deepstream_libraries-repository-setup)

4. Start the triton server.
   Update the `inference_backend` parameter in config.pbtxt to "pytorch" or "tensorrt". Default backend is "tensorrt"
      ```bash
      tritonserver --model-repository triton_models
      ```

### Triton Client instructions
1. Launch the triton client docker
      ```bash
      docker run -ti --net host --gpus '"device=0"' -v <deepstream_libraries_local_mount_path>:/deepstream_libraries -w /deepstream_libraries nvcr.io/nvidia/tritonserver:25.03-py3-sdk /bin/bash
      ```
    In case the client and server are on the same machine in a local-server setup, we can simply reuse the server image (and even docker exec into the same container) by installing the Triton client utilities:
      ```bash
      pip3 install tritonclient[all]
      ```

2. DeepStream Libraries Installation [steps](../README.md#deepstream_libraries-installation)

3. DeepStream Libraries Repository Setup [steps](../README.md#deepstream_libraries-repository-setup)

4. Run client script for different data modalities
    - Run segmentation on a single image
      ```bash
      python3 triton_client.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1
      ```

    - Run segmentation on folder containing images
      ```bash
      python3 triton_client.py --input_path ../assets/images/ --output_dir ./output --batch_size 2 --backend pytorch
      ```

    - Run segmentation on a video file
      ```bash
      python3 triton_client.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  --output_dir ./output --batch_size 4
      ```

    - Run segmentation on a video file with streamed encoding/decoding (highly recommended as performance is greatly improved in this mode), use --stream_video or -sv
      ```bash
      python3 triton_client.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  --output_dir ./output --batch_size 4 --stream_video
      ```

    - Run benchmark on segmentation app

      To benchmark this client run, we can use the benchmark.py in the following way. It should launch 1 process, ignore 1 batch from front and end as warmup batches, save per process and overall numbers as JSON files in /tmp directory. To understand more about performance benchmarking in CV-CUDA, please refer to [Performance Benchmarking README](https://gitlab-master.nvidia.com/cv/cvcuda/-/blob/main/samples/scripts/README.md)
        ```bash
        python3 ../scripts/benchmark.py -np 1 -w 1 -o /tmp triton_client.py -i ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  -b 4 -sv
        ```

Note: MultiGPU support with Triton is not available. As a workaround, please use --gpus '"device=0". This will restrict the container to only GPU0.