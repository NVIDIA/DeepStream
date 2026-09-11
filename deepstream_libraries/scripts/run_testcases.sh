#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script run various sample apps of DeepStream Libraries.
# run the script using this cmd "sh scripts/run_testcases.sh"

# Initialize variables for counting successes and failures
current_test_case=0
success_count=0
failure_count=0
failed_test_cases=""

# Function to execute a test case
# Arguments:
#   $1 - Test case number
#   $2 - Command to execute
run_test() {
    current_test_case=$((current_test_case + 1))
    echo "################################ Test Case $current_test_case #####################################"
    echo "#  $1  #"
    echo "Running command: $2"
    if eval "$2"; then
        echo "Test Case $current_test_case: PASSED"
        success_count=$((success_count + 1))
    else
        echo "Test Case $current_test_case: FAILED"
        failure_count=$((failure_count + 1))
        # Store the failed test case information
        failed_test_cases="${failed_test_cases} \tTest Case ${current_test_case}: $1"'\n'
    fi
}

# Function to display summary report
display_report() {
    echo "################################ Summary ########################################"
    echo "Total Test Cases: $((success_count + failure_count))"
    echo "Success Count: $success_count"
    echo "Failure Count: $failure_count"
    # Check if there are any failed test cases and print them
    if [ "$failure_count" -gt 0 ]; then
        echo "Failed Test Cases:\n$failed_test_cases"
    fi
    echo "#################################################################################"
}

# Move to parent directory
if [ ! -d "assets" ]; then
    echo "Moving to parent directory."
    cd ..
fi

# Test cases
cd resize_image
run_test "Resize Image on a single image with default pramas" "python3 resize.py ../assets/images/tabby_tiger_cat.jpg"
run_test "Resize Image on a single image with height=500 width=700" "python3 resize.py ../assets/images/tabby_tiger_cat.jpg 500 700"
cd ..

cd classification
# run_test "Classification with default params" "python3 main.py"
run_test "Classification on a single image with batch size 1 with TensorRT backend" "python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1 --target_img_height 224 --target_img_width 224 --device_id 0 --backend tensorrt"
run_test "Classification on a single image with batch size 1 with Pytorch backend" "python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1 --target_img_height 224 --target_img_width 224 --device_id 0 --backend pytorch"
run_test "Classification on folder containing images with Pytorch backend" "python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 2  --backend pytorch"
run_test "Classification on a video file with TensorRT backend" "python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4 --output_dir ./output --batch_size 4 --backend tensorrt"
run_test "Classification on a video file with Pytorch backend" "python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4 --output_dir ./output --batch_size 4 --backend pytorch"
cd ..

cd object_detection
# run_test "Object-Detection with default params" "python3 main.py"
run_test "Object-Detection on a single image with batch size 1 with TensorRT backend" "python3 main.py --input_path ../assets/images/peoplenet.jpg --output_dir ./output --target_img_height 544 --target_img_width 960 --device_id 0 --backend tensorrt --confidence_threshold 0.9 --iou_threshold 0.2"
run_test "Object-Detection on a single image with batch size 1 with TensorFlow backend" "python3 main.py --input_path ../assets/images/peoplenet.jpg --output_dir ./output --target_img_height 544 --target_img_width 960 --device_id 0 --backend tensorflow --confidence_threshold 0.9 --iou_threshold 0.2"
run_test "Object-Detection on folder containing images with TensorRT backend" "python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 3  --backend tensorrt"
run_test "Object-Detection on a video file with TensorRT backend" "python3 main.py --input_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --output_dir ./output --batch_size 2 --backend tensorrt"
run_test "Object-Detection on a video file with TensorFlow backend" "python3 main.py --input_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --output_dir ./output --batch_size 2 --backend tensorflow"
cd ..

cd segmentation
# run_test "Segmentation with default params" "python3 main.py"
run_test "Segmentation on a single image with Pytorch backend" "python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1 --backend pytorch"
run_test "Segmentation on a single image with TensorRT backend" "python3 main.py --input_path ../assets/images/tabby_tiger_cat.jpg --output_dir ./output --batch_size 1 --backend tensorrt"
run_test "Segmentation on folder containing images with pytorch backend" "python3 main.py --input_path ../assets/images/ --output_dir ./output --batch_size 2 --backend pytorch"
run_test "Segmentation on a video file with TensorRT backend" "python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  --output_dir ./output --batch_size 4 --backend tensorrt"
run_test "Segmentation on a video file with Pytorch backend" "python3 main.py --input_path ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4  --output_dir ./output --batch_size 4 --backend pytorch"
run_test "Benchmark on segmentation app" "python3 ../scripts/benchmark.py -np 1 -w 1 -o ./output main.py -b 4 -i ../assets/videos/pexels-ilimdar-avgezer-7081456.mp4"
cd ..


cd decode_video
run_test "Decode-Video with use_device_memory 0" "python3 decode.py --gpu_id 0 --encoded_file_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --raw_file_path output.yuv --use_device_memory 0"
run_test "Decode-Video with use_device_memory 1" "python3 decode.py --gpu_id 0 --encoded_file_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --raw_file_path output.yuv --use_device_memory 1"
cd ..

cd encode_video
run_test "Encode-Video" "python3 encode.py --gpu_id 0 --raw_file_path ../decode_video/output.yuv  --encoded_file_path output.h264 --size 1920x1080 --format nv12 --codec h264 --config_file ../assets/configs/encode_config.json"
cd ..

cd transcode_video
run_test "Transcode-Video" "python3 transcode.py --gpu_id 0 --in_file_path ../assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4 --out_file_path output.h264 --codec h264 --preset P1 --config_file ../assets/configs/encode_config.json"
cd ..

# Display summary report
display_report

# Exit with failure if there are failed test cases
if [ "$failure_count" -gt 0 ]; then
    exit 1
fi
