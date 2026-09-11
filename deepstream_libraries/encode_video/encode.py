# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import sys
import json
import argparse
sys.path.append('../')
from pathlib import Path
import PyNvVideoCodec as nvc
from common.nvc_utils import FetchGPUFrame, FetchCPUFrame, AppFrame

total_num_frames = 1000


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params):
    """
                This function illustrates encoding of frames using CUDA device buffers as input.

                The application reads the image data from file and loads it to CUDA input buffers using
                FetchGPUFrame(). The encoder subsequently copies the CUDA buffers and submits them to NVENC hardware
                for encoding as part of Encode() function. Video memory buffer allocated
                by the application to get the NVENC hardware output. This application copies the NVENC output
                from video memory buffer to host memory buffer in order to dump to a file, but this
                is not needed if application choose to use it in some other way.

                Parameters:
                    - gpu_id (int): Ordinal of GPU to use [Parameter not in use]
                    - dec_file_path (str): Path to
                file to be decoded
                    - enc_file_path (str): Path to output file into which raw frames are stored
                    - width (int): width of encoded frame
                    - height (int): height of encoded frame
                    - fmt (str) : surface format string in uppercase, for e.g. NV12
                    - config_params(key value pairs) : key value pairs providing fine-grained control on encoding

                Returns: - None.

                Example:
                >>> encode(0, "path/to/input/yuv/file","path/to/output/elementary/bitstream",1920,1080,"NV12")
                Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec and P4 preset
        """

    with open(dec_file_path, "rb") as decFile, open(enc_file_path, "wb") as encFile:
        nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)  # create encoder object
        input_frame_list = list([AppFrame(width, height, fmt) for x in range(1, 5)])
        for input_gpu_frame in FetchGPUFrame(input_frame_list,
                                             FetchCPUFrame(decFile, input_frame_list[0].frameSize),
                                             total_num_frames):
            bitstream = nvenc.Encode(input_gpu_frame)  # encode frame one by one
            bitstream = bytearray(bitstream)
            encFile.write(bitstream)

        bitstream = nvenc.EndEncode()  # flush encoder queue
        bitstream = bytearray(bitstream)
        encFile.write(bitstream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This sample application illustrates encoding of frames using CUDA device buffers as input."
    )

    parser.add_argument("-g", "--gpu_id", type=int, default=0, help="Unused variable, do not use", )
    parser.add_argument("-i", "--raw_file_path", type=Path, required=True, help="Raw video file (read from)", )
    parser.add_argument("-o", "--encoded_file_path", type=Path, required=True, help="Encoded video file (write to)", )
    parser.add_argument("-s", "--size", type=str, required=True, help="widthxheight of raw frame. Eg: 1920x1080", )
    parser.add_argument("-if", "--format", type=str, required=True, help="Format of input file", )
    parser.add_argument("-c", "--codec", type=str, required=True, help="h264, hevc, av1", )
    parser.add_argument("-json", "--config_file", type=str, default='', help="path of json config file", )

    args = parser.parse_args()
    config = {}

    if len(args.config_file):
        with open(args.config_file) as jsonFile:
            json_content = jsonFile.read()
        config = json.loads(json_content)
        config["preset"] = config["preset"].upper()

    args.codec = args.codec.lower()
    args.format = args.format.upper()
    config["codec"] = args.codec
    size = args.size.split("x")

    encode(args.gpu_id,
           args.raw_file_path.as_posix(),
           args.encoded_file_path.as_posix(),
           int(size[0]),
           int(size[1]),
           args.format,
           config)