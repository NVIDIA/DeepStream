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

import sys
import argparse
import numpy as np

sys.path.append('../')
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from common.nvc_utils import cast_address_to_1d_bytearray

def decode(gpu_id, enc_file_path, dec_file_path, use_device_memory):
    """
            Function to decode media file and write raw frames into an output file.

            This function will read a media file and split it into chunks of data (packets).
            A Packet contains elementary bitstream belonging to one frame and conforms to annex.b standard.
            Packet is sent to decoder for parsing and hardware accelerated decoding. Decoder returns list of raw YUV
            frames which can be iterated upon.

            Parameters: - gpu_id (int): Ordinal of GPU to use [Parameter not in use] - enc_file_path (str): Path to
            file to be decoded - enc_file_path (str): Path to output file into which raw frames are stored -
            use_device_memory (int): if set to 1 output decoded frame is CUDeviceptr wrapped in CUDA Array Interface
            else its Host memory Returns: - None.

            Example:
            >>> decode(0, "path/to/input/media/file","path/to/output/yuv", 1)
            Function to decode media file and write raw frames into an output file.
    """
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=use_device_memory)

    decoded_frame_size = 0
    raw_frame = None


    seq_triggered = False
    # printing out FPS and pixel format of the stream for convenience
    print("FPS = ", nv_dmx.FrameRate())
    # open the file to be decoded in write mode
    with open(dec_file_path, "wb") as decFile:
        # demuxer can be iterated, fetch the packet from demuxer
        for packet in nv_dmx:
            # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
            # size of (decode picture buffer) depends on GPU, fur Turing series its 8
            for decoded_frame in nv_dec.Decode(packet):
                # 'decoded_frame' contains list of views implementing cuda array interface
                # for nv12, it would contain 2 views for each plane and two planes would be contiguous 
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                    seq_triggered = True


                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                if use_device_memory:
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    bits = bytearray(raw_frame)
                    decFile.write(bits)
                else:
                    new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                    decFile.write(bytearray(new_array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This sample application illustrates the demuxing and decoding of a media file."
    )
    parser.add_argument(
        "-g", "--gpu_id", type=int, help="GPU id, check nvidia-smi, only for demo, do not use", )
    parser.add_argument(
        "-i", "--encoded_file_path", type=Path, required=True,
        help="Encoded video file (read from)", )
    parser.add_argument(
        "-o", "--raw_file_path", required=True, type=Path, help="Raw NV12 video file (write to)", )
    parser.add_argument(
        "-d", "--use_device_memory", required=True, type=int, help="Decoder output surface is in device memory else in "
                                                                   "host memory", )
    args = parser.parse_args()
    decode(args.gpu_id, args.encoded_file_path.as_posix(),
           args.raw_file_path.as_posix(),
           args.use_device_memory)
