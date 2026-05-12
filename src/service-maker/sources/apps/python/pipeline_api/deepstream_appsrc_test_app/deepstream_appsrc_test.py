#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, BufferRetriever, Receiver, BufferProvider, Feeder, ColorFormat, as_tensor, Buffer
from multiprocessing import Process
import sys
import torch
import platform
from queue import Queue, Empty

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
SAVE_TENSOR = False

class MyBufferRetriever(BufferRetriever):
    count = 0
    def __init__(self, queue):
        super().__init__()
        self.queue_ = queue

    def consume(self, buffer):
        tensor = buffer.extract(0).clone()
        self.queue_.put(tensor)
        self.count += 1
        print(f"consumed: {self.count}")
        return 1

class MyBufferProvider(BufferProvider):
    count = 0
    def __init__(self, queue):
        super().__init__()
        self.queue_ = queue

    def generate(self, size):
        try:
            tensor = self.queue_.get(timeout=2)
            torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
            if SAVE_TENSOR:
                torch.save(torch_tensor, f"image_{self.count}.pt")
            ds_tensor = as_tensor(torch_tensor, "HWC")
            buffer = ds_tensor.wrap(ColorFormat.RGB)
            self.count += 1
            print(f"generated: {self.count}")
            return buffer
        except Empty:
            print("Queue empty...")
            return Buffer()


def main(args):
    queue = Queue(maxsize=10)
    p1 = Pipeline("appsrc-test")
    p2 = Pipeline("appsink-test")

    #construct test pipeline for appsrc
    p1.add("appsrc", "src", {"caps": f"video/x-raw(memory:NVMM), format=RGB, width={VIDEO_WIDTH}, height={VIDEO_HEIGHT}, framerate=30/1", "do-timestamp": True})
    p1.add("nvvideoconvert", "convert", {"nvbuf-memory-type": 2, "compute-hw": 1})
    p1.add("capsfilter", "capsfilter", {"caps": "video/x-raw(memory:NVMM), format=NV12"})
    p1.add("nvstreammux", "mux", {"batch-size": 1, "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT})
    p1.add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink", {"sync": False})
    p1.link("src", "convert").link(("convert", "mux"), ("", "sink_%u")).link("mux", "sink")
    p1.attach("src", Feeder("feeder", MyBufferProvider(queue)), tips="need-data/enough-data").start()

    # construct test pipeline for appsink
    uri_list = args[1:]
    for index, uri in enumerate(uri_list):
        p2.add("nvurisrcbin", f"src_{index}", {"uri": uri})
    p2.add("nvstreammux", "mux", {"batch-size": len(uri_list), "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT})
    p2.add("nvvideoconvert", "converter")
    p2.add("capsfilter", "capsfilter", {"caps": "video/x-raw(memory:NVMM), format=RGB"})
    p2.add("appsink", "sink", {"emit-signals": True})
    for i, _ in enumerate(uri_list):
        p2.link((f"src_{i}", "mux"), ("", "sink_%u"))
    p2.link("mux", "converter", "capsfilter", "sink")
    p2.attach("sink", Receiver("receiver", MyBufferRetriever(queue)), tips="new-sample").start()

    p2.wait()
    p1.wait()



if __name__ == '__main__':
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv,))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()