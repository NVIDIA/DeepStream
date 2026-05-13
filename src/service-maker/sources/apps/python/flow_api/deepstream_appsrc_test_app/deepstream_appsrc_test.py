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

from pyservicemaker import Pipeline, Flow, BufferRetriever, BufferProvider, ColorFormat, as_tensor, Buffer
import torch
import sys
from queue import Queue, Empty
import threading

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
        print("MyBufferProvider")
        super().__init__()
        self.queue_ = queue
        self.format = "RGB"
        self.width = 1280
        self.height = 720
        self.framerate = 30
        self.device = 'gpu'

    def generate(self, size):
        try:
            tensor = self.queue_.get(timeout=2)
            torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
            if SAVE_TENSOR:
                torch.save(torch_tensor, f"image_{self.count}.pt")
            ds_tensor = as_tensor(torch_tensor, "HWC")
            self.count += 1
            print(f"generated: {self.count}")
            return ds_tensor.wrap(ColorFormat.RGB)
        except Empty:
            print("Queue empty...")
            return Buffer()

def run_appsink_flow(uri_list, queue):
    appsink_pipeline = Pipeline("appsink")
    appsink_flow = Flow(appsink_pipeline).batch_capture(uri_list).retrieve(MyBufferRetriever(queue))
    appsink_flow()

def run_appsrc_flow(queue):
    appsrc_pipeline = Pipeline("appsrc")
    appsrc_flow = Flow(appsrc_pipeline).inject([MyBufferProvider(queue)]).render()
    appsrc_flow()

def deepstream_appsrc_test_app(uri_list):
    queue = Queue(maxsize=10)

    thread1 = threading.Thread(target=run_appsink_flow, args=(uri_list, queue))
    thread2 = threading.Thread(target=run_appsrc_flow, args=(queue,))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread2.join()
    thread1.join()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % sys.argv[0])
        sys.exit(1)

    deepstream_appsrc_test_app(sys.argv[1:])