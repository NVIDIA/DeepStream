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

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from multiprocessing import Process
import sys

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            vehcle_count = 0
            person_count = 0
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                if class_id == 0:
                    vehcle_count += 1
                elif class_id == 2:
                    person_count += 1
            print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
                  f"Frame Number={frame_meta.frame_number},"
                  f"Vehicle Count={vehcle_count}, Person Count={person_count}")
            text = f"Person={person_count},Vehicle={vehcle_count}"
            display_meta = batch_meta.acquire_display_meta()
            label = osd.Text()
            label.display_text = text.encode('ascii')
            label.x_offset = 10
            label.y_offset = 12
            label.font.name = osd.FontFamily.Serif
            label.font.size = 12
            label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            label.set_bg_color = True
            label.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(label)
            frame_meta.append(display_meta)

def deepstream_test1_app(stream_file_path):
    pipeline = Pipeline("deepstream-test1")
    flow = Flow(pipeline).batch_capture([stream_file_path]).infer(CONFIG_FILE_PATH)
    flow.attach(what=Probe("counter", ObjectCounterMarker())).render()()

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> \n" % sys.argv[0])
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_test1_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()