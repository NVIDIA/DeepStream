# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, RenderMode
import threading, time
import os

RENDER_MODE = RenderMode.DISPLAY if os.environ.get("DISPLAY") else RenderMode.DISCARD

def stop_pipeline(pipeline, timeout=1):
    time.sleep(timeout)
    print("Stop")
    pipeline.stop()

class ObjectCounter(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.frames = 0

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
        self.frames += 1

class MetadataExtractor(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.extracted_count = 0
        self.total_objects = 0
        self.total_bboxes = 0
        self.has_segmaps = False

    def handle_metadata(self, batch_meta):
        # Test the C++ extract() method
        extracted_list = batch_meta.extract()

        # Verify it's a list with size max_frames
        assert isinstance(extracted_list, list)
        assert len(extracted_list) == batch_meta.max_frames

        # Process each extracted metadata
        for metadata in extracted_list:
            if metadata is not None:
                self.extracted_count += 1
                # Verify metadata structure (should be a dict now)
                assert isinstance(metadata, dict)
                assert 'shape' in metadata
                assert 'bboxes' in metadata
                assert 'probs' in metadata
                assert 'labels' in metadata
                assert 'objects' in metadata
                assert 'seg_maps' in metadata
                assert 'timestamp' in metadata

                # Count objects and bboxes
                self.total_objects += len(metadata['objects'])
                self.total_bboxes += len(metadata['bboxes'])

                # Check if we have segmentation maps
                if len(metadata['seg_maps']) > 0:
                    self.has_segmaps = True
                    # Verify seg_maps are numpy arrays of int type
                    import numpy as np
                    for seg_map in metadata['seg_maps']:
                        assert isinstance(seg_map, np.ndarray)
                        assert seg_map.dtype == np.int32 or seg_map.dtype == np.int64

source = "/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/source_list_static.yaml"

pgie_config = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

sgie_cofig_list = [
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie1_config.yml",
    "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test2/dstest2_sgie2_config.yml"
]

tracker_configs = [
    (
        "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    )
]

def test_inference():
    pipeline = Pipeline("test")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 120))
    thread.start()
    flow = Flow(pipeline).batch_capture(source, file_loop=False).infer(pgie_config)
    flow = flow.track(ll_config_file=tracker_configs[0][0], ll_lib_file=tracker_configs[0][1])
    for sgie_config in sgie_cofig_list:
        flow = flow.infer(config=sgie_config)
    r = ObjectCounter()
    flow.attach(what=Probe("counter", r)).render(RENDER_MODE)()
    assert r.frames > 0
    thread.join()

def test_tracker():
    for ll_config, ll_lib in tracker_configs:
        pipeline = Pipeline("test")
        thread = threading.Thread(target=stop_pipeline, args=(pipeline, 120))
        thread.start()
        Flow(pipeline).batch_capture(source, file_loop=False).infer(pgie_config).track(ll_config_file=ll_config, ll_lib_file=ll_lib).render(RENDER_MODE)()
        thread.join()

def test_metadata_extract():
    """Test the C++ batch_metadata.extract() method"""
    pipeline = Pipeline("test_extract")
    thread = threading.Thread(target=stop_pipeline, args=(pipeline, 120))
    thread.start()

    flow = Flow(pipeline).batch_capture(source, file_loop=False).infer(pgie_config)
    flow = flow.track(ll_config_file=tracker_configs[0][0], ll_lib_file=tracker_configs[0][1])

    extractor = MetadataExtractor()
    flow.attach(what=Probe("extractor", extractor)).render(RENDER_MODE)()

    # Verify extraction worked
    assert extractor.extracted_count > 0, "Should have extracted metadata from at least one frame"
    assert extractor.total_objects > 0, "Should have detected some objects"
    assert extractor.total_bboxes > 0, "Should have some bounding boxes"

    print(f"Metadata extraction test passed:")
    print(f"  - Frames extracted: {extractor.extracted_count}")
    print(f"  - Total objects: {extractor.total_objects}")
    print(f"  - Total bboxes: {extractor.total_bboxes}")
    print(f"  - Has segmentation maps: {extractor.has_segmaps}")

    thread.join()

if __name__ == '__main__':
    test_inference()
    test_tracker()
    test_metadata_extract()