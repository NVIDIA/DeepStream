#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd, postprocessing
from multiprocessing import Process
import sys
import platform
import os
import torch
import torchvision.ops as ops
import math
from typing import Dict, List

PIPELINE_NAME = "deepstream-test1"
CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"

STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

class ResnetDetectorConverter(postprocessing.ObjectDetectorOutputConverter):
    """Sample tensor output converter for Resnet object detector"""
    NETWORK_WIDTH = 960
    NETWORK_HEIGHT = 544
    def __init__(self):
        self._threshold = 0.2
        self._bbox_norm = 35.0

    def __call__(self, output_layers: Dict) -> List:
        """
        Convert output tensors to object detection results

        Args:
            output_layers: direct output layers from the model, dictionary of (name, tensor)
        return:
            List of BBox tensor ``(class_id, confidence, x1, y1, x2, y2)``
        """
        outputs = []
        bbox_data = output_layers.pop('output_bbox/BiasAdd:0', None)
        conv_data = output_layers.pop('output_cov/Sigmoid:0', None)
        if bbox_data is None or conv_data is None:
            print("Output layers not found!")
            return outputs
        n_classes = conv_data.shape[0]
        if bbox_data and conv_data:
            bbox_tensor = torch.utils.dlpack.from_dlpack(bbox_data).to('cpu')
            conv_tensor = torch.utils.dlpack.from_dlpack(conv_data).to('cpu')
            strides = (math.ceil(self.NETWORK_WIDTH/conv_tensor.shape[1]), math.ceil(self.NETWORK_HEIGHT/conv_tensor.shape[2]))
            center_x = [(i*strides[1]+0.5)/self._bbox_norm for i in range(conv_tensor.shape[2])]
            center_y = [(i*strides[1]+0.5)/self._bbox_norm for i in range(conv_tensor.shape[2])]
            mask = conv_tensor > self._threshold
            conv_tensor = conv_tensor[mask]
            indices = mask.nonzero(as_tuple=False)
            # calculate the bboxes based on the grid and confidence threshold
            objects_per_class = [[] for _ in range(n_classes)]
            for conv, i in zip(conv_tensor, indices):
                class_id = i[0]
                grid_y = i[1]
                grid_x = i[2]
                x1 = -(bbox_tensor[class_id*n_classes][grid_y][grid_x]-center_x[grid_x])*self._bbox_norm
                y1 = -(bbox_tensor[class_id*n_classes+1][grid_y][grid_x]-center_y[grid_y])*self._bbox_norm
                x2 = (bbox_tensor[class_id*n_classes+2][grid_y][grid_x]+center_x[grid_x])*self._bbox_norm
                y2 = (bbox_tensor[class_id*n_classes+3][grid_y][grid_x]+center_y[grid_y])*self._bbox_norm
                objects_per_class[int(class_id)].append([float(conv), float(x1), float(y1), float(x2), float(y2)])
            # do the grouping
            for c in range(n_classes):
                if objects_per_class[c]:
                    bboxes = torch.tensor([n[1:] for n in objects_per_class[c]], dtype=torch.float32)
                    scores = torch.tensor([n[0] for n in objects_per_class[c]], dtype=torch.float32)
                    keep_indices = ops.nms(bboxes, scores, 0.2)
                    bboxes = bboxes[keep_indices]
                    scores = scores[keep_indices]
                    objects = torch.cat([scores.unsqueeze(1), bboxes], dim=1).tolist()
                    for o in objects:
                        outputs.append([c] + o)
        return outputs

class ObjectCounterMarker(BatchMetadataOperator):
    def __init__(self, output_tensor_meta):
        super().__init__()
        self._use_tensor = output_tensor_meta
        self._postprocessing = ResnetDetectorConverter() if output_tensor_meta else None

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            vehicle_count = 0
            person_count = 0
            if self._use_tensor:
                for user_meta in frame_meta.tensor_items:
                    objects = self._postprocessing(user_meta.as_tensor_output().get_layers())
                    scales = [
                        STREAM_WIDTH/float(ResnetDetectorConverter.NETWORK_WIDTH),
                        STREAM_HEIGHT/float(ResnetDetectorConverter.NETWORK_HEIGHT)
                    ]
                    for o in objects:
                        if o[0] == 0:
                            vehicle_count += 1
                        elif o[0] == 2:
                            person_count += 1
                        # create object meta for downstream usage and osd display
                        object_meta = batch_meta.acquire_object_meta()
                        object_meta.class_id = o[0]
                        object_meta.confidence = o[1]
                        object_meta.rect_params.left = o[2] * scales[0]
                        object_meta.rect_params.top = o[3] * scales[1]
                        object_meta.rect_params.width = (o[4]- o[2]) * scales[0]
                        object_meta.rect_params.height = (o[5] - o[3]) * scales[1]
                        object_meta.rect_params.border_width = 1
                        object_meta.rect_params.border_color = osd.Color(1.0, 0, 0, 1.0)
                        frame_meta.append(object_meta)
            else:
                for object_meta in frame_meta.object_items:
                    class_id = object_meta.class_id
                    if class_id == 0:
                        vehicle_count += 1
                    elif class_id == 2:
                        person_count += 1
            print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
                  f"Frame Number={frame_meta.frame_number},"
                  f"Vehicle Count={vehicle_count}, Person Count={person_count}")
            display_text = f"Person={person_count},Vehicle={vehicle_count}"
            display_meta = batch_meta.acquire_display_meta()
            text = osd.Text()
            text.display_text = display_text.encode('ascii')
            text.x_offset = 10
            text.y_offset = 12
            text.font.name = osd.FontFamily.Serif
            text.font.size = 12
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(text)
            frame_meta.append(display_meta)

def main(file_path):
    file_ext = os.path.splitext(file_path)[1]

    if file_ext in [".yaml", ".yml"]:
        pipeline = Pipeline(PIPELINE_NAME, file_path)
        output_tensor_meta = pipeline["infer"].get("output-tensor-meta")
        if output_tensor_meta:
            # disabling object meta from nvinfer plugin for using customized tensor converter
            pipeline["infer"].set({"filter-out-class-ids": "0;1;2;3"})
        pipeline.attach("infer", Probe("counter", ObjectCounterMarker(output_tensor_meta))).start().wait()
    else:
        (Pipeline(PIPELINE_NAME).add("filesrc", "src", {"location": file_path}).add("h264parse", "parser").add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": STREAM_WIDTH, "height": STREAM_HEIGHT})
            .add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH})
            .add("nvosdbin", "osd").add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
            .link("src", "parser", "decoder").link(("decoder", "mux"), ("", "sink_%u")).link("mux","infer", "osd", "sink")
            .attach("infer", Probe("counter", ObjectCounterMarker(output_tensor_meta=False)))
            .attach("infer", "measure_fps_probe", "my probe")
            .start().wait())

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> OR <YAML config file>\n" % sys.argv[0])
        sys.exit(1)
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()