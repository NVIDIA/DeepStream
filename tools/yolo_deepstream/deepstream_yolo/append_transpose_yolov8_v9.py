# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("yolov9-t-converted.onnx"))
# graph = gs.import_onnx(onnx.load("yolov8-s.onnx"))
ori_output = graph.outputs[0]
trans_out  = gs.Variable(name="trans_out", dtype=np.float32, shape=(-1, 8400, 84))
trans_node = gs.Node(op="Transpose",name="transpose_output_node", attrs={"perm":np.array([0,2,1])}, inputs=[ori_output], outputs=[trans_out])
graph.nodes.append(trans_node)
graph.outputs = [trans_out]
graph.cleanup(remove_unused_graph_inputs=True).toposort()
model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(model, "yolov9-t-converted-trans-dynamic_batch_640.onnx")