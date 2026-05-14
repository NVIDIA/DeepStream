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