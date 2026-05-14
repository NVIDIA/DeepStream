pip3 install -e .
cd checkpoints; bash download_ckpts.sh; cd ..

# --model can be chosen from tiny, small, base_plus, large
MODEL_TYPE="large"
mkdir -p checkpoints/${MODEL_TYPE}
python3 export_sam2_onnx.py --model ${MODEL_TYPE}
echo "SAM2 ONNX models are exported to checkpoints/${MODEL_TYPE}"

# Copy models to DeepStream tracker path
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
cp checkpoints/${MODEL_TYPE}/*.onnx /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
echo "SAM2 ONNX models are copied to /opt/nvidia/deepstream/deepstream/samples/models/Tracker/"
