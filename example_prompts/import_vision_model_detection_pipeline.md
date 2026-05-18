Use deepstream-import-vision-model to onboard an object detection model end-to-end: from acquisition and engine build through performance benchmarking and report generation.

Before starting, prompt me for the following inputs. For each prompt, I will either provide a value or reply `y` (or `Y`) to accept the default shown in brackets:

- **Model URL or local path** [default: `https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/rtdetr_2d_warehouse`]
- **Sample video for validation and benchmarking** [default: `/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4`]

If I reply `y` (or leave the input blank), proceed with the default and clearly note in the output which defaults were used.

Once inputs are collected, run all steps unattended:

- Acquire the model (download ONNX or export SafeTensors → ONNX)
- Build a dynamic TensorRT engine
- Generate and compile a custom nvinfer bbox parser
- Run single-stream KITTI validation
- Run multi-stream benchmark sweep
- Generate benchmark charts and a PDF report

Save all artifacts under `models/<model_name>/`.

Also generate a `README.md` with setup instructions and how to run the application.
