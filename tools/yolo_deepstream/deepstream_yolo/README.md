# Deploy YOLO Models With DeepStream #

**This sample shows how to integrate YOLO models with customized output layer parsing for detected objects with DeepStreamSDK.**

## 1. Sample contents: ##
- `deepstream_app_config_yolo.txt`: DeepStream reference app configuration file for using YOLO models as the primary detector.
- `config_infer_primary_yoloV4.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV4 detector model.
- `config_infer_primary_yoloV7.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV7 detector model.
- `config_infer_primary_yoloV8.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV8 detector model.
- `config_infer_primary_yoloV9.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV9 detector model.
- `config_infer_primary_yoloV11.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV11 detector model.
- `config_infer_primary_yoloV11_obb.txt`: Configuration file for the GStreamer nvinfer plugin for the YOLOv11 OBB (Oriented Bounding Box) detector model.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`: Output layer parsing function for detected objects for the Yolo models.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu`: Output layer parsing function for detected objects for the Yolo models by CUDA.

## 2. Download and Build ##

```sh
  $ cd ~/
  $ git clone https://github.com/NVIDIA/DeepStream.git
  $ cd ~/DeepStream/tools/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo
  $ make
  $ cd ..
```
NOTE: To improve performance on specific GPUs, please add "-gencode=arch=compute_xx,code=sm_xx" in Makefile. Computing capability can be found in this link https://developer.nvidia.com/zh-cn/cuda-gpus#compute.

## 3. Prepare model and Run with deepstream-app ##

### TIPS: DeepStream 7.1+ is required ###

#### Yolov4 

- Go to this pytorch repository <https://github.com/Tianxiaomo/pytorch-YOLOv4> where you can convert YOLOv4 Pytorch model into **ONNX**
- Other famous YOLOv4 pytorch repositories as references:
  - <https://github.com/WongKinYiu/PyTorch_YOLOv4>
  - <https://github.com/bubbliiiing/yolov4-pytorch>
  - <https://github.com/maudzung/Complex-YOLOv4-Pytorch>
  - <https://github.com/AllanYiin/YoloV4>
- Or you can download reference ONNX model directly from here ([link](https://nvidia.box.com/s/achcifjwl1ac99tdvfwtfmxgec5d0pro)).  

- Run
  ```bash
  $ deepstream-app -c deepstream_app_config_yolo.txt
  ```
  The output result will output to `yolo.mp4`

#### YOLOv7
- Follow the guide https://github.com/WongKinYiu/yolov7#export, export a dynamic-batch-1-output onnx-model

```bash
$ python export.py --weights ./yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --dynamic-batch
```

- Generate the TensorRT engine with the following command:

```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7_dy.onnx --fp16 --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolov7_dy_gpu_b16.engine
```

- Run
```bash
$ deepstream-app -c deepstream_app_config_yolo.txt -t
```
The output result will output to `yolo.mp4`

#### YOLOv8
##### 1. Run on GPU with int8 precision with calibrated model(GPU Only)
- Convert model with trtexec

  The model can be found in [model-list](../README.md#model-list)

  ```bash
  # batchsize = 16
  $ /usr/src/tensorrt/bin/trtexec --onnx=yolov8s_640_dynamic.onnx --fp16 --verbose --saveEngine=yolov8s_640_dynamic.onnx_b16.engine --minShapes=x.1:16x3x640x640 --optShapes=x.1:16x3x640x640 --maxShapes=x.1:16x3x640x640
  ```

- Enable the configure on config files, edit [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt)
  ```ini
  # batch-size = 16
  config-file=config_infer_primary_yoloV8.txt
  ```

##### 2. Run
  ```bash
  $ deepstream-app -c deepstream_app_config_yolo.txt
  ```
  The output result will output to `yolo.mp4`

#### YOLOv9
- Get ONNX model by one of the following methods
  * Follow the guide https://github.com/WongKinYiu/yolov9/issues/2#issuecomment-1960519506, export a dynamic-batch-1-output onnx-model
    ```bash
    $ python export.py --weights yolov9-s-converted.pt --dynamic --include onnx
    ```
  * Download the ONNX model directly from [yolov9-s onnx model](https://nvidia.box.com/s/dzch7bx0xlap4hoc5nk9huy72w33wbc9)
- Put the ONNX model file under yolo_deepstream/deepstream_yolo directory
- Modify the [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt) file
  ```ini
  config-file=config_infer_primary_yoloV9.txt
  ```
- Run
  ```bash
  deepstream-app -c deepstream_app_config_yolo.txt -t
  ```
  The output result will output to `yolo.mp4`

#### YOLOv8/YOLOv9 TIPS

 - How to export yolov8 models with official repo
 Follow the guide https://github.com/ultralytics/ultralytics/tree/v8.2.103, export a dynamic-batch-1-output onnx-model

    ```python
    from ultralytics import YOLO
    # Load a model
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
    path = model.export(format="onnx")  # export the model to ONNX format
    ```

 - For yolov8 and yolov9 models user must add a transpose node at the end of the network through [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/release/9.2/tools/onnx-graphsurgeon) :
    ```bash
    $ python append_transpose_yolov8_v9.py
    ```

#### YOLOv11s

- Use the following command to download the yolov11s onnx-model

```bash
wget https://nvidia.box.com/shared/static/diao78cnzdhakz5fbx4ax5213z70ymi7 -O yolov11s_640_dynamic.onnx
```

- Convert model with trtexec
```bash
# batch size 16
/usr/src/tensorrt/bin/trtexec --minShapes=images:1x3x672x672 --optShapes=images:16x3x672x672 --maxShapes=images:16x3x672x672 --onnx=yolov11s_qat_int8_672_dynamic.onnx  --saveEngine=yolov11s_qat_int8_672_dynamic.onnx_b16_gpu0_int8.engine --int8 --fp16 --dumpLayerInfo --profilingVerbosity=detailed --exportLayerInfo=yolov11s_qat_int8_672_dynamic_layer.json --exportProfile=yolov11s_qat_int8_672_dynamic_profile.json
```

- Modify the [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt) file
```ini
config-file=config_infer_primary_yoloV11.txt
```

- Run
```bash
deepstream-app -c deepstream_app_config_yolo.txt -t
```
The output result will output to `yolo.mp4`

#### YOLOv11 OBB (Oriented Bounding Box)

- Export the YOLOv11 OBB model to ONNX (e.g. with Ultralytics). Requires DeepStream SDK for YOLOv11 OBB custom parser.

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n-obb.pt')
model.export(format='onnx', dynamic=False, simplify=True)
"
```

- Put `yolo11n-obb.onnx` and `labels_obb.txt` under this directory. nvinfer will build the engine on first run. Update `custom-lib-path` in [config_infer_primary_yoloV11_obb.txt](./config_infer_primary_yoloV11_obb.txt) if your DeepStream custom parser is installed elsewhere.

- Modify the [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt) file
  * Set `uri` under `[source0]` to your own aerial video (e.g. DOTA-style) to test with this model.
  * Under `[primary-gie]`, set `config-file` and `labelfile-path`.

    ```ini
    # [source0]
    uri=file:///path/to/your/aerial_video.mp4


    # [primary-gie]
    config-file=config_infer_primary_yoloV11_obb.txt
    labelfile-path=labels_obb.txt
    ```

- Run
```bash
deepstream-app -c deepstream_app_config_yolo.txt -t
```
The output result will output to `yolo.mp4`

## 4. CUDA Post Processing

this sample provide two ways of yolov7/yolov8/yolov9/yolov11 post-processing(decode yolo result, not include NMS), CPU version and GPU version
- CPU implement can be found in: [nvdsparsebbox_Yolo.cpp](nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp)
- CUDA implement can be found in: [nvdsparsebbox_Yolo_cuda.cu](nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu)

Default will use CUDA-post processing. To enable CPU post-processing:
in [config_infer_primary_yoloV7.txt](config_infer_primary_yoloV7.txt), [config_infer_primary_yoloV8.txt](config_infer_primary_yoloV8.txt), [config_infer_primary_yoloV9.txt](config_infer_primary_yoloV9.txt) or [config_infer_primary_yoloV11.txt](config_infer_primary_yoloV11.txt)

- `parse-bbox-func-name=NvDsInferParseCustomYoloV7_cuda` -> `parse-bbox-func-name=NvDsInferParseCustomYoloV7`
- `disable-output-host-copy=1` -> `disable-output-host-copy=0`

The performance of the CPU-post-processing and CUDA-post-processing result can be found in [Performance](../#Performance)
