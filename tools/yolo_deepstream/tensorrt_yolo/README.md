# Yolo TensorRT cpp

## Description
This is a yolo TensorRT cpp app. Fisrt, using trtexec to convert onnx model to FP32 or FP16 TensorRT engine ,or INT8 TensorRT engine from the QAT model finetuned from [yolov7_qat](../yolov7_qat) and [yolov8](../README.md#onnx-model-list).
Then you can use the `detect/video_detect` app to detect a list of images(images number must smaller than the batchsize of the model)/video. or use `validate_coco` app to test mAP of the TensorRT engine.
## Prerequisites
#### Install opencv
- Note: There are OpenCV4 dependencies in this program. 
Follow README and documents of this repository https://github.com/opencv/opencv to install OpenCV.
And, if you want use detect_video app, please install opencv with `ffmpeg` enabled

#### Install jsoncpp libs
jsoncpp lib is used to write coco-dataset-validate-result to json file. 
```bash
$ sudo apt-get install libjsoncpp-dev
```
## Build and Run yolo-TensorRT-app
### Build
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j4
```

### Prepare TensorRT engines
model can be found in [model-list](../README.md#model-list)
convert onnx model to tensorrt-engine
```bash
# fp32 model
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7fp32.engine
# fp16 model
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7fp16.engine --fp16
# int8 QAT model, the onnx model with Q&DQ nodes
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7qat.onnx --saveEngine=yolov7QAT.engine --fp16 --int8
# int8 implicit quant yolov8s model running on gpu
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov8s_640_dynamic.onnx --fp16 --int8 --verbose --calib=yolov8s_ptq_precision_config_calib.cache --saveEngine=yolov8s_ptq_640_gpu_b1.engine --minShapes=x.1:1x3x640x640 --optShapes=x.1:1x3x640x640 --maxShapes=x.1:1x3x640x640 --precisionConstraints=obey --layerPrecisions=Split_36:fp16,Reshape_37:fp16,Transpose_38:fp16,Softmax_39:fp16,Conv_41:fp16,Sub_64:fp16,Concat_65:fp16,Mul_67:fp16,Sigmoid_68:fp16,Concat_69:fp16
# int8 implicit quant DLA-spec-finetuned yolov8s model running on DLA
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s_DAT_640_noqdq.onnx --fp16 --int8 --verbose --calib=yolov8s_DAT_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions=Split_36:fp16,Reshape_37:fp16,Transpose_38:fp16,Softmax_39:fp16,Conv_41:fp16,Sub_64:fp16,Concat_65:fp16,Mul_67:fp16,Sigmoid_68:fp16,Concat_69:fp16 --saveEngine=yolov8s_DAT_noqdq_640_DLA.engine --useDLACore=0 --allowGPUFallback
```
### Detection & Validate
- detect with image:
    - Run on yolov7
    ```bash
    $ ./build/detect --engine=yolov7db4fp32.engine --img=./imgs/horses.jpg,./imgs/zidane.jpg --version=v7
    ```
    - Run on yolov8 gpu
    ```bash
    $ ./build/detect --engine=yolov8s_ptq_640_gpu_b1.engine --img=./imgs/horses.jpg,./imgs/zidane.jpg --version=v8
    ```
    - Run on yolov8 dla
    ```bash
    $ ./build/detect --engine=yolov8s_DAT_noqdq_640_DLA.engine --img=./imgs/horses.jpg,./imgs/zidane.jpg --version=v8
    ```

- detect with video:
    
    note: only support batchsize = 1 now.
    - Run on yolov7
    ```bash
    $ ./build/video_detect --engine=./yolov7fp32.engine --video=YOUR_VIDEO_PATH.mp4 --version=v7
    ```
    - Run on yolov8 gpu
    ```bash
    $ ./build/video_detect --engine=yolov8s_ptq_640_gpu_b1.engine --video=YOUR_VIDEO_PATH.mp4 --version=v8
    ```
    - Run on yolov8 dla
    ```bash
    $ ./build/video_detect --engine=yolov8s_DAT_noqdq_640_DLA.engine --video=YOUR_VIDEO_PATH.mp4 --version=v8
    ```

- validate mAP on dataset
    - note: validate_coco only support model inputsize `[batchsize, 3, 672, 672]` or `[batchsize, 3, 640, 640]`

    - Run on yolov7
        ```bash
        $ ./build/validate_coco --engine=./yolov7fp32.engine --coco=/YOUR/COCO/DATA/PATH/ --version=v7
        ```
    - Run on yolov8 on gpu
        ```bash
        $ ./build/validate_coco --engine=./yolov8s_DAT_noqdq_672_gpu.engine --coco=/YOUR/COCO/DATA/PATH/ --version=v8
        ```
    
    - Run on yolov8 on dla
        ```bash
        $ ./build/validate_coco --engine=./yolov8s_DAT_noqdq_672_DLA.engine --coco=/YOUR/COCO/DATA/PATH/ --version=v8
        ```

    output:
    ```
    --------------------------------------------------------
    Yolov7 initialized from: yolov7672.engine
    input : images , shape : [ 1,3,672,672,]
    output : output , shape : [ 1,27783,85,]
    --------------------------------------------------------
    5000 / 5000
    predict result has been written to ./predict.json
    ```
    validate output with `test_coco_map.py`:
    ```
    $ python test_coco_map.py --predict ./predict.json --coco /YOUR/COCO/DATA/PATH/
    ...
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51005
    ...
    ```
