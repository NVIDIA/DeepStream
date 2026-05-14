#! /bin/bash
wget https://nvidia.box.com/shared/static/ownxazhmtpnlo3jvbkx4r62ffccm8hu5 -O yolov8s_DAT_640_noqdq.onnx && \
wget https://nvidia.box.com/shared/static/6bua0bo57cb6s44048os9qq9i1xjw5u1 -O yolov8s_DAT_precision_config_calib.cache && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s_DAT_640_noqdq.onnx --fp16 --int8 --verbose --calib=yolov8s_DAT_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions=Split_36:fp16,Reshape_37:fp16,Transpose_38:fp16,Softmax_39:fp16,Conv_41:fp16,Sub_64:fp16,Concat_65:fp16,Mul_67:fp16,Sigmoid_68:fp16,Concat_69:fp16 --saveEngine=yolov8s_DAT_640_noqdq_DLA.engine --useDLACore=0 --allowGPUFallback
