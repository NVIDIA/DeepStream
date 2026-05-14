# Dockerfile for ds-cradio: PeopleNet Transformer + C-RADIOv3-H pipeline
# Uses DeepStream/nvinfer for detection and polygraphy (TensorRT) for embeddings

FROM "nvcr.io/nvidia/deepstream:9.0-triton-multiarch" AS inference_base

# Install Python dependencies required by this pipeline.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.7.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    transformers==4.54.1 \
    opencv-python

# Build TensorRT OSS to provide customized plugins (e.g. for PeopleNet
# Transformer ONNX model) and the trtexec tool.
ARG TRT_VERSION_MAJOR=10
ARG TRT_VERSION_MINOR=14
ARG TRT_VERSION_PATCH=1
ARG TRT_VERSION_BUILD=48

ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=13
ARG CUDA_VERSION_MINOR=1
ARG CUDA_VERSION_PATCH=80
ARG CUDA_VERSION_BUILD=36836380_0
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=9.17.0.29

ENV TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL

WORKDIR /tmp
ENV TRT_TAG="release/$TRT_VERSION_MAJOR_MINOR"
RUN mkdir trt_oss_src && \
    cd trt_oss_src && \
    echo "$PWD Building TRT OSS..." && \
    git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
    cd TensorRT && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    cmake .. \
    -DGPU_ARCHS="80;86;90;100;120" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_OUT_DIR=/tmp/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION \
    && for i in {1..50}; do make -j 12 && break; done
RUN cp /tmp/out/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
    cp /tmp/out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
    cp /tmp/out/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
    cp /tmp/out/trtexec /usr/local/bin/ && \
    cd ../../../ && \
    rm -rf trt_oss_src

# Build DeepStream TAO app post-processors (provides NvDsInferParseCustomDDETRTAO
# for PeopleNet Transformer).
ENV DS_TAO_APPS_REPO=https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
ENV DS_TAO_APPS_TAG="master"
ARG CACHE_BUSTER=unique_value_to_bypass_docker_cache
RUN mkdir ds_tao_apps_src && \
   cd ds_tao_apps_src && \
   echo "$PWD Pulling DS TAO apps repo..." && \
   git clone -b $DS_TAO_APPS_TAG $DS_TAO_APPS_REPO ds_tao_apps && \
   cd ds_tao_apps/post_processor && \
   echo "$PWD Building DS TAO apps post processing..." && \
   CUDA_MODULE_LOADING=LAZY CUDA_VER=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR make && \
   [ -f libnvds_infercustomparser_tao.so ] && echo "$PWD post processing build successful" || echo "$PWD post processing build failed" && \
   cp libnvds_infercustomparser_tao.so /opt/nvidia/deepstream/deepstream/lib && \
   cd ../../../ && \
   rm -rf ds_tao_apps_src

# Install Python dependencies for the polygraphy backend
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://pypi.nvidia.com \
    Pillow==11.1.0 \
    timm \
    torchvision \
    einops \
    onnx \
    onnxscript \
    tensorrt==10.14.1.48.post1 \
    polygraphy==0.49.24

WORKDIR /workspace

# Copy the generated inference pipeline
ADD ds-cradio.tgz /workspace

# Copy the ONNX export script (for model preparation inside container)
COPY export_onnx.py /workspace/export_onnx.py

# Set environment variables required for DeepStream runtime
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes

ENTRYPOINT ["python3", "/workspace/__main__.py"]
CMD []
