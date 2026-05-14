## Introduction

This document describes the sample deepstream_nvdsanalytics_test application.

This sample builds on top of the service-maker deepstream_test1 sample at at `/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/pipeline_api/deepstream_test1_app` to demonstrate how to:

* Use multiple sources in the pipeline.
* Use a uridecodebin so that any type of input (e.g. RTSP/File), any GStreamer
  supported container format, and any codec can be used as input.
* Configure the stream-muxer to generate a batch of frames and infer on the
  batch for better resource utilization.
* Perform analytics on metadata attached by nvinfer and nvtracker using nvdsanalytics plugin
* Extract the stream metadata, which contains useful information about the
  frames in the batched buffer. Extract the analytics object level and frame level metadata,
  which contains information about region of interest filtering, overcrowding detection, direction
  detection, and line crossing.

Refer to the service-maker deepstream_test1 sample documentation for an example of a
single-stream inference, bounding-box overlay, and rendering.

## Usage

Run with the uri(s).

```
$ python3 deepstream_nvdsanalytics.py <uri1> [uri2] ... [uriN]
e.g.
$ python3 deepstream_nvdsanalytics.py file:///home/ubuntu/video1.mp4 file:///home/ubuntu/video2.mp4
For URI(s) with special characters like @,& etc, you need to pass the uri within quotes
$ python3 deepstream_nvdsanalytics.py 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0' 'rtsp://user@ip/cam/realmonitor?channel=1&subtype=0'
```
