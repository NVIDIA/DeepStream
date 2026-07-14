Use DeepStream SDK pyservicemaker APIs to develop an efficient python application that can do the following.
- Read from 16 RTSP H264 1080p streams, decode the videos, batch frames together and infer using ResNet18 TrafficCamNet model.
- Target 30 FPS per stream.

**Important**
Use nvurisrcbin as source to automatically handle the RTSP URIs.
Profile the pipeline and report the bottleneck (decode / compute / memory-bandwidth), the max streams this GPU can sustain at 30 FPS, and which hardware upgrade would help.

Save the generated code in efficient_pipeline_app directory.
Also generate a README.md with setup instructions and how to run the application.
