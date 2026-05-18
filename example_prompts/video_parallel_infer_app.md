Use DeepStream SDK pyservicemaker Pipeline APIs to develop the python application that can do the following.
- Read 4 video streams from files, decode the videos.
- Two parallel inference branches:
  - Branch 1: Select streams 0, 1, 2, 3, run primary object detection using ResNet18 TrafficCamNet model, then track objects.
  - Branch 2: Select streams 1, 2, 3, run object detection using YOLO26s detection model, then track objects.
    - Download the YOLO26s detection model using the ultralytics library, then convert the model to ONNX model that supports dynamic batch,  in a Python virtual environment
    - Write a DeepStream custom parsing library for the model
- Merge metadata from streams 0, 1, and 2 of Branch 1 with metadata from streams 1, and 2 of Branch 2 to form the final result.
- Display the combined detection results with bounding boxes using OSD.

**Important**
- Use nvurisrcbin as source to automatically handle various types of video files.
- Multiple inference models process the same video stream parallelly and their metadata needs to be merged.

Save the generated code in video_parallel_infer_app directory.
Also generate a README.md with setup instructions and how to run the application.
