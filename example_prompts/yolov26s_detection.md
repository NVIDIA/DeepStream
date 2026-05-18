Download the YOLO26s detection model using the ultralytics library, then convert the model to ONNX model that supports dynamic batch,  in a Python virtual environment.
Write a DeepStream custom parsing library for the model.
Use DeepStream SDK pyservicemaker APIs to develop the python application that can do the following.
- Read from file, decode the video and infer using the model.
- The custom parsing library is used in nvinfer's configuration file.
- Display the bounding box around detected objects using OSD.


**Important**
Use nvurisrcbin as source to automatically handle various types of video files.

Save the generated code in yolo_detection directory.
Also generate a README.md with setup instructions and how to run the application.
