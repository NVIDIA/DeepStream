Use DeepStream SDK pyservicemaker APIs to develop the python application that can do the following.

- Stream from 4 RTSP cameras simultaneously, decode the videos, batch frames together and infer using ResNet18 TrafficCamNet model.
- Use tracker after infer to track the detected objects.
- display the bounding box around detected objects using OSD.
- Render all four video in 2x2 tiled window.
 
Save the generated code in multi_stream_tracker_app directory.
Also generate a README.md with setup instructions and how to run the application.
