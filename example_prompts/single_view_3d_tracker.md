Use DeepStream SDK pyservicemaker APIs to develop the python application that can do the following.

- Stream from a single video source (file or RTSP), decode the video.
- Implement Single-View 3D Tracking using nvtracker. configured with a camera projection matrix from a camera info YAML file, to estimate and track object states in the 3D physical world.
- Display the reconstructed 3D bounding boxes overlaid on the video using OSD and render output with EGL sink.
- Save the output video as out.mp4.

**Important**
Save the generated code in single_view_3d_tracker_app directory.
Also generate a README.md with setup instructions, how to download required models and camera calibration requirements, and how to run the application.
In tracker configuraton file, set minPoseConfidence to 0.925 and outputConvexHull to 0.