Use DeepStream SDK pyservicemaker APIs to develop the python application that can do the following.

- Read from file, decode the video and infer using ResNet18 TrafficCamNet model. 
- Convert the detected object bounding boxes into JSON format every 30 frames and send this information directly to the Kafka server.

**Important**
Use nvurisrcbin as source to automatically handle various types of video files.

Save the generated code in msgconv_kafka directory.
Also generate a README.md with setup instructions and how to run the application.