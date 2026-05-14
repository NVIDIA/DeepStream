# Using Inference Builder with AI Agents — Examples and Guidelines

## General Guidelines

1. **Start simple, then refine.** Begin with a short, clear prompt that states the core requirements. Use follow-up messages to add details or adjust behavior.

2. **Ask for self-check and verification.** Tell the agent to validate its work (e.g., run a smoke test or sanity check) so the result is correct before you rely on it.

3. **Request documentation and diagrams.** Ask the agent to produce a README and architecture diagrams. These make the solution easier to understand and easier to extend later.

4. **Use helper scripts for operations.** Request small scripts for common tasks (e.g., “create a script to start, stop, rebuild, get status, and fetch logs for the microservice”). These simplify testing and debugging.

5. **Correct course early.** If the agent goes off track, give a direct hint right away. For example:
   - If no free GPU is available, tell it to stop and retry later.
   - If it picks the wrong model, ask it to double-check the choice.

## Sample Prompts

### 1. Create a very basic Deepstream Application

***Use deepstream inference builder tool to create an object detection pipeline with PeopleNet Transformer model from NGC and verify it with a smoke test.***

The Agent is expected to draft a plan based on the prompt and generates required files accordingly (file names may differ in each run unless explicitly specified):

- peoplenet_transformer_pipeline.yaml (pipeline configuration yaml)
- Dockerfile
- nvdsinfer_config.yaml (nvinfer configuration yaml)
- peoplenet.tgz (python code for inference)
- models/peoplenet (folder for required model files)

The Agent is expected to fix all the build and runtime errors while doing the smoke test.

### 2. Create an advanced Deepstream Application  Supporting Tracker and Multi-stream Inputs

***Use deepstream inference builder tool to create an object detection pipeline with the PeopleNet transformer model from NGC; the pipeline should support 4 video inputs in parallel and track all the objects in the inputs using NVDCF tracker. Do a smoke test once done.***

The Agent is expected to draft a plan based on the prompt and generates required files accordingly (file names may differ in each run unless explicitly specified):

- peoplenet_transformer_pipeline.yaml (pipeline configuration yaml)
- Dockerfile
- nvdsinfer_config.yaml (nvinfer configuration yaml)
- peoplenet.tgz (python code for inference)
- models/peoplenet (folder for required model files)

The Agent is expected to fix all the build and runtime errors while doing the smoke test.

###  3. Create a Deepstream Inference Application with Customized Logic

***Use deepstream inferenc builder tool to create an object detection pipeline with PeopleNet transformer model from NGC; the pipeline accepts video url as input and output detected bounding boxes. Count the number of people in each frame and raise an alarm whenever the number exceeds 10 by writing it down to a file. Do a smoke test once done.***

The Agent is expected to draft a plan based on the prompt and generates required files accordingly (file names may differ in each run unless explicitly specified):

- peoplenet_transformer_pipeline.yaml (pipeline configuration yaml)
- Dockerfile
- nvdsinfer_config.yaml (nvinfer configuration yaml)
- processor.py (Customized processor for counting and raising alarm)
- peoplenet.tgz (python code for inference)
- models/peoplenet (folder for required model files)

The Agent is expected to fix all the build and runtime errors while doing the smoke test.


### 4. Create a Realtime Video Intelligence Microservice (rtvi) with vllm Backend and Deepstream Accelerated Decoder

#### 4.1. Start with the Master Prompt as below

***Locate the MCP tool deepstream-inference-builder and complete the task: Start a new project under name rtvi_vlm and create a video summarization/caption microservice there using Qwen3-VL-2B-Instruct model driven by vllm backend and Deepstream accelerated decoder. The microservice supports adding/deleting RTSP streams as assets and generating video caption for added streams. The given RTSP stream will be chunked to a batch of frames with a given interval for summarization. I also want to build a container image for the microservice to run on B200. Do a smoke test on the image once done.***

The Agent is expected to understand the prompt and creates a folder named “rtvi_vlm” and generates all the files there:
- config.yaml (pipeline configuration yaml)
- Dockerfile
- openapi.yaml (server specification)
- processors.py (customized code)
- rtvi_vlm.tgz (backbone python code)
- model-repo (folder for required model files)

The Agent is responsible for the smoke testing and troubleshooting of any issues encountered with the generated container image.

#### 4.2. Add Helper Scripts for Manual Testing with Follow-up Prompts

***Generate a service management script to start, restart, stop, rebuild the service and check the service status, also fetch the service logs. Generate a client script to add/delete streams and trigger a summarization request on streams.***

With the generated server and client script, we can ask Cursor to intensively test stream summarization, and during this process Cursor will fix the generated processor code based on the container log.
We'll need to go through this a few times because Cursor has to figure out the best model setup by running it.

You can ask Cursor to use your live stream: `rtsp://<server address>:<port>/<stream-name>`.

#### 4.3. New feature request

New features can always be added through follow-up prompts:

***Now let's add a new feature to the project. After the model generates summarizations, let's postprocess it for each chunk and send them to a Kafka server. Create an environment variable to toggle the feature.***

#### 4.4. Document the project for future development

***Write a detailed documentation for the project to ensure seamless and consistent future work.***

Subsequently, the project remains open, allowing for continued adjustment, addition, or removal of features using the same methods.

#### 4.5. YOLO26s Detection Deepstream Application

***Download the YOLO26s detection model using the ultralytics library, then convert the model to ONNX model that supports dynamic batch, in Python virtual environment. Write a DeepStream custom parser for the model. Use deepstream inference builder tool to create an object detection pipeline with the model and custom parser. Do a smoke test once done.***

The Agent is expected to draft a plan based on the prompt and generates required files accordingly (file names may differ in each run unless explicitly specified):

- yolo26s_pipeline.yaml (pipeline configuration yaml)
- Dockerfile
- nvdsinfer_config.yaml (nvinfer configuration yaml)
- yolo26s-detection.tgz (python code for inference)
- models/yolo26s (folder for required model files)
- custom_parser (DeepStream custom parser for the model)

The Agent is expected to fix all the build and runtime errors while doing the smoke test.
