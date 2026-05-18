Implement a Python application that uses a multi-modal VLM to summarize video frames and sends summaries to a remote server via Kafka.

### Architecture

1. **DeepStream Pipeline**: Use DeepStream pyservicemaker APIs to receive N RTSP streams, decode video, and convert frames to RGB format. Process each stream independently — do not mux streams together.

2. **Frame Sampling & Batching**: Use MediaExtractor to sample frames at a configurable interval (e.g. 1 frame every 10 seconds). When the VLM supports multi-frame input, batch sampled frames over a configurable duration (e.g. 1 minute) before sending to the model. Each batch must contain frames from a single stream only.

3. **vLLM Backend**: Implement a module that receives a batch of decoded video frames and returns a text summary from the multi-modal VLM.

4. **Kafka Output**: Send each text summary to a remote server using Kafka.

### Constraints
- Scalable to hundreds of RTSP streams across multiple GPUs on a single node. Distribute processing load across all available GPUs.
- Never mix frames from different RTSP streams in a single batch.

Store output in the `rtvi_app` directory.
Also generate a README.md with instructions to setup kafka server, vLLM, and how to run the application.
