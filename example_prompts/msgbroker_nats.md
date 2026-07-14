Build a DeepStream pyservicemaker application that publishes object detection results to NATS.

- Implement a NATS protocol adapter library:
    - Provide a configuration option to enable JetStream for persistent streams.
    - Authentication support (including NKeys, user credentials, token-based auth, etc.).
    - TLS encryption.

- Read from file, decode the video and infer using ResNet18 TrafficCamNet model.

Save the generated code in `msgbroker_nats_app` directory.
Also generate a `README.md` with setup instructions and how to run the application. The `README.md` must additionally describe how to set up a local test environment for verification, including:

- How to launch a local NATS server (e.g. via Docker or the `nats-server` binary) with JetStream enabled.
- How to run a subscriber client that subscribes to the subject published by the application and prints the received messages, so the user can confirm detection metadata is flowing end-to-end.
