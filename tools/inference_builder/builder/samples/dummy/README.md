This sample demonstrates how to build a inference pipeline with a dummy backend. The experiment is mainly for testing the basic flow.

The sample configuration supports multiple input types including image data, image assets and video assets in one endpoint

**Note:** Make sure you are in the root directory (`path/to/inference_builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference_builder root directory. Also ensure that your virtual environment is activated before running any commands.

Build the dummy inference flow:

```bash
python builder/main.py builder/samples/dummy/dummy.yaml -a builder/samples/dummy/openapi.yaml -c builder/samples/dummy/processors.py -o builder/samples/dummy  --server-type fastapi -t
```
Run the dummy inference flow:

```bash
cd builder/samples
docker compose up --build dry-run
```
Test the dummy inference flow:

Go to http://localhost:8800/docs to see the swagger ui.





