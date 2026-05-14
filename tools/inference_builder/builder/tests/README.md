# Docker Container Test Suite

This directory contains a comprehensive test suite for testing Docker container builds with different arguments and configurations.

## Overview

The test suite allows you to:
- Test Docker builds with different TensorRT, CUDA, and cuDNN versions
- Run containers with various command line arguments
- Test different environment configurations
- Validate volume mounts and data access
- Generate detailed test reports
- **Capture and save container logs to files for debugging**
- **Automatically detect ERROR logs and fail tests when errors are found**

## Files

- `test_docker_builds.py` - Main test script
- `test_configs.json` - Test configuration file
- `run_tests.sh` - Shell script wrapper for easy usage
- `setup_rtsp_server.sh` - Script to setup RTSP server for stream testing
- `Dockerfile` - Dockerfile to test
- `README.md` - This documentation
- `logs/` - Directory containing container logs (created automatically)
- `frame_sampling/` - Frame sampling test application directory
- `output_types/` - Output types compatibility test directory
- `concurrency/` - Concurrency test application directory

## Quick Start

### Run Tests

#### Standard Test Suite
```bash
chmod +x run_tests.sh
./run_tests.sh standard
```

#### Custom Configuration
```bash
./run_tests.sh custom -c my_config.json --log-dir custom_logs
```

#### Run Specific Test Cases
```bash
# Run only tests matching a specific name (partial match)
./run_tests.sh standard -t "frame_sampling"

# Run all tests including disabled ones
./run_tests.sh standard -t "*"
```

#### GPU Device Selection
```bash
# Use specific GPU device (device 0)
./run_tests.sh standard --gpus "device=0"

# Use multiple GPU devices
./run_tests.sh standard --gpus "device=0,1"

# Use all available GPUs (default)
./run_tests.sh standard --gpus "all"
```

## Test Configuration Structure

Each test configuration in `test_configs.json` has the following structure:

```json
{
  "name": "Test Name",
  "description": "Test description",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",

    "CACHE_BUSTER": "unique_value"
  },
  "test_config": {
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "/tmp/test_data": "/workspace/data"
    },
    "cmd": [
      "--input", "/workspace/data/video.mp4",
      "--output", "/workspace/data/output.mp4",
      "--batch-size", "4"
    ]
  }
}
```

### Build Arguments

#### Required/Common Arguments
- `TEST_APP_NAME`: Name of the test application (e.g., frame_sampling, output_types, concurrency)
- `CACHE_BUSTER`: Unique value to bypass Docker cache

#### Optional Arguments
- `SERVER_TYPE`: Server type for the application (e.g., fastapi, serverless). Default: serverless
- `OPENAPI_SPEC`: Path to OpenAPI specification file (for API-based tests)
- `TRT_VERSION_*`: TensorRT version components
- `CUDA_VERSION_*`: CUDA version components
- `CUDNN_VERSION`: cuDNN version
- `DS_TAO_APPS_TAG`: DS TAO apps git tag

#### Advanced Path Configuration
These arguments allow custom paths for code generation (useful for complex test setups):
- `APP_YAML_PATH`: Custom path to app.yaml (default: `{TEST_APP_NAME}/app.yaml`)
- `OUTPUT_DIR`: Custom output directory (default: `{TEST_APP_NAME}`)
- `PROCESSORS_PATH`: Custom processors.py path (default: auto-detected in output directory)

### Test Configuration

- `default_enable`: Whether the test is enabled by default (default: true). When false, only code generation runs unless `--test-case` is specified
- `timeout`: Timeout in seconds for the container test (default: 10 seconds)
- `env`: Environment variables to set in the container
- `volumes`: Volume mounts (host_path: container_path)
- `cmd`: Command line arguments to pass to the application
- `error_detection`: Configuration for detecting errors in container logs
  - `enabled`: Whether to check for error patterns (default: true)
  - `patterns`: List of error patterns to detect (optional, uses defaults if not specified)
- `prerequisite_script`: Script to run before launching the docker container (optional)
  - Can be any shell command or script path
  - Useful for setting up test environments (e.g., starting RTSP servers)
  - Script logs are saved to `logs/prerequisite_{test_id}.log`
  - If script fails, the test is marked as failed
  - Automatic cleanup is performed after the test completes

## Example Test Configurations

### 1. Basic Test
```json
{
  "name": "Basic Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "default_test"
  },
  "test_config": {
    "timeout": 10,
    "cmd": ["--help"]
  }
}
```

### 2. Video Files Test
```json
{
  "name": "Video Files Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "files_test"
  },
  "test_config": {
    "default_enable": true,
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "frame_sampling/models": "/workspace/models"
    },
    "cmd": [
      "--video-files", "34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10&chunks=10"
    ],
    "error_detection": {
      "enabled": false
    }
  }
}
```

### 3. High Performance Mode
```json
{
  "name": "High Performance Mode",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "high_perf_test"
  },
  "test_config": {
    "timeout": 60,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes",
      "NVDS_ENABLE_LATENCY_MEASUREMENT": "1"
    },
    "volumes": {
      "/tmp/test_data": "/workspace/data"
    },
    "cmd": [
      "--input", "/workspace/data/sample_video.mp4",
      "--output", "/workspace/data/output.mp4",
      "--batch-size", "4",
      "--fps", "30",
      "--gpu-id", "0"
    ]
  }
}
```

### 4. RTSP Stream Test with Prerequisite Setup
```json
{
  "name": "RTSP Stream Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "rtsp_test"
  },
  "test_config": {
    "default_enable": false,
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "frame_sampling/models": "/workspace/models"
    },
    "cmd": [
      "--video-streams", "fcf490e3-a64b-4dc9-95db-dcdba90032b3?frames=10&interval=100000000&chunks=100"
    ],
    "error_detection": {
      "enabled": false
    },
    "prerequisite_script": "./setup_rtsp_server.sh frame.jpg -m test -f 30 --daemon"
  }
}
```

### 5. FastAPI Server Test
```json
{
  "name": "FastAPI Concurrency Test",
  "description": "Build concurrency app and run threaded client",
  "build_args": {
    "TEST_APP_NAME": "concurrency",
    "CACHE_BUSTER": "concurrency_fastapi",
    "SERVER_TYPE": "fastapi",
    "OPENAPI_SPEC": "builder/samples/dummy/openapi.yaml"
  },
  "test_config": {
    "default_enable": true,
    "timeout": 30,
    "volumes": {
      "concurrency/models": "/workspace/models"
    },
    "cmd": []
  }
}
```

## Selective Test Execution

### Using default_enable

Tests can be selectively enabled or disabled using the `default_enable` field in the test configuration:

- **`default_enable: true`** (default): Test runs normally (code generation + Docker build + test)
- **`default_enable: false`**: Only code generation runs, Docker build and test are skipped

This is useful for:
- Tests that require special setup or resources (e.g., RTSP servers)
- Long-running tests that should only run when explicitly requested
- Tests under development that aren't ready for regular CI runs

### Running Disabled Tests

You can run disabled tests using the `-t` or `--test-case` argument:

```bash
# Run a specific disabled test
./run_tests.sh standard -t "RTSP Stream Test"

# Run all tests including disabled ones
./run_tests.sh standard -t "*"
```

When `-t` or `--test-case` is specified, it forces the full flow (code generation + Docker build + test) even for tests with `default_enable: false`.

## GPU Device Selection

By default, all tests run with `--gpus all`, which makes all GPUs available to the Docker containers. You can override this to run tests on specific GPU devices:

```bash
# Run on GPU device 0 only
./run_tests.sh standard --gpus "device=0"

# Run on GPU devices 0 and 1
./run_tests.sh standard --gpus "device=0,1"

# Run with all GPUs (default)
./run_tests.sh standard --gpus "all"

# Python script directly
python3 test_docker_builds.py --config-file test_configs.json --gpus "device=0"
```

**Note**: The GPU device string is passed directly to Docker's `--gpus` flag. Refer to [Docker's GPU documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu) for more advanced options.

## Command Line Arguments

The test script supports various command line arguments:

### Python Script (`test_docker_builds.py`)
```bash
python3 test_docker_builds.py [OPTIONS]

Options:
  --dockerfile PATH     Path to Dockerfile (default: Dockerfile)
  --base-dir PATH       Base directory for Docker build context (default: .)
  --config-file PATH    JSON file with test configurations (REQUIRED)
  --output PATH         Output file for test report
  --log-dir DIR         Directory to save container logs (default: logs)
  --no-cleanup          Don't cleanup images after testing
  --gitlab-token TOKEN  GitLab token for authentication (can also use GITLAB_TOKEN env var)
  --test-case NAME      Run only test cases matching this name (partial match). Use '*' for all.
  --gpus DEVICES        GPU devices to use (default: 'all'). Examples: 'all', 'device=0', 'device=0,1'
```

### Shell Script (`run_tests.sh`)
```bash
./run_tests.sh [COMMAND] [OPTIONS]

Commands:
  standard           Run standard test suite with all configurations from test_configs.json
  custom             Run test with custom configuration file
  help               Show help message

Options:
  --no-cleanup           Don't cleanup Docker images after testing
  --output FILE          Save test report to specified file
  -c, --config FILE      Use custom configuration file (required for custom)
  --log-dir DIR          Directory to save container logs (default: logs)
  -t, --test-case NAME   Run only test cases matching this name (partial match). Use '*' for all.
  --gpus DEVICES         GPU devices to use (default: 'all'). Examples: 'all', 'device=0', 'device=0,1'
```

## Test Applications

The test suite includes several test applications:

### frame_sampling
Tests frame sampling functionality with video files and streams. Supports both file-based and RTSP stream inputs.

### output_types
Tests compatibility of different output types with FastAPI backend using a simplified dummy backend.

### concurrency
Tests concurrent request handling with FastAPI server using a threaded client.

## Test Reports

Test reports are generated in JSON format and include:

- Summary statistics (total tests, passed, failed, success rate)
- Detailed results for each test
- Build and test outputs
- Configuration used for each test

Example report structure:
```json
{
  "summary": {
    "total_tests": 2,
    "passed": 2,
    "failed": 0,
    "success_rate": "100.0%"
  },
  "results": [
    {
      "test_id": 1,
      "name": "Default Configuration",
      "status": "PASSED",
      "build_success": true,
      "test_success": true,
      "build_output": "...",
      "test_output": "..."
    }
  ]
}
```

## Advanced Usage

### Custom Test Configurations

Create your own test configuration file:

```json
[
  {
    "name": "My Custom Test",
    "description": "Testing with custom parameters",
    "build_args": {
      "TEST_APP_NAME": "frame_sampling",
      "TRT_VERSION_MAJOR": "10",
      "TRT_VERSION_MINOR": "8",
      "CACHE_BUSTER": "my_test"
    },
    "test_config": {
      "default_enable": true,
      "timeout": 45,
      "env": {
        "CUSTOM_VAR": "custom_value"
      },
      "volumes": {
        "/path/to/my/data": "/workspace/data"
      },
      "cmd": [
        "--my-custom-arg", "value",
        "--another-arg", "another_value"
      ]
    }
  }
]
```

### Running Specific Tests

Use the `-t` or `--test-case` argument to run specific tests by name (partial matching supported):

```bash
# Run tests matching "frame_sampling" (partial match)
./run_tests.sh standard -t "frame_sampling"

# Run tests matching "FastAPI"
./run_tests.sh standard -t "FastAPI"

# Run all tests including disabled ones
./run_tests.sh standard -t "*"
```

Alternatively, create a custom configuration file with a subset of tests:

```bash
# Run with custom configuration
./run_tests.sh custom -c my_custom_tests.json
```

### Continuous Integration

For CI/CD pipelines, you can run tests with specific configurations:

```bash
# Run tests and save report
./run_tests.sh standard --output test_report.json

# Run tests on specific GPU device
./run_tests.sh standard --gpus "device=0" --output test_report.json

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **Docker not running**: Ensure Docker daemon is running

2. **Permission issues**: Run with appropriate permissions or use `sudo`

3. **Build failures**: Check if GitLab token has proper access to repositories

4. **Test timeouts**: Increase timeout in the script if needed

### Debug Mode

Run tests with verbose output:

```bash
# Enable debug logging
export PYTHONPATH=.
python3 -u test_docker_builds.py --config-file test_configs.json --no-cleanup
```

### Cleanup

Clean up test images manually if needed:

```bash
# List test images
docker images | grep test-inference_builder

# Remove all test images
docker images | grep test-inference_builder | awk '{print $3}' | xargs docker rmi
```

## Security Notes

- **Never commit GitLab tokens to version control**
- Use environment variables or CI/CD secrets for token storage
- Rotate tokens regularly
- Use tokens with minimal required permissions

## Contributing

To add new test configurations:

1. Add your configuration to `test_configs.json`
2. Update this README if needed
3. Test your configuration locally
4. Submit a pull request

## License

This test suite is part of the inference_builder project and follows the same license terms.

## Log Dumping

The test suite automatically captures and saves container logs to files for debugging and analysis.

### Log File Structure

Each test run creates a log file with the following structure:
```
=== Test Configuration ===
Image: test-inference_builder-1-1234567890
Command: docker run --rm -e NVSTREAMMUX_ADAPTIVE_BATCHING=yes test-inference_builder-1-1234567890 --video-files 34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10&chunks=10
Return Code: 0
Timestamp: 2025-01-15 10:30:45

=== STDOUT ===
[Application output here]

=== STDERR ===
[Error messages here]

=== END LOG ===
```

### Log File Naming

Log files are named using the pattern: `test_{test_id}_{image_name}.log`
- `test_id`: Sequential test number
- `image_name`: Docker image name (sanitized for filesystem)

### Log Directory

- **Default**: `logs/` directory in the current working directory
- **Custom**: Specify with `--log-dir` option
- **Auto-creation**: Directory is created automatically if it doesn't exist

### Example Usage

```bash
# Use default log directory
./run_tests.sh standard

# Use custom log directory
./run_tests.sh standard --log-dir /tmp/test_logs

# Python script directly
python3 test_docker_builds.py --log-dir my_logs --config-file test_configs.json
```

### Log File Locations in Reports

The test report includes the path to each log file:

```json
{
  "results": [
    {
      "test_id": 1,
      "log_file": "logs/test_1_test-inference_builder-1-1234567890.log",
      "status": "PASSED"
    }
  ]
}
```

## Error Detection

The test script includes built-in error detection that can be configured per test:

### Default Error Patterns

When `error_detection.enabled` is set to `true` (the default), the following patterns are detected:

- `[ERROR]`, `[CRITICAL]`, `[FATAL]`
- Various error/exception keywords (case-insensitive)
- Stack traces and tracebacks

You can customize these patterns in your test configuration:

```json
{
  "test_config": {
    "error_detection": {
      "enabled": true,
      "patterns": [
        "[ERROR]",
        "[CRITICAL]",
        "CustomErrorPattern"
      ]
    }
  }
}
```

### Real-time Output Streaming

When `error_detection.enabled` is set to `false`, the container output is streamed to the host's stdout in real-time instead of being captured for analysis. This is useful for:

- **Interactive debugging**: See container output as it happens
- **Long-running processes**: Monitor progress without waiting for completion
- **Development workflows**: Get immediate feedback during testing

Example configuration for real-time output:
```json
{
  "name": "Real-time Output Test",
  "description": "Test with error detection disabled for real-time output streaming",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "realtime_test"
  },
  "test_config": {
    "default_enable": true,
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "frame_sampling/models": "/workspace/models"
    },
    "cmd": [
      "--video-files", "34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10&chunks=10"
    ],
    "error_detection": {
      "enabled": false
    }
  }
}
```

**Behavior differences:**
- **Error detection enabled**: Output is captured and analyzed for error patterns. Test fails if errors are detected.
- **Error detection disabled**: Output is streamed to host stdout in real-time. Test only fails if container exits with non-zero return code.

**Note**: When error detection is disabled, logs are still saved to files for later review, but the output is also displayed in real-time on the host console.

## Volume Mounting

### Pre-build Commands

The test script automatically executes a pre-build command before each Docker build. This command is hardcoded and uses the `TEST_APP_NAME` from the build arguments to determine the sample folder:

**Hardcoded Command:**
```bash
python builder/main.py builder/tests/{TEST_APP_NAME}/app.yaml -o builder/tests/{TEST_APP_NAME} -c builder/tests/{TEST_APP_NAME}/processors.py --server-type serverless -t
```

**Example:**
- If `TEST_APP_NAME=frame_sampling`, the command becomes:
  ```bash
  python builder/main.py builder/tests/frame_sampling/app.yaml -o builder/tests/frame_sampling -c builder/tests/frame_sampling/processors.py --server-type serverless -t
  ```

This pre-build step is useful for:
- **Code generation**: Running scripts that generate code or configuration files
- **Dependency preparation**: Setting up dependencies or downloading assets
- **Environment setup**: Preparing the build environment
- **Validation**: Running tests or checks before building

**Pre-build Command Behavior:**
- **Automatic execution**: Runs before every Docker build in the test suite
- **Dynamic folder**: Uses `TEST_APP_NAME` from build arguments to determine the test application folder
- **OPENAPI_SPEC support**: If `OPENAPI_SPEC` is specified in build args, it is copied to the test app folder
- **Custom paths**: Supports `APP_YAML_PATH`, `OUTPUT_DIR`, and `PROCESSORS_PATH` for flexible configurations
- **Auto-detection**: Automatically detects processors.py in the test app folder if not explicitly specified
- **Execution**: Runs in the current working directory before Docker build
- **Timeout**: 10-minute timeout (600 seconds)
- **Failure handling**: If pre-build command fails, the test is marked as failed
- **Output**: Pre-build command output is logged and included in test results
- **Code generation only**: When `default_enable` is false, only code generation runs (skips Docker build/test)

### Timeout Configuration

Each test can specify a custom timeout value. If the container doesn't complete within the specified time, the test is marked as failed.

**Example:**
```json
{
  "test_config": {
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "cmd": [
      "--video-streams", "34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10"
    ]
  }
}
```

**Timeout Behavior:**
- **Default**: 10 seconds if not specified
- **Failure**: Test fails if container doesn't complete within timeout
- **Logging**: Timeout information is logged and saved to log files
- **Flexible**: Different timeouts can be set for different tests