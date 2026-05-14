{#
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#}

import unittest
import json
import yaml
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# # Add builder/validator.py path to Python path
# VALIDATOR_DIR = Path("{{ validator_dir }}")
# if str(VALIDATOR_DIR) not in sys.path:
#     sys.path.append(str(VALIDATOR_DIR))

# Will load validate.py in the same directory
import validate

# Common paths relative to test runner location
GENERATED_CLIENT_DIR = Path(__file__).parent / validate.GENERATED_CLIENT_DIR
TEST_CASES_FILE = Path(__file__).parent / validate.TEST_CASES_FILE

# Setup environment before imports
if not validate.check_client_exists():
    if not validate.setup_environment(GENERATED_CLIENT_DIR):
        logger.error("Failed to setup client environment")
        sys.exit(1)

from openapi_client.api.nvidiametropolisinferenceapi_api import NVIDIAMETROPOLISINFERENCEAPIApi
from openapi_client.api_client import ApiClient
from openapi_client.configuration import Configuration
from openapi_client.models.inference_request import InferenceRequest

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all test methods.

        This is a class method because it sets up shared resources (host, api client)
        that will be used by all test methods. It runs once before any tests.
        """
        cls.host = os.getenv("TEST_HOST", "http://127.0.0.1:8800")
        cls.tolerance = float(os.getenv("TEST_TOLERANCE", "1e-5"))

        logger.info("Setting up test environment")
        logger.info(f"Server host: {cls.host}")
        logger.info(f"Tolerance: {cls.tolerance}")

        with open(TEST_CASES_FILE) as f:
            cls.test_cases = yaml.safe_load(f)
        logger.info(f"Loaded {len(cls.test_cases)} test cases from {TEST_CASES_FILE}")

        config = Configuration(host=cls.host)
        api_client = ApiClient(config)
        cls.api = NVIDIAMETROPOLISINFERENCEAPIApi(api_client)

    def setUp(self):
        """Set up test fixtures before each test method.

        Instance method because it's specific to each test run.
        Creates a logger for the specific test instance.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.id()}")
        self.logger.info(f"Starting test: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method.

        Instance method because it's specific to each test run.
        """
        self.logger.info(f"Completed test: {self._testMethodName}")

    def test_health_check(self):
        """Test health check endpoint"""
        # self.logger.info("Running health check...")
        # response = self.api.health_ready_v1_health_ready_get_with_http_info()
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(response.data["status"], "ready")
        # self.logger.info("Health check successful")
        self.logger.info(f"Skip health check for {self.host}")

    def test_inference(self):
        """Test inference with pre-built requests"""

        for test in self.test_cases:
            with self.subTest(test=test["name"]):
                test_logger = logging.getLogger(f"{__name__}.{self.id()}.{test['name']}")
                test_logger.info(f"Starting test case: {test['name']}")
                try:
                    # Load pre-built request
                    request_path = Path(test["request"])
                    test_logger.debug(f"Loading request from: {request_path}")
                    with open(request_path) as f:
                        request_data = json.load(f)
                    request = InferenceRequest.from_dict(request_data)

                    # Load expected output
                    expected_path = Path(test["expected"])
                    test_logger.debug(f"Loading expected output from: {expected_path}")
                    with open(expected_path) as f:
                        expected = json.load(f)

                    # Run inference
                    test_logger.info("Executing inference request...")
                    response = self.api.inference(request)
                    actual = response.to_dict()

                    # Compare results
                    if expected:
                        test_logger.info("Validating response...")
                        self.assertTrue(
                            # TODO: We can template below line for different validators
                            # validator = {{ my_validator }}
                            # validator.compare_responses(actual, expected, self.tolerance)
                            # or make abstract validator class and ask users to implement their own validator
                            validate.CvValidator.compare_responses(actual, expected, self.tolerance),
                            f"Test {test['name']}: Response doesn't match expected output under tolerance {self.tolerance}. "
                            f"Actual: {actual}, Expected: {expected}"
                        )
                    else:
                        test_logger.info("No expected output, skipping validation")
                    test_logger.info(f"Test case '{test['name']}' passed successfully")

                except Exception as e:
                    test_logger.error(f"Test case '{test['name']}' failed: {str(e)}")
                    raise

if __name__ == '__main__':
    # Create a test runner that outputs both to console and file
    runner = unittest.TextTestRunner(
        verbosity=2,  # Increased verbosity
        stream=sys.stdout,  # Output to console
    )

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInference)

    # Run tests
    result = runner.run(suite)

    # Exit with appropriate code so the test framework detects failures
    sys.exit(0 if result.wasSuccessful() else 1)