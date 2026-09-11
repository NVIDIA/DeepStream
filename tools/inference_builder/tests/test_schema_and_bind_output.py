# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for config schema validation and ModelOperator.bind_output() dispatch.

T3 acceptance criteria:
- TYPE_CUSTOM_VIDEO_OUTPUT_FILE and TYPE_CUSTOM_VIDEO_OUTPUT_ASSET present in tensorSpec.data_type oneOf.
- TYPE_CUSTOM_VIDEO_OUTPUT_FILE and TYPE_CUSTOM_VIDEO_OUTPUT_ASSET present in common dataTypes enum.
- TYPE_CUSTOM_VIDEO_OUTPUT_FILE and TYPE_CUSTOM_VIDEO_OUTPUT_ASSET rejected from input tensor specs.
- TYPE_CUSTOM_LONG_VIDEO_ASSETS and TYPE_CUSTOM_VIDEO_CHUNK_ASSETS rejected from output tensor specs.
- bind_output() with either custom video output type creates VideoOutputDataFlow.
- Existing output binding behaviour unchanged for known data types (no regression).

Test tier: Tier 2 (no GPU required).

To run:
    pytest tests/test_schema_and_bind_output.py -v
"""

import json
import pathlib

from jsonschema import Draft7Validator
import pytest

SCHEMA_PATH = pathlib.Path(__file__).parent.parent / "schemas" / "config.schema.json"
COMMON_DEFINITIONS_PATH = (
    pathlib.Path(__file__).parent.parent / "schemas" / "common" / "definitions.schema.json"
)
BASE_MODEL_SCHEMA_PATH = (
    pathlib.Path(__file__).parent.parent / "schemas" / "common" / "base-model.schema.json"
)
INPUT_ONLY_VIDEO_TYPES = (
    "TYPE_CUSTOM_LONG_VIDEO_ASSETS",
    "TYPE_CUSTOM_VIDEO_CHUNK_ASSETS",
)
OUTPUT_ONLY_VIDEO_TYPES = (
    "TYPE_CUSTOM_VIDEO_OUTPUT_FILE",
    "TYPE_CUSTOM_VIDEO_OUTPUT_ASSET",
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaVideoOutputType:
    """Verify custom video output types are registered in the config schema."""

    @pytest.fixture(scope="class")
    def schema(self):
        with open(SCHEMA_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def common_definitions(self):
        with open(COMMON_DEFINITIONS_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def base_model_schema(self):
        with open(BASE_MODEL_SCHEMA_PATH) as f:
            return json.load(f)

    def _minimal_config(self):
        return {
            "name": "test_project",
            "model_repo": "/tmp/models",
            "input": [{"name": "input", "data_type": "TYPE_FP32", "dims": [-1]}],
            "output": [{"name": "output", "data_type": "TYPE_FP32", "dims": [-1]}],
            "models": [
                {
                    "name": "test_model",
                    "backend": "dummy",
                    "input": [{"name": "input", "data_type": "TYPE_FP32", "dims": [-1]}],
                    "output": [{"name": "output", "data_type": "TYPE_FP32", "dims": [-1]}],
                }
            ],
        }

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_in_one_of(self, schema, data_type):
        """Custom video output types must appear in tensorSpec.data_type oneOf."""
        one_of = schema["definitions"]["tensorSpec"]["properties"]["data_type"]["oneOf"]
        consts = [entry.get("const") for entry in one_of]
        assert data_type in consts, f"{data_type} not found in schema. Existing consts: {consts}"

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_has_description(self, schema, data_type):
        """Custom video output type entries must have non-empty descriptions."""
        one_of = schema["definitions"]["tensorSpec"]["properties"]["data_type"]["oneOf"]
        entry = next((e for e in one_of if e.get("const") == data_type), None)
        assert entry is not None
        assert "description" in entry
        assert len(entry["description"]) > 10

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_in_common_definitions(self, common_definitions, data_type):
        """Custom video output types must appear in common dataTypes."""
        data_types = common_definitions["definitions"]["dataTypes"]["enum"]
        assert data_type in data_types

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_rejected_from_top_level_input(self, schema, data_type):
        """Custom video output types are output-only and cannot be top-level inputs."""
        config = self._minimal_config()
        config["input"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert errors, f"{data_type} should be rejected from top-level input"

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_rejected_from_model_input(self, schema, data_type):
        """Custom video output types are output-only and cannot be model inputs."""
        config = self._minimal_config()
        config["models"][0]["input"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert errors, f"{data_type} should be rejected from model input"

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_type_custom_video_output_allowed_on_top_level_output(self, schema, data_type):
        """Custom video output types remain valid on output specs."""
        config = self._minimal_config()
        config["output"][0] = {
            "name": "frames_out",
            "data_type": data_type,
            "dims": [-1, -1, 3],
        }

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert not errors, [error.message for error in errors]

    @pytest.mark.parametrize("data_type", INPUT_ONLY_VIDEO_TYPES)
    def test_video_asset_types_rejected_from_top_level_output(self, schema, data_type):
        """Video asset types are input-only and cannot be top-level outputs."""
        config = self._minimal_config()
        config["output"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert errors, f"{data_type} should be rejected from top-level output"

    @pytest.mark.parametrize("data_type", INPUT_ONLY_VIDEO_TYPES)
    def test_video_asset_types_rejected_from_model_output(self, schema, data_type):
        """Video asset types are input-only and cannot be model outputs."""
        config = self._minimal_config()
        config["models"][0]["output"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert errors, f"{data_type} should be rejected from model output"

    @pytest.mark.parametrize("data_type", INPUT_ONLY_VIDEO_TYPES)
    def test_video_asset_types_allowed_on_top_level_input(self, schema, data_type):
        """Video asset types remain valid on input specs."""
        config = self._minimal_config()
        config["input"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert not errors, [error.message for error in errors]

    @pytest.mark.parametrize("data_type", INPUT_ONLY_VIDEO_TYPES)
    def test_video_asset_types_allowed_on_model_input(self, schema, data_type):
        """Video asset types remain valid on model input specs."""
        config = self._minimal_config()
        config["models"][0]["input"][0]["data_type"] = data_type

        errors = list(Draft7Validator(schema).iter_errors(config))

        assert not errors, [error.message for error in errors]

    def test_common_base_model_input_uses_input_tensor_spec(self, base_model_schema):
        """Backend schemas inherit input/output directionality restrictions."""
        assert (
            base_model_schema["properties"]["input"]["items"]["$ref"]
            == "#/definitions/inputTensorSpec"
        )
        assert (
            base_model_schema["properties"]["output"]["items"]["$ref"]
            == "#/definitions/outputTensorSpec"
        )

    def test_existing_types_still_present(self, schema):
        """Existing data types must not have been removed."""
        one_of = schema["definitions"]["tensorSpec"]["properties"]["data_type"]["oneOf"]
        consts = {entry.get("const") for entry in one_of}
        required = {
            "TYPE_UINT8",
            "TYPE_FP32",
            "TYPE_STRING",
            "TYPE_CUSTOM_IMAGE_BASE64",
            "TYPE_CUSTOM_LONG_VIDEO_ASSETS",
            "TYPE_CUSTOM_VIDEO_CHUNK_ASSETS",
        }
        missing = required - consts
        assert not missing, f"Previously existing data types removed: {missing}"


# ---------------------------------------------------------------------------
# bind_output() dispatch tests (mocked ModelOperator)
# ---------------------------------------------------------------------------


class TestBindOutputDispatch:
    """Verify ModelOperator.bind_output() creates the right DataFlow subclass."""

    def _make_operator(self):
        """Return a ModelOperator with a dummy model config (no GPU work)."""
        from lib.inference import ModelOperator

        model_config = {
            "name": "test_model",
            "backend": "dummy",
            "input": [{"name": "input", "data_type": "TYPE_FP32", "dims": [-1]}],
            "output": [{"name": "output", "data_type": "TYPE_FP32", "dims": [-1]}],
        }
        op = ModelOperator(model_config=model_config, model_repo="/tmp/models")
        return op

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_bind_output_custom_video_output_creates_video_output_dataflow(self, data_type):
        """bind_output with custom video output types creates VideoOutputDataFlow."""
        from lib.inference import VideoOutputDataFlow

        op = self._make_operator()
        configs = [{"name": "frames_out", "data_type": data_type, "dims": [-1, -1, 3]}]
        flow = op.bind_output(configs)
        assert isinstance(
            flow, VideoOutputDataFlow
        ), f"Expected VideoOutputDataFlow, got {type(flow).__name__}"

    def test_bind_output_plain_type_creates_dataflow(self):
        """bind_output with TYPE_FP32 creates plain DataFlow (no regression)."""
        from lib.inference import DataFlow, VideoOutputDataFlow

        op = self._make_operator()
        configs = [{"name": "output", "data_type": "TYPE_FP32", "dims": [-1]}]
        flow = op.bind_output(configs)
        assert isinstance(flow, DataFlow)
        assert not isinstance(flow, VideoOutputDataFlow)

    def test_bind_output_string_type_creates_dataflow(self):
        """bind_output with TYPE_STRING creates plain DataFlow (no regression)."""
        from lib.inference import DataFlow, VideoOutputDataFlow

        op = self._make_operator()
        configs = [{"name": "text_out", "data_type": "TYPE_STRING", "dims": [-1]}]
        flow = op.bind_output(configs)
        assert isinstance(flow, DataFlow)
        assert not isinstance(flow, VideoOutputDataFlow)

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_bind_output_video_output_flow_is_outbound(self, data_type):
        """The VideoOutputDataFlow created by bind_output must have outbound=True."""
        op = self._make_operator()
        configs = [{"name": "frames_out", "data_type": data_type, "dims": [-1, -1, 3]}]
        flow = op.bind_output(configs)
        assert flow._outbound is True

    @pytest.mark.parametrize("data_type", OUTPUT_ONLY_VIDEO_TYPES)
    def test_bind_input_rejects_custom_video_output(self, data_type):
        """Custom video output types are output-only at runtime too."""
        op = self._make_operator()
        configs = [{"name": "frames_in", "data_type": data_type, "dims": [-1, -1, 3]}]

        with pytest.raises(ValueError, match="can only be used for output tensors"):
            op.bind_input(configs)

        assert op.inputs == []

    @pytest.mark.parametrize("data_type", INPUT_ONLY_VIDEO_TYPES)
    def test_bind_output_rejects_video_asset_input_types(self, data_type):
        """Video asset input types are rejected from runtime output binding."""
        op = self._make_operator()
        configs = [{"name": "video_out", "data_type": data_type, "dims": [-1]}]

        with pytest.raises(ValueError, match="can only be used for input tensors"):
            op.bind_output(configs)

        assert op.outputs == []

    def test_outbound_dataflow_mapping_contains_video_output(self):
        """outbound_dataflow_mapping must map both custom video output types."""
        from lib.inference import VideoOutputDataFlow, outbound_dataflow_mapping

        for data_type in OUTPUT_ONLY_VIDEO_TYPES:
            assert data_type in outbound_dataflow_mapping
            assert outbound_dataflow_mapping[data_type] is VideoOutputDataFlow
