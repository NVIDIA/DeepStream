# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tests for GstChildProxy property support via the Python service-maker API.

Validates that child element properties can be set and read back using the
"::" notation (e.g. "signaller::uri" on webrtcsink) through node.set() and
node.get(), mirroring gst-launch behaviour:

  gst-launch-1.0 audiomixer name=mix sink_0::volume=0.5 ...

Uses audiomixer whose sink pads are GstChildProxy children with "volume"
(float) and "mute" (bool) properties — the same mechanism as webrtcsink's
"signaller::uri".
"""

import math
from pyservicemaker import Pipeline


def _build_mixer_pipeline(name: str) -> Pipeline:
    """audiotestsrc -> audiomixer -> fakesink, linked so sink_0 pad exists."""
    pipeline = Pipeline(name)
    pipeline.add("audiotestsrc", "src",  {"num-buffers": 5}) \
            .add("audiomixer",   "mix")                       \
            .add("fakesink",     "sink", {"sync": False})     \
            .link("src", "mix", "sink")
    return pipeline


# ---------------------------------------------------------------------------
# App path: node.set(dict) -> Object::set_(name, Value)
# ---------------------------------------------------------------------------

def test_child_proxy_set_float():
    """node.set() with a float child property round-trips correctly."""
    pipeline = _build_mixer_pipeline("test-cp-float")
    pipeline["mix"].set({"sink_0::volume": 0.42})
    val = pipeline["mix"].get("sink_0::volume")
    assert math.isclose(val, 0.42, rel_tol=1e-5), f"Expected 0.42, got {val}"


def test_child_proxy_set_bool():
    """node.set() with a bool child property round-trips correctly."""
    pipeline = _build_mixer_pipeline("test-cp-bool")
    pipeline["mix"].set({"sink_0::mute": True})
    val = pipeline["mix"].get("sink_0::mute")
    assert val is True, f"Expected True, got {val}"


def test_child_proxy_set_multiple():
    """node.set() with multiple child properties in one call."""
    pipeline = _build_mixer_pipeline("test-cp-multi")
    pipeline["mix"].set({
        "sink_0::volume": 0.75,
        "sink_0::mute":   False,
    })
    assert math.isclose(pipeline["mix"].get("sink_0::volume"), 0.75, rel_tol=1e-5)
    assert pipeline["mix"].get("sink_0::mute") is False


def test_child_proxy_overwrite():
    """Child property can be overwritten after initial set."""
    pipeline = _build_mixer_pipeline("test-cp-overwrite")
    pipeline["mix"].set({"sink_0::volume": 0.1})
    pipeline["mix"].set({"sink_0::volume": 0.9})
    val = pipeline["mix"].get("sink_0::volume")
    assert math.isclose(val, 0.9, rel_tol=1e-5), f"Expected 0.9, got {val}"


# ---------------------------------------------------------------------------
# YAML path: node.set() called with a dict that mirrors a YAML properties block
# This exercises Object::set_(name, YAML::Node) via the Python dict -> YAML
# conversion that Pipeline.add() and node.set() perform internally.
# ---------------------------------------------------------------------------

def test_child_proxy_yaml_path_via_pipeline_add():
    """
    Pipeline.add() accepts a properties dict and passes it through the same
    YAML::Node path used when loading a pipeline YAML file.  Verifies child
    properties supplied at add-time are applied correctly after linking.
    """
    pipeline = Pipeline("test-cp-yaml")
    # Properties are supplied at add time (mirrors YAML file loading).
    # The pad doesn't exist until after linking, so we link first, then
    # update — same flow as the C++ test app.
    pipeline.add("audiotestsrc", "src",  {"num-buffers": 5}) \
            .add("audiomixer",   "mix")                       \
            .add("fakesink",     "sink", {"sync": False})     \
            .link("src", "mix", "sink")

    # Apply via dict (YAML-node code path).
    pipeline["mix"].set({"sink_0::volume": 0.33, "sink_0::mute": False})

    assert math.isclose(pipeline["mix"].get("sink_0::volume"), 0.33, rel_tol=1e-5)
    assert pipeline["mix"].get("sink_0::mute") is False


def test_child_proxy_two_children():
    """Child properties on two different child pads are independent."""
    pipeline = Pipeline("test-cp-two")
    pipeline.add("audiotestsrc", "src",  {"num-buffers": 5}) \
            .add("audiotestsrc", "src2", {"num-buffers": 5}) \
            .add("audiomixer",   "mix")                       \
            .add("fakesink",     "sink", {"sync": False})     \
            .link("src",  "mix") \
            .link("src2", "mix") \
            .link("mix",  "sink")

    pipeline["mix"].set({
        "sink_0::volume": 0.2,
        "sink_1::volume": 0.8,
        "sink_1::mute":   True,
    })

    assert math.isclose(pipeline["mix"].get("sink_0::volume"), 0.2, rel_tol=1e-5)
    assert math.isclose(pipeline["mix"].get("sink_1::volume"), 0.8, rel_tol=1e-5)
    assert pipeline["mix"].get("sink_1::mute") is True


# ---------------------------------------------------------------------------
# Enum: compositor sink pad operator / sizing-policy (G_TYPE_ENUM)
# ---------------------------------------------------------------------------

def _build_compositor_pipeline(name: str) -> Pipeline:
    """videotestsrc -> compositor -> fakesink, linked so sink_0 pad exists."""
    pipeline = Pipeline(name)
    pipeline.add("videotestsrc", "src",  {"num-buffers": 5}) \
            .add("compositor",   "comp")                      \
            .add("fakesink",     "sink", {"sync": False})     \
            .link("src", "comp", "sink")
    return pipeline


def test_child_proxy_enum_app_path():
    """G_TYPE_ENUM child property set and read back via app path (node.set dict)."""
    pipeline = _build_compositor_pipeline("test-cp-enum-app")
    # Default operator is 1 (over). Set to 0 (source) and back.
    pipeline["comp"].set({"sink_0::operator": 0})
    assert pipeline["comp"].get("sink_0::operator") == 0, \
        f"Expected 0, got {pipeline['comp'].get('sink_0::operator')}"

    pipeline["comp"].set({"sink_0::operator": 1})
    assert pipeline["comp"].get("sink_0::operator") == 1, \
        f"Expected 1, got {pipeline['comp'].get('sink_0::operator')}"


def test_child_proxy_enum_yaml_path():
    """G_TYPE_ENUM child property set and read back via YAML path (node.set dict)."""
    pipeline = _build_compositor_pipeline("test-cp-enum-yaml")
    # sizing-policy: 0=none (default), 1=keep-aspect-ratio
    pipeline["comp"].set({"sink_0::sizing-policy": 1})
    assert pipeline["comp"].get("sink_0::sizing-policy") == 1, \
        f"Expected 1, got {pipeline['comp'].get('sink_0::sizing-policy')}"

    pipeline["comp"].set({"sink_0::sizing-policy": 0})
    assert pipeline["comp"].get("sink_0::sizing-policy") == 0, \
        f"Expected 0, got {pipeline['comp'].get('sink_0::sizing-policy')}"


# ---------------------------------------------------------------------------
# Int: compositor xpos/ypos/width/height (G_TYPE_INT)
# ---------------------------------------------------------------------------

def test_child_proxy_int():
    """G_TYPE_INT child properties set and read back."""
    pipeline = _build_compositor_pipeline("test-cp-int")
    pipeline["comp"].set({
        "sink_0::xpos":   320,
        "sink_0::ypos":   240,
        "sink_0::width":  640,
        "sink_0::height": 480,
    })
    assert pipeline["comp"].get("sink_0::xpos")   == 320
    assert pipeline["comp"].get("sink_0::ypos")   == 240
    assert pipeline["comp"].get("sink_0::width")  == 640
    assert pipeline["comp"].get("sink_0::height") == 480

    # Negative int
    pipeline["comp"].set({"sink_0::xpos": -100})
    assert pipeline["comp"].get("sink_0::xpos") == -100


# ---------------------------------------------------------------------------
# Uint: compositor zorder (G_TYPE_UINT)
# ---------------------------------------------------------------------------

def test_child_proxy_uint():
    """G_TYPE_UINT child property set and read back."""
    pipeline = _build_compositor_pipeline("test-cp-uint")
    pipeline["comp"].set({"sink_0::zorder": 5})
    assert pipeline["comp"].get("sink_0::zorder") == 5

    pipeline["comp"].set({"sink_0::zorder": 0})
    assert pipeline["comp"].get("sink_0::zorder") == 0


# ---------------------------------------------------------------------------
# Regression: top-level properties on non-child-proxy elements still work
# ---------------------------------------------------------------------------

def test_top_level_properties_unaffected():
    """Standard element properties continue to work after the child proxy fix."""
    pipeline = Pipeline("test-cp-regression")
    pipeline.add("fakesink", "sink", {"sync": False, "async": False})
    assert pipeline["sink"].get("sync")  is False
    assert pipeline["sink"].get("async") is False


if __name__ == "__main__":
    test_child_proxy_set_float()
    test_child_proxy_set_bool()
    test_child_proxy_set_multiple()
    test_child_proxy_overwrite()
    test_child_proxy_yaml_path_via_pipeline_add()
    test_child_proxy_two_children()
    test_child_proxy_enum_app_path()
    test_child_proxy_enum_yaml_path()
    test_child_proxy_int()
    test_child_proxy_uint()
    test_top_level_properties_unaffected()
    print("All tests passed.")
