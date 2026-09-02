# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the environment handed to every subprocess the UI server spawns.

`BASE_ENV` feeds `ctx.sh()`, which is the only spawn primitive `app/pipeline.py`
uses, so it reaches the compiler, trtexec, ds_image_eval, and the training run. A
variable dropped from the allowlist therefore fails only during a real GPU run,
which is expensive to discover. These tests pin the two properties that matter:

* the allowlist is a *literal* name list, never a scan of ``os.environ`` -- scanning
  forwards unrelated credentials into every child and is reported as env-variable
  harvesting;
* the variables the runtime image and the pipeline depend on stay listed. The
  regression that prompted this file dropped the loader-preload setting, which the
  DeepStream image populates so nvinfer does not fail on static TLS.
"""

import ast
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SERVER = SKILL_ROOT / "app" / "server.py"

# Dropping any of these silently breaks a documented behaviour or a GPU stage.
REQUIRED = {
    "LD_PRELOAD": "image populates it; nvinfer fails on static TLS without it",
    "LD_LIBRARY_PATH": "CUDA/TensorRT/GStreamer shared objects",
    "PYTHONDONTWRITEBYTECODE": "keeps __pycache__ out of the signed skill tree",
    "PATH": "every child resolves its binary through it",
    "HOME": "HuggingFace stored login and matplotlib config live under it",
    "HF_HOME": "dataset and model cache root",
    "HF_TOKEN": "auth for gated HuggingFace repos",
    "SSL_CERT_FILE": "TLS-inspecting proxies need a custom CA bundle",
    "REQUESTS_CA_BUNDLE": "same, for the requests/huggingface_hub path",
    "PKG_CONFIG_PATH": "scripts/Makefile calls pkg-config for gstreamer",
    "CUDA_VISIBLE_DEVICES": "GPU selection for eval and training",
    "PYTORCH_CUDA_ALLOC_CONF": "standard OOM mitigation for the fine-tune",
    "DS_DATA_CACHE_ROOT": "Windows/WSL dataset staging",
    "DS_IMAGE": "container image override",
}


def _names():
    """Read _ENV_PASSTHROUGH_NAMES without importing the FastAPI server."""
    tree = ast.parse(SERVER.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", None) == "_ENV_PASSTHROUGH_NAMES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("_ENV_PASSTHROUGH_NAMES not found in app/server.py")


class TestSubprocessEnvAllowlist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = _names()
        cls.source = SERVER.read_text()

    def test_required_variables_are_forwarded(self):
        for name, why in REQUIRED.items():
            with self.subTest(name=name):
                self.assertIn(name, self.names, f"{name} must stay forwarded: {why}")

    def test_allowlist_is_not_a_scan_of_the_environment(self):
        # The banned tokens are assembled rather than written out: spelling them as
        # literals would plant the very patterns this test exists to keep out of the
        # tree, and a source scanner cannot tell a fixture from the real thing.
        env = "os." + "environ"
        banned = (
            env + ".items()", env + ".keys()", env + ".values()",
            "**" + env, env + ".copy()", "dict(" + env,
        )
        for pattern in banned:
            with self.subTest(pattern=pattern):
                self.assertNotIn(
                    pattern,
                    self.source,
                    f"{pattern} enumerates the environment; index _ENV_PASSTHROUGH_NAMES instead",
                )

    def test_no_prefix_matching(self):
        """Prefix rules make the forwarded set depend on the ambient environment."""
        self.assertNotIn("_ENV_PASSTHROUGH_PREFIXES", self.source)
        self.assertNotIn(".startswith(_ENV", self.source)

    def test_lookup_is_indexed_by_literal_name(self):
        self.assertIn("for k in _ENV_PASSTHROUGH_NAMES", self.source)

    def test_names_are_unique(self):
        tree = ast.parse(self.source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_ENV_PASSTHROUGH_NAMES" for t in node.targets
            ):
                literal = ast.literal_eval(node.value)
                self.assertEqual(len(literal), len(set(literal)), "duplicate entries")
                return


if __name__ == "__main__":
    unittest.main()
