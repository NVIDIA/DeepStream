# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real shell publisher with a private synthetic Python builder."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class WheelPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wheel fixture ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        script = Path(__file__).resolve().parents[2] / "src/service-maker/sources/python/build.sh"
        self.script = self.root / "repo/src/service-maker/sources/python/build.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(script, self.script)
        for relative in ("repo/src/service-maker/includes", "repo/includes", "output", "bin", "scratch"):
            (self.root / relative).mkdir(parents=True)
        self.source = self.script.parent / "src"
        (self.source / "dist").mkdir(parents=True)
        (self.source / "dist/pyservicemaker-stale.whl").write_bytes(b"stale")
        (self.script.parent / "aarch64-toolchain.cmake").write_text("user-owned toolchain")
        builder = self.root / "bin/python3.12"
        builder.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
import sys
if int(os.environ.get('BUILD_EXIT', '0')):
    sys.exit(int(os.environ['BUILD_EXIT']))
out = Path(sys.argv[sys.argv.index('--outdir') + 1])
for index in range(int(os.environ.get('WHEEL_COUNT', '1'))):
    (out / ('pyservicemaker-fresh' + str(index) + '.whl')).write_bytes(b'fresh')
""")
        builder.chmod(0o755)

    def run_builder(self, arch, **overrides):
        env = os.environ.copy()
        env.update(PATH=str(self.root / "bin") + os.pathsep + env["PATH"],
                   TMPDIR=str(self.root / "scratch"), **overrides)
        return subprocess.run(["bash", str(self.script), "--minor-ver", "12", "--arch", arch,
                               "--output-dir", str(self.root / "output")],
                              env=env, text=True, capture_output=True, timeout=15)

    def assert_private_cleanup(self):
        self.assertEqual(list((self.root / "scratch").iterdir()), [])
        self.assertEqual((self.script.parent / "aarch64-toolchain.cmake").read_text(), "user-owned toolchain")
        self.assertEqual((self.source / "dist/pyservicemaker-stale.whl").read_bytes(), b"stale")

    def test_failing_python_propagates_and_never_publishes_stale_wheel(self):
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch):
                process = self.run_builder(arch, BUILD_EXIT="42")
                self.assertEqual(process.returncode, 42, process.stdout + process.stderr)
                self.assertEqual(list((self.root / "output").iterdir()), [])
                self.assert_private_cleanup()

    def test_exactly_one_fresh_wheel_required(self):
        for count in ("0", "2"):
            with self.subTest(count=count):
                process = self.run_builder("aarch64", WHEEL_COUNT=count)
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(list((self.root / "output").iterdir()), [])
                self.assert_private_cleanup()

    def test_success_with_spaced_paths_publishes_only_this_build(self):
        process = self.run_builder("aarch64")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual([path.name for path in (self.root / "output").iterdir()], ["pyservicemaker-fresh0.whl"])
        self.assert_private_cleanup()


if __name__ == "__main__":
    unittest.main()
