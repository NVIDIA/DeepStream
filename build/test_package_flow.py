#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only packaging contracts using an owned, minimal repository/runtime.

Run with ``python3 -m unittest discover -s build -p test_package_flow.py -v``.
No SDK build, download, package installation or privileged operation is needed.
Two tiny fixture DSOs are compiled to exercise real ELF metadata inspection;
they are not genuine SDK binaries and do not qualify a GVS runtime.
"""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import unittest


BUILD_DIR = Path(__file__).resolve().parent
SDK_PATH = "opt/nvidia/deepstream/deepstream-9.2"


@unittest.skipUnless(shutil.which("cc") and shutil.which("git") and shutil.which("readelf"),
                     "requires cc, git and readelf for owned ELF fixtures")
@unittest.skipUnless(platform.machine() == "x86_64", "fixture compiler and package cases target x86_64")
class PackageFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ds-package-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "fixture repo"
        self.runtime = self.root / "runtime input"
        self.output = self.root / "package output"
        self.stage = self.root / "package stage"
        self.forbidden = self.root / "forbidden-commands.log"
        self.repo.mkdir()
        (self.repo / "build").mkdir()
        for name in ("build.sh", "package.sh"):
            shutil.copy2(BUILD_DIR / name, self.repo / "build" / name)
        self.env = os.environ.copy()
        for name in ("NVDS_VERSION", "PKG_VERSION", "CUDA_VER", "BASH_ENV", "ENV"):
            self.env.pop(name, None)
        self.env["NVDS_VERSION"] = "9.2.0"
        self.env["CUDA_VER"] = "13.2"
        self.env["PACKAGE_TEST_FORBIDDEN"] = str(self.forbidden)
        self.tools = self.root / "guarded tools"
        self.tools.mkdir()
        for command in ("sudo", "make", "cmake", "apt", "apt-get", "dpkg", "wget", "curl", "pip", "pip3"):
            self.write(self.tools / command,
                       '#!/bin/bash\nprintf "%s\\n" "$0" >> "$PACKAGE_TEST_FORBIDDEN"\nexit 97\n',
                       executable=True)
        self.env["PATH"] = str(self.tools) + os.pathsep + self.env["PATH"]
        self.make_sources()
        self.make_runtime()

    @staticmethod
    def write(path, content, executable=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        if executable:
            path.chmod(0o755)

    def make_sources(self):
        files = {
            ".gitignore": "*.so\n*.o\n.build/\n",
            "includes/public.h": "/* public fixture header */\n",
            "src/utils/public_source.cpp": "// tracked fixture, index version\n",
            "src/gst-utils/gst-nvdsinferbase/Makefile": "INC:= ../../utils\n",
            "src/service-maker/cmake/nvds_service_makerConfig.cmake":
                'set(INC "${CMAKE_CURRENT_LIST_DIR}/../../../includes")\n',
            "src/service-maker/sources/core/Makefile":
                "NVDS_VERSION:=9.2\nAPP_COMMON_DIR:= ../../../apps/common\nINC:= -I/usr/local/cuda-$(CUDA_VER)/include\n",
            "src/service-maker/sources/engine/Makefile":
                "NVDS_VERSION:=9.2\nINC:= -I/usr/local/cuda-$(CUDA_VER)/include\n",
            "src/apps/common/common.c": "/* common fixture */\n",
            "src/apps/sample_apps/deepstream-server/Makefile": "LIBS:= -lnvds_rest_server -lm\n",
            "src/apps/sample_apps/deepstream-infer-tensor-meta-test/Makefile":
                "SRC:= ../../../utils/nvdsinfer_customparser\n",
            "src/utils/ds3d/dataloader/lidarsource/Makefile": "INC:= /opt/deepstream/sources/includes\n",
            "tools/deepstream-otelcol/README.md": "Collector fixture documentation.\n",
            "scripts/install.sh": "#!/bin/bash\nexit 98 # must not execute while packaging\n",
            "scripts/uninstall.sh":
                'remove_if_deb "deepstream-sample-data"\nif [[ -n "${PLATFORM}" ]]; then\n'
                '  remove_if_deb "deepstream-binaries-${PLATFORM}"\nfi\n',
            "scripts/rtpmanager/fixture.patch": "fixture patch, do not apply\n",
            "deepstream_libraries/build/build_package.sh":
                '#!/bin/bash\nprintf "wheel-build\\n" >> "$PACKAGE_TEST_FORBIDDEN"\nexit 97\n',
        }
        plugins = (
            "gst-nvinfer", "gst-nvinferserver", "gst-nvmsgconv", "gst-nvmultistream2",
            "gst-nvmultiurisrcbin", "gst-nvds3dbridge", "gst-nvds3dfilter", "gst-nvds3dmixer",
            "gst-nvdsbins", "gst-nvdslogger", "gst-nvmodelmux", "gst-nvmultistream",
            "gst-nvsegvisual", "gst-nvtiler", "gst-nvdsucx", "gst-nvinfereval",
        )
        for plugin in plugins:
            files[f"src/gst-plugins/{plugin}/Makefile"] = "INC:= ../../utils ../../gst-utils\n"
        files["src/gst-plugins/gst-nvmultiurisrcbin/Makefile"] += (
            "SRC:= ../../utils/nvds_msgapi/common_src/nvds_utils.cpp\n"
        )
        files["src/gst-plugins/gst-nvdsanalytics/Makefile"] = "INC:= -I ../../../utils/nvds_analytics\n"
        for app in ("deepstream-app", "deepstream-test5", "deepstream-transfer-learning-app"):
            files[f"src/apps/sample_apps/{app}/Makefile"] = (
                "SRC:= ../../../../src/utils/nvds_msgapi/common_src/nvds_utils.cpp\n"
                "INC:= ../../../gst-utils\n"
            )
        files["src/apps/sample_apps/deepstream-test5/Makefile"] += (
            "INC+= -I../../../src/utils/nvds_msgapi/inc\n"
        )
        for subpath in (
            "TritonBackendEnsemble/nvdsinferserver_custom_impl_ensemble",
            "TritonOnnxYolo/nvdsinfer_custom_impl_yolo",
            "TritonOnnxYolo/nvdsinferserver_custom_impl_yolo",
        ):
            files[f"src/apps/sample_apps/{subpath}/Makefile"] = (
                "INC:= /opt/deepstream/sources/includes\n"
            )
        for helper in ("prepare_classification_test_video.sh", "prepare_ds_triton_tao_model_repo.sh",
                       "prepare_ds_triton_model_repo.sh", "triton_backend_setup.sh"):
            files[f"scripts/{helper}"] = "#!/bin/bash\nexit 98 # package only\n"
        for helper in ("prepare_model.py", "prepare_samples.py", "requirements.txt", "README.md"):
            files[f"tools/modelopt/{helper}"] = "model preparation fixture; never executed\n"
        for path, content in files.items():
            self.write(self.repo / path, content)
        self.run_command(["git", "init", "--quiet", str(self.repo)])
        self.run_command(["git", "-C", str(self.repo), "add", "."])
        self.write(self.repo / "src/utils/public_source.cpp", "// tracked fixture, current worktree\n")
        self.write(self.repo / "src/utils/new_source.cpp", "// untracked fixture source\n")
        self.write(self.repo / "src/utils/ignored.so", b"must not ship compiled source artifact")
        self.write(self.repo / "src/utils/.build/ignored.o", b"must not ship compiled source object")

    def make_runtime(self):
        self.write(self.runtime / "bin/deepstream-fixture", b"GVS fixture executable\x00\xff", executable=True)
        self.write(self.runtime / "lib/libfixture.so.1", b"GVS fixture library\x00\x80\xff")
        (self.runtime / "lib/libfixture.so").symlink_to("libfixture.so.1")
        self.write(self.runtime / "samples/models/model.bin", b"GVS model fixture\x00\xfe")
        self.write(self.runtime / "service-maker/lib/libfixture-service.so", b"GVS service library fixture")
        self.write(self.runtime / "LicenseAgreement.pdf", b"fixture license asset, not a real agreement")
        self.make_infer_fixture(11)

    def make_infer_fixture(self, major):
        """Create an actual ELF DT_NEEDED fixture, not a TensorRT implementation."""
        dependency = f"libnvinfer.so.{major}"
        for command, source in (
            (["cc", "-shared", "-fPIC", "-x", "c", "-", f"-Wl,-soname,{dependency}",
              "-o", str(self.runtime / "lib" / dependency)], "int fixture_dependency(void) { return 1; }\n"),
            (["cc", "-shared", "-fPIC", "-x", "c", "-", "-L" + str(self.runtime / "lib"),
              "-Wl,--no-as-needed", "-l:" + dependency, "-o", str(self.runtime / "lib/libnvds_infer.so")],
             "int fixture_inference(void) { return 1; }\n"),
        ):
            run = subprocess.run(command, input=source, env=self.env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout)

    def run_command(self, command, expected=0):
        run = subprocess.run(command, env=self.env, cwd=self.repo, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        if expected is not None:
            self.assertEqual(run.returncode, expected, run.stdout)
        return run

    def package_command(self, *extra):
        return ["bash", str(self.repo / "build/package.sh"), "--format=tar", "--arch=amd64",
                "--platform=x86", f"--runtime-root={self.runtime}",
                f"--output-dir={self.output}", f"--stage-dir={self.stage}", *extra]

    def assert_no_build_or_install(self):
        self.assertFalse(self.forbidden.exists(), self.forbidden.read_text() if self.forbidden.exists() else "")

    def test_help_is_read_only(self):
        for script in ("build.sh", "package.sh"):
            run = self.run_command(["bash", str(self.repo / "build" / script), "--help"])
            self.assertIn("package", run.stdout.lower())
        self.assertFalse(self.stage.exists())
        self.assertFalse(self.output.exists())
        self.assert_no_build_or_install()

    def test_missing_runtime_stops_before_build_or_install(self):
        run = self.run_command(["bash", str(self.repo / "build/build.sh"),
                                f"--package-from={self.root / 'absent'}"], expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("missing", run.stdout.lower())
        self.assert_no_build_or_install()

    def test_empty_runtime_argument_is_rejected_before_build(self):
        run = self.run_command(["bash", str(self.repo / "build/build.sh"), "--package-from="], expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("package-from", run.stdout)
        self.assert_no_build_or_install()

    def test_direct_empty_runtime_argument_is_rejected_before_staging(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "owned preexisting staging marker\n")
        run = self.run_command(self.package_command("--runtime-root="), expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("runtime-root", run.stdout)
        self.assertTrue(marker.is_file())
        self.assert_no_build_or_install()

    def test_package_from_conflicts_are_rejected_before_actions(self):
        for conflict in ("--only=utils", "--resume", "--deepstream-libraries-wheel"):
            with self.subTest(conflict=conflict):
                run = self.run_command(["bash", str(self.repo / "build/build.sh"),
                                        f"--package-from={self.runtime}", conflict], expected=None)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn("cannot be combined", run.stdout)
                self.assert_no_build_or_install()

    def test_driver_package_options_require_runtime(self):
        for option in ("--package-platform=x86", "--include-untracked-sources"):
            with self.subTest(option=option):
                run = self.run_command(["bash", str(self.repo / "build/build.sh"), option], expected=None)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn("require --package-from", run.stdout)
        self.assert_no_build_or_install()

    def test_invalid_platform_and_architecture_are_rejected(self):
        for option in ("--platform=unknown", "--platform=jetson", "--arch=ppc64el"):
            with self.subTest(option=option):
                run = self.run_command(self.package_command(option), expected=None)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertFalse(self.stage.exists())
        self.assert_no_build_or_install()

    def test_stage_cannot_overlap_runtime_or_repository(self):
        before = (self.runtime / "lib/libfixture.so.1").read_bytes()
        alias = self.root / "runtime alias"
        alias.symlink_to(self.runtime, target_is_directory=True)
        for stage in (self.runtime, self.runtime / "inside", self.root, self.repo, alias):
            with self.subTest(stage=stage):
                run = self.run_command(self.package_command(f"--stage-dir={stage}"), expected=None)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertTrue("overlap" in run.stdout or "inside runtime" in run.stdout, run.stdout)
                self.assertEqual((self.runtime / "lib/libfixture.so.1").read_bytes(), before)
                self.assertTrue((self.repo / "build/package.sh").is_file())
        self.assert_no_build_or_install()

    def test_stage_cannot_delete_tracked_source_subtree(self):
        header = self.repo / "includes/public.h"
        before = header.read_bytes()
        run = self.run_command(self.package_command(f"--stage-dir={header.parent}"), expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertTrue(header.is_file(), "packaging deleted a tracked source subtree")
        self.assertEqual(header.read_bytes(), before)
        self.assert_no_build_or_install()

    def test_stage_cannot_delete_build_scripts(self):
        script = self.repo / "build/package.sh"
        before = script.read_bytes()
        run = self.run_command(self.package_command(f"--stage-dir={script.parent}"), expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertTrue(script.is_file(), "packaging deleted its own build-script directory")
        self.assertEqual(script.read_bytes(), before)
        self.assert_no_build_or_install()

    def test_wrong_runtime_metadata_stops_before_removing_stage(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "owned preexisting staging marker\n")
        self.make_infer_fixture(10)
        run = self.run_command(self.package_command(), expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("TensorRT 11", run.stdout)
        self.assertTrue(marker.is_file())
        self.make_infer_fixture(11)
        inference = self.runtime / "lib/libnvds_infer.so"
        elf = bytearray(inference.read_bytes())
        # Fixture compiler is native x86_64. Change e_machine to AArch64 only.
        elf[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(elf)
        run = self.run_command(self.package_command(), expected=None)
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("TensorRT 11", run.stdout)
        self.assertTrue(marker.is_file())
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("dpkg-deb"), "requires rsync and dpkg-deb")
    def test_runtime_bytes_symlinks_and_tracked_source_policy(self):
        self.run_command(self.package_command("--format=both", "--keep-stage"))
        package = self.output / "deepstream_sdk_v9.2.0_x86_64.tbz2"
        with tarfile.open(package, "r:bz2") as archive:
            for path in ("bin/deepstream-fixture", "lib/libfixture.so.1", "lib/libnvds_infer.so", "samples/models/model.bin",
                         "service-maker/lib/libfixture-service.so", "LicenseAgreement.pdf"):
                self.assertEqual(archive.extractfile(f"{SDK_PATH}/{path}").read(),
                                 (self.runtime / path).read_bytes())
            self.assertEqual(archive.getmember(f"{SDK_PATH}/lib/libfixture.so").linkname, "libfixture.so.1")
            self.assertEqual(archive.extractfile(f"{SDK_PATH}/sources/libs/public_source.cpp").read(),
                             b"// tracked fixture, current worktree\n")
            names = archive.getnames()
            self.assertNotIn(f"{SDK_PATH}/sources/libs/new_source.cpp", names)
            self.assertNotIn(f"{SDK_PATH}/sources/libs/ignored.so", names)
            self.assertNotIn(f"{SDK_PATH}/sources/libs/.build/ignored.o", names)
            self.assertIn(f"{SDK_PATH}/samples/modelopt/README.md", names)
        deb = self.output / "deepstream-9.2_9.2.0-1_amd64.deb"
        metadata = self.run_command(["dpkg-deb", "-f", str(deb), "Architecture", "Depends"]).stdout
        self.assertIn("amd64", metadata)
        self.assertIn("libnvinfer11", metadata)
        self.assertNotIn("libnvinfer10", metadata)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_driver_packages_untracked_sources_without_building(self):
        library = self.runtime / "lib/libfixture.so.1"
        before = hashlib.sha256(library.read_bytes()).hexdigest()
        self.run_command(["bash", str(self.repo / "build/build.sh"),
                          f"--package-from={self.runtime}", "--package-platform=x86",
                          "--package-format=tar", "--include-untracked-sources"])
        package = self.repo / "build/deepstream_sdk_v9.2.0_x86_64.tbz2"
        with tarfile.open(package, "r:bz2") as archive:
            names = archive.getnames()
            self.assertIn(f"{SDK_PATH}/sources/libs/new_source.cpp", names)
            self.assertNotIn(f"{SDK_PATH}/sources/libs/ignored.so", names)
        self.assertEqual(hashlib.sha256(library.read_bytes()).hexdigest(), before)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_driver_forwards_arm_platform_architecture(self):
        # Metadata-only fixture: changing e_machine is not an ARM build/run.
        inference = self.runtime / "lib/libnvds_infer.so"
        elf = bytearray(inference.read_bytes())
        elf[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(elf)
        for target in ("jetson", "sbsa"):
            with self.subTest(target=target):
                self.run_command(["bash", str(self.repo / "build/build.sh"),
                                  f"--package-from={self.runtime}", f"--package-platform={target}",
                                  "--package-format=tar"])
                package = self.repo / f"build/deepstream_sdk_v9.2.0_{target}.tbz2"
                self.assertTrue(package.is_file())
                with tarfile.open(package, "r:bz2") as archive:
                    self.assertEqual(archive.extractfile(f"{SDK_PATH}/lib/libnvds_infer.so").read(), bytes(elf))
        self.assert_no_build_or_install()


if __name__ == "__main__":
    unittest.main()
