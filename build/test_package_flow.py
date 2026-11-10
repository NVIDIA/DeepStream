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
import re
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
        for name in ("NVDS_VERSION", "PKG_VERSION", "CUDA_VER", "TRT_VER", "CUDNN_VERSION",
                     "SKIP_SPARSE4D_OCR", "DEEPSTREAM_VARIANT", "PLATFORM",
                     "TRITON_PKG_PATH", "TRITON_CLIENT_SDK_PKG_PATH", "BASH_ENV", "ENV",
                     "TRT_INCLUDE_DIR", "CC"):
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
            "src/gst-plugins/gst-nvdsvisionencoder-c/Makefile":
                (BUILD_DIR.parent / "src/gst-plugins/gst-nvdsvisionencoder-c/Makefile").read_text(),
            "src/utils/nvdsinfer/nvdsinfer_model_builder.h": "/* inference header fixture */\n",
            "src/utils/nvdsinfer/nvdsinfer_onnx_paths.h":
                (BUILD_DIR.parent / "src/utils/nvdsinfer/nvdsinfer_onnx_paths.h").read_text(),
            "src/utils/nvdsinferutils/include/nvdsinfer_tlt.h": "/* TLT header fixture */\n",
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
            files[f"scripts/{helper}"] = "#!/bin/bash\nset -e\nexit 98 # package only\n"
        files["scripts/triton_backend_setup.sh"] = (
            BUILD_DIR.parent / "scripts/triton_backend_setup.sh").read_text()
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

    def test_runtime_directory_symlinks_rejected_before_staging_changes(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "owned staging must remain unchanged on rejected input\n")
        self.write(Path(str(self.stage) + ".deepstream-package-owner"), str(self.repo.resolve()) + "\n")
        for relative in ("bin", "lib", "samples", "service-maker", "samples/configs/deepstream-app",
                         "sources", "sources/sparse4d", "sources/sparse4d/configs"):
            with self.subTest(relative=relative):
                original = self.runtime / relative
                original.mkdir(parents=True, exist_ok=True)
                external = self.root / ("external-" + relative.replace("/", "-"))
                original.rename(external)
                original.symlink_to(external, target_is_directory=True)
                sentinel = external / "LicenseAgreement.pdf"
                self.write(sentinel, "runtime input must not be modified\n")
                try:
                    result = self.run_command(self.package_command(), expected=None)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("directory symlink", result.stdout)
                    self.assertTrue(sentinel.is_file(), "input was modified through a stage symlink")
                    self.assertTrue(marker.is_file(), "rejected input discarded the existing stage")
                finally:
                    original.unlink()
                    external.rename(original)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_internal_sample_directory_aliases_package_without_input_changes(self):
        model = self.runtime / "samples/triton_model_repo/Primary_Detector/model.bin"
        self.write(model, b"model alias fixture\x00\xff")
        aliases = {
            "triton_tao_model_repo/Primary_Detector": "../triton_model_repo/Primary_Detector",
            "trtis_model_repo": "triton_model_repo",
            "another_model_repo": "triton_model_repo",
            "chained_model_repo": "another_model_repo",
        }
        for relative, target in aliases.items():
            link = self.runtime / "samples" / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=True)
        self.run_command(self.package_command("--keep-stage"))
        archive = next(self.output.glob("*.tbz2"))
        with tarfile.open(archive) as packaged:
            for relative, target in aliases.items():
                original = self.runtime / "samples" / relative
                staged = self.stage / SDK_PATH / "samples" / relative
                self.assertEqual(original.readlink(), Path(target))
                self.assertTrue(staged.is_symlink())
                expected = (self.stage / SDK_PATH / "samples/triton_model_repo" /
                            ("Primary_Detector" if relative.endswith("Primary_Detector") else ""))
                self.assertEqual(staged.resolve(), expected.resolve())
                entry = packaged.getmember(f"{SDK_PATH}/samples/{relative}")
                self.assertTrue(entry.issym())
                self.assertEqual(entry.linkname, str(staged.readlink()))
        self.assertEqual(model.read_bytes(), b"model alias fixture\x00\xff")
        self.assertTrue((self.stage / SDK_PATH / "lib/libfixture.so").is_symlink())
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_sample_alias_overlays_and_pruning_change_only_staging(self):
        configs = self.runtime / "samples/original-configs"
        foreign = configs / "source1_csi_dec_infer_resnet.txt"
        self.write(foreign, "input configuration must survive\n")
        alias = self.runtime / "samples/configs/deepstream-app"
        alias.parent.mkdir(parents=True)
        alias.symlink_to("../original-configs", target_is_directory=True)
        (self.runtime / "samples/modelopt").symlink_to("models", target_is_directory=True)
        self.run_command(self.package_command("--keep-stage"))
        staged = self.stage / SDK_PATH / "samples"
        self.assertFalse((staged / "configs/deepstream-app" / foreign.name).exists())
        self.assertEqual(foreign.read_text(), "input configuration must survive\n")
        self.assertTrue((staged / "modelopt/prepare_model.py").is_file())
        self.assertFalse((self.runtime / "samples/models/prepare_model.py").exists())
        self.assertEqual(alias.readlink(), Path("../original-configs"))
        self.assert_no_build_or_install()

    def test_unsafe_sample_directory_aliases_preserve_input_and_existing_stage(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "preserve existing stage\n")
        self.write(Path(str(self.stage) + ".deepstream-package-owner"), str(self.repo.resolve()) + "\n")
        inside = self.runtime / "samples/model-data"
        outside = self.root / "external-model-data"
        self.write(inside / "model.bin", b"inside model")
        self.write(outside / "model.bin", b"external model")
        link = self.runtime / "samples/alias"
        # The last target escapes to / and re-enters the original samples tree.
        # Canonical input containment alone would accept it, but cp -a would
        # preserve a route out of disposable staging back to the original SDK.
        reentry = "../" * (len(link.parent.parts) + 2) + str(inside).lstrip("/")
        targets = (str(inside), str(outside), "../../external-model-data", reentry,
                   "missing-directory", ".", "alias")
        for target in targets:
            with self.subTest(target=target):
                self.write(marker, "preserve existing stage\n")
                link.symlink_to(target, target_is_directory=True)
                try:
                    result = self.run_command(self.package_command(), expected=None)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertTrue(marker.is_file(), "rejected input discarded existing stage")
                    self.assertEqual((inside / "model.bin").read_bytes(), b"inside model")
                    self.assertEqual((outside / "model.bin").read_bytes(), b"external model")
                finally:
                    link.unlink()
        self.assert_no_build_or_install()

    def test_sample_directory_alias_graph_cycles_are_rejected(self):
        first = self.runtime / "samples/first/link"
        second = self.runtime / "samples/second/link"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.symlink_to("../second", target_is_directory=True)
        second.symlink_to("../first", target_is_directory=True)
        result = self.run_command(self.package_command(), expected=None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.stage.exists())
        self.assertFalse(self.output.exists())
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_runtime_root_symlink_alias_is_supported(self):
        alias = self.root / "runtime-root-alias"
        alias.symlink_to(self.runtime, target_is_directory=True)
        self.run_command(self.package_command(f"--runtime-root={alias}", "--keep-stage"))
        self.assertTrue((self.stage / SDK_PATH / "lib/libfixture.so").is_symlink())
        self.assertEqual((self.runtime / "lib/libfixture.so").readlink(), Path("libfixture.so.1"))
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_source_overlay_replaces_file_symlink_without_modifying_target(self):
        original = self.root / "external-service-source"
        self.write(original, "private runtime input must survive source overlay\n")
        destination = self.runtime / "service-maker/sources/core/Makefile"
        destination.parent.mkdir(parents=True)
        destination.symlink_to(original)
        self.run_command(self.package_command("--keep-stage"))
        self.assertEqual(original.read_text(), "private runtime input must survive source overlay\n")
        staged = self.stage / SDK_PATH / "service-maker/sources/core/Makefile"
        self.assertFalse(staged.is_symlink())
        self.assertIn("INC_DIR?=", staged.read_text())
        self.assert_no_build_or_install()

    def test_jetson_triton_setup_requires_both_providers_before_mutation(self):
        setup = BUILD_DIR.parent / "scripts/triton_backend_setup.sh"
        command = ["bash", "-c",
                   'mkdir() { echo PREPARATION_REACHED; return 97; }; '
                   'export -f mkdir; exec bash "$1"', "guarded-setup", str(setup)]
        server = "https://example.invalid/v2.74.0/server.zip"
        client = "https://example.invalid/v2.74.0/client.zip"
        for variant in ("jetson", "tegra", "aarch64"):
            self.env["DEEPSTREAM_VARIANT"] = variant
            for providers in ({}, {"TRITON_PKG_PATH": server},
                              {"TRITON_CLIENT_SDK_PKG_PATH": client},
                              {"TRITON_PKG_PATH": "", "TRITON_CLIENT_SDK_PKG_PATH": client}):
                with self.subTest(variant=variant, providers=providers):
                    self.env.pop("TRITON_PKG_PATH", None)
                    self.env.pop("TRITON_CLIENT_SDK_PKG_PATH", None)
                    self.env.update(providers)
                    result = self.run_command(command, expected=None)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("explicit", result.stdout)
                    self.assertNotIn("PREPARATION_REACHED", result.stdout)
        self.env.update(TRITON_PKG_PATH=server, TRITON_CLIENT_SDK_PKG_PATH=client)
        # A reviewed explicit pair reaches the first possible mutation, which
        # is intercepted; no installer, download or mkdir actually executes.
        result = self.run_command(command, expected=97)
        self.assertIn("PREPARATION_REACHED", result.stdout)
        self.env["TRITON_CLIENT_SDK_PKG_PATH"] = "https://example.invalid/v2.73.0/client.zip"
        result = self.run_command(command, expected=1)
        self.assertIn("versions must match", result.stdout)
        self.assertNotIn("PREPARATION_REACHED", result.stdout)
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
    def test_igx_legacy_package_keeps_selected_trt10_and_cuda13(self):
        self.env.pop("CUDA_VER", None)
        self.env["TRT_VER"] = "10.13.3.9"  # fixture selection, not a qualified IGX version
        self.make_infer_fixture(10)
        inference = self.runtime / "lib/libnvds_infer.so"
        metadata = bytearray(inference.read_bytes())
        metadata[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(metadata)  # metadata-only ARM fixture; never executed
        before = inference.read_bytes()
        self.run_command(self.package_command("--arch=arm64", "--platform=igx",
                                              "--format=both", "--keep-stage"))
        self.assertTrue((self.output / "deepstream_sdk_v9.2.0_igx.tbz2").is_file())
        deb = self.output / "deepstream-9.2_9.2.0-1_arm64.deb"
        depends = self.run_command(["dpkg-deb", "-f", str(deb), "Depends"]).stdout
        for name in ("libnvinfer10", "libnvinfer-dev", "libnvonnxparsers10",
                     "libnvonnxparsers-dev", "libnvinfer-plugin10", "libnvinfer-plugin-dev"):
            self.assertIn(name + " (>= 10.13.3.9)", depends)
            self.assertIn(name + " (<< 10.13.3.10)", depends)
        self.assertIn("cuda-cudart-13-0", depends)
        self.assertIn("libnvvpi4,", depends)
        self.assertNotIn("libnvvpi4 (", depends)
        self.assertNotIn("libnvinfer11", depends)
        self.assertNotIn("cuda-cudart-13-4", depends)
        self.assertEqual(inference.read_bytes(), before)
        setup = self.stage / SDK_PATH / "samples/triton_backend_setup.sh"
        command = ["bash", "-c",
                   'mkdir() { echo PREPARATION_REACHED; return 97; }; '
                   'export -f mkdir; exec bash "$1"', "guarded-igx-setup", str(setup)]
        result = self.run_command(command, expected=1)
        self.assertIn("IGX", result.stdout)
        self.assertIn("explicit", result.stdout)
        self.assertNotIn("PREPARATION_REACHED", result.stdout)
        # Fixture URLs test selection only; they are never fetched or qualified.
        self.env.update(TRITON_PKG_PATH="https://example.invalid/v2.70.0/server.zip",
                        TRITON_CLIENT_SDK_PKG_PATH="https://example.invalid/v2.70.0/client.zip")
        result = self.run_command(command, expected=97)
        self.assertIn("Explicit IGX", result.stdout)
        self.assertIn("PREPARATION_REACHED", result.stdout)
        self.env["TRITON_CLIENT_SDK_PKG_PATH"] = "https://example.invalid/v2.69.0/client.zip"
        result = self.run_command(command, expected=1)
        self.assertIn("versions must match", result.stdout)
        self.assertNotIn("PREPARATION_REACHED", result.stdout)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_offline_igx_driver_forwards_arm_architecture(self):
        self.env.pop("CUDA_VER", None)
        self.env["TRT_VER"] = "10.13.3.9"
        self.make_infer_fixture(10)
        inference = self.runtime / "lib/libnvds_infer.so"
        metadata = bytearray(inference.read_bytes())
        metadata[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(metadata)
        self.run_command(["bash", str(self.repo / "build/build.sh"),
                          f"--package-from={self.runtime}", "--package-platform=igx",
                          "--package-format=tar"])
        self.assertTrue((self.repo / "build/deepstream_sdk_v9.2.0_igx.tbz2").is_file())
        self.assert_no_build_or_install()

    def test_igx_rejects_unknown_version_or_wrong_runtime_without_changing_stage(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "preflight must not discard staging\n")
        for version, major, arch in ((None, 10, "arm64"), ("11.4.0.106", 11, "arm64"),
                                     ("10.13.3.9", 11, "arm64"), ("10.13.3.9", 10, "amd64")):
            with self.subTest(version=version, major=major, arch=arch):
                self.env.pop("TRT_VER", None)
                if version is not None:
                    self.env["TRT_VER"] = version
                self.make_infer_fixture(major)
                inference = self.runtime / "lib/libnvds_infer.so"
                metadata = bytearray(inference.read_bytes())
                metadata[18:20] = (183).to_bytes(2, "little")
                inference.write_bytes(metadata)
                result = self.run_command(self.package_command("--platform=igx", "--arch=" + arch),
                                          expected=None)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertTrue(marker.is_file())
        self.assert_no_build_or_install()

    def test_igx_header_version_reader_uses_selected_compiler_and_rejects_bad_macros(self):
        source = (BUILD_DIR / "package.sh").read_text()
        match = re.search(r"^read_trt_header_version\(\) \{\n.*?^\}", source, re.M | re.S)
        self.assertIsNotNone(match, "native IGX packaging needs an explicit header-version reader")
        include = self.root / "selected headers"
        header = include / "NvInferVersion.h"
        valid = "#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 13\n" \
                "#define NV_TENSORRT_PATCH 3\n#define NV_TENSORRT_BUILD 9\n"
        self.env["TRT_INCLUDE_DIR"] = str(include)
        self.env["CC"] = shutil.which("cc")
        self.write(header, valid)
        result = self.run_command(["bash", "-ec", match.group() + "\nread_trt_header_version"])
        self.assertEqual(result.stdout.strip(), "10.13.3.9")
        for invalid in (valid.replace("#define NV_TENSORRT_BUILD 9\n", ""),
                        valid.replace("NV_TENSORRT_BUILD 9", "NV_TENSORRT_BUILD nope")):
            self.write(header, invalid)
            result = self.run_command(["bash", "-ec", match.group() + "\nread_trt_header_version"], expected=None)
            self.assertNotEqual(result.returncode, 0, result.stdout)
        self.write(header, valid)
        self.env["CC"] = str(self.root / "absent compiler")
        result = self.run_command(["bash", "-ec", match.group() + "\nread_trt_header_version"], expected=None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assert_no_build_or_install()

    def test_native_packaging_forwards_selected_platform(self):
        source = (BUILD_DIR / "build.sh").read_text()
        match = re.search(r"^run_package\(\) \{\n.*?^\}", source, re.M | re.S)
        self.assertIsNotNone(match)
        for stack, target in (("x86", "x86"), ("aarch64", "jetson"),
                              ("sbsa", "sbsa"), ("igx", "igx")):
            with self.subTest(stack=stack):
                self.env.update(STACK_PLATFORM=stack, NVDS_FULL_VERSION="9.2.0",
                                TRT_VER="10.13.3.9" if stack == "igx" else "11.4.0.106",
                                PACKAGE_FORMAT="both", SCRIPT_DIR=str(BUILD_DIR))
                result = self.run_command(["bash", "-ec", match.group() +
                    '\nenv() { printf "ARG=%s\\n" "$@"; }\nrun_package'])
                self.assertIn("ARG=--platform=" + target + "\n", result.stdout)
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

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("dpkg-deb"), "requires rsync and dpkg-deb")
    def test_default_version_packages_921_in_92_install_tree(self):
        self.env.pop("NVDS_VERSION")
        self.run_command(self.package_command("--format=both", "--keep-stage"))
        package = self.output / "deepstream_sdk_v9.2.1_x86_64.tbz2"
        self.assertTrue(package.is_file(), "default tarball must use SDK version 9.2.1")
        with tarfile.open(package, "r:bz2") as archive:
            self.assertEqual(archive.extractfile(f"{SDK_PATH}/version").read().splitlines()[0],
                             b"Version: 9.2.1")
            self.assertEqual(archive.getmember("opt/nvidia/deepstream/deepstream").linkname,
                             "deepstream-9.2")
            self.assertFalse(any(name.startswith("opt/nvidia/deepstream/deepstream-9.2.1/")
                                 for name in archive.getnames()))
        deb = self.output / "deepstream-9.2_9.2.1-1_amd64.deb"
        self.assertTrue(deb.is_file(), "default Debian filename must use 9.2.1 and package family 9.2")
        self.assertEqual(self.run_command(["dpkg-deb", "-f", str(deb), "Package"]).stdout.strip(),
                         "deepstream-9.2")
        self.assertEqual(self.run_command(["dpkg-deb", "-f", str(deb), "Version"]).stdout.strip(),
                         "9.2.1-1")
        self.assertEqual((self.stage / SDK_PATH / "version").read_text().splitlines()[0],
                         "Version: 9.2.1")
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("dpkg-deb"), "requires rsync and dpkg-deb")
    def test_visionencoder_inference_dependencies_survive_packaging(self):
        relative = "sources/gst-plugins/gst-nvdsvisionencoder-c/Makefile"
        original = self.repo / "src/gst-plugins/gst-nvdsvisionencoder-c/Makefile"
        original_bytes = original.read_bytes()
        inference = self.runtime / "lib/libnvds_infer.so"
        original_elf = inference.read_bytes()
        self.env.pop("CUDA_VER", None)
        for target, arch, machine in (("x86", "amd64", 62), ("jetson", "arm64", 183),
                                      ("sbsa", "arm64", 183)):
            with self.subTest(platform=target):
                # Only the metadata fixture changes architecture. No ARM code
                # is compiled or executed by this package-layout check.
                metadata = bytearray(original_elf)
                metadata[18:20] = machine.to_bytes(2, "little")
                inference.write_bytes(metadata)
                self.run_command(self.package_command("--format=both", "--keep-stage",
                                                      "--arch=" + arch, "--platform=" + target))
                staged = self.stage / SDK_PATH / relative
                text = staged.read_text()
                self.assertIn("-I../../libs/nvdsinfer", text)
                self.assertIn("-I../../libs/nvdsinferutils/include", text)
                self.assertNotIn("../../utils/", text)
                self.assertIn("-lnvds_infer", text)
                for include in ("../../libs/nvdsinfer/nvdsinfer_model_builder.h",
                                "../../libs/nvdsinfer/nvdsinfer_onnx_paths.h",
                                "../../libs/nvdsinferutils/include/nvdsinfer_tlt.h"):
                    self.assertTrue((staged.parent / include).is_file(), include)
                suffix = {"x86": "x86_64", "jetson": "jetson", "sbsa": "sbsa"}[target]
                with tarfile.open(self.output / ("deepstream_sdk_v9.2.0_" + suffix + ".tbz2"),
                                  "r:bz2") as archive:
                    self.assertEqual(archive.extractfile(SDK_PATH + "/" + relative).read(),
                                     staged.read_bytes())
                    self.assertEqual(archive.extractfile(SDK_PATH +
                        "/sources/libs/nvdsinfer/nvdsinfer_onnx_paths.h").read(),
                        (BUILD_DIR.parent / "src/utils/nvdsinfer/nvdsinfer_onnx_paths.h").read_bytes())
                extracted = self.root / ("deb-extracted-" + target)
                self.run_command(["dpkg-deb", "-x", str(self.output / (
                    "deepstream-9.2_9.2.0-1_" + arch + ".deb")), str(extracted)])
                self.assertEqual((extracted / SDK_PATH / relative).read_bytes(), staged.read_bytes())
                self.assertEqual((extracted / SDK_PATH /
                    "sources/libs/nvdsinfer/nvdsinfer_onnx_paths.h").read_bytes(),
                    (BUILD_DIR.parent / "src/utils/nvdsinfer/nvdsinfer_onnx_paths.h").read_bytes())
                self.assertEqual(original.read_bytes(), original_bytes)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_driver_defaults_to_921_without_version_override(self):
        self.env.pop("NVDS_VERSION")
        self.run_command(["bash", str(self.repo / "build/build.sh"),
                          f"--package-from={self.runtime}", "--package-platform=x86",
                          "--package-format=tar"])
        package = self.repo / "build/deepstream_sdk_v9.2.1_x86_64.tbz2"
        self.assertTrue(package.is_file(), "build driver must forward its default SDK version 9.2.1")
        with tarfile.open(package, "r:bz2") as archive:
            self.assertEqual(archive.extractfile(f"{SDK_PATH}/version").read().splitlines()[0],
                             b"Version: 9.2.1")
            self.assertEqual(archive.getmember("opt/nvidia/deepstream/deepstream").linkname,
                             "deepstream-9.2")
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

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("dpkg-deb"), "requires rsync and dpkg-deb")
    def test_platform_debian_stack_defaults(self):
        self.env.pop("CUDA_VER", None)
        self.env.pop("TRT_VER", None)
        self.env.pop("SKIP_SPARSE4D_OCR", None)
        inference = self.runtime / "lib/libnvds_infer.so"
        native = inference.read_bytes()
        for target, arch, cuda, trt, upper in (
            ("x86", "amd64", "13-4", "11.4.0.106", "11.4.0.107"),
            ("jetson", "arm64", "13-2", "11.4.1.31", "11.4.1.32"),
            ("sbsa", "arm64", "13-4", "11.4.0.106", "11.4.0.107"),
        ):
            with self.subTest(target=target):
                metadata = bytearray(native)
                if arch == "arm64":
                    metadata[18:20] = (183).to_bytes(2, "little")
                inference.write_bytes(metadata)
                self.run_command(self.package_command("--format=deb", f"--arch={arch}",
                                                       f"--platform={target}"))
                package = self.output / f"deepstream-9.2_9.2.0-1_{arch}.deb"
                dependencies = self.run_command(["dpkg-deb", "-f", str(package), "Depends"]).stdout
                self.assertIn(f"cuda-cudart-{cuda},", dependencies)
                for library in ("libnvinfer11", "libnvinfer-dev", "libnvonnxparsers11",
                                "libnvonnxparsers-dev", "libnvinfer-plugin11", "libnvinfer-plugin-dev"):
                    self.assertIn(f"{library} (>= {trt})", dependencies)
                    self.assertIn(f"{library} (<< {upper})", dependencies)
        self.assert_no_build_or_install()

    def test_invalid_stack_selection_preserves_existing_stage(self):
        marker = self.stage / "must-survive.txt"
        self.write(marker, "existing unrelated content\n")
        for name, value in (("TRT_VER", "10.16.0.72"), ("TRT_VER", ""),
                            ("CUDA_VER", "invalid"), ("SKIP_SPARSE4D_OCR", "yes")):
            with self.subTest(name=name, value=value):
                previous = self.env.get(name)
                self.env[name] = value
                result = self.run_command(self.package_command(), expected=None)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(name, result.stdout)
                self.assertTrue(marker.is_file())
                if previous is None:
                    self.env.pop(name)
                else:
                    self.env[name] = previous
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_sbsa_packaged_setup_rejects_jetson_defaults_before_mutation(self):
        inference = self.runtime / "lib/libnvds_infer.so"
        metadata = bytearray(inference.read_bytes())
        metadata[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(metadata)
        self.env.pop("CUDA_VER", None)
        self.run_command(self.package_command("--arch=arm64", "--platform=sbsa", "--keep-stage"))
        setup = self.stage / SDK_PATH / "samples/triton_backend_setup.sh"
        # Intercept the first possible mutation if the early guard regresses.
        command = ['bash', '-c',
                   'mkdir() { echo mkdir >> "$PACKAGE_TEST_FORBIDDEN"; return 97; }; '
                   'export -f mkdir; exec bash "$1"', 'guarded-setup', str(setup)]
        result = self.run_command(command, expected=None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("SBSA", result.stdout)
        self.assertIn("26.10", result.stdout)
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync"), "requires rsync")
    def test_arm_reduced_ocr_survives_source_overlay_and_external_config_import(self):
        sparse = "src/gst-plugins/gst-nvdsvideotemplate/ds-sparse4d"
        configs = self.repo / sparse / "configs"
        self.write(configs / "config.yaml", "generate_2d_bbox: True\ngenerate_3d_bbox: True\n")
        for name in ("deepstream_test5_app_sparse4d.txt", "deepstream_test5_app_sparse4d_filesrc.txt",
                     "deepstream_test5_app_sparse4d_realtime.txt"):
            self.write(configs / name, "[primary-gie]\nenable=1\n[secondary-pre-process0]\nenable=1\n")
        setup = ('#!/bin/bash\nset -e\nCONFIG_FILE_PATH="$1"\ncp "$2" "$CONFIG_FILE_PATH"\n'
                 'OCR_INFERENCE=$(sed -n "s/generate_2d_bbox: //p" "$CONFIG_FILE_PATH")\n'
                 'if [[ "$OCR_INFERENCE" == True ]]; then touch "$3"; fi\n')
        self.write(configs / "sparse4d_setup.sh", setup, executable=True)
        for name in ("config_preprocess_ocr.txt", "config_ocr.yaml"):
            self.write(configs / name, "OCR-only fixture configuration\n")
        self.write(self.repo / sparse / "Makefile",
                   "INC:= -I../../../utils/nvdsinfer -I../../../utils/nvdsinferutils/include\n")
        self.write(self.repo / sparse / "README", "Synthetic Sparse4D source fixture\n")
        self.write(self.repo / "src/utils/nvmultiobjecttracker/Makefile",
                   "INC:= ../nvds_msgapi/mqtt_protocol_adaptor ../nvds_msgapi/common_src\n")
        self.run_command(["git", "add", "src"])
        original_yaml = (configs / "config.yaml").read_bytes()
        for name in ("libcustom_preprocess_ocr.so", "libnvocdr.so", "libmsda_fp16.so"):
            self.write(self.runtime / "sources/sparse4d" / name, b"opaque runtime fixture")
        inference = self.runtime / "lib/libnvds_infer.so"
        metadata = bytearray(inference.read_bytes())
        metadata[18:20] = (183).to_bytes(2, "little")
        inference.write_bytes(metadata)
        self.env.pop("CUDA_VER", None)
        self.run_command(self.package_command("--arch=arm64", "--platform=sbsa", "--keep-stage"))
        for relative in ("sources/sparse4d/configs",
                         "sources/gst-plugins/gst-nvdsvideotemplate/ds-sparse4d/configs"):
            root = self.stage / SDK_PATH / relative
            self.assertIn("generate_2d_bbox: False", (root / "config.yaml").read_text())
            self.assertIn("generate_3d_bbox: True", (root / "config.yaml").read_text())
            self.assertFalse((root / "config_ocr.yaml").exists())
            for config in root.glob("deepstream_test5_app_sparse4d*.txt"):
                text = config.read_text()
                self.assertIn("[primary-gie]\nenable=1", text)
                self.assertIn("[secondary-pre-process0]\nenable=0", text)
            imported = self.root / "imported.yaml"
            external = self.root / "external.yaml"
            ocr_action = self.root / "ocr-should-not-run"
            self.write(external, "generate_2d_bbox: True\n")
            self.run_command(["bash", str(root / "sparse4d_setup.sh"), str(imported), str(external), str(ocr_action)])
            self.assertIn("generate_2d_bbox: False", imported.read_text())
            self.assertFalse(ocr_action.exists())
            self.assertTrue((root / "sparse4d_setup.sh").stat().st_mode & 0o111)
        runtime = self.stage / SDK_PATH / "sources/sparse4d"
        self.assertFalse((runtime / "libnvocdr.so").exists())
        self.assertFalse((runtime / "libcustom_preprocess_ocr.so").exists())
        self.assertTrue((runtime / "libmsda_fp16.so").exists())
        self.assertTrue((self.runtime / "sources/sparse4d/libnvocdr.so").exists())
        self.assertEqual((configs / "config.yaml").read_bytes(), original_yaml)
        source = self.stage / SDK_PATH / "sources/gst-plugins/gst-nvdsvideotemplate/ds-sparse4d/Makefile"
        self.assertIn("../../../libs/nvdsinferutils", source.read_text())
        tracker = self.stage / SDK_PATH / "sources/libs/nvmultiobjecttracker/Makefile"
        self.assertIn("../nvds_msgapi_common_src", tracker.read_text())
        self.assert_no_build_or_install()

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("dpkg-deb"), "requires rsync and dpkg-deb")
    def test_explicit_stack_overrides_reach_package_metadata(self):
        self.env["CUDA_VER"] = "13.2"
        self.env["TRT_VER"] = "11.4.1.31"
        self.run_command(self.package_command("--format=deb"))
        package = self.output / "deepstream-9.2_9.2.0-1_amd64.deb"
        dependencies = self.run_command(["dpkg-deb", "-f", str(package), "Depends"]).stdout
        self.assertIn("cuda-cudart-13-2,", dependencies)
        self.assertIn("libnvinfer11 (>= 11.4.1.31)", dependencies)
        self.assertIn("libnvinfer11 (<< 11.4.1.32)", dependencies)
        self.assert_no_build_or_install()


@unittest.skipUnless(shutil.which("make"), "requires make for owned build fixtures")
class BuildSourcePurityTests(unittest.TestCase):
    """Execute real build entrypoints with local, non-installing tool fixtures."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ds-source-purity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.trace = self.root / "trace"
        self.env = os.environ.copy()
        for key in ("BASH_ENV", "ENV", "MAKEFLAGS", "MFLAGS", "MAKELEVEL", "CMAKE_ARGS",
                    "CUDA_VER", "PREBUILT_SO", "_PYTHON_HOST_PLATFORM", "ARCHFLAGS"):
            self.env.pop(key, None)
        self.env.update(PATH=str(self.tools) + os.pathsep + self.env["PATH"],
                        TMPDIR=str(self.scratch), PURITY_TRACE=str(self.trace))
        for command in ("sudo", "apt", "apt-get", "pip", "pip3", "curl", "wget", "cmake"):
            self.write(self.tools / command, "#!/bin/bash\nexit 97\n", executable=True)

    write = staticmethod(PackageFlowTests.write)

    def run_command(self, command, cwd=None, expected=0):
        run = subprocess.run(command, cwd=cwd or self.repo, env=self.env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        if expected is not None:
            self.assertEqual(run.returncode, expected, run.stdout)
        return run

    def make_python_fixture(self):
        python = self.repo / "src/service-maker/sources/python"
        python.mkdir(parents=True)
        shutil.copy2(BUILD_DIR.parent / "src/service-maker/sources/python/build.sh", python)
        self.project = python / "src"
        self.write(self.project / "pyservicemaker/module.py", "# source fixture\n")
        self.write(self.project / "bindings/source.h", "// binding fixture\n")
        self.write(self.project / "pyservicemaker.egg-info/PKG-INFO", "existing metadata\n")
        self.write(self.project / "build/stale.o", "existing build product\n")
        self.write(self.project / "dist/pyservicemaker-stale.whl", "stale wheel\n")
        self.write(self.project / "pyservicemaker/_pydeepstream.so", "stale extension\n")
        for headers in (self.repo / "src/service-maker/includes", self.repo / "includes"):
            self.write(headers / "fixture.h", "// matching checkout headers\n")
        self.env["PURITY_PROJECT"] = str(self.project)
        self.env["PURITY_SM_INCLUDE"] = str(self.repo / "src/service-maker/includes")
        self.env["PURITY_DS_INCLUDE"] = str(self.repo / "includes")
        interpreter = '''#!/bin/bash
set -e
printf '%s\\n' "$PWD" >> "$PURITY_TRACE"
[[ "$1 $2 $3" == '-m build --outdir' ]]
[[ "$PWD" != "$PURITY_PROJECT" ]]
cmp pyservicemaker/module.py "$PURITY_PROJECT/pyservicemaker/module.py"
cmp bindings/source.h "$PURITY_PROJECT/bindings/source.h"
[[ ! -e pyservicemaker.egg-info && ! -e build && ! -e dist ]]
[[ ! -e pyservicemaker/_pydeepstream.so ]]
[[ "$CMAKE_ARGS" == *"-DSM_INCLUDE_DIR=$PURITY_SM_INCLUDE"* ]]
[[ "$CMAKE_ARGS" == *"-DDS_INCLUDE_DIR=$PURITY_DS_INCLUDE"* ]]
[[ -f "$PURITY_SM_INCLUDE/fixture.h" && -f "$PURITY_DS_INCLUDE/fixture.h" ]]
if [[ "${_PYTHON_HOST_PLATFORM:-}" == linux-aarch64 ]]; then
    [[ "$CMAKE_ARGS" == *"-DCMAKE_TOOLCHAIN_FILE="* ]]
    [[ -f "${CMAKE_ARGS##*-DCMAKE_TOOLCHAIN_FILE=}" ]]
fi
mkdir -p pyservicemaker.egg-info build
printf 'new metadata\\n' > pyservicemaker.egg-info/PKG-INFO
if [[ "${PURITY_BUILD_STATUS:-0}" != 0 ]]; then exit "$PURITY_BUILD_STATUS"; fi
for ((i=0; i<${PURITY_WHEEL_COUNT:-1}; i++)); do
    printf 'fresh wheel %s\\n' "$i" > "$4/pyservicemaker-$i.whl"
done
'''
        for name in ("python3", "python3.12"):
            self.write(self.tools / name, interpreter, executable=True)
        self.output = self.root / "wheels"
        self.output.mkdir()
        return ["bash", str(python / "build.sh"), "--output-dir", str(self.output)]

    def assert_python_sources_unchanged(self):
        self.assertEqual((self.project / "pyservicemaker.egg-info/PKG-INFO").read_text(),
                         "existing metadata\n")
        self.assertEqual((self.project / "build/stale.o").read_text(), "existing build product\n")
        self.assertEqual((self.project / "dist/pyservicemaker-stale.whl").read_text(), "stale wheel\n")
        self.assertEqual((self.project / "pyservicemaker/_pydeepstream.so").read_text(), "stale extension\n")
        self.assertEqual(list(self.scratch.iterdir()), [], "private build must be cleaned")

    def test_python_wheel_build_uses_private_source_and_matching_headers(self):
        command = self.make_python_fixture()
        for options in ([], ["--minor-ver", "12", "--arch", "aarch64"]):
            with self.subTest(options=options):
                self.run_command(command + options)
                self.assertEqual((self.output / "pyservicemaker-0.whl").read_text(), "fresh wheel 0\n")
                self.assert_python_sources_unchanged()

    def test_python_wheel_failure_preserves_status_sources_and_existing_output(self):
        command = self.make_python_fixture()
        self.env["PURITY_BUILD_STATUS"] = "7"
        self.write(self.output / "old.whl", "existing output\n")
        self.run_command(command, expected=7)
        self.assertEqual((self.output / "old.whl").read_text(), "existing output\n")
        self.assert_python_sources_unchanged()

    def test_python_wheel_requires_exactly_one_fresh_output(self):
        command = self.make_python_fixture()
        for count in ("0", "2"):
            with self.subTest(count=count):
                self.env["PURITY_WHEEL_COUNT"] = count
                self.run_command(command, expected=1)
                self.assertEqual(list(self.output.iterdir()), [])
                self.assert_python_sources_unchanged()

    def make_nvocdr_fixture(self):
        source = BUILD_DIR.parent / "src/apps/tao_apps/apps/tao_others/deepstream-nvocdr-app"
        app = self.repo / "app"
        app.mkdir()
        for name in ("Makefile", "Makefile.nvocdr"):
            shutil.copy2(source / name, app / name)
        self.write(app / "deepstream_nvocdr_app.cpp", "// app fixture\n")
        self.write(self.repo / "common/ds_yml_parse.cpp", "// common fixture\n")
        self.write(app / "nvocdr/Makefile",
                   '.PHONY: all clean\nall:\n\t@echo core-all >> "$$PURITY_TRACE"\n'
                   '\t@touch libnvocdr.so\nclean:\n\t@echo core-clean >> "$$PURITY_TRACE"\n'
                   '\t@rm -f libnvocdr.so\n')
        self.original_makefile = app / "nvocdr/deepstream/Makefile"
        self.write(self.original_makefile, "$(error tracked submodule Makefile must stay unused)\n")
        self.write(app / "nvocdr/deepstream/nvocdrlib_impl.cpp", "// wrapper fixture\n")
        self.write(self.tools / "pkg-config", "#!/bin/bash\nexit 0\n", executable=True)
        self.write(self.tools / "gcc", "#!/bin/bash\necho x86_64-linux-gnu\n", executable=True)
        self.write(self.tools / "g++", '''#!/bin/bash
set -e
printf 'compile %s\\n' "$*" >> "$PURITY_TRACE"
while [[ $# -gt 0 ]]; do
    if [[ "$1" == -o ]]; then printf 'fixture binary\\n' > "$2"; exit 0; fi
    shift
done
exit 96
''', executable=True)
        return app

    def test_nvocdr_all_and_install_revisit_libraries_without_source_copy(self):
        app = self.make_nvocdr_fixture()
        original = self.original_makefile.read_bytes()
        self.run_command(["make", "-j4", "CUDA_VER=13.4"], cwd=app)
        wrapper = app / "nvocdr/deepstream/libnvocdr_impl.so"
        self.assertTrue(wrapper.is_file())
        wrapper.unlink()  # Remove only this fixture's generated output.
        self.run_command(["make", "CUDA_VER=13.4"], cwd=app)
        self.assertTrue(wrapper.is_file(), "current app must not skip missing library rebuild")
        self.assertEqual(self.trace.read_text().count("core-all\n"), 2)
        libraries, binaries = self.root / "lib", self.root / "bin"
        libraries.mkdir()
        binaries.mkdir()
        self.run_command(["make", "install", "CUDA_VER=13.4",
                          "LIB_INSTALL_DIR=" + str(libraries), "APP_INSTALL_DIR=" + str(binaries)], cwd=app)
        for name in ("libnvocdr.so", "libnvocdr_impl.so"):
            self.assertTrue((libraries / name).is_file())
        self.assertTrue((binaries / "deepstream-nvocdr-app").is_file())
        self.assertEqual(self.original_makefile.read_bytes(), original)

    def test_nvocdr_provider_edits_recompile_and_clean_needs_no_cuda(self):
        app = self.make_nvocdr_fixture()
        wrapper = app / "nvocdr/deepstream"
        provider = app / "Makefile.nvocdr"
        self.run_command(["make", "-f", "../../Makefile.nvocdr", "CUDA_VER=13.4"], cwd=wrapper)
        # Fixed fixture mtimes isolate the selected provider from the unused Makefile.
        for path in (self.original_makefile, wrapper / "nvocdrlib_impl.cpp"):
            os.utime(path, (1, 1))
        for path in (wrapper / "nvocdrlib_impl.o", wrapper / "libnvocdr_impl.so"):
            os.utime(path, (2, 2))
        os.utime(provider, (3, 3))
        previous = self.trace.read_text()
        self.run_command(["make", "-f", "../../Makefile.nvocdr", "CUDA_VER=13.4"], cwd=wrapper)
        self.assertGreater(self.trace.read_text().count("compile "), previous.count("compile "))
        self.run_command(["make", "clean"], cwd=app)
        self.assertFalse((wrapper / "libnvocdr_impl.so").exists())
        self.assertFalse((wrapper / "nvocdrlib_impl.o").exists())
        self.assertIn("core-clean\n", self.trace.read_text())
        self.assertIn("must stay unused", self.original_makefile.read_text())

    def test_tao_driver_stage_does_not_overwrite_submodule_makefile(self):
        source = (BUILD_DIR / "build.sh").read_text()
        stage = source.split("if begin_stage tao_apps; then\n", 1)[1].split(
            "# Stage: reference_apps", 1)[0]
        script = '''
begin_stage() { return 0; }
git() { printf ' initialized submodule\\n'; }
run_as_root() { echo 'unexpected privileged source mutation' >&2; return 98; }
run_submake() { printf '%s\\n' "$*" >> "$PURITY_TRACE"; }
stage_had_failures() { return 1; }
finish_stage() { :; }
FAILED_BUILDS=()
MK=()
if begin_stage tao_apps; then
''' + stage
        self.run_command(["bash", "-ec", script])
        self.assertIn("-C src/apps/tao_apps", self.trace.read_text())


if __name__ == "__main__":
    unittest.main()
