#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e
set -u
set -o pipefail

# Release selection is not platform qualification. An AArch64 early-access ZIP
# is not automatically a JetPack7.2.2/TRT11.4.1 provider.
readonly TRITON_REQUIRED_VERSION=2.74.0
TRITON_VERSION=$TRITON_REQUIRED_VERSION
TRITON_EXPECTED_TRT=11
TRITON_PKG_PATH=${TRITON_PKG_PATH:-}
TRITON_CLIENT_SDK_PKG_PATH=${TRITON_CLIENT_SDK_PKG_PATH:-}
DEEPSTREAM_LIB_DIR=${DEEPSTREAM_LIB_DIR:-/opt/nvidia/deepstream/deepstream/lib}
TRITON_INSTALL_DIR=${TRITON_INSTALL_DIR:-/opt/tritonserver}

fail() { echo "Error: $*" >&2; exit 1; }

case "${1:-}" in
    --help)
        echo "Jetson Triton26.10/${TRITON_REQUIRED_VERSION} native setup; exact JetPack qualification is pending."
        echo "Set TRITON_PKG_PATH and TRITON_CLIENT_SDK_PKG_PATH to reviewed matching HTTPS ZIP URLs."
        echo "Jetson URLs must contain /v${TRITON_REQUIRED_VERSION}/; no default or older fallback is selected."
        echo "DEEPSTREAM_VARIANT=igx uses the Jetson Triton and TensorRT 11 selection."
        echo "SBSA/DGX Spark uses Triton26.04/2.68.0 in containers; native apply is refused."
        echo "--check-only checks URL selection only, without downloads or system changes."
        echo "Provision the matching platform dependencies and stop DeepStream/Triton before installation."
        exit 0 ;;
    ''|--check-only) ;;
    *) fail "Unknown option: $1" ;;
esac
[[ $# -le 1 ]] || fail 'Expected at most one option.'

# Keep IGX as its own variant. Its compute stack matches Jetson.
TRITON_SETUP_VARIANT=${DEEPSTREAM_VARIANT:-${PLATFORM:-}}
case "$TRITON_SETUP_VARIANT" in
    jetson|tegra|aarch64) TRITON_SETUP_VARIANT=jetson ;;
    igx) ;;
    sbsa) ;;
    '')
        TRITON_SETUP_VARIANT=jetson
        if [[ "$(uname -m)" == aarch64 && ! -f /etc/nv_tegra_release ]]; then
            TRITON_SETUP_VARIANT=sbsa
        fi ;;
    *) fail 'Select DEEPSTREAM_VARIANT=jetson|tegra|aarch64|sbsa|igx for ARM backend setup.' ;;
esac
if [[ "$TRITON_SETUP_VARIANT" == sbsa && "${1:-}" != --check-only ]]; then
    fail 'SBSA/DGX Spark is container-only; native Triton setup is not supported. Use --check-only for selection inspection.'
fi

TRITON_SELECTION_VERSION=$TRITON_REQUIRED_VERSION
if [[ "$TRITON_SETUP_VARIANT" == sbsa ]]; then
    TRITON_SELECTION_VERSION=2.68.0
fi

for selection in TRITON_PKG_PATH TRITON_CLIENT_SDK_PKG_PATH; do
    value=${!selection}
    [[ -n "$value" ]] || fail "An explicit matching ${selection} is required; no default archive is selected."
    [[ "$value" =~ ^https://[^?#[:space:]]+/v([0-9]+\.[0-9]+\.[0-9]+)/[[:alnum:]_.+-]+\.zip(\?[^#[:space:]]*)?$ ]] ||
        fail "${selection} must select a reviewed /vMAJOR.MINOR.PATCH/ HTTPS ZIP."
    selected_version=${BASH_REMATCH[1]}
    if [[ "$selected_version" != "$TRITON_SELECTION_VERSION" ]]; then
        fail "${TRITON_SETUP_VARIANT} selection targets Triton ${TRITON_SELECTION_VERSION}, not ${selected_version}."
    fi
    if [[ "$selection" == TRITON_PKG_PATH ]]; then
        TRITON_VERSION=$selected_version
    else
        [[ "$selected_version" == "$TRITON_VERSION" ]] || fail 'Server and client SDK archive versions must match.'
    fi
done
[[ "$TRITON_PKG_PATH" != "$TRITON_CLIENT_SDK_PKG_PATH" ]] || fail 'Select separate server and client SDK archives.'
echo "Selected Triton ${TRITON_VERSION} for ${TRITON_SETUP_VARIANT}; exact target-stack compatibility must be qualified separately."
[[ "${1:-}" != --check-only ]] || exit 0

[[ $EUID -eq 0 ]] || fail 'Must be run as root or sudo.'
[[ "$(uname -m)" == aarch64 ]] || fail 'This packaged helper is for native ARM targets, not x86.'
[[ -d "$DEEPSTREAM_LIB_DIR" ]] || fail 'Install the matching DeepStream SDK before setting up Triton.'
[[ "$TRITON_INSTALL_DIR" == /*/tritonserver ]] || fail 'TRITON_INSTALL_DIR must be an absolute tritonserver directory.'
[[ ! -L "$TRITON_INSTALL_DIR" ]] || fail 'Use a real tritonserver directory, not a destination symlink.'
[[ "$DEEPSTREAM_LIB_DIR" == /*/lib || "$DEEPSTREAM_LIB_DIR" == /*/lib64 ]] || fail 'DEEPSTREAM_LIB_DIR must be an absolute SDK library directory.'
for dependency in python3 wget readelf ldd ldconfig gst-inspect-1.0; do
    command -v "$dependency" >/dev/null || fail "Missing prerequisite ${dependency}; provision the selected platform dependencies first."
done

# Never build legacy OpenSSL/Boost, add CUDA repositories or pin an older DCGM
# here. A qualified 26.10 provider needs its own matching dependency closure.
TRITON_DOWNLOADS=$(mktemp -d /tmp/deepstream-triton-${TRITON_VERSION}.XXXXXX)
replacement=
complete=0
rollback_targets=()
rollback_backups=()
pending_files=()
cleanup() {
    local status=$? index target backup
    trap - EXIT
    set +e
    if [[ "$complete" != 1 && -n "$replacement" ]]; then
        # Restore already-published components in reverse order. Keep failed new
        # components and the recovery directory instead of deleting their data.
        for ((index=${#rollback_targets[@]}-1; index>=0; index--)); do
            target=${rollback_targets[index]}
            backup=${rollback_backups[index]}
            if [[ -e "$target" || -L "$target" ]]; then
                mv -- "$target" "$replacement/failed-$index" || {
                    echo "Error: recovery needs manual attention in $replacement" >&2
                    continue
                }
            fi
            if [[ -e "$backup" || -L "$backup" ]]; then
                mv -- "$backup" "$target" || echo "Error: restore $target from $replacement" >&2
            fi
        done
        echo "Triton setup did not complete; inspect retained recovery files in $replacement" >&2
    fi
    for target in "${pending_files[@]}"; do rm -f -- "$target"; done
    rm -rf -- "$TRITON_DOWNLOADS"
    exit "$status"
}
trap cleanup EXIT
chmod 700 "$TRITON_DOWNLOADS"
wget -q --https-only --timeout=60 --tries=2 -O "$TRITON_DOWNLOADS/server.zip" -- "$TRITON_PKG_PATH"
wget -q --https-only --timeout=60 --tries=2 -O "$TRITON_DOWNLOADS/client.zip" -- "$TRITON_CLIENT_SDK_PKG_PATH"

# Validate/extract in a private workspace. Never execute the downloaded server.
python3 -I -B - "$TRITON_DOWNLOADS" "$TRITON_VERSION" "$TRITON_EXPECTED_TRT" > "$TRITON_DOWNLOADS/layout" <<'PY'
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import zipfile

work = Path(sys.argv[1])
version = sys.argv[2]
trt_abi = sys.argv[3]

def extract(name):
    root = work / name
    root.mkdir()
    links = []
    with zipfile.ZipFile(work / (name + '.zip')) as archive:
        seen = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or any(ord(c) < 32 for c in member.filename) or str(path) in seen):
                raise ValueError('Unsafe or duplicate archive member')
            seen.add(str(path))
            destination = root.joinpath(*path.parts)
            mode = member.external_attr >> 16
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISLNK(mode):
                links.append((destination, archive.read(member).decode('utf-8')))
            else:
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError('Unsupported archive member type')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
                destination.chmod(0o755 if mode & 0o111 else 0o644)
    for destination, target in links:
        if Path(target).is_absolute():
            raise ValueError('Absolute archive symlink')
        destination.parent.resolve().relative_to(root.resolve())
        (destination.parent / target).resolve().relative_to(root.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
    for destination, _ in links:
        destination.resolve(strict=True).relative_to(root.resolve())
    return root

def unique(candidates, description):
    paths = {p.resolve(strict=True) for p in candidates if p.is_file()}
    if len(paths) != 1:
        raise ValueError('Expected exactly one ' + description)
    return paths.pop()

def elf(path, executable=False):
    with path.open('rb') as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:6] != b'\x7fELF\x02\x01'
            or struct.unpack_from('<H', header, 18)[0] != 183
            or struct.unpack_from('<H', header, 16)[0] not in ((2, 3) if executable else (3,))):
        raise ValueError('Expected a genuine ELF64/AArch64 binary: ' + path.name)

try:
    server = extract('server')
    client = extract('client')
    binary = unique(server.glob('**/bin/tritonserver'), 'server executable')
    root = binary.parent.parent
    library = unique(root.glob('**/libtritonserver.so*'), 'server library')
    backend = root / 'backends/tensorrt/libtriton_tensorrt.so'
    grpc = unique(client.glob('**/libgrpcclient.so*'), 'client runtime')
    for archive, description in ((server, 'Server'), (client, 'Client SDK')):
        markers = [p for p in archive.rglob('TRITON_VERSION') if p.is_file()]
        if not markers or any(p.read_text().strip() != version for p in markers):
            raise ValueError(description + ' archive lacks the selected TRITON_VERSION marker')
    # Only these subtrees are published. A link into an archive-only sibling
    # (for example ../../lib/backend.so) would become dangling after copying.
    # Keep relative links within their own component, never between components.
    for component in ('bin', 'backends'):
        directory = root / component
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('Expected a real provider directory: ' + component)
        for path in directory.rglob('*'):
            if path.is_symlink():
                path.resolve(strict=True).relative_to(directory)
    elf(binary, executable=True)
    for path in (library, backend, grpc):
        elf(path)
    dynamic = subprocess.check_output(['readelf', '-d', str(backend)], text=True)
    needed = re.findall(r'Shared library: \[([^\]]+)\]', dynamic)
    if f'libnvinfer.so.{trt_abi}' not in needed or 'libcudart.so.13' not in needed:
        raise ValueError(f'TensorRT backend must target TensorRT{trt_abi}/CUDA13')
    for path in (root, library, grpc):
        print(path)
except (OSError, UnicodeError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
    raise SystemExit('Error: invalid Triton provider: ' + str(error))
PY
mapfile -t layout < "$TRITON_DOWNLOADS/layout"
[[ ${#layout[@]} -eq 3 ]] || fail 'Invalid provider layout.'
TRITON_ROOT_DIR=${layout[0]}
TRITON_SERVER_LIBRARY=${layout[1]}
TRITON_CLIENT_LIBRARY=${layout[2]}

# Inspect the native dependency closure without starting the server. Inputs
# resolved inside the download tree are not installed by this minimal helper.
for library in "$TRITON_SERVER_LIBRARY" "$TRITON_CLIENT_LIBRARY" \
        "$TRITON_ROOT_DIR/bin/tritonserver" "$TRITON_ROOT_DIR/backends/tensorrt/libtriton_tensorrt.so"; do
    if ! dependencies=$(LD_LIBRARY_PATH="$DEEPSTREAM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$library" 2>&1); then
        fail 'Cannot inspect provider dependencies; no runtime files were replaced.'
    fi
    if [[ "$dependencies" == *'not found'* || "$dependencies" == *"$TRITON_DOWNLOADS"* ]]; then
        fail 'Provider dependencies are missing or confined to the download tree; provision the reviewed native dependency closure first.'
    fi
done

# Stage the full replacement before publishing; preserve existing components
# for recovery, and do not merge stale 26.04 backends into the new set.
mkdir -p "$TRITON_INSTALL_DIR"
replacement=$(mktemp -d "$TRITON_INSTALL_DIR/.triton-${TRITON_VERSION}.XXXXXX")
cp -a "$TRITON_ROOT_DIR/bin" "$replacement/bin"
cp -a "$TRITON_ROOT_DIR/backends" "$replacement/backends"
chmod 0755 "$replacement/bin/tritonserver"
# Prepare both SDK files before publishing any server/backend changes.
for component in libtritonserver.so libgrpcclient.so; do
    source=$TRITON_SERVER_LIBRARY
    [[ "$component" != libgrpcclient.so ]] || source=$TRITON_CLIENT_LIBRARY
    if [[ -e "$DEEPSTREAM_LIB_DIR/$component" || -L "$DEEPSTREAM_LIB_DIR/$component" ]]; then
        cp -a -- "$DEEPSTREAM_LIB_DIR/$component" "$replacement/previous-$component"
    fi
    temporary=$(mktemp "$DEEPSTREAM_LIB_DIR/.${component}.XXXXXX")
    pending_files+=("$temporary")
    install -m 0644 -- "$source" "$temporary"
done
for component in bin backends; do
    backup="$replacement/previous-$component"
    if [[ -e "$TRITON_INSTALL_DIR/$component" || -L "$TRITON_INSTALL_DIR/$component" ]]; then
        mv -- "$TRITON_INSTALL_DIR/$component" "$backup"
    fi
    rollback_targets+=("$TRITON_INSTALL_DIR/$component")
    rollback_backups+=("$backup")
    mv -- "$replacement/$component" "$TRITON_INSTALL_DIR/$component"
done
index=0
for component in libtritonserver.so libgrpcclient.so; do
    rollback_targets+=("$DEEPSTREAM_LIB_DIR/$component")
    rollback_backups+=("$replacement/previous-$component")
    mv -fT -- "${pending_files[index]}" "$DEEPSTREAM_LIB_DIR/$component"
    index=$((index + 1))
done
ldconfig
if ! gst-inspect-1.0 nvinferserver >/dev/null; then
    fail 'nvinferserver is not loadable after Triton backend setup.'
fi
complete=1
echo "Installed selected Triton ${TRITON_VERSION} server/backends and fresh client runtime."
echo "Previous components, if any, are retained in: $replacement"
echo 'Installation is not platform/inference qualification. Restart and qualify the selected native workload.'
