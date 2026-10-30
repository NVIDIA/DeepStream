<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMC Service Setup

Use this workflow when the user asks to launch, deploy, verify, stop, or inspect AutoMagicCalib.

## Inputs

Collect or infer:

- `<DEEPSTREAM_REPO>`: DeepStream checkout root
- `$AMC_ROOT`: `$DEEPSTREAM_REPO/tools/auto-magic-calib`
- `$AMC_COMPOSE`: `$AMC_ROOT/compose/compose.yml`
- `$AMC_RUNTIME_CONFIG`: non-secret port/host configuration, default `$DEEPSTREAM_REPO/data/amc/amc-runtime.conf`
- backend port, default `8000`, and UI port, default `5000`
- host network IP for `HOST_IP`
- Ubuntu 24.04 on an x86_64 host, NVIDIA Driver 590, and an NVIDIA GPU with NVENC support
- Docker configured for non-root use and NVIDIA Container Toolkit
- NGC access for the images declared by the active compose file
- HuggingFace access when VGGT is requested, including multi-camera tuning

Do not use compose files from another repository. If `$AMC_COMPOSE` is missing, report the missing path and stop.

## Preflight

Docker must run without `sudo`:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
export AMC_COMPOSE="$AMC_ROOT/compose/compose.yml"
test -f "$AMC_COMPOSE" || { echo "ERROR: missing $AMC_COMPOSE"; exit 1; }
docker ps
test "$(uname -m)" = x86_64 || { echo "ERROR: AMC requires an x86_64 host"; exit 1; }
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 24.04 || { echo "ERROR: AMC requires Ubuntu 24.04"; exit 1; }
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
printf '%s\n' "$DRIVER_MAJOR" | grep -Eq '^[0-9]+$' || { echo "ERROR: cannot determine NVIDIA driver version"; exit 1; }
test "$DRIVER_MAJOR" -ge 590 || { echo "ERROR: AMC requires NVIDIA Driver 590 or newer"; exit 1; }
ENCODER_SESSIONS="$(nvidia-smi -i 0 --query-gpu=encoder.stats.sessionCount --format=csv,noheader,nounits 2>/dev/null || true)"
printf '%s\n' "$ENCODER_SESSIONS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' || { echo "ERROR: GPU 0 does not expose required NVENC support"; exit 1; }
docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi
```

If `docker ps` reports permission denied, ask the user to run `sudo usermod -aG docker $USER && newgrp docker`, then confirm `docker ps` works before continuing. Do not silently switch to `sudo docker`.

## Credentials

Ask the user to authenticate to NGC through their normal secret-handling path. Use `--password-stdin`; do not echo or store the API key.

```bash
read -rsp "NGC API key: " NGC_API_KEY; printf '\n'
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
unset NGC_API_KEY
```

Independent VGGT calibration requires the `facebook/VGGT-1B-Commercial` model license to be accepted and a HuggingFace token. Pass that token through `HF_TOKEN`, not a command-line `--token` argument. Download VGGT before changing ownership of `models/`.

## VGGT Prerequisite for Tuning

For explicit multi-camera tuning, require the checkpoint at `$AMC_ROOT/models/vggt/vggt_1B_commercial.pt`; do not ask whether to enable VGGT. Reuse an existing valid model. Single-camera tuning does not use VGGT, and normal AMC-only calibration remains supported.

If the checkpoint is missing, check `command -v hf` and `hf --help`. If unavailable, explain the planned CLI installation and obtain approval before installing tools. If both `uv` and `pipx` are absent, separately obtain approval for the privileged `apt-get update` and `apt-get install pipx` operations below; unattended execution without that approval must stop. Install the CLI in isolation; do not assume `pip` is installed or use `--break-system-packages`, `sudo pip`, or a user-site pip install:

```bash
if command -v uv >/dev/null 2>&1; then
  export PATH="$(uv tool dir --bin):$PATH"
  if ! command -v hf >/dev/null 2>&1 || ! hf --help >/dev/null 2>&1; then
    uv tool install --force huggingface_hub
  fi
else
  if ! command -v pipx >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y pipx
  fi
  HF_BIN_DIR="$(pipx environment --value PIPX_BIN_DIR)"
  export PATH="${HF_BIN_DIR}:$PATH"
  if ! command -v hf >/dev/null 2>&1 || ! hf --help >/dev/null 2>&1; then
    pipx install --force huggingface_hub --include-deps
  fi
fi
command -v hf && hf --help
```

If an existing `models/` directory is not writable, download through the user's `hf` process into a user-owned staging directory under `$DEEPSTREAM_REPO/data/amc/`. After approval of the exact destination, have the backend owner install only the verified checkpoint into `models/vggt/` with the container's ownership. Do not run token-bearing downloads as root or recursively transfer existing model ownership merely to make the download work.

Run installation and verification yourself when permitted; pause for required privilege approval or an installation failure. Advance only after both checks succeed. Handle token entry separately, never as part of the CLI installation response.

Then ask the user to accept the [model terms](https://huggingface.co/facebook/VGGT-1B-Commercial), create a [read token](https://huggingface.co/settings/tokens), and run the following on the AMC host. Never request the token in chat; wait for confirmation that the download succeeded.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
VGGT_DOWNLOAD_DIR="$AMC_ROOT/models/vggt"
if ! { mkdir -p "$VGGT_DOWNLOAD_DIR" 2>/dev/null && test -w "$VGGT_DOWNLOAD_DIR"; }; then
  VGGT_DOWNLOAD_DIR="$DEEPSTREAM_REPO/data/amc/vggt-download"
fi
mkdir -p "$VGGT_DOWNLOAD_DIR" || exit 1
read -rsp "Hugging Face token: " HF_TOKEN; printf '\n'
export HF_TOKEN
hf download facebook/VGGT-1B-Commercial vggt_1B_commercial.pt --local-dir "$VGGT_DOWNLOAD_DIR"
HF_DOWNLOAD_STATUS=$?
unset HF_TOKEN
test "$HF_DOWNLOAD_STATUS" -eq 0 || exit 1
if [ "$VGGT_DOWNLOAD_DIR" != "$AMC_ROOT/models/vggt" ]; then
  printf 'Downloaded to %s; have the backend owner install this checkpoint into %s before continuing.\n' "$VGGT_DOWNLOAD_DIR" "$AMC_ROOT/models/vggt"
  exit 1
fi
test -s "$AMC_ROOT/models/vggt/vggt_1B_commercial.pt"
```

If the backend was already running when the model was installed, wait until it is idle, restart `auto-magic-calib-ms` with the same runtime configuration, and verify readiness. In a verified multi-camera project, require `vggt_state == READY` before launching VGGT.

## Configure Compose

Configure Compose with process environment variables before launch. Read Compose to resolve image and mount settings; do not modify or back up checked-in Compose files. Set explicit project/model paths so no release defaults are inferred.

Ask for the IPv4 address of this AMC host reachable from the user's browser. Show `hostname -I` candidates if useful, but do not select the first interface automatically on a multi-homed/VPN host. Set `HOST_IP` explicitly before this block; verify the published backend and UI addresses after launch and ask the user to confirm browser access before reporting setup complete.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
cd "$AMC_ROOT/compose"

for port in $(seq 8000 8009); do
  if ! lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then MS_PORT=$port; break; fi
done
for port in $(seq 5000 5009); do
  if ! lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then UI_PORT=$port; break; fi
done
test -n "$MS_PORT" || { echo "ERROR: no free AMC backend port in 8000-8009"; exit 1; }
test -n "$UI_PORT" || { echo "ERROR: no free AMC UI port in 5000-5009"; exit 1; }

: "${HOST_IP:?set the user-confirmed browser-reachable AMC host IPv4 address}"
export HOST_IP
HOST_IP="$(python3 -c 'import ipaddress,os; print(ipaddress.IPv4Address(os.environ["HOST_IP"]))')" || exit 1

export AUTO_MAGIC_CALIB_MS_PORT="$MS_PORT"
export AUTO_MAGIC_CALIB_UI_PORT="$UI_PORT"
export PROJECT_DIR="$AMC_ROOT/projects"
export MODEL_DIR="$AMC_ROOT/models"
export HOST_IP
export AMC_RUNTIME_CONFIG="${AMC_RUNTIME_CONFIG:-$DEEPSTREAM_REPO/data/amc/amc-runtime.conf}"
mkdir -p "$(dirname "$AMC_RUNTIME_CONFIG")"
umask 077
{
  printf 'AUTO_MAGIC_CALIB_MS_PORT=%s\n' "$AUTO_MAGIC_CALIB_MS_PORT"
  printf 'AUTO_MAGIC_CALIB_UI_PORT=%s\n' "$AUTO_MAGIC_CALIB_UI_PORT"
  printf 'HOST_IP=%s\n' "$HOST_IP"
} > "$AMC_RUNTIME_CONFIG"
```

For RTSP support, set `VIOS_BASE_URL` only to a VIOS endpoint that has already passed the `/vst/api/v1/sensor/list` health check.

## Permissions

The containers run as UID/GID `1000`. Changing ownership recursively affects existing files and may remove the original user's access; it is not automatically undone. Verify that this UID is the intended container owner on this host. Resolve and display both exact directories and obtain explicit approval before changing them. Preserve current ownership/ACLs outside those directories so they can be restored with `sudo setfacl --restore=<backup>` when no AMC workload is using them. If ACL tools are unavailable, stop and resolve that prerequisite before an ownership change.

For a non-interactive agent shell, ask the user about the displayed targets first, then set `AMC_OWNERSHIP_APPROVAL` to the exact target string printed by the gate and rerun. Never populate this variable merely to satisfy the check without the user's approval. Run setup snippets in their own tool shells; their fail-closed `exit` ends that step, not the separate tuning-lock holder.

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
: "${DEEPSTREAM_REPO:?set the checkout root}"
AMC_ROOT="$(realpath -e "$AMC_ROOT")" || exit 1
test -f "$AMC_ROOT/compose/compose.yml" || exit 1
for name in projects models; do
  test ! -L "$AMC_ROOT/$name" || { echo "ERROR: refusing symlink target $AMC_ROOT/$name"; exit 1; }
done
mkdir -p "$AMC_ROOT/projects" "$AMC_ROOT/models"
printf 'Ownership targets: %s\n' "$AMC_ROOT/projects" "$AMC_ROOT/models"
if test -t 0; then
  read -r -p "Change ownership of these exact directories and their contents to 1000:1000? [y/N]: " ownership_answer
  case "$ownership_answer" in y|Y) ;; *) echo "Stopped before ownership changes"; exit 1 ;; esac
else
  EXPECTED_OWNERSHIP_APPROVAL="1000:1000|$AMC_ROOT/projects|$AMC_ROOT/models"
  test "${AMC_OWNERSHIP_APPROVAL:-}" = "$EXPECTED_OWNERSHIP_APPROVAL" || {
    printf 'ERROR: ask the user to approve these exact paths first, then rerun with AMC_OWNERSHIP_APPROVAL=%s\n' "$EXPECTED_OWNERSHIP_APPROVAL"
    exit 1
  }
fi
command -v getfacl >/dev/null && command -v setfacl >/dev/null || exit 1
mkdir -p "$DEEPSTREAM_REPO/data/amc"
OWNERSHIP_BACKUP="$(mktemp "$DEEPSTREAM_REPO/data/amc/ownership-XXXXXX.acl")" || exit 1
sudo getfacl -R -p "$AMC_ROOT/projects" "$AMC_ROOT/models" > "$OWNERSHIP_BACKUP" || exit 1
printf 'Ownership restore file: %s\n' "$OWNERSHIP_BACKUP"
sudo chown 1000:1000 -R "$AMC_ROOT/projects" "$AMC_ROOT/models"
```

## Launch

Read image names from the active compose file instead of hardcoding tags:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
export AMC_COMPOSE="$AMC_ROOT/compose/compose.yml"
export AMC_RUNTIME_CONFIG="${AMC_RUNTIME_CONFIG:-$DEEPSTREAM_REPO/data/amc/amc-runtime.conf}"
test -r "$AMC_RUNTIME_CONFIG" || { echo "ERROR: missing $AMC_RUNTIME_CONFIG; run Configure Compose first"; exit 1; }
read_runtime_value() { sed -n "s/^$1=//p" "$AMC_RUNTIME_CONFIG" | head -n 1; }
export AUTO_MAGIC_CALIB_MS_PORT="$(read_runtime_value AUTO_MAGIC_CALIB_MS_PORT)"
export AUTO_MAGIC_CALIB_UI_PORT="$(read_runtime_value AUTO_MAGIC_CALIB_UI_PORT)"
export HOST_IP="$(read_runtime_value HOST_IP)"
test -n "$AUTO_MAGIC_CALIB_MS_PORT" && test -n "$AUTO_MAGIC_CALIB_UI_PORT" && test -n "$HOST_IP" || { echo "ERROR: incomplete $AMC_RUNTIME_CONFIG"; exit 1; }
export PROJECT_DIR="$AMC_ROOT/projects"
export MODEL_DIR="$AMC_ROOT/models"
cd "$AMC_ROOT/compose"

docker compose -f "$AMC_COMPOSE" config --images
for image in $(docker compose -f "$AMC_COMPOSE" config --images | sort -u); do
  docker manifest inspect "$image" >/dev/null || {
    echo "ERROR: NGC key cannot access $image"
    exit 1
  }
done
docker compose -f "$AMC_COMPOSE" up -d
docker compose -f "$AMC_COMPOSE" ps
```

## Verification

Check readiness on `/v1/ready` and the UI root:

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
export AMC_RUNTIME_CONFIG="${AMC_RUNTIME_CONFIG:-$DEEPSTREAM_REPO/data/amc/amc-runtime.conf}"
test -r "$AMC_RUNTIME_CONFIG" || { echo "ERROR: missing $AMC_RUNTIME_CONFIG; run Configure Compose first"; exit 1; }
read_runtime_value() { sed -n "s/^$1=//p" "$AMC_RUNTIME_CONFIG" | head -n 1; }
MS_PORT="$(read_runtime_value AUTO_MAGIC_CALIB_MS_PORT)"
UI_PORT="$(read_runtime_value AUTO_MAGIC_CALIB_UI_PORT)"
HOST_IP="$(read_runtime_value HOST_IP)"
test -n "$MS_PORT" && test -n "$UI_PORT" && test -n "$HOST_IP" || { echo "ERROR: incomplete $AMC_RUNTIME_CONFIG"; exit 1; }

curl -fsS "http://localhost:${MS_PORT}/v1/ready"
curl -fsS "http://localhost:${MS_PORT}/openapi.yaml" >/tmp/amc-openapi.yaml
curl -fsS -o /dev/null "http://localhost:${UI_PORT}"
curl -fsS "http://${HOST_IP}:${MS_PORT}/v1/ready"
curl -fsS -o /dev/null "http://${HOST_IP}:${UI_PORT}"
```

Local success does not prove remote browser access. If the published addresses fail or the user's browser cannot reach them, stop and correct `HOST_IP`, routing, or firewall configuration before giving UI instructions.

If readiness fails, inspect container state and logs:

```bash
docker compose -f "$AMC_ROOT/compose/compose.yml" ps
docker compose -f "$AMC_ROOT/compose/compose.yml" logs --tail=200 auto-magic-calib-ms
```

## Stop

```bash
export DEEPSTREAM_REPO="/path/to/DeepStream"
export AMC_ROOT="$DEEPSTREAM_REPO/tools/auto-magic-calib"
docker compose -f "$AMC_ROOT/compose/compose.yml" down
```

Only remove volumes or generated data if the user explicitly asks.

## Report

Return:

- compose file path used: `$DEEPSTREAM_REPO/tools/auto-magic-calib/compose/compose.yml`
- microservice URL: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/v1`
- Swagger URL: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/docs`
- UI URL: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>`
- readiness response and container status
- project/model/data directory paths
- any missing prerequisites
