# Platform Guide: Choosing the Right Dockerfile and Base Image

Use this guide to select the correct Dockerfile template and DeepStream base image
for your target hardware **before** writing any Dockerfile or pipeline config.

---

## Quick Selection Table

| Platform | Architecture | Example Hardware | Dockerfile Template | DeepStream Base Image |
|---|---|---|---|---|
| **x86_64 datacenter** | x86_64 | A100, H100, B200, RTX | `samples://dockerfile/ds_app/Dockerfile` | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| **Jetson / Tegra** | aarch64 | Orin, AGX Orin, Thor | `samples://dockerfile/ds_app/Dockerfile.tegra` | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| **arm-sbsa server** | aarch64 | GB10, GB300, DGX Spark | `samples://dockerfile/ds_app/Dockerfile.dgxspark` | `nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark` |

---

## How to Identify Your Platform

### Step 1 — Check architecture
```bash
uname -m
```
- `x86_64` → use the **x86_64 datacenter** Dockerfile
- `aarch64` → continue to Step 2

### Step 2 — Distinguish Tegra (Jetson) from arm-sbsa (server)
```bash
# Tegra devices expose a device-tree model string:
cat /proc/device-tree/model 2>/dev/null
# e.g. "NVIDIA Jetson AGX Orin" → Tegra

# Server ARM64 (GB10, GB300, DGX Spark) report GPU via nvidia-smi:
nvidia-smi -L
# e.g. "GPU 0: NVIDIA GB10" → arm-sbsa server
```

If `/proc/device-tree/model` contains "Jetson" → use **Dockerfile.tegra**.
If `nvidia-smi` reports GB10 / GB300 / GH200 / Grace and no Jetson device-tree → use **Dockerfile.dgxspark**.

---

## Key Differences by Platform

### x86_64 (`Dockerfile`)
- PyTorch installed via pip from `https://download.pytorch.org/whl/cu130`
- TRT OSS GPU archs: `80;86;90;100` (Ampere, Ada, Hopper, Blackwell-x86)
- CUDA headers at `/usr/local/cuda-<ver>/include`

### Jetson / Tegra (`Dockerfile.tegra`)
- PyTorch **copied** from `nvcr.io/nvidia/pytorch:25.08-py3` (no pip wheel for Tegra)
- TRT OSS GPU archs: `110;120;121` (Ampere/Orin, Blackwell/Thor)
- Tegra BSP libraries (`libnvbufsurface.so`, etc.) **mounted at runtime** by the
  NVIDIA Container Toolkit — container will fail without `--runtime nvidia` on
  actual Jetson hardware
- Base image: Tegra multiarch variant (`-ma` suffix)

### arm-sbsa server (`Dockerfile.dgxspark`)
- PyTorch **copied** from `nvcr.io/nvidia/pytorch:25.08-py3` (same as Tegra)
- TRT OSS GPU archs: `110;120;121` (same as Tegra)
- CUDA headers present in the arm-sbsa base image (no Tegra BSP dependency)
- Base image: `arm-sbsa-spark` variant — **not** the `-ma` Tegra image
- Requires `apt-get install cmake libyaml-cpp-dev pkg-config` before TRT OSS build

---

## Common Mistakes to Avoid

| Mistake | Symptom | Fix |
|---|---|---|
| Using Tegra base image on GB10/GB300 | `libnvbufsurface.so.1.0.0: cannot open shared object file` | Switch to `Dockerfile.dgxspark` and the `arm-sbsa-spark` base image |
| Installing PyTorch via pip on Tegra/arm-sbsa | No compatible wheel found, or wrong CUDA variant | Copy torch from `nvcr.io/nvidia/pytorch:25.08-py3` |
| Missing `cmake` before TRT OSS build | `cmake: command not found` | Add `apt-get install cmake` (or skip TRT OSS build if not needed) |
| Using wrong GPU arch flags | TRT plugins fail to load or slow JIT compilation | Match `-DGPU_ARCHS` to the GPU table above |
| Tegra container without `--runtime nvidia` | Missing Tegra BSP `.so` files at runtime | Run with `--runtime nvidia` via NVIDIA Container Toolkit on Jetson |

---

## TAO Post-Processor Library (`libnvds_infercustomparser_tao.so`)

Required for TAO-trained detection models (e.g. PeopleNet Transformer) that use
`NvDsInferParseCustomDDETRTAO`. It is **not** bundled in any base image and must
be built from source:

```bash
# Inside the container (or in the Dockerfile RUN step):
git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
cd deepstream_tao_apps/post_processor
CUDA_VER=<major>.<minor> make DS_SRC_PATH=/opt/nvidia/deepstream/deepstream
cp libnvds_infercustomparser_tao.so /opt/nvidia/deepstream/deepstream/lib/
```

CUDA version by platform:
- x86_64 datacenter: `CUDA_VER=13.1`
- Jetson / arm-sbsa server: `CUDA_VER=13.0`

On arm-sbsa (GB10/GB300), if CUDA headers are not in the container, mount them
from the host at `/usr/local/cuda-13.0` or pre-build the `.so` on the host and
`COPY` it into the Dockerfile.
