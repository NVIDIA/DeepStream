# AutoMagicCalib

AutoMagicCalib (AMC) is an automated calibration tool that estimates both intrinsic and extrinsic camera parameters for multi-camera and single-camera systems. It provides camera projection matrices and lens distortion coefficients essential for accurate 3D reconstruction and multi-view applications.

AMC eliminates the need for traditional calibration patterns (like checkerboards) by using tracked moving objects in the scene as natural features for calibration. It leverages DeepStream's object detection and tracking capabilities to identify and follow objects (particularly people) across frames, then analyzes these trajectories across camera views to automatically derive camera parameters from regular operational footage. This approach enables calibration without interrupting normal operations, allows retroactive calibration using archived footage, and performs calibration in the actual deployment environment.

The service supports both a geometry-based approach (AMC) using object trajectories and geometric relationships, and a model-based approach (VGGT) that leverages learned models for higher accuracy and robustness.

## Features
- Estimate camera lens distortion parameter (k1)
- Estimate 3x4 camera projection matrix (focal length, rotation, translation)
- Ground truth focal length override: Use known focal lengths while preserving GeoCalib rotation intelligence
- Output calibration results in YAML format
- Dedicated rectification workflow (Auto, Manual, or Videos Are Rectified) with frame preview, full-video generation, live logs, and rectified-video download
- Independent AMC and VGGT calibration choices (VGGT can run without a prior AMC run)
- Shared layout post-processing for trajectory overlays, optional GT evaluation, MV3DT, and virtual-GT overlays
- Visualization tools:
  - Score metrics graphs of parameter estimation
  - Rectified video generation with estimated lens parameters
  - Trajectory and virtual-object overlay imagery on Results
  - Visual overlay video generation (SV3DT) with estimated camera projection matrix
- Complete end-to-end pipeline for single-camera and multi-camera calibration
- Bundle adjustment for improved accuracy
- Evaluation against ground truth data
- **Web UI workflow**: 7-step guided calibration — project setup → video configuration → parameters → rectification → manual alignment → execute (AMC/VGGT + shared post-process) → results (ROI/tripwire drawing, verification, export)

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
  - [System Requirements](#system-requirements)
  - [NGC Setup](#ngc-setup)
  - [Project Setup](#project-setup)
  - [Sample Data Setup](#sample-data-setup)
- [Calibration Workflow (UI)](#calibration-workflow-ui)
  - [Step 1: Project Setup](#step-1-project-setup)
  - [Step 2: Video Configuration](#step-2-video-configuration)
  - [Step 3: Parameters](#step-3-parameters)
  - [Step 4: Rectification](#step-4-rectification)
  - [Step 5: Manual Alignment](#step-5-manual-alignment)
  - [Step 6: Execute Calibration](#step-6-execute-calibration)
  - [Step 7: Results](#step-7-results)
- [Assumptions](#assumptions)
  - [Tracklet-Based Calibration: Input Video Requirements](#tracklet-based-calibration-input-video-requirements)
  - [Lens Distortion Output (`distortion.yaml`)](#lens-distortion-output-distortionyaml)
    - [Distortion model](#distortion-model)
- [Custom Dataset](#custom-dataset)
  - [Input Requirements](#input-requirements)
  - [Alignment Data (alignment_data.json)](#alignment-data-alignment_datajson)
  - [Guidelines for Input Videos](#guidelines-for-input-videos-to-achieve-optimal-calibration-results)
  - [Ground Truth Data Format](#ground-truth-data-format)
    - [calibration.json](#calibrationjson)
    - [ground\_truth.json](#ground_truthjson)
- [Troubleshooting](#troubleshooting)
- [License](#license)
  - [Repository Licenses](#repository-licenses)
  - [Proprietary Container Notices](#proprietary-container-notices-automagiccalib-and-automagiccalibui)

<br><br>
# Quick Start

### Agent Skills
AutoMagicCalib workflows use the existing DeepStream agent skills under
[`skills/`](../../skills/README.md). Select the setup, sample, video, or RTSP
calibration skill from the DeepStream repository root. Full calibration runs
require the runtime prerequisites below and are not expected to complete in
restricted or no-GPU sandboxes; in those environments, agents should limit work
to planning, configuration review, and command preparation.

Quick install:

```bash
# Claude Code
mkdir -p ~/.claude/skills
cp -r ../../skills/amc-* ~/.claude/skills/

# Codex
mkdir -p ~/.codex/skills
cp -r ../../skills/amc-* ~/.codex/skills/
```

Restart or reopen the coding assistant after copying the skills. See
[`skills/README.md`](../../skills/README.md) for the full skill catalog and
example prompts.

Expected runtime varies by host, image/model downloads, video length, detector choice, and whether you run Auto/Manual rectification plus optional VGGT and layout post-processing. The bundled sample can take several minutes up to about 30 minutes, custom video calibration can take 10-60+ minutes, and optional VGGT refinement usually adds a few minutes.

### System Requirements
- x86_64 system
- OS Ubuntu 24.04
- NVIDIA GPU with hardware encoder (NVENC)
- NVIDIA driver 590
- Docker (setup to run without sudo privilege)
- NVIDIA container toolkit (see [NVIDIA DeepStream Docker Prerequisites](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html#prerequisites))
- [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) (required to
  download the optional NGC sample dataset)

### NGC Setup
This step is needed to pull AutoMagicCalib docker images and download resources
such as `nvidia/amc-nv-warehouse`.

1. Install the [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) for
   Linux and verify that it is available:

   ```bash
   ngc --version
   ```

2. Visit the NGC sign-in page, enter your email address, and click **Next**, or
   create an account.
3. Choose your organization/team.
4. Generate an API key following the instructions.
5. Configure the NGC CLI with that API key:

   ```bash
   ngc config set
   ```

6. Log in to the NGC Docker registry:

   ```bash
   docker login nvcr.io
   ```

   When prompted, enter `$oauthtoken` as the username and paste your NGC API
   key as the password.

### Project Setup
AutoMagicCalib is distributed as part of the DeepStream repository. Clone
DeepStream, then run the commands below from
`tools/auto-magic-calib`. This directory contains this README along with
`compose/`, `assets/`, `models/`, and `projects/`.

All paths in the rest of this guide are relative to the AutoMagicCalib root,
except for `PROJECT_DIR` and `MODEL_DIR` in `compose/.env`, which Docker Compose
resolves relative to the `compose/ms/` directory (hence the `../../` prefix in
their defaults).

#### Download and set up VGGT model
Optionally you can download VGGT model for model based calibration

Download the VGGT commercial model from [HuggingFace](https://huggingface.co/facebook/VGGT-1B-Commercial). Downloaded model must be copied to appropriate model directory as mentioned below.

> **Note:** You need to sign up for a HuggingFace account and accept the model license agreement before downloading the model.

#### Optional VIOS Setup for RTSP Capture
RTSP support requires `VIOS_BASE_URL`. If a VIOS server is already reachable from the AMC host, verify it before setting `VIOS_BASE_URL` in `compose/.env`. Without VIOS, the rest of the calibration workflow can run, but RTSP support remains disabled.

If VIOS is not deployed yet, use the VIOS deployment assets from the VSS repository:

```text
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/main/services/vios/deployment
```

Follow `1click_README.md` in that directory. A typical development deployment command is:

```bash
python3 oneclick_dc_deployment_for_dev.py --auto
```

Verify VIOS before enabling it:

```bash
curl http://<vios-host>:30888/vst/api/v1/sensor/list
```

#### Configure Environment Variables
Edit the `compose/.env` file to set the required environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `HOST_IP` | **Yes** | — | IP address of the host machine |
| `AUTO_MAGIC_CALIB_MS_PORT` | No | `8000` | Port for the microservice API |
| `AUTO_MAGIC_CALIB_UI_PORT` | No | `5000` | Port for the web UI |
| `PROJECT_DIR` | No | `../../projects` | Path to the projects directory |
| `MODEL_DIR` | No | `../../models` | Path to the models directory |
| `VIOS_BASE_URL` | No | — | VIOS server URL for RTSP capture APIs |

If you want to enable VGGT, VGGT model should be copied inside `$MODEL_DIR/vggt/`.

```dotenv
AUTO_MAGIC_CALIB_MS_PORT=8000
AUTO_MAGIC_CALIB_UI_PORT=5000
PROJECT_DIR=../../projects
MODEL_DIR=../../models
HOST_IP=<your_host_ip>
# VIOS_BASE_URL=http://<VIOS_HOST_IP>:30888  # Uncomment and update this to support RTSP Stream
```

`HOST_IP` must not be empty. If it is empty, the UI API URL renders as `http://:<port>/v1`.

Leave `VIOS_BASE_URL` unset to keep RTSP capture disabled.

#### Set Directory Permissions
The `projects` and `models` directories must be owned by UID/GID 1000 for the containers to read/write properly.
```bash
sudo chown 1000:1000 -R projects
sudo chown 1000:1000 -R models
```

#### Launch Services
Start all services using Docker Compose. Container images will be pulled automatically on the first run.
```bash
cd compose
docker compose up -d
```
The microservice will be available at `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>` (default port 8000) and the UI at `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>` (default port 5000).

To stop the running containers,
```bash
docker compose down
```

### Sample Data Setup
From the AutoMagicCalib directory, unzip the compressed sample data file `assets/sdg_08_2_sample_data_010926.zip`. The sample folder includes 4 different types of data to help you run end-to-end calibration and evaluation.
1. Input video files
2. Ground truth data
3. BirdEyeView map image
4. Pre-calibrated transform for BirdEyeView map

```text
assets/sdg_08_2_sample_data_010926.zip
├── alignment_data
│   ├── alignment_data.json     # Pre-calibrated transform from `cam_00` reference frame to BirdEyeView map image 
│   └── layout.png              # BirdEyeView map image required for visualization
├── GT.zip                      # Ground truth data (camera info, extrinsics, object trajectories)
└── videos                      # Input video files
    ├── cam_00.mp4
    ├── cam_01.mp4
    ├── cam_02.mp4
    └── cam_03.mp4

```

Now you're ready to start the calibration process.

To try real world case, we have another sample data file [nv_warehouse_090926.zip](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/amc-nv-warehouse). The sample folder includes 4 different files. It does not have ground-truth data. Additionally it has `nv_warehouse_config.json`, which should be uploaded in the [config param step](#configuring-settings). For AMC calibration in the Execute step set the `Detector Type` as `Transformer`.

To download the dataset use the following command:
```bash
ngc registry resource download-version "nvidia/amc-nv-warehouse"

```

In case you want to try your own dataset, please verify requirements (files, directories, formats) explained in [Assumptions](#assumptions) section.


# Calibration Workflow (UI)

Once the microservice and UI containers are running, open your browser and navigate to `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>` (default port `5000`).

The UI presents a **7-step stepper workflow**. Each step validates its inputs before allowing you to proceed to the next. Rectification must reach **COMPLETED** before you can advance to Manual Alignment and later steps.

---

## Step 1: Project Setup

The Project Setup step allows you to create and manage calibration projects.

![Project Setup Step](resources/images/vss-autocalib-ui/main-ui-2.jpg)

### Creating a New Project

1. Enter a project name in the text field
   - **Requirements**: 3–50 characters
   - **Example**: `warehouse_cam_2024`, `parking_lot_north`
2. Click the **Create** button
3. The new project appears in the "Existing Projects" list below

![Create New Project](resources/images/vss-autocalib-ui/create_new_project.jpg)

**Project Name Validation**
- ✓ Valid: `warehouse_calibration`, `site_01`, `parking-lot-A`
- ✗ Invalid: `ab` (too short)

### Selecting a Project

1. Browse the list of existing projects
2. Click the **Select** button on the desired project card
3. The selected project is highlighted with a green border and checkmark
4. Project information is displayed at the bottom: "Project 'name' selected"

![Select Project](resources/images/vss-autocalib-ui/select_project.jpg)

**Project Card Information**

Each project card displays:
- **Project Name**: The name you assigned
- **Project ID**: Unique identifier (UUID)
- **Project State**: Current status badge
  - `INIT` (gray): Initial state, files not yet uploaded
  - `READY` (green): Ready for calibration
  - `RUNNING` (orange): Calibration in progress
  - `COMPLETED` (green): Calibration finished successfully
  - `ERROR` (red): Calibration failed
- **Video Count**: Number of uploaded video files
- **File Status**: Checkmarks for uploaded files
  - GT (Ground Truth): ✓ or ✗
  - Layout: ✓ or ✗
  - Alignment: ✓ or ✗

### Managing Projects

**Refreshing the Project List**

Click the **Refresh** button in the top-right corner to reload the project list from the server.

**Deleting a Project**

1. Click the trash icon (🗑️) on the project card
2. Confirm deletion in the dialog that appears
3. The project and all associated data are permanently deleted

**Deleting Selected Projects**

Use the checkboxes on project cards to multi-select projects for bulk delete (this is separate from the workflow **Select** button).

1. Check one or more project cards
2. In the Project Setup header, click **Delete selected (N)**
3. Confirm in the **Delete N selected project(s)?** dialog
4. Selected removable projects are permanently deleted; the project list reloads afterward

**Deleting All Projects**

1. In the Project Setup header (next to **Refresh**), click **Delete all**
2. Confirm in the **Delete all projects?** dialog
3. Every removable project is permanently deleted; the project list reloads afterward

**Delete all** is disabled when there are no projects. For **Delete selected** and **Delete all**, projects with calibration **running** or **RTSP capture in progress** are skipped and not removed.

> **Warning:** Deleting a project (including **Delete selected** or **Delete all**) cannot be undone. Export any important calibration results before deletion.

---

## Step 2: Video Configuration

Upload camera videos, layout image, ground truth data, and optional alignment file.

![Video Configuration Step](resources/images/vss-autocalib-ui/video_configuration_step.jpg)

### Upload Status Overview

At the top of the page, you'll see a status summary showing:
- **Videos**: Count of uploaded videos (minimum 1 required)
- **Ground Truth (Optional)**: Upload status
- **Layout**: Upload status (required)
- **Alignment (Optional)**: Upload status

![Upload Status](resources/images/vss-autocalib-ui/upload_status.jpg)

### Uploading Video Files

**Requirements**
- **Minimum**: 1 video file (2 or more for multi-camera calibration)
- **Formats**: MP4 . AutoMagicCalib does **not** support video formats that are not supported by DeepStream.
- **Required Video Resolution**: 1920×1080
- **Input assumptions**: See [Assumptions](#assumptions) for video content and recording requirements

Provide camera inputs using **either** file upload **or** RTSP capture (one camera for single-camera calibration, two or more for multi-camera). **RTSP capture is available only when VIOS is configured on the Auto Calibration server**; otherwise use file upload. The UI does not allow an active file-upload queue and RTSP capture at the same time; remove file-uploaded clips before switching to RTSP, and vice versa.

**Option A: Upload video files**

1. Click the **Select Videos** button to choose video files from your computer (MP4)
2. Selected videos appear in a list where you can reorder them by dragging
3. Reorder videos to match your desired camera order (`cam_00`, `cam_01`, etc.)—maintain **order of overlapping field of view (FOV)**
4. Click **Upload N File(s)** to upload all selected videos
5. Wait for the upload progress bar to complete

![Video Upload](resources/images/vss-autocalib-ui/video_upload.jpg)

**Managing Video Files**
- **View List**: All selected or uploaded videos are listed with their filenames
- **Reorder**: Drag and drop videos to change their order before uploading
- **Delete Video**: Click the trash icon (🗑️) next to a video to remove it
- **Re-upload**: Delete and upload again if needed

**Option B: RTSP capture (VIOS)**

*(Shown only when the Auto Calibration service exposes RTSP capture; VIOS is configured on the server side.)*

1. Finish or clear any pending **Video Files** selection or upload before starting RTSP; if the project already has clips from file upload, remove them under **Video Files** first
2. In the **RTSP capture (VIOS)** card, set **Duration (seconds, min 60)** (minimum **60** seconds, per server requirement)
3. Under **Streams**, enter **all** **RTSP URLs** for the project before capturing. Use **Add stream** for additional cameras. Optionally set **Camera name** and **Sensor ID**
4. Click **Capture and add to project** **once** for the full stream list (all cameras record together). Do not add streams to an in-flight session or run separate captures for different camera subsets when you need multi-camera time synchronization
5. Wait for the status chip and progress bar (**STARTING** → **RECORDING** → **STOPPING** / **INGESTING** as applicable → **INGESTED**)
6. Optionally click **Stop early** after at least **60** seconds of **RECORDING**; early stop continues into automatic ingest
7. When the session reaches **INGESTED**, captured clips appear under **Video Files**

While RTSP capture or ingest is running, **Video Files** upload is disabled until the pipeline completes.

![RTSP Input](resources/images/vss-autocalib-ui/rtsp_input.jpg)

> RTSP streams must be **time-synchronized**: one **Capture and add to project** with every stream configured together—no staggered captures for different camera subsets. List each **RTSP URL** under **Streams** in **order of overlapping FOV** (first stream = first camera in the overlap chain).
> For **VIOS pre-registered RTSP streams**, use the **source URL** (for example, the **NVStreamer** URL if the stream originates from NVStreamer) rather than the VIOS-proxied URL.

### Uploading Ground Truth Data

Ground truth data is optional and used for calibration evaluation.

**Requirements**
- **Format**: ZIP file
- **Content**: Ground truth calibration data

**Upload Process**

1. Click **Upload Ground Truth (Optional)** button
2. Select your ZIP file
3. Wait for upload confirmation
4. Status changes to "Ground truth uploaded ✓"

**Deleting Ground Truth**

If ground truth is already uploaded, the button changes to **Delete Ground Truth**. Click it to remove the file.

![Ground Truth Delete](resources/images/vss-autocalib-ui/gt_delete.jpg)

### Uploading Layout Image

The layout image is required and represents the top-down view or map of your surveillance area.

**Requirements**
- **Format**: PNG
- **Content**: Bird's eye view map or layout diagram
- **Recommended**: High resolution for better accuracy

**Upload Process**

1. Click **Upload Layout** button
2. Select your image file
3. Wait for upload confirmation
4. Status changes to "Layout image uploaded ✓"

**Deleting Layout**

If layout is already uploaded, the button changes to **Delete Layout**. Click it to remove the file.

![Layout Delete](resources/images/vss-autocalib-ui/layout_delete.jpg)

### Uploading Alignment Data

Alignment data is optional at this step. You can either upload a pre-existing alignment file here or create / view / edit it interactively in Step 5.

**Requirements**
- **Format**: JSON file
- **Content**: Alignment point data (4+ point sets)

**Upload Process**

1. Click **Upload Alignment (Optional)** button
2. Select your JSON file
3. Wait for upload confirmation
4. Status changes to "Alignment file uploaded ✓"

**Deleting Alignment**

If alignment is already uploaded, the button changes to **Delete Alignment**. Click it to remove the file.

![Alignment Delete](resources/images/vss-autocalib-ui/alignment_delete.jpg)

### Requirements Note

**Required for Calibration:**
- At least 1 video file (2 or more for multi-camera calibration)
- Layout image (PNG)
- Alignment data (can be created in Manual Alignment step)
- Rectified / linear media (completed in Step 4: Rectification)

**Optional:**
- Ground truth data (ZIP file) — for evaluation purposes

> You can proceed past Video Configuration even if ground truth and alignment are not uploaded. Complete Rectification (Step 4) before Alignment. Alignment can be created, viewed, or edited interactively in Step 5.

---

## Step 3: Parameters

Set floor-plan scale and optional focal lengths. 

![Parameters Step](resources/images/vss-autocalib-ui/parameters_step.jpg)

### Floor-Plan to 3D Scale

Initialize **layout pixels per meter** (`layout_px_per_m`) so overlays and world-coordinate exports match the physical floor plan.

**Requirements**
- Layout image uploaded in Step 2

**How to Configure**

1. Open the **Floor-plan to 3D scale** card
2. Either:
   - 2A. Type a positive **layout px/m** value directly, **or**
   - 2B. Click two points on the layout map, enter the real-world distance in meters, and use the calculated px/m value
3. Click **Save Scale** to persist `layout_px_per_m` or **Reset** to recalculate layout_px_per_m

**Controls**
- **Scroll**: Zoom the layout canvas
- **Drag**: Pan when zoomed
- **Reset**: Clear measurement points and restore the last saved px/m

![Floor Plan to 3D Scale](resources/images/vss-autocalib-ui/floor_plan.jpg)

> Wrong scale causes misaligned BEV / trajectory overlays after layout post-processing. Re-run shared post-processing after changing scale if calibration already completed.

### Focal Length Configuration

Focal lengths are optional but can improve calibration accuracy.

**Requirements**
- One value per camera
- Comma-separated list
- Positive numbers only
- Count must match video count

**How to Configure**

1. Find the **Focal Length (Optional)** card
2. Enter focal lengths separated by commas (e.g., `1269.01, 1099.50, 1099.50, 1099.50`)
3. Click **Save Focal Length**
4. Confirmation message appears

**Clearing Focal Lengths**

1. Delete all text from the input field
2. Click **Save Focal Length**
3. Focal lengths are cleared from the project

![Focal Length Configuration](resources/images/vss-autocalib-ui/focal_length_configuration.jpg)

> Scale and focal length are view-only while AMC or VGGT calibration is running.

### Configuring Settings

On the Parameters step, you can customize calibration settings before running the pipeline. The settings control is **only visible on this step**.

![Settings Dialog](resources/images/vss-autocalib-ui/settings_dialog.jpg)

**Configuration Options**
- **Option 1: Upload** — upload a pre-configured settings file to apply all parameters at once
- **Option 2: Manual Configuration** — modify each parameter individually through the settings interface

**Additional Actions**
- **Download**: Export the current settings configuration to a file
- **Reset to Defaults**: Restore all settings to their default values
- **Save Settings**: Save your changes

![Settings Update](resources/images/vss-autocalib-ui/settings_update.jpg)

> **Warning:** Do not change settings while AMC or VGGT calibration is running. Make configuration changes before starting calibration in Step 6: Execute. Rectification model / K1 search settings are configured on Step 4, not here.

---

## Step 4: Rectification

Generate **linear (rectified) camera media** before alignment and calibration. AMC and VGGT consume staged `rectified.mp4` / `rectified.jpg`; they no longer perform implicit in-calibration rectification.

![Rectification Step](resources/images/vss-autocalib-ui/rectification_step.jpg)

Rectification must reach **COMPLETED** before you can proceed to Manual Alignment. Verification on Execute also requires completed linear media.

### Choose a Rectification Path

| Path | When to use | Outcome |
|---|---|---|
| **Auto Rectification** | Estimate distortion from frame 0 (GeoCalib) | Review previews → **Generate Rectified Videos** → COMPLETED |
| **Manual Rectification** | Tune distortion model/parameters per camera | Preview frame → **Generate Rectified Videos** → COMPLETED |
| **Videos Are Rectified** | Inputs are already linear / undistorted | Stages source videos as linear media immediately → COMPLETED |

### Auto Rectification

1. Select **Auto Rectification**
2. Choose distortion model: `simple_divisional` (default), `simple_radial`, or `radial`
3. Optionally adjust K1 search range (`K1 minimum`, `K1 maximum`, `K1 steps`)
4. Click **Save Settings & Start Auto Rectification**
5. Wait for state **Ready for Review**
6. Review side-by-side original / GeoCalib rectified previews
7. Click **Generate Rectified Videos** to write full videos and `distortion.yaml`

![Auto Rectification Step](resources/images/vss-autocalib-ui/auto_rectification_step.jpg)

### Manual Rectification

> **Note:** You may run Manual Rectification directly, or do **Auto Rectification → Manual Rectification**. If Auto Rectification has run, Manual Rectification automatically sets the distortion parameters to those estimates as initial values.

1. Select **Manual Rectification**
2. Click **Start Manual Rectification**
3. Per camera, edit absolute distortion parameters (`model`, `k1`, and `k2` when model is `radial`). **Optionally** apply the same params from cam_00 to all cameras
4. Move sliders to set distortion parameters. Use live grid overlay feedback. Curved grid contours show the approximate look of straight objects/lines in the scene at the current distortion parameters set in the slider(s). Users may adjust **Contour rotation** to make in-view rotation for easier tuning; click **Rectify Image** for a frame-0 remapped preview
5. Click **Generate Rectified Videos**

![Manual Rectification Step](resources/images/vss-autocalib-ui/manual_rectification.jpg)

### Videos Are Rectified

1. Select **Videos Are Rectified**
2. Click **Stage Linear Videos**
3. Source videos are staged as linear media (no distortion estimate); state becomes **COMPLETED**

![Linear Rectification Step](resources/images/vss-autocalib-ui/linear_media_rectification.jpg)

### After Completion

- Download rectified media and distortion.yaml via **Download rectified videos** (ZIP of single-view results)
- Rectification path, model, and parameter details are shown after rectification completes.
- Alignment and calibration unlock

![Download Rectified Params](resources/images/vss-autocalib-ui/download_rectified_params.jpg)

### Re-rectification

Re-running Auto, Manual, or Videos Are Rectified after COMPLETED **invalidates downstream work**. You must:

1. Re-verify the project on Execute
2. Relaunch AMC and/or VGGT calibration
3. Re-run shared layout post-processing (multi-camera)

Concurrent rectification / calibration / post-processing is rejected.

> Distortion models supported in the UI: `simple_divisional` (**default**), `simple_radial`, `radial`. Already-linear inputs use the **Videos Are Rectified** path instead of a separate `pinhole` toggle.

---

## Step 5: Manual Alignment

Create, upload, view, or edit alignment data by selecting corresponding points across camera views and the layout map. This step is required for calibration. Interactive points are collected on **rectified / linear** frames from Step 4; **file upload** expects original / pre-rectification camera coordinates (see Option 1).

### Three Options for Alignment

**Option 1: Upload Existing Alignment**

If you already have an `alignment_data.json` file (for example from CLI / original **pre-rectification** frames):

1. Click **Upload alignment_data.json**
2. Select your JSON file
3. Wait for upload confirmation — the UI uploads with `coord_space=original`; the microservice converts camera points to rectified-space alignment after Step 4 when needed
4. The alignment tool opens in **view** mode for visual inspection; switch to **edit** to adjust points

> **Important:** External file upload expects **original / distorted** camera coordinates. Do **not** supply already-rectified camera points on this path — that causes an incorrect second conversion. Interactive draw/save (Option 2) uses `coord_space=rectified` because points are picked on rectified frames.

**Option 2: Create Alignment Interactively**

1. Click **Open Alignment Tool** / create controls
2. The interactive alignment interface opens (points are collected on **rectified / linear** frames)
3. Follow the point selection process
4. Save when at least 4 point sets are complete (saved as rectified-space)

**Option 3: View / Edit Existing Alignment**

When alignment already exists:

1. Click **View** to inspect points overlaid on camera and layout images
2. Click **Edit** to adjust points, then **Save Changes**
3. Or **Delete Alignment Data** and recreate from scratch

![Alignment Option](resources/images/vss-autocalib-ui/manual_alignment.jpg)

![Alignment Option](resources/images/vss-autocalib-ui/manual_alignment_2.jpg)

![Manual Alignment Tool](resources/images/vss-autocalib-ui/step4_manual_alignment_tool.jpg)

### Alignment Status

At the top of the page, you'll see the current alignment status:
- **Green Badge**: "Alignment data exists" — file already uploaded or created
- **Gray Badge**: "No alignment data" — need to upload or create alignment

![Alignment Status](resources/images/vss-autocalib-ui/alignment_status.jpg)

### Prerequisites Check

Before creating alignment interactively, the system checks:
- ✓ At least 1 video uploaded
- ✓ Layout image uploaded
- ✓ Rectification completed (linear media ready)

If prerequisites are not met, you'll see a warning message directing you to Step 2 or Step 4.

### Interactive Alignment Tool

**Interface Overview**

The alignment tool shows one concatenated canvas. The layout depends on how many videos are in the project:

**Multi-camera (2 or more videos)**
- **Left**: Camera 0 (cam_00.mp4)
- **Center**: Camera 1 (cam_01.mp4)
- **Right**: Layout map (BEV — bird's eye view)

**Single-camera (1 video)**
- **Left**: Camera (cam_00.mp4)
- **Right**: Layout map (BEV)

![Alignment Canvas](resources/images/vss-autocalib-ui/alignment_canvas.jpg)

**Progress Indicator**

At the top, you'll see:
- **Progress Bar**: Visual progress (0–100%)
- **Completion Status**: "X / Y sets (Min 4 required)" or "(Ready to save)"
- **Current Action**: Prompt shows the next panel to click—for example **Camera 0**, **Camera 1**, **Layout Map**, or **Camera** (single-camera)

### Point Selection Process

**Multi-camera (3 clicks per point set)**

1. **Select Point on Camera 0** — click on a distinct feature in Camera 0 (left section); a colored circle with number "1" appears; system prompts "Click on: Camera 1"
2. **Select Corresponding Point on Camera 1** — click on the same physical location in Camera 1 (center section); system prompts "Click on: Layout Map"
3. **Select Corresponding Point on Layout** — click on the same physical location on the layout map (right section); **Point Set 1 complete**
4. **Repeat for Additional Points** — repeat for at least **4** total point sets; each set uses a different color (green, blue, red, yellow)

**Single-camera (2 clicks per point set)**

1. **Select Point on Camera** — click on a distinct feature in the camera view (left section); system prompts "Click on: Layout Map"
2. **Select Corresponding Point on Layout** — click on the same physical location on the layout map (right section); **Point Set 1 complete**
3. **Repeat for Additional Points** — repeat for at least **4** total point sets (same minimum as multi-camera)

![Point Selection Process](resources/images/vss-autocalib-ui/point_selection_process.jpg)

**Point Selection Tips**
- Choose points on the **ground plane**
- Select **distinct features** (corners, markings, poles)
- Ensure each point is **visible in every panel** for that project (all cameras and the layout map, or camera + layout for single-camera)
- Distribute points across **different depths and locations**
- Avoid points on **moving objects**
- Use **zoom controls** for precision

### Zoom and Navigation

**Zoom Controls** (located above the canvas):
- **Zoom In** (🔍+): Increase zoom level
- **Zoom Out** (🔍-): Decrease zoom level
- **Reset (100%)**: Return to original zoom level
- **Current Zoom**: Displayed as percentage (e.g., "Zoom: 150%")

**Navigation**
- **Scroll Wheel**: Zoom in/out on the canvas
- **Click + Drag**: Pan around when zoomed in
- **Zoom Range**: 50% to 300%

![Zoom Controls](resources/images/vss-autocalib-ui/zoom_controls.jpg)

### Point Set Management

- **Undo Last Point**: Click the **Undo** button to remove the most recently placed point
- **Reset All Points**: Click the **Reset All** button to clear all points and start over
- **Add More Points**: After completing 4 point sets, click **Add More Points** to add additional sets for improved accuracy

![Point Management](resources/images/vss-autocalib-ui/point_management.jpg)

### Saving Alignment Data

**Requirements**
- Minimum **4** complete point sets
- **Multi-camera**: each set must include Camera 0, Camera 1, and layout map points
- **Single-camera**: each set must include camera and layout map points (2 points per set)

**Save Process**

1. Complete at least 4 point sets
2. The **Save Alignment** button becomes enabled
3. Button shows: "Save Alignment (X sets)" where X is the count
4. Click **Save Alignment (X sets)**
5. System generates and uploads the alignment JSON file
6. Success message appears
7. Alignment tool closes automatically

![Save Alignment](resources/images/vss-autocalib-ui/save_alignment.jpg)

Click the **Cancel** button to exit the alignment tool without saving.

### Alignment Data Format

The **generated / interactively saved** alignment data is a JSON array. Each outer element is one point set; camera coordinates are in pixel space of the **rectified / linear** images. For **external file upload** via the UI, supply **original / pre-rectification** camera coordinates instead — see Option 1 above and [Alignment Data (alignment_data.json)](#alignment-data-alignment_datajson).

**Multi-camera** — three `[x, y]` pairs per set (camera 0, camera 1, layout):

```json
[
  [[x0_cam0, y0_cam0], [x0_cam1, y0_cam1], [x0_layout, y0_layout]],
  [[x1_cam0, y1_cam0], [x1_cam1, y1_cam1], [x1_layout, y1_layout]],
  ...
]
```

**Single-camera** — two `[x, y]` pairs per set (camera, layout):

```json
[
  [[x0_cam, y0_cam], [x0_layout, y0_layout]],
  [[x1_cam, y1_cam], [x1_layout, y1_layout]],
  ...
]
```

### Deleting Alignment Data

If alignment data already exists and you want to recreate it:

1. The interface shows: "Alignment data already exists for this project"
2. Click **Delete Alignment Data** button
3. Confirm deletion
4. Create new alignment using either upload or interactive method

> **Warning:** Deleting alignment data cannot be undone. You'll need to recreate or re-upload it.

### Best Practices

**Point Selection Strategy**
- **Minimum 4 points**: Required for calibration
- **Recommended 6–8 points**: Better accuracy and robustness

**Point Distribution**
- Spread points across the entire area
- Include points at different depths (near and far)
- Cover all quadrants of the layout
- Avoid clustering points in one area

**Point Quality**
- Use sharp, distinct features
- Avoid ambiguous or blurry areas
- Prefer corners and intersections
- Ensure good contrast

**Common Mistakes to Avoid**
- ✗ Selecting points on walls or elevated surfaces
- ✗ Choosing points only in the center
- ✗ Using points on moving objects
- ✗ Clicking too quickly without precision
- ✗ Forgetting to zoom in for accuracy

---

## Step 6: Execute Calibration

Verify project requirements and run calibration. AMC and VGGT are **independent** calibration choices: each can be launched or relaunched on its own. Multi-camera projects finish raw calibration first, then run **shared layout post-processing** for overlays, evaluation, MV3DT, and virtual-GT imagery.

![Execute Step](resources/images/vss-autocalib-ui/execute_step.jpg)

### Project State Overview

At the top of the page, you'll see the current project state:
- **INIT** (gray): Initial state
- **READY** (blue): Ready to run calibration
- **RUNNING** (orange): Calibration in progress
- **COMPLETED** (green): Calibration finished
- **ERROR** (red): Calibration failed

When RUNNING, an elapsed time counter and progress bar are displayed.

![Project State](resources/images/vss-autocalib-ui/project_state.jpg)

### Requirements Checklist

The system validates all required files before allowing calibration:

- ✓ **Videos (minimum 1)**: Shows count of uploaded videos
- ✓ **Layout Image**: Confirms layout is uploaded
- ✓ **Alignment Data**: Confirms alignment is uploaded or created

If any requirement is not met, you'll see a warning message: "Complete required inputs before verification."

![Requirements Checklist](resources/images/vss-autocalib-ui/project_inputs_req.jpg)

### Optional Configuration

The system also displays optional configuration status:

- **Ground Truth Data**: ✓ Uploaded (for evaluation purposes) or ⊙ Not provided (optional)
- **Focal Length**: ✓ X value(s) shown, or ⊙ Not provided (optional)

![Optional Configuration](resources/images/vss-autocalib-ui/project_inputs_optional.jpg)

### Verification Process

Before running AMC or VGGT, you must verify the project. Verification is **shared** for both methods.

**How to Verify**

1. Ensure all requirements are met (green checkmarks).
2. Click the **Verify Project** button
3. System validates all files and configurations (rejects nonlinear / stale input)
4. Success message appears: "Project verified successfully"
5. Project state changes to "READY"
6. AMC and VGGT start controls become enabled (when applicable)

![Verify Project](resources/images/vss-autocalib-ui/verify_project.jpg)

> Re-rectification after a previous verify invalidates downstream work: verify again, relaunch calibration, then re-run post-processing.

### Calibration

You can run AMC calibration and VGGT calibration in any order. VGGT calibration runs quickly. AMC calibration is a tracklet-based method, so it takes longer to run. First, run VGGT calibration and check its output. Then run AMC calibration if the video has sufficient people movement. After you run both calibration methods, pick the most accurate calibration from the output visualization in the last step.

### Running AMC Calibration

AMC (Auto Magic Calibration) is the geometry-based / tracklet-based calibration method. It consumes staged rectified media only (no implicit rectification inside the run).

**How to Start**

1. Open the **AMC** tab
2. After verification, select **Detector Type** (`Resnet` or `Transformer`) if shown
3. Click **Start AMC Calibration** (or **Relaunch AMC Calibration** after a prior run)
4. Calibration pipeline begins immediately
5. Project / AMC state changes to "RUNNING"
6. Progress indicators and live AMC logs appear

**Detector selection guidance**

- **ResNet**: Use the default for clear video with good visibility when faster
  calibration is preferred.
- **Transformer**: Use for crowded scenes, frequent occlusions, low contrast, or
  when ResNet produces too few or unreliable person tracks. It generally
  provides stronger detection at the cost of longer runtime.

![AMC Calibration Running](resources/images/vss-autocalib-ui/amc_after_verification.jpg)

**During Calibration**
- **Elapsed Time**: Updates every second
- **Progress Bar**: Animated progress indicator
- **Status Message**: "AMC calibration is running..."
- **Info Alert**: "This may take several minutes. You can close this page and return later."
- **AMC Live Logs**: Real-time calibration logs displayed during execution
- **Auto-refresh**: Status updates every 3 seconds

![AMC Calibration Running](resources/images/vss-autocalib-ui/amc_calib_running.jpg)

**Stopping Calibration**

1. Click **Stop Calibration** (appears when AMC is RUNNING)
2. Calibration process terminates
3. You can relaunch after fixing inputs

> **Warning:** Stopping calibration will discard partial AMC results for that run. VGGT state is not overwritten by AMC state transitions.

### AMC Completion

When AMC finishes successfully:
- **Success Alert**: "✅ AMC Calibration completed successfully!"
- **Message**: Multi-camera — re-run AMC, run VGGT, or proceed to layout post-processing / results; single-camera — re-run AMC or proceed to Results
- **AMC State**: Shows "COMPLETED"
- AMC produces **raw** calibration outputs; layout overlays / GT eval / MV3DT for multi-camera come from shared post-processing

![AMC Calibration Completed](resources/images/vss-autocalib-ui/amc_completed.jpg)

### Calibration Failure

If AMC calibration fails (for example during multi-view **tracklet matching**):
- **Error Alert**: "❌ Calibration failed!"
- **AMC State**: **ERROR** (VGGT state is tracked separately and is not incorrectly overwritten)
- **Reset Option**: "Reset Project" button appears

**How to Recover**

*Option 1: Relaunch AMC Calibration*

1. Click **Relaunch AMC Calibration**
2. Or re-verify if inputs / rectification changed
3. Start AMC again

*Option 2: Run VGGT instead (multi-camera)*

VGGT does **not** require a successful AMC run. After verify + completed rectification, open the **VGGT** tab and start VGGT independently.

*Option 3: Reset Project*

1. Click **Reset Project**
2. Project state returns to "INIT"
3. Go back to previous steps, fix inputs, re-rectify if needed, then verify again

![Project Reset](resources/images/vss-autocalib-ui/project_reset.jpg)

### VGGT Calibration

VGGT (Vision-Geometry Graph Transformer) is an optional **multi-camera** calibration method. It runs directly from staged linear camera videos and is **not gated on AMC success**.

> VGGT requires **VGGT support on the backend** (model and dependencies installed). It is **not offered for single-camera** projects.

**When Available**
- **Multi-camera only** (two or more videos)
- Backend has VGGT installed
- Rectification state is **COMPLETED**
- Project is verified (AMC/E2E is optional and not required)

**How to Run VGGT**

1. Open the **VGGT** tab
2. Confirm project is verified and rectification is complete
3. Click **Start VGGT Calibration** (or **Relaunch VGGT Calibration**)
4. Progress indicators and live VGGT logs appear

**VGGT Features**
- **Independent of AMC**: Can run without a prior or successful AMC run
- **Relaunchable**: Overwrites prior VGGT raw outputs on re-run
- **Duration**: Typically a few minutes
- Produces **raw** calibration outputs; shared post-processing builds layout-derived artifacts

![VGGT Calibration](resources/images/vss-autocalib-ui/vggt_calib.jpg)

**VGGT Completion**

When VGGT finishes successfully:
- **Success Alert**: "✅ VGGT calibration completed successfully!"
- **Message**: Re-run VGGT, or proceed to layout post-processing / results
- **VGGT State**: **COMPLETED**

**VGGT Not Available**

If VGGT is not installed on the backend:
- **Info Alert**: "VGGT Calibration Not Available"
- **Action**: Use AMC results, or install VGGT support and retry

### Layout Post-processing (Multi-camera)

After AMC and/or VGGT complete, run **shared layout post-processing**. This stage is separate from AMC/VGGT launchers and handles:

- Layout scaling / world conversion
- Trajectory overlays
- Optional GT evaluation
- MV3DT exports
- Virtual-GT overlays

**How to Run**

1. Complete AMC or VGGT (or both) and ensure layout alignment is present
2. Click **Run Post-processing** (or **Re-run Post-processing**)
3. Wait for **postprocess** state **COMPLETED**
4. Open **Results** for trajectory / virtual-object imagery and exports

Post-processing is disabled while AMC or VGGT is RUNNING, and stays in INIT until at least one calibration path and alignment are ready. Re-run after another calibration, alignment, scale, or rectification change.

![SV Post Process](resources/images/vss-autocalib-ui/post_processing.jpg)

> Single-camera projects use the AMC single-view post-processing path; the shared multi-camera Layout Post-processing card is shown for two or more cameras.

### Calibration Information

At the bottom of the AMC panel, you'll see a summary:
- **Project ID**: Unique identifier
- **Videos**: Number of cameras
- **Focal Lengths**: Provided or Not provided
- **AMC State**: Current AMC state
- **VGGT State**: Current VGGT state (multi-camera)

![Calibration Information](resources/images/vss-autocalib-ui/calib_info.jpg)

### Best Practices

**Before Calibration**
- Complete rectification (Step 4) and confirm linear media
- Double-check uploaded files and floor-plan scale
- Verify alignment points are accurate (on rectified frames)
- Ensure stable network connection

**After Calibration**
- Run layout post-processing (multi-camera) before expecting trajectory / virtual-GT Results imagery
- Run or relaunch VGGT independently when desired
- Export results before making changes
- After re-rectification: verify → relaunch calibration → re-run post-process

### Troubleshooting

**Verification Fails**
- Check that all required files are uploaded
- Confirm rectification is COMPLETED
- Ensure video files are not corrupted
- Verify alignment data has at least 4 point sets
- Try re-uploading files

**Calibration Takes Too Long**
- Normal duration: 5–15 minutes depending on video length
- Check server resources (CPU, GPU, memory)
- Verify network connection is stable
- Contact administrator if it exceeds 30 minutes

**Calibration Fails**
- Check video file formats and quality
- Verify alignment points are on the ground plane (rectified space)
- Ensure layout image matches physical space
- Review server logs for detailed errors
- Try the other calibration path (AMC ↔ VGGT) independently

---

## Step 7: Results

View calibration results, draw optional ROIs/tripwires on rectified frames, evaluate accuracy, and export calibration data.

![Results Step](resources/images/vss-autocalib-ui/results_step.jpg)

### Results Availability

The **Results** step is available when at least one calibration path has completed successfully:

- **AMC completed successfully**, or
- **VGGT completed successfully** (even if AMC was never run or failed)

For multi-camera trajectory / virtual-GT overlays and evaluation metrics, run **Layout Post-processing** on Execute after calibration.

**If No Results Are Available Yet**
- **Running**: "Calibration is still running — Please wait for calibration to complete."
- **Error**: "Calibration failed — Please check your input files and try again." (if neither AMC nor VGGT has completed)
- **Init/Ready**: "Please run calibration in the Execute step"

![Results Not Ready](resources/images/vss-autocalib-ui/step6_if_amc_is_running.jpg)

### Overlay Image

The overlay image shows calibration results on the layout map. Switch overlay source between:

- **Trajectory Overlay** — reconstructed object trajectories. Generated only if the **AMC** calibration method has run, because this overlay uses AMC tracklets.
- **GT Virtual Objects** — virtual-GT object imagery per camera (when post-processing produced them)

> **VGGT-only behavior:** **VGGT overlay image not available** is expected when
> VGGT runs without AMC. VGGT does not create AMC tracklet-matching artifacts,
> which are required for trajectory overlays; evaluation metrics, camera
> parameters, and exports remain available. If AMC also runs, its artifacts
> enable trajectory overlays for VGGT results.

**Features**
- **View**: Displays cameras' fields of view / trajectories on the layout (or virtual-GT per camera)
- **Download**: Save the overlay image
- **Result Type Tabs**: Switch between AMC and VGGT results (if available)

**How to View**
1. The overlay image loads automatically when post-processed artifacts exist
2. Toggle **Trajectory Overlay** / **GT Virtual Objects**
3. Use AMC / VGGT tabs to switch result types
4. Click **Download** to save the image

![Overlay Image](resources/images/vss-autocalib-ui/overlay_image.jpg)

**GT Virtual Objects**
- Per-camera overlay produced by layout post-processing: a **green** world floor grid (`z = 0`, typically 1 m spacing) and **red** fixed-height poles (current default **1.75 m**) at each grid intersection, both projected with the calibrated camera matrix (AMC or VGGT).
- Use it as a visual check — if calibration is good, the green grid sits on the real floor and the red poles look upright and correctly scaled; a sliding grid or leaning poles usually means that camera’s pose is off. Requires ground truth / post-processed virtual-GT artifacts; pick a camera to inspect each view, then compare AMC vs VGGT if both exist.

![GT Virtual Objects Image](resources/images/vss-autocalib-ui/GT_virtual_object.jpg)

### Evaluation Metrics

If ground truth data was uploaded and post-processing ran, evaluation metrics are available.

**Metrics Display**
- **Layout Visualization**: 3D points plotted on layout showing accuracy
- **Statistics Card**: L2 distance statistics in meters
  - Average L2 distance
  - Standard deviation
  - Maximum distance
  - Minimum distance
- **Result Type Tabs**: Switch between AMC and VGGT evaluation

![Evaluation Metrics](resources/images/vss-autocalib-ui/evaluation_metrics.jpg)

**Interpreting Metrics**
- **Lower Average**: Better calibration accuracy
- **Lower Std Dev**: More consistent calibration
- **Compare AMC vs VGGT**: VGGT often shows improvement when both are available

> Evaluation metrics are only available if ground truth data was uploaded in Step 2 and layout post-processing completed.

### Camera Parameters

View detailed calibration parameters for each camera.

**Features**
- **Camera Tabs**: Switch between cameras (Camera 0, Camera 1, etc.)
- **Result Type Tabs**: Switch between AMC and VGGT parameters
- **YAML Format**: Parameters displayed in YAML format
- **Export Button**: Export all camera parameters

**How to View**
1. Click on a camera tab (e.g., "Camera 0")
2. Parameters load and display in a code block
3. Switch between AMC and VGGT tabs to compare
4. Click **Export AMC** or **Export VGGT** to download all parameters

![Camera Parameters](resources/images/vss-autocalib-ui/cam_params.jpg)

**Parameter Contents**

The YAML file contains:
- **Camera Projection Matrix (3×4)**: Camera projection matrix
- **Additional Metadata**: Project ID, timestamp, etc.

### ROI & Tripwire Drawing (Optional)

ROI and tripwire editing is an **optional post-calibration** activity on Results. Draw on **rectified** camera frames or the layout map. Annotations save automatically.

#### Annotation Target

Choose **Camera** or **Global ROIs / tripwires** at the top left of the card before drawing.

![Annotation Target](resources/images/vss-autocalib-ui/annotation_target.jpg)

**Camera (per-stream annotations)**

1. Select **Camera** in the annotation target toggle
2. Choose a stream from the **Select Camera** dropdown
3. The first frame of the selected **rectified** video loads on the canvas
4. Switch between cameras to draw ROIs, tripwire lines, and tripwire directions on each one

![Camera Selection](resources/images/vss-autocalib-ui/cam_selection.jpg)

**Global ROIs / tripwires (layout-map annotations)**

1. Upload a layout image in Step 2 (required for this mode)
2. Select **Global ROIs / tripwires** in the annotation target toggle (disabled until a layout is uploaded)
3. The layout map loads on the canvas instead of a video frame
4. Use the same drawing tools as for camera mode; global shapes are stored separately and listed as **Global (layout map)** in the right panel

> Global and per-camera annotations use the same tools and auto-save behavior.

#### Drawing Tools

**Available Tools**
- **Draw ROI**: Create polygonal regions of interest
- **Draw Tripwire**: Create tripwire lines for counting
- **Tripwire Direction**: Create directional tripwires with arrows
- **Show/Hide**: Toggle visibility of annotations
- **Reset**: Clear all annotations for the active target (current camera or global layout map)

![Drawing Tools](resources/images/vss-autocalib-ui/drawing_tools.jpg)

#### Drawing ROIs

ROIs define areas of interest for detection and tracking.

**How to Draw**

1. Click the **Draw ROI** button (it becomes highlighted)
2. Click on the video frame to add points
3. Add at least 3 points to form a polygon
4. Finish the ROI by pressing the `F` key or right-clicking on the canvas
5. The ROI is automatically saved with a green color

**ROI Features**
- **Color**: Green (#00ff00)
- **Minimum Points**: 3
- **Maximum Points**: Unlimited
- **Auto-save**: Saved immediately upon completion

![ROI Drawing](resources/images/vss-autocalib-ui/roi_drawing.jpg)

**Editing ROIs**
- **Delete**: Click the delete button next to the ROI in the right panel
- **Redraw**: Delete the existing ROI and draw a new one

#### Drawing Tripwire Lines

Tripwire lines are used for counting objects crossing a line.

**How to Draw**

1. Click the **Draw Tripwire** button
2. Click once to set the start point
3. Click again to set the end point
4. The tripwire line is automatically saved with a red color

**Tripwire Line Features**
- **Color**: Red (#ff0000)
- **Points**: Exactly 2 (start and end)
- **Auto-save**: Saved immediately upon completion
- **Use Case**: Bidirectional counting

![Tripwire Line](resources/images/vss-autocalib-ui/tripwire_line.jpg)

#### Drawing Tripwire Directions

Tripwire directions are used for unidirectional counting with an arrow indicator.

**How to Draw**

1. Click the **Tripwire Direction** button
2. Click once to set the start point
3. Click again to set the end point (direction of arrow)
4. The tripwire direction is automatically saved with a yellow color and arrow

**Tripwire Direction Features**
- **Color**: Yellow (#ffff00)
- **Arrow**: Shows direction from start to end
- **Points**: Exactly 2 (start and end)
- **Auto-save**: Saved immediately upon completion
- **Use Case**: Unidirectional counting (e.g., entry/exit)

![Tripwire Direction](resources/images/vss-autocalib-ui/tripwire_direction.jpg)

#### Canvas Controls

**Zoom and Pan**
- **Scroll Wheel**: Zoom in/out on the canvas
- **Click + Drag**: Pan around when zoomed in
- **Show/Hide Button**: Toggle visibility of all annotations
- **Reset Button**: Clear all annotations for the active annotation target

**Visual Feedback**
- **Drawing Mode**: Active tool is highlighted in the toolbar
- **Cursor**: Changes to crosshair when in drawing mode
- **Point Markers**: Visible while drawing
- **Completed Annotations**: Rendered with solid colors

#### Annotation List (Right Panel)

The right panel shows all annotations for the active target—the selected **Camera** or **Global (layout map)**.

- **ROIs Section**: Count of completed ROIs; each ROI shown as a green chip with point count; delete button for each
- **Tripwire Lines Section**: Count of completed tripwire lines; each line shown as a red chip; delete button for each
- **Tripwire Directions Section**: Count of completed tripwire directions; each direction shown as a yellow chip with arrow; delete button for each

![Annotation List](resources/images/vss-autocalib-ui/annotation_list.jpg)

#### Export Image-Mode JSON

Use this when you need ROIs and tripwires in **pixel coordinates** without world-coordinate conversion.

**How to Export**

1. In the right panel, open the **Export image-mode JSON** card
2. Click **Export image-mode JSON**
3. The browser downloads `<project_name>_image_mode_exported.json`

![Export Image-Mode JSON](resources/images/vss-autocalib-ui/export_image_mode.jpg)

**What Is Exported**

- ROIs and tripwires from all cameras plus any **global** layout annotations, in **pixel space**
- `calibrationType` is **image** (same JSON shape as cartesian export)
- Drawn on **rectified** frames / layout map on the Results step


### ROI & Tripwire Verification

Verify per-camera and **global** ROIs and tripwires: how they appear on each rectified stream, on the layout map (pixel space), and on the **world map** (BEV) after calibration projection.

**Features**
- **Side-by-side layout**: Left panel (camera or layout map) and **World map (layout + projection)** on the right
- **Annotation target**: **Camera** (per-stream rectified view) or **Global ROIs / tripwires** (layout `layout.png` pixels; requires layout from Step 2)
- **Result type tabs** (right panel): **AMC** or **VGGT** world-map projection when both calibrations completed
- **Global features on world map**: Global ROIs and tripwires projected into world coordinates and drawn in **purple** on the BEV
- **Global features on camera view**: The same global ROIs/tripwires **re-projected** onto the selected camera when visible in that field of view—also **purple**
- **Sensor assignment checkboxes** (camera target): Include or exclude the selected camera from each global ROI/tripwire `sensors` list in the export JSON

**How to Use**

1. Click **Show ROI & Tripwire Verification**
2. Choose **Camera** or **Global ROIs / tripwires** in the annotation target toggle
3. **Camera** target: pick a stream from **Select Camera**; review the left rectified frame and the right world map; use **AMC** / **VGGT** tabs on the world map as needed
4. **Global** target: left panel shows global shapes on **layout.png** (pixel coordinates); right panel shows their **world-map** projection in purple
5. Under the camera view, use checkboxes labeled **Global ROI:** / **Global tripwire:** to include or exclude that camera in each ROIs/Tripwires `sensors` list (only global items projected to that camera are listed)
6. Use world-map zoom controls for detail; click **Close** when finished

![Show ROI and Tripwires](resources/images/vss-autocalib-ui/show_roi_and_tripwires.jpg)

![ROI and Tripwire Verification](resources/images/vss-autocalib-ui/roi_and_tripwire_verification.jpg)

**Left Panel — Camera target**
- Rectified video frame for the selected camera
- **Per-camera** annotations: ROIs (green polygons), tripwire lines (red), tripwire directions (yellow arrows)
- **Global** ROIs (purple regions) and global tripwires (purple lines) when they project into this camera's view
- **Global sensor checkboxes**: For each global ROI/tripwire that applies to this camera, check to keep the camera in that feature's `sensors` list in the export JSON; uncheck to exclude it

**Left Panel — Global target**
- **Layout map (pixel coordinates)**: Global ROIs and tripwires drawn on `layout.png` (green / red / yellow in layout space)
- Compare with the right panel to confirm world projection matches the layout drawing

**Right Panel — World map (layout + projection)**
- Bird's-eye / world-coordinate map with all projected annotations for the active **AMC** or **VGGT** result
- **Per-camera** ROIs and tripwires for all streams, plus **global** ROIs and tripwires in **purple**
- Zoom: 50% to 500%; pan by dragging when zoomed

**Zoom Controls**
- **Zoom In** (🔍+): Increase zoom level
- **Zoom Out** (🔍-): Decrease zoom level
- **Reset** (↻): Return to 100% zoom
- **Current Zoom**: Displayed as percentage

![BEV Zoom Controls](resources/images/vss-autocalib-ui/bev_zoom_controls.jpg)

### Export Calibration Data

Export complete calibration data in various formats.

**Export Options**

1. **Full Export** — Opens a dialog for optional metadata and download. Complete calibration JSON with ROI/tripwire **world coordinates**. **Choose AMC or VGGT inside the dialog** when both calibrations completed (toggle: **Download / edit AMC export** vs **Download / edit VGGT export**); otherwise the available result type is used automatically. **AMC** uses the AMC projection matrix; saved/downloaded as `{project_name}_exported.json`. **VGGT** uses the VGGT projection matrix (multi-camera, when VGGT completed); saved/downloaded as `{project_name}_exported_vggt.json`.
2. **MV3DT ZIP AMC** — MV3DT-compatible format for verification; ZIP archive; filename: `{project_name}_mv3dt.zip`
3. **MV3DT ZIP VGGT** *(if available)* — MV3DT-compatible format with VGGT results; ZIP archive; filename: `{project_name}_vggt_mv3dt.zip`
4. **Delete Results** — removes all calibration results; project returns to READY state; allows re-running calibration

![Export Options](resources/images/vss-autocalib-ui/export_options.jpg)

**How to Export**

- **Full Export**: Click **Full Export**, choose **AMC** or **VGGT** in the dialog when both are available, optionally edit metadata or open the manual JSON editor by clicking **Full Control**, then click **Download JSON**. The file downloads to your browser's download folder.

![Export JSON Editor](resources/images/vss-autocalib-ui/export_json.jpg)

> This is an advanced user feature. Edit the JSON only if you understand the calibration schema; any changes should be made carefully to avoid invalid or incorrect calibration output.

- **Reset to Original Calibration** (Advanced Export Metadata): In the Full Export dialog, discard Full Control edits and structured metadata changes (including global ROI/tripwire, sensor group, and region metadata) and regenerate the project export from the original calibration results. Confirm in the **Reset to Original Calibration?** dialog — this cannot be undone.

- **Other exports (MV3DT ZIP)**:

  1. Click the desired export button
  2. Wait for processing (may take a few seconds)
  3. File downloads automatically to your browser's download folder
  4. Success message confirms export

> **Export Options Explained:**
> - **Full Export**: Complete calibration with ROI/tripwire world coordinates; pick AMC or VGGT in the dialog when both exist
> - **Reset to Original Calibration**: Restore export JSON from original calibration output
> - **MV3DT ZIP**: MV3DT-compatible format for verification (separate AMC and VGGT buttons)


### Deleting Results

If you need to re-run calibration with different parameters:

1. Click **Delete Results** button
2. Confirm deletion in the dialog
3. All calibration results are removed
4. Project state returns to "READY"
5. Files (videos, layout, alignment) and staged linear media remain

> **Warning:** Deleting results cannot be undone. Export important data before deletion.

### Completion Message

At the bottom of the page, a success message confirms calibration is complete.

![Calibration Complete](resources/images/vss-autocalib-ui/calib_completed.jpg)

**Message**
- **Title**: "🎉 Calibration Complete!"
- **Text**: "All calibration results are ready. You can export the data and use it in your applications."

### How to Interpret Calibration Outputs

Upon completion, the UI presents overlay images and metric numbers depending on whether ground truth data was provided and post-processing ran.

**Case 1: Ground Truth Data Exists**

If ground truth data was uploaded, the tool calculates the **L2 distance** as the primary evaluation metric — the Euclidean distance between the 3D ground truth object location and the estimated location determined by triangulation.

Statistics displayed:
- **Average**: Mean L2 distance across all points
- **Standard Deviation**: Measure of consistency
- **Maximum**: Worst-case error
- **Minimum**: Best-case error

Since a lower L2 distance indicates better accuracy, compare these metrics between AMC and VGGT results to select the superior calibration.

Additionally, calibration results from the two methods can be compared visually using the overlay visualization. Object trajectories reconstructed using the camera matrices are shown as colored lines; ground truth trajectories are displayed in white. A close alignment of the colored trajectories with the white lines signifies accurate camera parameters.

> When comparing AMC and VGGT results: look for lower L2 distance values (better accuracy), compare overlay images for trajectory alignment, and check consistency of colored lines with white ground truth lines.

**Case 2: No Ground Truth Data**

When ground truth data is unavailable, calibration results can be compared qualitatively using overlay images, which display:
- **Reconstructed object trajectories**: Shown as colored lines
- **Estimated camera locations**: Shown as colored dots with corresponding camera IDs
- **Virtual-object overlays** (when generated): Qualitative check of projected virtual GT objects per camera

**Qualitative Evaluation Tips:**
- Camera positions should match expected physical locations
- Object trajectories should follow logical paths on the floor map
- FOV (Field of View) boundaries should align with physical constraints
- Compare AMC and VGGT overlays to identify which better matches the layout

### Best Practices

**Reviewing Results**
- Confirm layout post-processing completed (multi-camera) before judging overlays
- Check trajectory and virtual-object overlays
- Verify evaluation metrics if ground truth is available
- Compare AMC and VGGT results if both available
- Review camera parameters for reasonableness

**Exporting Data**
- Export both AMC and VGGT results for comparison
- Use **Reset to Original Calibration** if Full Control metadata edits should be discarded
- Keep MV3DT ZIP for verification purposes
- Store exports with descriptive names and dates
- Maintain backups of important calibration data

**Verification**
- Draw ROIs/tripwires on Results when needed, then verify projections
- Check all cameras, not just one
- Use zoom to inspect details
- Compare AMC vs VGGT projections

**Before Deleting**
- Export all needed data first
- Verify exports are complete and valid
- Document any issues or observations
- Consider keeping project for reference

### Next Steps

After completing calibration:
- Use exported data in your surveillance application
- Integrate calibration parameters with your tracking system
- Set up ROIs and tripwires in your production environment
- Monitor and validate calibration accuracy in real-world scenarios


# Assumptions

AutoMagicCalib makes several assumptions about input data structure. Please ensure your data follows these requirements before you deploy the service or record footage.

## Tracklet-Based Calibration: Input Video Requirements

AutoMagicCalib (AMC) estimates camera parameters by detecting and tracking **people** in your footage, then matching those trajectories (**tracklets**) across camera views. Calibration quality depends heavily on input video content.

> **Note:** These requirements apply to the **tracklet-based AMC pipeline** (the default calibration path). This is not applicable to the optional VGGT model-based calibration path.

**Video content (required for AMC)**

- **Moving people**: People must be clearly visible and moving throughout the clip. AMC relies on PeopleNet detection and 3D tracking of people in the scene.
- **Scene coverage**: Trajectories should span as much of each camera's field of view as possible. More unique walkers and broader coverage produce more usable tracklets.
- **Recommended headcount**: Plan for **many** moving people in the scene. As a practical guideline, aim for **at least 10 unique individuals** walking through the monitored area during the recording window (more is better for multi-camera overlap).

**Video duration**

- **Recommended**: **five minutes or longer** per camera. There is no strict minimum, but given the size of the space to calibrate and normal walking speed, longer clips let the tracker build stable trajectories across the field of view; short clips often fail during multi-view **tracklet matching**.

**Resolution and format**

- **Resolution**: **1920 × 1080**
- **Format** (file upload): MP4 . AutoMagicCalib does **not** support video formats that are not supported by DeepStream.

**Multi-camera specifics**

- **Camera count**: Two or more time-synchronized videos or RTSP streams (one stream for single-camera calibration).
- **Time synchronization**: All cameras must cover the **same time window**. Use clips recorded in sync or post-processed to be in sync. For RTSP, use one combined **Capture and add to project** for every stream—staggered or later-added streams break calibration.
- **FOV order**: List cameras in **order of overlapping field of view** (`cam_00` overlaps `cam_01`, `cam_01` overlaps `cam_02`, and so on).
- **FOV overlap**: For the best calibration quality, more overlap between consecutive camera pairs is better. Aim for **at least 30% FOV overlap**; insufficient overlap reduces matched tracklets and often causes multi-view calibration to fail.

**Tracklet thresholds (why the above matters)**

AMC filters and matches tracklets before multi-view calibration. Default pipeline settings include:

- **Minimum tracklet length**: 90 frames (approximately three seconds at 30 fps)—short or jittery tracks are discarded.
- **Minimum matched tracklets** (multi-camera): six cross-camera tracklet correspondences per camera pair (three in robust two-view fallback).

If your videos lack enough moving people or are too short, calibration may fail at detection, tracking, or **tracklet matching**. You may need to adjust the configuration parameters through the UI or capture better videos. See [Custom Dataset](#custom-dataset) for capture best practices and [Troubleshooting](#troubleshooting) if tracklet matching fails.

## Lens Distortion Output (`distortion.yaml`)

AMC can accept videos that contain **lens distortion**. Distortion handling is a **dedicated Rectification stage (Step 4)** with Auto, Manual, or **Videos Are Rectified** paths. AMC and VGGT consume staged `rectified.mp4` / `rectified.jpg` under each camera's single-view folder; they **do not** perform implicit in-calibration rectification.

When Auto or Manual rectification generates linear media (distortion model `simple_divisional`, `simple_radial`, or `radial`), the pipeline:

1. Estimates or accepts per-camera distortion parameters (`model`, **k1**, and **k2** when model is `radial`)
2. **Explicitly** writes full rectified videos / stills and canonical `distortion.yaml` under `single_view_results/cam_XX/`
3. Unlocks alignment and calibration only after rectification state is **COMPLETED**

Use **Videos Are Rectified** when inputs are already linear; that path stages source videos as linear media and clears prior distortion caches.

**Important for downstream applications**

Rectified videos (`rectified.mp4`) are produced under each camera's single-view output folder on the server, but they are **not** automatically substituted for your original camera feeds in downstream applications. Exported calibration JSON and MV3DT ZIP files describe cameras in the **rectified (undistorted) image space**.

To stay consistent with AMC calibration results, downstream applications must either:

- Prefer the per-camera `rectified.mp4` files from the calibration output (downloadable from the Rectification step as a ZIP; not wired automatically by the microservice), **or**
- Apply `distortion.yaml` using the model-specific remapping in [Distortion model](#distortion-model) before using AMC projection matrices (do **not** assume OpenCV `cv2.undistort()`)

**Output location**

Per camera (example paths on the calibration server):

```text
<project_output>/single_view_results/cam_00/distortion.yaml
<project_output>/single_view_results/cam_00/geocalib_distortion.yaml   # when GeoCalib estimates distortion
<project_output>/single_view_results/cam_00/rectified.mp4
<project_output>/single_view_results/cam_00/rectified.jpg
```

**`geocalib_distortion.yaml` format** (when Auto distortion estimation runs)

GeoCalib writes a richer record that includes the distortion model name and both coordinate conventions:

```yaml
model: simple_divisional    # or simple_radial, radial (must match rectification config)
k1: -1.2345678e-06          # AMC pixel-centered convention (meaning depends on model)
k1_geocalib: -4.5678901e-03 # GeoCalib internal normalized coordinates
focal_length: 1269.0        # focal length used for k1 conversion
source: geocalib
k2: ...                     # present only when model is radial
```

### Distortion model

AMC supports:

- `simple_divisional` — default
- `simple_radial`
- `radial`

Each camera’s `distortion.yaml` records the selected model and coefficients:

```yaml
model: simple_divisional
k1: ...
```

The `radial` model also includes `k2`. Coefficient meanings depend on `model`; in particular, `simple_divisional.k1` is not an OpenCV radial coefficient.

Parameters use full-resolution pixel coordinates with:

```text
cx = width // 2
cy = height // 2
```

#### Reproducing AMC rectification

AMC does not pass these coefficients to an OpenCV distortion model. It computes model-specific destination-to-source maps itself and uses `cv2.remap()` only to resample the image.

For each rectified output pixel `(x, y)`:

```text
dx = x - cx
dy = y - cy
r² = dx² + dy²

simple_radial:
s = 1 + k1*r²

radial:
s = 1 + k1*r² + k2*r⁴

simple_divisional:
s = (1 - sqrt(max(0, 1 - 4*k1*r²))) / (2*k1*r²)
s = 1 when |2*k1*r²| is near zero

source_x = cx + s*dx
source_y = cy + s*dy
```

Use `source_x` and `source_y` as the OpenCV remap coordinates:

```python
rectified = cv2.remap(
    distorted,
    map_x.astype(np.float32),
    map_y.astype(np.float32),
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
)
```

`cv2.remap()` maps destination pixels to source pixels, so the equations are evaluated on the rectified output grid.

Prefer using AMC’s generated rectified video directly. It already matches AMC calibration space and must not be undistorted again. If processing raw video separately, read `model` and all applicable coefficients from `distortion.yaml`; do not pass `simple_divisional` coefficients to `cv2.undistort()`.

Apply coefficients at their original image resolution—prefer rectifying before resizing. When **Videos Are Rectified** is selected, rectification is bypassed and `distortion.yaml` is not required.

Changing or repeating rectification invalidates downstream calibration and post-processing outputs.

See also [Custom Dataset](#custom-dataset) (lens distortion guidance) and the AutoMagicCalib `rectification_config.yaml` in the source repository for tuning search ranges and model selection.

# Custom Dataset

For a custom dataset, you should prepare the following items:

- **Input videos or RTSP streams** — Camera video files **or** time-synchronized RTSP streams
- **A floor map** — Layout/map image of the surveillance area (PNG)
- **Alignment data** — `alignment_data.json` (upload or create in the UI; see [Alignment Data](#alignment-data-alignment_datajson))
- **Layout Pixels Per Meter** — Number of pixels per meter in the layout floor map. Initialize this on [Step 3: Parameters](#step-3-parameters) (Floor-plan to 3D scale) or under [Configuring Settings](#configuring-settings).
- **Ground truth data (optional)** — For calibration evaluation
- **Rectified / linear media** — Complete [Step 4: Rectification](#step-4-rectification) (Auto, Manual, or Videos Are Rectified) before alignment and calibration

## Input Requirements

**Video input (file upload or RTSP)**

- **Formats** (file upload): MP4 . AutoMagicCalib does **not** support video formats that are not supported by DeepStream.
- **Resolution**: **1920 × 1080** is required for uploaded videos (matches the workflow and evaluation pipeline)
- **Camera count**: One video or stream for **single-camera** calibration; **two or more** for multi-camera
- **Time synchronization**: All multi-camera videos or RTSP streams must cover the **same time window**—use one combined RTSP **Capture and add to project** for every stream, or upload clips that were recorded in sync. Staggered or later-added streams break calibration.
- **Order**: List or upload streams in **order of overlapping field of view (FOV)** (first = first camera in the overlap chain), whether using files or RTSP URLs.

**Single-camera datasets**

A valid single-camera project needs:

- One synchronized video (or one RTSP URL)
- One layout/map image
- `alignment_data.json` with at least **4** point sets; each set has **two** `[x, y]` pairs (camera + layout/BEV)—see [Alignment Data](#alignment-data-alignment_datajson)

**Multi-camera datasets**

- Two or more time-synchronized videos or RTSP streams
- One layout/map image
- `alignment_data.json` with at least **4** point sets; each set has one point per camera plus the layout (see [Alignment Data](#alignment-data-alignment_datajson))

Users should pay close attention to upload and stream order, as this order implicitly determines camera pairing. For optimal results, consecutive camera pairs should have a significant amount of overlapping Field of View (FOV).

## Alignment Data (alignment_data.json)

Alignment data maps corresponding points between camera views and the layout (bird's eye view). Prepare or create this file before running calibration.

**Requirements**

- Minimum **4** complete point sets
- Layout coordinates are always pixel positions on the layout / BEV image
- **Camera coordinate space depends on how you provide the file** (see below)

**Multi-camera** — three `[x, y]` pairs per set (camera 0, camera 1, layout):

```json
[
  [[x0_cam0, y0_cam0], [x0_cam1, y0_cam1], [x0_layout, y0_layout]],
  [[x1_cam0, y1_cam0], [x1_cam1, y1_cam1], [x1_layout, y1_layout]]
]
```

**Single-camera** — two `[x, y]` pairs per set (camera, layout):

```json
[
  [[x0_cam, y0_cam], [x0_layout, y0_layout]],
  [[x1_cam, y1_cam], [x1_layout, y1_layout]]
]
```

For projects with more than two cameras, each point set includes one pair per camera plus the layout point (same pattern as multi-camera above, extended to all views).

**Coordinate space (important)**

| How you provide alignment | Expected camera coordinates | What the microservice does |
|---|---|---|
| **UI file upload** (`Upload alignment_data.json`) | **Original / pre-rectification** (distorted) pixel space | Uploads with `coord_space=original`; converts to rectified space after Step 4 when needed |
| **Interactive draw / save** in the Manual Alignment tool | **Rectified / linear** pixel space | Saves with `coord_space=rectified` (points are clicked on rectified frames) |
| **API upload** | Follow the `coord_space` query parameter you pass (`original` or `rectified`) | Converts only when `coord_space=original` |

Do **not** upload already-rectified camera points through the UI file-upload path — that triggers a second conversion and corrupts alignment. For interactive creation and the on-disk rectified alignment used by calibration, points must match the **rectified / linear** frames from Step 4. See [Step 5: Manual Alignment](#step-5-manual-alignment).

## Guidelines for Input Videos to Achieve Optimal Calibration Results

To ensure the most accurate camera calibration, careful consideration should be given to how the input videos provided. The following points detail how to maximize the quality of the calibration outcome.

### 1. Minimizing Lens Distortion

The current calibration methodology performs best when input videos are "linear," meaning they exhibit no lens distortion. Use **Step 4: Rectification** (Auto or Manual) when distortion is present, or **Videos Are Rectified** when footage is already linear. While the tool can handle minor distortion after rectification, optimal results are achieved when the staged linear media has negligible residual distortion.

### 2. Maximizing Camera Overlap

Accurate calibration requires a significant degree of overlap between the fields of view of the different cameras. It is essential to maximize the overlap between cameras as much as possible.

### 3. Leveraging Unique Scene Features

The presence of diverse and unique objects in the input videos contributes significantly to calibration accuracy. Our automatic calibration tool specifically utilizes people moving within the field of view, so videos with many moving people are ideal. The trajectories of these moving subjects should cover the Field of View (FOV) as broadly as possible.

Additionally, large, unique objects can enhance accuracy. For instance, in a setting like a warehouse with multiple cameras, views can become challenging due to repetitive elements (e.g., similar racks). In such environments, large, distinct objects, like forklifts, are beneficial for better calibration accuracy.

## Ground Truth Data Format

If you want to evaluate the camera calibration results using ground truth data, you should have a ZIP file containing the following data files:

- `calibration.json`
- `ground_truth.json`

### calibration.json

This file has camera parameters including intrinsic and extrinsic parameters. The JSON schema definition for calibration is as follows:

```json
{
   "sensors": [
       {
           "id": "Camera",
           "intrinsicMatrix": [
               [1269.00511584492, -3.730349362740526e-14, 959.9999999999999],
               [0.0, 1269.0051158449194, 539.9999999999999],
               [0.0, 0.0, 0.9999999999999998]
           ],
           "extrinsicMatrix": [
               [0.9999941499743863, 0.0020258073539418126, 0.00275610623331978, 7.506433779240641],
               [0.00329149786382878, -0.3506837842628175, -0.9364881470135763, 1.2002890745303207],
               [-0.0009306228113685242, 0.936491740251709, -0.3506884006942753, 11.111379874347342]
           ],
           "attributes": [
               {"name": "frameWidth", "value": 1920},
               {"name": "frameHeight", "value": 1080}
           ],
           "cameraMatrix": [
               [1268.1042942335746, 901.6028305375089, -333.16335175660936, 20192.627546980937],
               [3.6743913098523424, 60.686023462551134, -1377.7799858632666, 7523.318108219307],
               [-0.0009306228113685238, 0.9364917402517088, -0.35068840069427526, 11.111379874347342]
           ]
       },
       {
           "id": "Camera_01",
           "intrinsicMatrix": [
               [1099.498973963849, -4.707345624410664e-14, 960.0],
               [0.0, 1099.4989739638488, 539.9999999999998],
               [0.0, 0.0, 1.0]
           ],
           "extrinsicMatrix": [
               [-0.9999609312669344, -0.008839453589732555, 5.147844000033541e-11, -7.521032053009582],
               [-0.004417374837733223, 0.4997143960386968, -0.866178970647073, -0.1501353870483639],
               [0.007656548785712605, -0.8661451301323095, -0.49973392001021566, 10.265551144735602]
           ],
           "attributes": [
               {"name": "frameWidth", "value": 1920},
               {"name": "frameHeight", "value": 1080}
           ],
           "cameraMatrix": [
               [-1092.1057310976453, -841.2182950793291, -479.7445631532065, 1585.5620735129166],
               [-0.7223627574165982, 81.71709544806465, -1222.2192063010361, 5378.3239141418835],
               [0.0076565487857126035, -0.8661451301323094, -0.4997339200102156, 10.2655511447356]
           ]
       }
   ]
}
```

**Parameter Descriptions:**

| Parameter | Description |
|---|---|
| `id` | Unique string identifier for the sensor (e.g., Camera, Camera_01, Camera_02, …). **Must match exactly** the camera keys used under `2d bounding box visible` in `ground_truth.json` for that sensor—mismatched IDs will break evaluation. |
| `intrinsicMatrix` | 3×3 camera intrinsic parameter matrix. Follows the same definition in [OpenCV documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). |
| `extrinsicMatrix` | 3×4 camera extrinsic parameter matrix. Follows the same definition in [OpenCV documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). |
| `cameraMatrix` | 3×4 combined camera projection matrix. Follows the same definition in [OpenCV documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). |
| `attributes` | Array of name-value pairs for additional sensor attributes. `frameHeight`: image height resolution, `frameWidth`: image width resolution. |

### ground_truth.json

This file has object information including 3D locations and bounding boxes. The JSON schema definition for ground truth object data is as follows:

```json
{
    "0": [
        {
            "object id": 0,
            "object type": "person",
            "object name": "male_adult_police_04",
            "3d location": [-7.82265567779541, 4.5983476638793945, -9.851457150045206e-11],
            "2d bounding box visible": {
                "Camera": [912, 362, 955, 507],
                "Camera_01": [960, 664, 1062, 941]
            }
        },
        {
            "object id": 2,
            "object type": "person",
            "object name": "female_adult_police_01",
            "3d location": [-17.455900192260742, 15.370429992675781, 0.02103900909423828],
            "2d bounding box visible": {
                "Camera": [447, 245, 470, 276]
            }
        },
        {
            "object id": 4,
            "object type": "person",
            "object name": "female_adult_police_03",
            "3d location": [-13.054417610168457, 2.3046987056732178, 0.02103901281952858],
            "2d bounding box visible": {
                "Camera": [391, 418, 443, 576],
                "Camera_01": [1668, 481, 1805, 688],
                "Camera_02": [1084, 398, 1125, 530]
            }
        }
    ],
    "1": [
        {
            "object id": 0,
            "object type": "person",
            "object name": "male_adult_police_04",
            "3d location": [-7.822440147399902, 4.597992420196533, -1.1969732149896828e-10],
            "2d bounding box visible": {
                "Camera": [912, 362, 955, 507],
                "Camera_01": [960, 664, 1062, 609]
            }
        }
    ]
}
```

**Parameter Descriptions:**

| Parameter | Description |
|---|---|
| frame index | Video frame index (0, 1, …) — the top-level keys |
| `object id` | Object index (integer value) |
| `object type` | Object class (person, fork lift, etc.) |
| `object name` | Unique object name |
| `3d location` | Object's 3D location in meters [x, y, z] |
| `2d bounding box visible` | 2D bounding boxes in each camera view [x_min, y_min, x_max, y_max] |


# Troubleshooting

This section provides solutions to common issues and error messages.

## Cannot Access the UI

**Symptom**: Browser shows "Unable to connect" or "Connection refused"

**Possible Causes**: Backend server not running; incorrect URL or port; network connectivity issues; firewall blocking access

**Solutions**:

1. **Verify server is running**
   ```bash
   docker ps | grep auto-calib
   # Or from compose directory:
   docker compose ps
   ```
2. **Verify URL and port** — check browser address bar; try `http://localhost:<AUTO_MAGIC_CALIB_UI_PORT>` from the server machine
3. **Check network connectivity**
   ```bash
   ping <server-ip>
   nc -vz <server-ip> <port>
   ```
4. **Check firewall settings** — ensure the UI port is allowed

## Port Already in Use

**Symptom**: Docker container fails to start with "port is already allocated" error

**Solution**:

1. Check what's using the port:
   ```bash
   sudo lsof -i :5000
   sudo lsof -i :8000
   ```
2. Change ports in `compose/.env` (`AUTO_MAGIC_CALIB_MS_PORT`, `AUTO_MAGIC_CALIB_UI_PORT`) and restart:
   ```bash
   docker compose down
   docker compose up -d
   ```
3. Or stop the conflicting process/container

## API_URL_NOT_PROVIDED Error

**Symptom**: UI loads but shows "API_URL_NOT_PROVIDED" error

**Cause**: Docker Compose started without proper `HOST_IP` environment variable

**Solution**:

1. Stop services: `docker compose down`
2. Verify `HOST_IP=<your_host_ip>` is set in `compose/.env`
3. Restart: `docker compose up -d`

## Verification Fails

**Symptom**: "Verify Project" button shows error, or project never reaches READY

**Solutions**:

1. **Check requirements checklist**
   - ✓ At least **1** video or RTSP clip for **single-camera**; **2 or more** for **multi-camera**
   - ✓ Layout image uploaded
   - ✓ Alignment data uploaded or created
   - ✓ Rectification state **COMPLETED** (Step 4) — verification rejects stale/nonlinear input
2. **Verify alignment data**
   - Minimum 4 complete point sets
   - **Multi-camera**: each set must include points for every camera view plus the layout (BEV) point
   - **Single-camera**: each set must include **camera + layout** point pairs (two `[x, y]` pairs per set)
3. **Check video files** — ensure videos are not corrupted; re-upload if needed
4. **After re-rectification** — verify again, then relaunch calibration and re-run post-processing
5. **Check server logs**: `docker compose logs | grep -i error`

## Calibration Fails or Takes Too Long

**Symptom**: Calibration fails, runs too long (>30 minutes), or results look wrong

**Solutions**:

1. **Check input files** — video quality, layout image scale/orientation, alignment on ground plane (rectified space)
2. **Check video synchronization** — same time window for all cameras; for RTSP, one combined capture for all streams
3. **Review floor-plan scale** — set **Layout Pixels Per Meter** (`layout_px_per_m`) on Parameters (Floor-plan to 3D scale) or Settings; wrong scale causes misaligned BEV/overlays
4. **Check server resources**: `top`, `nvidia-smi`
5. **Try the other path independently** — VGGT does not require a successful AMC run (multi-camera, after rectification + verify)
6. **Run / re-run layout post-processing** (multi-camera) after AMC or VGGT for overlays and evaluation
7. **Reset project** and retry after fixing inputs

## RTSP Capture Issues

**Symptom**: RTSP capture card missing, capture fails, or ingested clips are out of sync

**Solutions**:

1. **Verify VIOS configuration** — confirm VIOS is running; set `VIOS_BASE_URL` in deployment `.env` if using VSS/VIOS deployment
2. **Use source RTSP URLs** — for VIOS pre-registered streams, use the **source URL** (e.g., NVStreamer URL), not the VIOS-proxied URL
3. **Capture timing** — configure **all** streams, then run **one** **Capture and add to project**; minimum **60** seconds of **RECORDING**; use **Stop early** only after that; wait for **INGESTED** (ingest is automatic—no separate ingest step)
4. **Check server logs**: `docker compose logs | grep -i -E 'rtsp|vios|capture|ingest'`

## Rectification Issues

**Symptom**: Cannot advance past Rectification, Generate Videos fails, or calibration rejects nonlinear input

**Solutions**:

1. Confirm videos are uploaded before starting Auto / Manual / Videos Are Rectified
2. For Auto/Manual: reach **Ready for Review**, then click **Generate Rectified Videos** (COMPLETED is required — READY_FOR_REVIEW alone does not unlock Next)
3. Use **Videos Are Rectified** only when source footage is already linear
4. After re-rectification, re-verify and relaunch AMC/VGGT, then re-run post-processing
5. Check live **Rectification Logs** and `docker compose logs | grep -i rectif`

## VGGT Calibration Issues

**Symptom**: VGGT section unavailable, fails, or does not appear on Results

**Solutions**:

1. **Confirm project type** — VGGT is **multi-camera only** (two or more videos); not supported for single-camera
2. **Verify model installation** — download `vggt_1B_commercial.pt` and place in `$MODEL_DIR/vggt/`
3. **Run at the right time** — complete Rectification, verify the project, then **Start VGGT Calibration**; AMC is **not** required
4. **View VGGT results** — after VGGT completes, run layout post-processing (multi-camera) for overlays; open **Results** and use the **VGGT** tab
5. **Check server logs**: `docker compose logs | grep -i vggt`

## Results and Export Issues

**No results available** — Results is available when AMC **or** VGGT completes successfully. For multi-camera trajectory / virtual-GT overlays, also run **Layout Post-processing** on Execute.

**Export fails** — check browser download settings, popup blocker, disk space; try a different browser; check `docker compose logs | grep -i export`

**ROI verification not working or BEV looks wrong**

1. Draw ROIs/tripwires on **Results** (optional post-calibration), then click **Show ROI & Tripwire Verification**
2. Verify calibration completed (AMC and/or VGGT) and post-processing completed when overlays are missing
3. Check **Layout Pixels Per Meter** (`layout_px_per_m`) on the Parameters step
4. Compare layout pixels (Global target) with world-map projection; switch AMC/VGGT tabs if one result looks better
5. Revisit alignment points and time-synchronized inputs
6. Use **Reset to Original Calibration** in Full Export if metadata edits corrupted the export JSON

## Performance Issues

**UI slow or canvas laggy** — refresh browser, close other tabs, enable hardware acceleration, reduce zoom level, use latest Chrome.


# License

## Repository Licenses
This repository contains materials released under different licenses:
- The scripts and code are licensed under the Apache License 2.0.
- The assets are licensed under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license.

## Proprietary Container Notices (AutoMagicCalib and AutoMagicCalibUI)
The scripts in this repository interact with and pull the proprietary AutoMagicCalib and AutoMagicCalibUI containers. The use of these containers, and any software, data, or intellectual property contained within them, is governed by a separate set of licenses and third-party notices.

The applicable End User License Agreement (EULA), 3rd-party notice, and reference information for the release images can be found in:
- [`nvcr.io/nvidia/auto-magic-calib:3.2.1`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/auto-magic-calib?version=3.2.1)
- [`nvcr.io/nvidia/auto-magic-calib-ui:3.2.1`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/auto-magic-calib-ui?version=3.2.1)
