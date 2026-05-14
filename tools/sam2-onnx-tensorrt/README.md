#  SAM2-ONNX-TRT
Export all the Segment Anything 2 (SAM2) network modules to ONNX files, so TensorRT can create inference
engines for accelerated inference. [DeepStream MaskTracker](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-masktracker) uses them to perform
multi-object tracking. The output includes below modules:

## Table of Contents
- [Overview](#overview)
- [Network Architecture](#network-architecture)
  - [Image Encoder](#image-encoder)
  - [Mask Decoder](#mask-decoder)
  - [Memory Attention](#memory-attention)
  - [Memory Encoder](#memory-encoder)
- [Run](#run)
- [References](#references)

## Overview

The SAM2-ONNX-TRT project exports four key network modules from Segment Anything 2 (SAM2) to ONNX format for TensorRT acceleration:

### 1. Image Encoder
A transformer-based encoder network that extracts visual features from each frame. This network processes the frame once and shares the extracted features across all targets in that frame.

### 2. Memory Attention
This module generates conditional features for each target by combining the current frame's image encoder features with spatial memory features and object pointers from previous frames, which are stored in the memory bank.

### 3. Mask Decoder (including prompt encoder)
This component takes bounding box prompts (if provided) and conditional features from memory attention to generate segmentation masks and object pointers. The object pointers capture high-level semantic information to uniquely identify each target.

### 4. Memory Encoder
This module downsamples and fuses the segmentation mask with the image encoder features to create spatial memory features for the current frame, which are then stored in the memory bank for future reference.

## Network Architecture

The following tables show the input/output specifications for each network module. All inputs and outputs across all networks use float data type.

### Image Encoder
| Index | Type | Name | Size | Description |
|-------|------|------|------|-------------|
| 0 | Input | image | [1, 3, 1024, 1024] | Input RGB image |
| 1 | Output | pix_feat | [1, 256, 64, 64] | Pixel-level features |
| 2 | Output | high_res_feat0 | [1, 32, 256, 256] | High-resolution features (level 0) |
| 3 | Output | high_res_feat1 | [1, 64, 128, 128] | High-resolution features (level 1) |
| 4 | Output | vision_feats | [1, 256, 64, 64] | Vision features |
| 5 | Output | vision_pos_embed | [4096, 1, 256] | Vision positional embeddings |

### Mask Decoder
| Index | Type | Name | Size | Description |
|-------|------|------|------|-------------|
| 0 | Input | point_coords | [-1, -1, 2] | Point coordinates (dynamic batch) |
| 1 | Input | point_labels | [-1, -1] | Point labels (dynamic batch) |
| 2 | Input | image_embed | [-1, 256, 64, 64] | Image embeddings (dynamic batch) |
| 3 | Input | high_res_feats_0 | [1, 32, 256, 256] | High-resolution features (level 0) |
| 4 | Input | high_res_feats_1 | [1, 64, 128, 128] | High-resolution features (level 1) |
| 5 | Output | obj_ptr | [-1, 256] | Object pointer (dynamic batch) |
| 6 | Output | mask_for_mem | [-1, 1, 1024, 1024] | Mask for memory storage (dynamic batch) |
| 7 | Output | pred_mask | [-1, 1, 256, 256] | Predicted segmentation mask (dynamic batch) |
| 8 | Output | iou | [-1, 1] | Intersection over Union score (dynamic batch) |
| 9 | Output | occ_logit | [-1, 1] | Occlusion logit (dynamic batch) |

### Memory Attention
| Index | Type | Name | Size | Description |
|-------|------|------|------|-------------|
| 0 | Input | current_vision_feat | [1, 256, 64, 64] | Current frame vision features |
| 1 | Input | current_vision_pos_embed | [4096, 1, 256] | Current frame positional embeddings |
| 2 | Input | memory_0 | [1, -1, 256] | Memory bank features (dynamic) |
| 3 | Input | memory_1 | [1, -1, 64, 64, 64] | Memory bank spatial features (dynamic) |
| 4 | Input | memory_pos_embed | [1, -1, 64] | Memory positional embeddings (dynamic) |
| 5 | Input | cond_frame_id_diff | [] | Conditional frame ID difference (scalar) |
| 6 | Output | image_embed | [1, 256, 64, 64] | Enhanced image embeddings |

### Memory Encoder
| Index | Type | Name | Size | Description |
|-------|------|------|------|-------------|
| 0 | Input | mask_for_mem | [1, 1, 1024, 1024] | Mask for memory encoding |
| 1 | Input | pix_feat | [1, 256, 64, 64] | Pixel features |
| 2 | Input | occ_logit | [1, 1] | Occlusion logit |
| 3 | Output | maskmem_features | [1, 64, 64, 64] | Mask memory features |
| 4 | Output | maskmem_pos_enc | [1, 4096, 64] | Mask memory positional encodings |
| 5 | Output | temporal_code | [7, 1, 1, 64] | Temporal encoding |

## Run

Below script will install SAM2 dependencies, download the SAM2 model checkpoints and export them to ONNX files. The ONNX files are saved in `checkpoints/${MODEL_TYPE}` directory.

```bash
# this script will install Python dependencies and export ONNX models
# better to run it in a Python virtual environment or docker container.
bash run.sh
```

> **Note:** It is normal to see TracerWarning messages during the export process. These warnings can be ignored as they don't affect the functionality of the exported ONNX models.

## References

This project uses information from the following repositories:

* SAM2 Meta Inc Repo: https://github.com/facebookresearch/segment-anything-2.git
* SAM2Export: https://github.com/Aimol-l/SAM2Export
* SAM2 ONNX Export Fixes: https://github.com/axinc-ai/segment-anything-2.git
