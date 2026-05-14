# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import os

# cv2 is imported only in BBoxOverlayVisualizer (lazy import in __init__)

def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.stack(np.nonzero(special_tokens_mask), axis=1)

    # generate attention mask and positional ids
    attention_mask = (
        np.tile(np.expand_dims(np.eye(num_token, dtype=bool), axis=0), (bs, 1, 1))
    )
    position_ids = np.zeros((bs, num_token))
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col in (0, num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1: col + 1, previous_col + 1: col + 1] = True
            position_ids[row, previous_col + 1: col + 1] = np.arange(
                0, col - previous_col
            )
            c2t_maski = np.zeros((num_token), dtype=bool)
            c2t_maski[previous_col + 1: col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    return attention_mask, position_ids


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j

    Args:
        tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = np.zeros((len(tokens_positive), max_text_len), dtype=float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except Exception:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except Exception:
                end_pos = None

        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos: end_pos + 1].fill(1)
    return positive_map

def tokenize_captions(tokenizer, cat_list, caption, max_text_len=256):
    """tokenize captions."""
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tokenized = tokenizer(caption, padding="max_length", return_tensors="np", max_length=max_text_len)

    label_list = np.arange(len(cat_list))
    pos_map = create_positive_map(tokenized, label_list, cat_list, caption[0], max_text_len=max_text_len)

    (
        text_self_attention_masks,
        position_ids,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens)

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : max_text_len, : max_text_len]

        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    input_ids = tokenized["input_ids"].astype(int).squeeze(0)
    attention_mask = tokenized["attention_mask"].astype(bool).squeeze(0)
    position_ids = position_ids.astype(int).squeeze(0)
    token_type_ids = tokenized["token_type_ids"].astype(int).squeeze(0)
    text_self_attention_masks = text_self_attention_masks.astype(bool).squeeze(0)

    return input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map

class GDinoTokenizer:
    name = "gdino-tokenizer"
    def __init__(self, config):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_text_len = 256

    def __call__(self, *args):
        labels = args[0]
        if isinstance(labels, str):
            labels = labels.split(",")
        caption = [" . ".join(labels) + ' .']
        input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(self.tokenizer, labels, caption, self.max_text_len)
        return input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map


class GDinoPostProcessor:
    name = "gdino-postprocessor"
    def __init__(self, config):
        model_home = config["model_home"]
        infer_config_path = config.get("infer_config_path", None)

        # Handle path resolution
        if infer_config_path:
            # Check if it's an absolute path
            if os.path.isabs(infer_config_path):
                self.infer_config_path = infer_config_path
                print(f"infer_config_path is an absolute path: {self.infer_config_path}")
            else:
                # If relative path, join with model_home
                self.infer_config_path = os.path.join(model_home, infer_config_path)
                print(f"infer_config_path is a relative path to model_home: {self.infer_config_path}")
            # Check if the file exists
            if not os.path.exists(self.infer_config_path):
                print(f"Warning: infer_config_path does not exist: {self.infer_config_path}")
                self.infer_config_path = None
        else:
            self.infer_config_path = None
            print(f"Warning: infer_config_path is not set")

        # set default values
        self.top_k = 300
        self.item_threshold = 0.5  # threshold for the score per inferenced item
        self.segmentation_threshold = 0.5  # threshold per pixel for segmentation mask
        # load top_k and threshold from nvdsinfer_config.yaml
        self._load_config()

        # other members
        self.shape = None
        self.has_masks = False

    def _load_config(self):
        """Load configuration from nvdsinfer_config.yaml file."""
        import yaml
        if not self.infer_config_path:
            return
        try:
            with open(self.infer_config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Get value from property section
            if "property" in config:
                property = config["property"]
                if "segmentation-threshold" in property:
                    self.segmentation_threshold = float(property["segmentation-threshold"])

            # Get values from class-attrs-all section
            if 'class-attrs-all' in config:
                class_attrs = config['class-attrs-all']
                # Update threshold if pre-cluster-threshold is specified
                if 'pre-cluster-threshold' in class_attrs:
                    self.item_threshold = float(class_attrs['pre-cluster-threshold'])
                # Update top_k if topk is specified
                if 'topk' in class_attrs:
                    self.top_k = int(class_attrs['topk'])
        except Exception as e:
            print(f"Warning: Failed to load config from {self.infer_config_path}: {str(e)}")

    def __call__(self, *args):
        # TODO: overflow observed
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def box_cxcywh_to_xyxy(x):
            """Convert box from cxcywh to xyxy."""
            x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return np.stack(b, axis=-1)

        # NOTE: move resize to client side
        # from scipy.ndimage import zoom
        # def _resize_func(x, width, height):
        #     # Calculate zoom factors
        #     zoom_y = height / x.shape[0]
        #     zoom_x = width / x.shape[1]

        #     # For 3D arrays (height, width, channels)
        #     if x.ndim == 3:
        #         return zoom(x, (zoom_y, zoom_x, 1), order=1)
        #     # For 2D arrays (height, width)
        #     return zoom(x, (zoom_y, zoom_x), order=1)
        pred_logits = args[0]  # [900, 256]
        pred_boxes = args[1]   # [900, 4]
        pred_masks = args[2]   # [900, 1, 136, 240]
        pos_maps = args[3]     # [num_labels, 256]

        if pred_masks is not None:
            self.has_masks = True
        else:
            self.has_masks = False

        """
        Extend leading dimension to match the implementation:
        - pred_logits: (B x NQ x 4) where B=1, NQ=900
        - pred_boxes: (B x NQ x 4) where B=1, NQ=900
        - pred_masks: (B x NQ x h x w)
        - target_sizes: (B x 4) [w, h, w, h] where B=1, w, h any value
        """
        pred_logits = np.expand_dims(pred_logits, axis=0)  # [1, 900, 256]
        pred_boxes = np.expand_dims(pred_boxes, axis=0)    # [1, 900, 4]
        if self.has_masks:
            pred_masks = np.expand_dims(pred_masks, axis=0)    # [1, 900, 1, 136, 240]
            self.shape = np.array(pred_masks.shape[3:])        # [136, 240]
        else:
            self.shape = np.array([1, 1])                      # [1, 1]
        target_sizes = np.array([[self.shape[1], self.shape[0],
                                 self.shape[1], self.shape[0]]])  # [1, 4]

        """Below is adapted from model owner"""
        bs = pred_logits.shape[0]
        # Sigmoid
        # prob_to_token = sigmoid(pred_logits).reshape((bs, pred_logits.shape[1], -1))
        prob_to_token = sigmoid(pred_logits)

        # TODO: why normalize across all tokens? that will weaken the probability of each label in that row.
        # Unless each row has only one label?
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

        """
        For each prediction, shrink the probability space from sparse per-token to per-label only
        label is a subset of tokens. Max token is 256.
        Example data, shows the first 5 out of 900 predictions
        Assume there are 2 labels in the query caption, out of the 256 tokens
        prob_to_label = [[[0.1, 0.8],    # pred 0: [label1=0.1, label2=0.8]
                          [0.4, 0.3],    # pred 1: [label1=0.4, label2=0.3]
                          [0.7, 0.9],    # pred 2: [label1=0.7, label2=0.9]
                          [0.2, 0.5],    # pred 3: [label1=0.2, label2=0.5]
                          [0.6, 0.1]]]   # pred 4: [label1=0.6, label2=0.1]
        """
        prob_to_label = prob_to_token @ pos_maps.T
        prob = prob_to_label  # 1, 900, 2

        # Get topk scores
        """
        When flattened to (1, 10), it becomes:
        [0.1, 0.8, 0.4, 0.3, 0.7, 0.9, 0.2, 0.5, 0.6, 0.1]
         p0c1 p0c2 p1c1 p1c2 p2c1 p2c2 p3c1 p3c2 p4c1 p4c2

        Let's say topk_indices after sorting and selecting top 4 being [5, 1, 4, 8],
        which corresponds to values [0.9, 0.8, 0.7, 0.6]
        To get the row index, we do topk_indices // 2 = [2, 0, 2, 4];
        To get the column index, we do topk_indices % 2 = [1, 1, 0, 0];

        This is global topk across probability to all classes,
        which can lead to repeat predictions index hence duplicate bboxes.
        - Captures all high-confidence predictions, even if multiple classes from same prediction
        - Might be useful when a single region could validly contain multiple objects
        - e.g., "person" and "athlete" could both be valid for the same box
        """
        topk_indices = np.argsort(prob.reshape((bs, -1)), axis=1)[:, ::-1][:, :self.top_k]  # [1, 10]

        """
        prob.reshape((bs, -1)) shape: (1, 1800)  # 900 predictions x 2 labels
        per_batch_prob = [[0.1, 0.8, 0.4, 0.3, 0.7, 0.9, 0.2, 0.5, 0.6, 0.1, ...]]

        topk_indices shape: (1, 10)
        ind = [[5, 1, 4, 8, ...]]  # Top 10 indices

        scores shape: (1, 10)
        scores = [[0.9, 0.8, 0.7, 0.6, ...]]  # Top 10 scores
        """
        scores = [per_batch_prob[ind] for per_batch_prob, ind in zip(prob.reshape((bs, -1)), topk_indices)]  # [1, 10]
        scores = np.array(scores)


        """
        If self.item_threshold = 0.7, then
        scores shape: (1, 10)
        scores = [[0.9, 0.8, 0.7, 0.6]]
        threshold_mask = [[True, True, False, False]]

        After filtering:
        scores shape: (1, 2)
        scores = [[0.9, 0.8]]
        result_indices shape: (1, 2)
        result_indices = [[5, 1]]
        """
        # Apply mask while preserving batch dimension
        threshold_mask = scores > self.item_threshold  # shape: (1, 10)
        scores = scores[threshold_mask]  # Flatten to 1D array of valid scores
        threshold_topk_indices = topk_indices[threshold_mask]  # Flatten to 1D array of valid indices
        # Reshape to ensure (bs, N) shape
        scores = scores.reshape(bs, -1)  # shape: (1, N), N <= 10
        threshold_topk_indices = threshold_topk_indices.reshape(bs, -1)  # shape: (1, N), N <= 10


        """
        After filtering with threshold:
        scores = [[0.9, 0.8, 0.7, 0.6]]           # shape: (1, 4)
        result_box_indices = [[2, 0, 2, 4]]               # shape: (1, 4)
        result_label_indices = [[1, 1, 0, 0]]                   # shape: (1, 4)

        This means:
        - Score 0.9 is for box 2, label 1
        - Score 0.8 is for box 0, label 1
        - Score 0.7 is for box 2, label 0
        - Score 0.6 is for box 4, label 0
        """
        result_box_indices = threshold_topk_indices // prob.shape[2]  # shape: (1, N), N <= 10
        result_label_indices = threshold_topk_indices % prob.shape[2]       # shape: (1, N), N <= 10

        # Take corresponding topk boxes
        """
        Assume
        - pred_boxes = np.array([[[1,2,3,4],       # box 0
                                  [5,6,7,8],       # box 1
                                  [9,10,11,12],    # box 2
                                  [13,14,15,16],   # box 3
                                  [17,18,19,20]]]) # box 4
        - result_box_indices = [[2, 0, 2, 4]], shape = (1, 4)

        1. np.expand_dims(result_box_indices, -1)
        The -1 means add dimension at the end:
        From: [[2, 0, 2, 4]]
        To:   [[[2],
                [0],
                [2],
                [4]]]
        Now, shape = (1, 4, 1)
        2. np.repeat(..., 4, axis=-1)

        Repeats each value 4 times along the last axis:
        To:   [[[2, 2, 2, 2],
                [0, 0, 0, 0],
                [2, 2, 2, 2],
                [4, 4, 4, 4]]]
        Now, shape = (1, 4, 4)

        3. np.take_along_axis(boxes, indices, axis=1)
        Uses the indices to gather values from boxes along axis 1
        Result will be:
        [[[ 9,10,11,12 ],    # box 2
          [ 1,2,3,4 ],       # box 0
          [ 9,10,11,12 ],    # box 2 again
          [ 17,18,19,20 ]]]  # box 4

        NOTE: This is copied from model owner. More memory efficient and readable way is in each batch directly select the rows:
        boxes = np.stack([boxes[b][result_box_indices[b]] for b in range(boxes.shape[0])])
        # print(boxes[0][[2, 0, 2, 4]])
        # Result will be:
        # [[ 9,10,11,12],    # box 2
        #  [ 1,2,3,4 ],      # box 0
        #  [ 9,10,11,12 ],   # box 2 again
        #  [ 17,18,19,20 ]]  # box 4
        """
        boxes = np.take_along_axis(pred_boxes, np.repeat(np.expand_dims(result_box_indices, -1), 4, axis=-1), axis=1)

        # Convert to x1, y1, x2, y2 format
        boxes = box_cxcywh_to_xyxy(boxes)

        # Scale back the bounding boxes to the original image size
        target_sizes = np.array(target_sizes)
        boxes = boxes * target_sizes[:, None, :]

        result_label_indices = result_label_indices.squeeze(0)
        if self.has_masks:
            masks = []
            # Clamp bounding box coordinates
            for i, target_size in enumerate(target_sizes):
                w, h = target_size[0], target_size[1]
                boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
                boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)
                m = pred_masks[i][result_box_indices[i], ...]  # Shape: (10, 1, 136, 240), if top 10
                m = m.squeeze(axis=1)  # Shape becomes: (10, 136, 240), if top 10
                m = np.transpose(m, (1, 2, 0))  # Shape: (136, 240, 10)

                # NOTE: move resize to client side
                # from functools import partial
                # N = 2  # small number, divisible by n_queries
                # m_split = np.split(m, N, axis=2)
                # m_split = list(map(partial(_resize_func, width=w, height=h), m_split))
                # m_scaled = np.concatenate(m_split, axis=2)
                # m_scaled = sigmoid(m_scaled) # Shape: (544, 960, 10)

                m_scaled = sigmoid(m) # Shape: (136, 240, 10)
                # Use moveaxis directly on m_scaled
                mask_array = np.moveaxis(m_scaled, 2, 0)  # reshape from (136, 240, 10) to (10, 136, 240)
                # For each mask, assign its corresponding label index where probability > threshold
                for idx, label_idx in enumerate(result_label_indices):  # result_label_indices is already squeezed to 1D
                    # mask_array[idx] shape: (136, 240)
                    # Example: if label_idx = 1 and threshold = 0.5
                    # mask_array[idx] = [
                    #   [0.7, 0.2, 0.8],
                    #   [0.1, 0.6, 0.3]
                    # ]
                    # After threshold:
                    # mask_array[idx] = [
                    #   [2, 0, 2],  # 2 because (label_idx=1 + 1)
                    #   [0, 2, 0]
                    # ]
                    mask_array[idx] = ((mask_array[idx] > self.segmentation_threshold) * (label_idx + 1)).astype(np.uint8)  # uint8 [0, 255]
                masks.append(mask_array)
            # Convert to flattened masks
            mask_list = [mask.flatten().astype(np.uint8).tolist() for mask in masks[0]]
        else:
            mask_list = [[]]

        boxes = boxes.squeeze(0)
        scores = scores.squeeze(0)

        return {
            "shape": self.shape.tolist(),
            "bboxes": boxes.tolist(),
            "probs": scores.tolist(),
            "labels": [[str(i)] for i in result_label_indices.tolist()],
            "mask": mask_list
        }


class BBoxOverlayVisualizer:
    """Postprocessor that overlays bounding boxes on decoded image frames.

    Draws bounding boxes with labels and confidence scores on the RGB HWC
    image tensor extracted from the DeepStream pipeline and saves annotated
    frames to disk.

    Config:
        output_dir: Directory to save annotated frames (default: /tmp/bbox_overlay)

    Input:
        output: DS metadata dict with 'bboxes', 'labels', 'probs' keys
        decoded_frames: RGB uint8 image tensor in HWC layout [H, W, 3]
    Output:
        output: DS metadata dict (passed through unchanged)
    """
    name = "bbox-overlay-visualizer"

    # Color palette for different labels (BGR for OpenCV)
    COLORS = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (128, 255, 0),  # spring green
        (255, 128, 0),  # azure
    ]

    def __init__(self, config):
        import cv2
        self._cv2 = cv2
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._thickness = 2
        self._frame_count = 0
        self._output_dir = config.get("output_dir", "/tmp/bbox_overlay")
        os.makedirs(self._output_dir, exist_ok=True)

    def __call__(self, metadata, image):
        if image is None:
            return metadata

        # Framework passes batch lists; unwrap to single items
        import torch
        if isinstance(metadata, list):
            metadata = metadata[0]
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        annotated = self._cv2.cvtColor(np.asarray(image), self._cv2.COLOR_RGB2BGR)

        if metadata is not None:
            bboxes = metadata.get("bboxes", [])
            labels = metadata.get("labels", [])
            probs = metadata.get("probs", [])
            shape = metadata.get("shape", [])

            # Scale bboxes from metadata shape to image shape if needed
            img_h, img_w = annotated.shape[:2]
            scale_x, scale_y = 1.0, 1.0
            if len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
                scale_x = img_w / shape[1]
                scale_y = img_h / shape[0]

            for i, bbox in enumerate(bboxes):
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)

                color = self.COLORS[i % len(self.COLORS)]
                self._cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self._thickness)

                # Build label text
                label_str = labels[i][0] if i < len(labels) and labels[i] else ""
                prob_str = f"{probs[i]:.2f}" if i < len(probs) else ""
                text = f"{label_str} {prob_str}".strip()

                if text:
                    (tw, th), _ = self._cv2.getTextSize(text, self._font, self._font_scale, 1)
                    self._cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    self._cv2.putText(annotated, text, (x1, y1 - 2),
                                self._font, self._font_scale, (255, 255, 255), 1, self._cv2.LINE_AA)

        # Save annotated frame to disk
        out_path = os.path.join(self._output_dir, f"frame_{self._frame_count:04d}.jpg")
        self._cv2.imwrite(out_path, annotated)
        self._frame_count += 1

        return metadata


