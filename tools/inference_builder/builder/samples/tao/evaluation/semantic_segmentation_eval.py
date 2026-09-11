# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import sys
import logging
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

# ---- Logger setup ----
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# ----------------------

# Add the project root to the Python path to import nim_client
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from nim_client import main as nim_client_main, convert_masks_to_image_size
from validation_utils import (
    validate_safe_path, validate_config_path, validate_dump_vis_path,
    validate_directory_path, validate_split_name
)



class SegmentationEvaluator:
    """Class responsible for collecting predictions and computing segmentation metrics."""

    def __init__(self, host: str, port: str, model_name: str = "nvdino-v2",
                 config_path: str = None):
        """Initialize the segmentation evaluator.

        Example config YAML structure:
        dataset:
          segment:
            root_dir: /path/to/dataset
            validation_split: val
            palette:
              - label_id: 0
                mapping_class: background
                rgb: [0, 0, 0]
                seg_class: background
              - label_id: 1
                mapping_class: foreground
                rgb: [255, 255, 255]
                seg_class: foreground
        """
        self.host = host
        self.port = port
        self.model_name = model_name

        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Extract dataset info
            dataset_config = self.config['dataset']['segment']
            raw_root_dir = dataset_config['root_dir']
            raw_val_split = dataset_config['validation_split']
            
            # Validate extracted config values for security
            if not validate_directory_path(raw_root_dir):
                raise ValueError(
                    f"Invalid root directory path in config: {raw_root_dir}")

            if not validate_split_name(raw_val_split):
                raise ValueError(
                    f"Invalid validation split name in config: {raw_val_split}")
            
            self.root_dir = raw_root_dir
            self.val_split = raw_val_split
            self.palette = dataset_config['palette']
            self.num_classes = len(self.palette)

            # Create label mappings
            # Example: if palette has background (id=0) and foreground (id=1)
            # label_id_train_id_mapping = {0: 0, 1: 1}
            self.label_id_train_id_mapping = {
                item['label_id']: idx for idx, item in enumerate(self.palette)}
            # train_id_name_mapping = {0: ["background"], 1: ["foreground"]}
            self.train_id_name_mapping = {
                idx: [item['mapping_class']]
                for idx, item in enumerate(self.palette)}

    def _load_ground_truth(self, mask_path: str) -> np.ndarray:
        """Load ground truth mask and convert to training IDs.

        Example:
        Input mask (2x2 pixels):
        [[0, 255],
         [255, 0]]

        After normalization (divide by 255):
        [[0, 1],
         [1, 0]]

        After label mapping (if label_id_train_id_mapping = {0: 0, 1: 1}):
        [[0, 1],
         [1, 0]]
        """
        mask = np.array(Image.open(mask_path))
        logger.info(f"Original mask unique values: {np.unique(mask)}")

        # Normalize mask values from [0, 255] to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        logger.info(f"Normalized mask unique values: {np.unique(mask)}")

        # Convert label IDs to training IDs, replacing None with 0
        converted_mask = np.zeros_like(mask)
        for label_id, train_id in self.label_id_train_id_mapping.items():
            converted_mask[mask == label_id] = train_id

        logger.info(f"Converted mask unique values: {np.unique(converted_mask)}")
        return converted_mask

    def _process_prediction(self, inference_response: Dict, target_shape: tuple) -> np.ndarray:
        """Process inference response to get prediction mask and resize to target shape.

        Example inference_response:
        {
            "data": [{
                "index": 0,
                "shape": [224, 224],
                "masks": [[0, 1, 1, 0, ...]],  # Flattened 224x224 mask
                "timestamp": 26837406483
            }],
            "model": "nvidia/nvdinov2"
        }

        Example processing:
        1. Original flattened mask (224x224):
           [0, 1, 1, 0, ...]

        2. Reshaped to original dimensions:
           [[0, 1],
            [1, 0],
            ...]

        3. Resized to target shape (e.g., 512x512):
           [[0, 1, 1, 0],
            [1, 0, 0, 1],
            ...]
        """
        if not inference_response or "data" not in inference_response:
            return None

        data = inference_response["data"][0]  # We expect only one item in data
        original_shape = tuple(data["shape"])
        masks = data["masks"][0]  # Get the first (and only) mask

        # Reshape the flattened mask to match original dimensions
        pred_mask = np.array(masks).reshape(original_shape)

        # Resize prediction to match target shape
        resized_masks = convert_masks_to_image_size([pred_mask], original_shape, target_shape)
        if not resized_masks:
            return None

        return resized_masks[0]

    def evaluate(self, dump_vis_path: str = None) -> Dict:
        """Evaluate segmentation model on validation set."""
        # Get list of validation images
        val_img_dir = os.path.join(self.root_dir, "images", self.val_split)
        val_mask_dir = os.path.join(self.root_dir, "masks", self.val_split)

        if not os.path.exists(val_img_dir) or not os.path.exists(val_mask_dir):
            raise ValueError(f"Validation directories not found: {val_img_dir} or {val_mask_dir}")

        # Resolve validation directories for security checks
        resolved_val_img_dir = os.path.realpath(val_img_dir)
        resolved_val_mask_dir = os.path.realpath(val_mask_dir)

        all_ground_truths = []
        all_predictions = []

        # Process each image
        try:
            img_files = os.listdir(val_img_dir)
        except Exception as e:
            logger.error(f"Error accessing validation image directory {val_img_dir}: {e}")
            raise ValueError(f"Cannot access validation image directory: {val_img_dir}")

        for img_name in tqdm(img_files, desc="Processing validation images"):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                logger.warning(f"Skipping non-image file: {img_name}")
                continue

            # Validate image filename for security
            if not validate_safe_path(img_name):
                logger.warning(f"Skipping potentially unsafe image file: {img_name}")
                continue

            img_path = os.path.join(val_img_dir, img_name)
            mask_path = os.path.join(val_mask_dir, img_name)

            # Additional security check: ensure the resolved paths are within expected directories
            try:
                resolved_img_path = os.path.realpath(img_path)
                resolved_mask_path = os.path.realpath(mask_path)

                if not resolved_img_path.startswith(resolved_val_img_dir):
                    logger.warning(f"Skipping image file outside validation dir: {img_name}")
                    continue

                if not resolved_mask_path.startswith(resolved_val_mask_dir):
                    logger.warning(f"Skipping mask file outside validation dir: {img_name}")
                    continue
            except Exception as e:
                logger.warning(f"Error resolving path for {img_name}: {e}")
                continue

            if not os.path.exists(mask_path):
                logger.warning(f"Ground truth mask not found for {img_name}")
                continue

            # Load ground truth first to get target shape
            try:
                gt_mask = self._load_ground_truth(mask_path)
                target_shape = gt_mask.shape
            except Exception as e:
                logger.warning(f"Error loading ground truth mask for {img_name}: {e}")
                continue

            # Get prediction from inference server
            response = nim_client_main(
                host=self.host,
                port=self.port,
                model=self.model_name,
                files=[img_path],
                text=None,
                dump_response=False,
                upload=False,
                return_response=True,
                vis_dir=dump_vis_path
            )

            pred_mask = self._process_prediction(response, target_shape)
            if pred_mask is None:
                logger.error(f"Failed to get prediction for {img_name}")
                raise ValueError(f"Failed to get prediction for {img_name}")

            # Verify shapes match
            if pred_mask.shape != gt_mask.shape:
                logger.error(f"Shape mismatch for {img_name}: pred {pred_mask.shape} vs gt {gt_mask.shape}")
                raise ValueError(f"Shape mismatch for {img_name}: pred {pred_mask.shape} vs gt {gt_mask.shape}")

            all_ground_truths.append(gt_mask)
            all_predictions.append(pred_mask)

        # Compute metrics
        metrics = self.compute_metrics_masks(all_ground_truths, all_predictions)

        # Print results
        logger.info("Evaluation Results:")
        logger.info(f"Mean IoU: {metrics['mean_iou_index']:.4f}")
        logger.info(f"Mean Precision: {metrics['prec']:.4f}")
        logger.info(f"Mean Recall: {metrics['rec']:.4f}")
        logger.info(f"Mean F1 Score: {metrics['fmes']:.4f}")

        # Print per-class results
        logger.info("\nPer-class Results:")
        for class_name, class_metrics in metrics['results_dic'].items():
            logger.info(f"\n{class_name}:")
            for metric_name, value in class_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def compute_metrics_masks(self, ground_truths, predictions):
        """Compute segmentation metrics using the existing implementation.

        Example inputs:
        ground_truths = [
            [[0, 1], [1, 0]],  # First image
            [[0, 0], [1, 1]]   # Second image
        ]
        predictions = [
            [[0, 1], [1, 1]],  # First image
            [[0, 0], [1, 1]]   # Second image
        ]
        """
        # Initialize confusion matrix
        # For 2 classes, this creates a 2x2 matrix:
        # conf_mat = [[0, 0],
        #             [0, 0]]
        conf_mat = np.zeros([self.num_classes, self.num_classes], dtype=np.float32)

        for pred, gt in tqdm(zip(predictions, ground_truths), desc="Computing metrics"):
            # First iteration:
            # pred = [[0, 1], [1, 1]]
            # gt = [[0, 1], [1, 0]]

            # Flatten the arrays and convert to integers
            pred = pred.flatten().astype(np.int32)
            # pred = [0, 1, 1, 1]
            gt = gt.flatten().astype(np.int32)
            # gt = [0, 1, 1, 0]

            # Debug: Check unique values and ranges
            logger.info(f"Unique values in ground truth: {np.unique(gt)}")
            logger.info(f"Unique values in prediction: {np.unique(pred)}")
            logger.info(f"Ground truth range: [{np.min(gt)}, {np.max(gt)}]")
            logger.info(f"Prediction range: [{np.min(pred)}, {np.max(pred)}]")

            # Initialize result matrix for this image
            # result = [[0, 0],
            #           [0, 0]]
            result = np.zeros((self.num_classes, self.num_classes))

            # Count predictions
            for i in range(len(gt)):
                if gt[i] >= self.num_classes or pred[i] >= self.num_classes:
                    logger.warning(f"Invalid class index - gt: {gt[i]}, pred: {pred[i]}, num_classes: {self.num_classes}")
                    continue
                result[gt[i]][pred[i]] += 1
                # After all iterations for first image:
                # result = [[1, 0],  # Class 0: 1 correct prediction
                #           [1, 1]]  # Class 1: 1 correct, 1 wrong prediction

            conf_mat += np.matrix(result)
            # After first image:
            # conf_mat = [[1, 0],
            #             [1, 1]]

        # Calculate metrics
        metrics = {}

        # True Positives (diagonal elements)
        # For first image:
        # perclass_tp = [1, 1]  # One correct prediction for each class
        perclass_tp = np.diagonal(conf_mat).astype(np.float32)

        # False Positives (column sums minus diagonal)
        # For first image:
        # perclass_fp = [0, 1]  # One false positive for class 1
        perclass_fp = conf_mat.sum(axis=0) - perclass_tp

        # False Negatives (row sums minus diagonal)
        # For first image:
        # perclass_fn = [0, 1]  # One false negative for class 1
        perclass_fn = conf_mat.sum(axis=1) - perclass_tp

        # Calculate IoU for each class
        # IoU = TP / (FP + TP + FN)
        # For first image:
        # iou_per_class = [1.0, 0.33]  # Class 0: perfect IoU, Class 1: 1/(1+1+1)
        iou_per_class = perclass_tp / (perclass_fp + perclass_tp + perclass_fn)

        # Calculate precision for each class
        # Precision = TP / (FP + TP)
        # For first image:
        # precision_per_class = [1.0, 0.5]  # Class 0: perfect precision, Class 1: 1/(1+1)
        precision_per_class = perclass_tp / (perclass_fp + perclass_tp)

        # Calculate recall for each class
        # Recall = TP / (TP + FN)
        # For first image:
        # recall_per_class = [1.0, 0.5]  # Class 0: perfect recall, Class 1: 1/(1+1)
        recall_per_class = perclass_tp / (perclass_tp + perclass_fn)

        f1_per_class = []
        final_results_dic = {}

        for num_class in range(self.num_classes):
            # Example: self.train_id_name_mapping = {
            #     0: ["background"],
            #     1: ["foreground"]
            # }
            name_class = "/".join(self.train_id_name_mapping[num_class])
            # name_class = "background" for class 0
            # name_class = "foreground" for class 1

            per_class_metric = {}
            prec = precision_per_class[num_class]
            rec = recall_per_class[num_class]
            iou = iou_per_class[num_class]
            f1 = (2 * prec * rec) / float((prec + rec)) if (prec + rec) > 0 else 0
            # For class 1: f1 = (2 * 0.5 * 0.5) / (0.5 + 0.5) = 0.5

            f1_per_class.append(f1)
            per_class_metric["precision"] = prec
            per_class_metric["Recall"] = rec
            per_class_metric["F1 Score"] = f1
            per_class_metric["iou"] = iou

            final_results_dic[name_class] = per_class_metric
            # final_results_dic = {
            #     "background": {"precision": 1.0, "Recall": 1.0, "F1 Score": 1.0, "iou": 1.0},
            #     "foreground": {"precision": 0.5, "Recall": 0.5, "F1 Score": 0.5, "iou": 0.33}
            # }

        # Calculate mean metrics
        def getScoreAverage(scores):
            valid_scores = [s for s in scores if not np.isnan(s)]
            return np.mean(valid_scores) if valid_scores else 0

        metrics["rec"] = getScoreAverage(recall_per_class)
        metrics["prec"] = getScoreAverage(precision_per_class)
        metrics["fmes"] = getScoreAverage(f1_per_class)
        metrics["mean_iou_index"] = getScoreAverage(iou_per_class)
        metrics["results_dic"] = final_results_dic

        return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Run semantic segmentation evaluation')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment config YAML file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Inference server host')
    parser.add_argument('--port', type=str, default='8800',
                      help='Inference server port')
    parser.add_argument('--dump-vis-path', type=str, default=None,
                      help='Path to dump visualized predictions result images')
    
    # Parse arguments with security validation
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"Argument parsing failed: {str(e)}")
        sys.exit(1)

    # Comprehensive security validation
    validation_errors = []
    
    # Validate config path
    if not validate_config_path(args.config):
        validation_errors.append(f"Invalid config file path: {args.config}")
    
    # Validate host
    if not validate_safe_path(args.host):
        validation_errors.append(f"Invalid host parameter: {args.host}")
    
    # Validate port
    if not validate_safe_path(args.port):
        validation_errors.append(f"Invalid port parameter: {args.port}")
    
    # Validate dump visualization path
    if not validate_dump_vis_path(args.dump_vis_path):
        validation_errors.append(f"Invalid dump visualization path: {args.dump_vis_path}")
    
    # Exit if any validation errors
    if validation_errors:
        logger.error("Security validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Additional security checks
    try:
        # Validate config file existence
        if not os.path.isfile(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        # Validate dump visualization directory creation if provided
        if args.dump_vis_path:
            try:
                os.makedirs(args.dump_vis_path, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create dump visualization directory: {str(e)}")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Security validation failed: {str(e)}")
        sys.exit(1)

    return args

def main():
    args = parse_args()

    # Initialize evaluator with error handling
    try:
        evaluator = SegmentationEvaluator(
            host=args.host,
            port=args.port,
            config_path=args.config
        )
    except Exception as e:
        logger.error(f"Failed to initialize SegmentationEvaluator: {str(e)}")
        sys.exit(1)

    # Run evaluation with error handling
    try:
        metrics = evaluator.evaluate(dump_vis_path=args.dump_vis_path)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()