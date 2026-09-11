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
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

from nim_client import main as nim_client_main, convert_bboxes_to_image_size

class PredictionCollector:
    """Class responsible for collecting predictions from the inference server."""

    def __init__(self, host: str, port: str, model_name: str = "nvdino-v2"):
        self.host = host
        self.port = port
        self.model_name = model_name

    def _convert_to_coco_format(self, inference_response: Dict, image_id: int, target_shape: tuple) -> List[Dict]:
        """Convert a single inference response to COCO prediction format."""
        predictions = []

        for data in inference_response["data"]:
            original_shape = tuple(data["shape"])
            bboxes = data["bboxes"]
            scores = data["probs"]
            labels = data["labels"]

            converted_bboxes = convert_bboxes_to_image_size(bboxes, original_shape, target_shape)
            if not converted_bboxes:
                continue

            for bbox, score, label in zip(converted_bboxes, scores, labels):
                x = bbox[0]
                y = bbox[1]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                prediction = {
                    "image_id": image_id,
                    "bbox": [x, y, width, height],
                    "score": score,
                    "category_id": int(label[0])  # it has to be int, otherwise COCOeval result in 0 score
                }
                predictions.append(prediction)

        return predictions

    def collect_predictions(self, val_json: str, image_base_dir: str, output_path: str, dump_vis_path: str) -> List[Dict]:
        """Collect predictions for all images in the validation set."""
        # Load validation config
        with open(val_json, 'r') as f:
            config = json.load(f)

        all_predictions = []

        # Process each image in the config
        for image_entry in tqdm(config["images"], desc="Processing images"):
            image_name = image_entry["file_name"]
            image_id = image_entry["id"]
            target_shape = (image_entry["height"], image_entry["width"])

            image_path = os.path.join(image_base_dir, image_name)

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            # Call inference server
            response = nim_client_main(
                host=self.host,
                port=self.port,
                model=self.model_name,
                files=[image_path],
                text=None,
                dump_response=False,
                upload=False,
                return_response=True,
                vis_dir=dump_vis_path
            )

            if response is None:
                logger.warning(f"Failed to get response for {image_path}")
                continue

            # Convert to COCO format
            coco_predictions = self._convert_to_coco_format(response, image_id, target_shape)
            all_predictions.extend(coco_predictions)

        # Save predictions to file
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)

        logger.info(f"Saved predictions to: {output_path}")
        return all_predictions

class COCOEvaluator:
    """Class responsible for COCO evaluation metrics computation."""

    def __init__(self, val_json: str, ann_type: str = 'bbox'):
        self.val_json = val_json
        self.ann_type = ann_type
        self.coco_gt = COCO(val_json)

    def evaluate(self, predictions: List[Dict]) -> Dict[str, float]:
        """Evaluate predictions against ground truth and return metrics."""
        logger.info("Running COCO evaluation...")
        coco_dt = self.coco_gt.loadRes(predictions)

        coco_eval = COCOeval(self.coco_gt, coco_dt, self.ann_type)
        image_ids = list(set(pred['image_id'] for pred in predictions))
        coco_eval.params.imgIds = sorted(image_ids)
        # coco_eval.params.useCats = 0

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            'AP': coco_eval.stats[0],    # AP @[0.5:0.95]
            'AP50': coco_eval.stats[1],  # AP @0.5
            'AP75': coco_eval.stats[2],  # AP @0.75
            'APs': coco_eval.stats[3],   # AP small
            'APm': coco_eval.stats[4],   # AP medium
            'APl': coco_eval.stats[5],   # AP large
            'AR1': coco_eval.stats[6],   # AR @1
            'AR10': coco_eval.stats[7],  # AR @10
            'AR100': coco_eval.stats[8], # AR @100
            'ARs': coco_eval.stats[9],   # AR small
            'ARm': coco_eval.stats[10],  # AR medium
            'ARl': coco_eval.stats[11]   # AR large
        }

        # Print results
        logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation and compute COCO metrics')
    parser.add_argument('--val-json', type=str, required=True,
                      help='Path to COCO format validation JSON file containing both images and ground truth annotations')
    parser.add_argument('--image-dir', type=str, required=True,
                      help='Base directory containing the images')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Inference server host')
    parser.add_argument('--port', type=str, default='8800',
                      help='Inference server port')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save COCO predictions')
    parser.add_argument('--ann-type', type=str, default='bbox',
                      choices=['bbox', 'segm'],
                      help='Annotation type (bbox or segm)')
    parser.add_argument('--dump-vis-path', type=str, default=None,
                      help='Path to dump visualized predictions result images')
    return parser.parse_args()

def main():
    args = parse_args()

    # Collect predictions
    collector = PredictionCollector(host=args.host, port=args.port)
    predictions = collector.collect_predictions(
        val_json=args.val_json,
        image_base_dir=args.image_dir,
        output_path=args.output,
        dump_vis_path=args.dump_vis_path
    )

    # Evaluate predictions
    evaluator = COCOEvaluator(val_json=args.val_json, ann_type=args.ann_type)
    metrics = evaluator.evaluate(predictions)

if __name__ == '__main__':
    main()