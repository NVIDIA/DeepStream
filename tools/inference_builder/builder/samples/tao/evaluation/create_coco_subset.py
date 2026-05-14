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

import json
import random
import sys
import argparse

def create_coco_subset(input_file, output_file, num_images=2, supercategory=None, category=None, image_names=None):
    """
    Create a debug subset of a COCO format JSON file.

    Args:
        input_file (str): Path to the input COCO format JSON file
        output_file (str): Path where the subset JSON will be saved
        num_images (int): Number of images to select (default: 2)
        supercategory (str): Supercategory to filter images (e.g., "vehicle")
        category (str): Category name to filter images (e.g., "car")
        image_names (list): List of specific image filenames to select

    Expected JSON Schema:
    {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 800,
                "height": 600
            },
            ...
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [x, y, width, height],
                "area": 100.0,
                "iscrowd": 0
            },
            ...
        ],
        "categories": [
            {
                "id": 1,
                "name": "car",
                "supercategory": "vehicle"
            },
            {
                "id": 2,
                "name": "truck",
                "supercategory": "vehicle"
            },
            ...
        ]
    }
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    if image_names:
        # If image names are provided, select those specific images
        selected_image_ids = []
        for img in data['images']:
            if img['file_name'] in image_names:
                selected_image_ids.append(img['id'])

        if not selected_image_ids:
            print(f"Warning: No images found with the provided filenames")
            return

        print(f"Found {len(selected_image_ids)} images matching the provided filenames")
    else:
        # If category or supercategory is specified, find matching images
        category_ids = []

        # First find category IDs based on category name
        if category:
            category_ids = [cat['id'] for cat in data['categories']
                          if cat.get('name', '').lower() == category.lower()]

            if not category_ids:
                print(f"Warning: No categories found with name '{category}'")
                return

            # If supercategory is also specified, filter by both
            if supercategory:
                category_ids = [cat['id'] for cat in data['categories']
                              if cat.get('name', '').lower() == category.lower() and
                                 cat.get('supercategory', '').lower() == supercategory.lower()]

                if not category_ids:
                    print(f"Warning: No categories found with name '{category}' and supercategory '{supercategory}'")
                    return

        # If only supercategory is specified
        elif supercategory:
            category_ids = [cat['id'] for cat in data['categories']
                          if cat.get('supercategory', '').lower() == supercategory.lower()]

            if not category_ids:
                print(f"Warning: No categories found with supercategory '{supercategory}'")
                return

        if category_ids:
            # Find annotations with these category IDs
            relevant_annotations = [ann for ann in data['annotations']
                                  if ann['category_id'] in category_ids]

            # Get unique image IDs from these annotations
            available_image_ids = list(set(ann['image_id'] for ann in relevant_annotations))

            if not available_image_ids:
                print(f"Warning: No images found with the specified category criteria")
                return

            # Select random images from available ones
            num_images = min(num_images, len(available_image_ids))
            selected_image_ids = random.sample(available_image_ids, num_images)
        else:
            # If no category specified, just select random images
            selected_image_ids = random.sample([img['id'] for img in data['images']], num_images)

    # Filter images
    data['images'] = [img for img in data['images'] if img['id'] in selected_image_ids]

    # Filter annotations
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in selected_image_ids]

    # Keep all categories (they might be needed for the annotations)
    # data['categories'] remains unchanged

    # Write the subset to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Created debug subset with {len(data['images'])} images and {len(data['annotations'])} annotations")
    print(f"Selected image IDs: {selected_image_ids}")
    if category:
        print(f"Selected images with category: {category}")
    if supercategory:
        print(f"Selected images with supercategory: {supercategory}")
    if image_names:
        print("Selected images:")
        for img in data['images']:
            print(f"- {img['file_name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a debug subset of COCO format JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected JSON Schema:
{
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "width": 800,
            "height": 600
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "bbox": [x, y, width, height],
            "area": 100.0,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "car",
            "supercategory": "vehicle"
        }
    ]
}

Examples:
    python create_coco_subset.py /media/scratch.metropolis3/yuw/datasets/coco/annotations/instances_val2017.json /tmp/val2017_subset.json --supercategory vehicle --num_images 5

    1. Basic usage - select 2 random images:
       python create_coco_subset.py input.json output.json

    2. Select specific number of random images:
       python create_coco_subset.py input.json output.json --num_images 5

    3. Select images with specific supercategory:
       python create_coco_subset.py input.json output.json --supercategory vehicle

    4. Select images with specific category:
       python create_coco_subset.py input.json output.json --category car

    5. Select images with both category and supercategory:
       python create_coco_subset.py input.json output.json --category car --supercategory vehicle

    6. Select specific images by their filenames:
       python create_coco_subset.py input.json output.json --image_names image1.jpg image2.jpg
        """
    )
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--num_images', type=int, default=2,
                       help='Number of images to select (default: 2)')
    parser.add_argument('--supercategory', type=str,
                       help='Supercategory to filter images (e.g., "vehicle")')
    parser.add_argument('--category', type=str,
                       help='Category name to filter images (e.g., "car")')
    parser.add_argument('--image_names', type=str, nargs='+',
                       help='List of image filenames to select (e.g., "image1.jpg image2.jpg")')

    args = parser.parse_args()
    create_coco_subset(args.input_file, args.output_file, args.num_images,
                       args.supercategory, args.category, args.image_names)