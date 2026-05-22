################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

"""Script to prepare resized images/labels for LPD."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import cv2
from PIL import Image


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(description='Prepare resized images/labels dataset for LPD')


    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory to OpenALPR's benchmark end2end us license plates."
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Ouput directory to resized images/labels."
    )


    parser.add_argument(
        "--target_width",
        type=int,
        required=True,
        help="target width for resized images/labels."
    )


    parser.add_argument(
        "--target_height",
        type=int,
        required=True,
        help="target height for resized images/labels."
    )
    return parser.parse_args(args)




def prepare_data(input_dir, img_list, output_dir, output_size):
    """Crop the license plates from the orginal images."""


    w, h = output_size


    target_img_path = os.path.join(output_dir, "image")
    target_label_path = os.path.join(output_dir, "label")


    if not os.path.exists(target_img_path):
        os.makedirs(target_img_path)


    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)


    for img_name in img_list:
        img_path = os.path.join(input_dir, img_name)
        label_path = os.path.join(input_dir,
                                  img_name.split(".")[0] + ".txt")
        print("Resizing {} and its label".format(img_path))


        # resize labels
        img = cv2.imread(img_path)
        (height, width, _) = img.shape
        ratio_w = float(float(w)/float(width) )
        ratio_h = float(float(h)/float(height) )


        with open(label_path, "r") as f:
            label_lines = f.readlines()
            assert len(label_lines) == 1
            label_items = label_lines[0].split()


        assert img_name == label_items[0]
        xmin = int(label_items[1])
        ymin = int(label_items[2])
        width = int(label_items[3])
        xmax = xmin + width
        height = int(label_items[4])
        ymax = ymin + height




        x1 = float(xmin) * ratio_w
        y1 = float(ymin) * ratio_h
        x2 = float(xmax) * ratio_w
        y2 = float(ymax) * ratio_h




        with open(os.path.join(target_label_path,
                               img_name.split(".")[0] + ".txt"), "w") as f:
            f.write("lpd 0.0 0 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(x1, y1, x2, y2))


        # resize images
        image = Image.open(img_path)
        scale_image = image.resize(output_size, Image.ANTIALIAS)
        scale_image.save(os.path.join(target_img_path, img_name))


def main(args=None):
    """Main function for data preparation."""


    args = parse_args(args)


    w = args.target_width
    h = args.target_height
    output_size = (w,h)


    img_files = []
    for file_name in os.listdir(args.input_dir):
        if file_name.split(".")[-1] == "jpg":
            img_files.append(file_name)


    image_dir = os.path.join(args.output_dir, "data")


    prepare_data(args.input_dir, img_files, image_dir, output_size)
 
    print("\nDone. Resized images/labels are saved at {} folder".format(args.output_dir))




if __name__ == "__main__":
    main()