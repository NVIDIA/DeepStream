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

"""Prepare the ImageNet dataset"""
import os
import argparse
import tarfile
import pickle
import gzip
import subprocess
from tqdm import tqdm

_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_VAL_TAR = 'ILSVRC2012_img_val.tar'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup the ImageNet dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', required=True,
                        help="The directory that contains downloaded tar files")
    parser.add_argument('--target-dir', required=True,
                        help="The directory to store extracted images")
    parser.add_argument('--num-thread', type=int, default=1,
                        help="Number of threads to use when building image record file.")
    args = parser.parse_args()
    return args

def extract_train(tar_fname, target_dir, num_thread=1):
    os.makedirs(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()

def extract_val(tar_fname, target_dir, num_thread=1):
    os.makedirs(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pklz')
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))

def main():
    args = parse_args()
    target_dir = os.path.expanduser(args.target_dir)
    if os.path.exists(target_dir):
        raise ValueError('Target dir ['+target_dir+'] exists. Remove it first')
    download_dir = os.path.expanduser(args.download_dir)
    train_tar_fname = os.path.join(download_dir, _TRAIN_TAR)
    val_tar_fname = os.path.join(download_dir, _VAL_TAR)
    extract_train(train_tar_fname, os.path.join(target_dir, 'train'), args.num_thread)
    extract_val(val_tar_fname, os.path.join(target_dir, 'val'), args.num_thread)

if __name__ == '__main__':
    main()
