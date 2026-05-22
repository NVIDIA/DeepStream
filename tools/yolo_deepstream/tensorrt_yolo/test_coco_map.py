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
import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--predict', type=str, default='./predict.json', help='model.pt path(s)')
    parser.add_argument('--coco', type=str, default='./coco/', help='*.data path')
    opt = parser.parse_args()
    print('\nEvaluating pycocotools mAP... saving %s...' % opt.predict)
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        anno = COCO(opt.coco+"/annotations/instances_val2017.json")  # init annotations api
        pred = anno.loadRes(opt.predict)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        # if is_coco:
        #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        print(f'pycocotools unable to run: {e}')
