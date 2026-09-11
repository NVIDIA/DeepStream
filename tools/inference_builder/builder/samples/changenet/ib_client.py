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
import requests, base64
import time
import numpy as np
import os
import io
from PIL import Image

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"


def main(host , port, files, out_format):
  if not files:
    print("Need the file path for inference")
    return

  # invoke_url = "http://localhost:8001/inference"
  invoke_url = "http://" + host + ":" + port + "/v1/infer"

  file_exts = []
  b64_images = []
  for file in files:
    file_exts.append(os.path.splitext(file)[1][1:])
    with open(file, "rb") as f:
      b64_images.append(base64.b64encode(f.read()).decode())

# assert len(image_b64) < 180_000, \
#   "To upload larger images, use the assets API (see docs)"

  headers = {
    "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
    "Accept": "application/json",
    # "NVCF-ASSET-DIR": "some-temp-dir",
    # "NVCF-FUNCTION-ASSET-IDS": "udjflsjo-jfoisjof-lsdfjofdj"
  }

  payload = {
    "input": [f"data:image/{e};base64,{i}" for e, i in zip(file_exts, b64_images)],
    "model": "nvidia/changenet"
  }
  start_time = time.time()
  response = requests.post(invoke_url, headers=headers, json=payload)
  infer_time = time.time() - start_time
  print(response)
  print(infer_time)

  if response.status_code == 200:
    output = response.json()
    print(f"Usage: num_images= {output['usage']['num_images']}")
    mask = base64.b64decode(output["data"])
    image_buffer = io.BytesIO(mask)
    image = Image.open(image_buffer)
    image.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--host", type=str,
                    help= "Server IP Address", default="0.0.0.0")
  parser.add_argument("--port", type=str,
                    help="Server port", default="8000")
  parser.add_argument("--file", type=str, help="File to send for inference", nargs='*', default=None)
  parser.add_argument("--format", type=str, help="Output embedding format, integer or base64",
                      default="integer")

  args = parser.parse_args()
  main(args.host, args.port, args.file, args.format)
