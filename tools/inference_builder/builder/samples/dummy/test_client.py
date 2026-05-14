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
import os
import mimetypes
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


def main(host, port, image_path, video_path, text):
    invoke_url = "http://" + host + ":" + port + "/v1/inference"
    upload_url = "http://" + host + ":" + port + "/v1/files"
    image_data =[]
    video_data = []
    if image_path:
        for image in image_path:
            multipart_data = MultipartEncoder(
                fields={
                    'file': (os.path.basename(image), open(image, 'rb'), mimetypes.guess_type(image)[0])
                }
            )
            response = requests.post(upload_url, headers={"Content-Type": multipart_data.content_type}, data=multipart_data)
            image_data.append(response.json()["data"])
        print(f"image_data uploaded: {image_data}")
        payload = {
            "input": image_data,
            "text": text,
            "model": "nvidia/nvdino-v2"
        }
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        response = requests.post(invoke_url, headers=headers, json=payload)
        print(f"Inference result from image data: {response.json()}")

    if video_path:
        for video in video_path:
            multipart_data = MultipartEncoder(
                fields={
                    'file': (os.path.basename(video), open(video, 'rb'), mimetypes.guess_type(video)[0])
                }
            )
            response = requests.post(upload_url, headers={"Content-Type": multipart_data.content_type}, data=multipart_data)
            video_data.append(response.json()["data"])
        print(f"video_data uploaded: {video_data}")

        payload = {
            "input": video_data,
            "text": text,
            "model": "nvidia/nvdino-v2"
        }
        headers = {
            "Content-Type": "application/json",
            "accept": "application/x-ndjson"
        }
        stream_response = requests.post(invoke_url, headers=headers, json=payload, stream=True)
        for line in stream_response.iter_lines():
            if line:
                print(f"Inference result from video data: {line}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str,
                      help= "Server IP Address", default="127.0.0.1")
    parser.add_argument("--port", type=str,
                      help="Server port", default="8000")
    parser.add_argument("--image-path", type=str, help="Image path to send for inference", nargs='*', default=None)
    parser.add_argument("--video-path", type=str, help="Video path to send for inference", nargs='*', default=None)
    parser.add_argument("--text", type=str, help="Extra text to send for inference", nargs='*', default=None)
    args = parser.parse_args()
    main(args.host, args.port, args.image_path, args.video_path, args.text)