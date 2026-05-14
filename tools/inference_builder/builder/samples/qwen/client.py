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

import time
import argparse
from openai import OpenAI
import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Image paths to send for inference", default=None, nargs="*")
    parser.add_argument("--videos", type=str, help="Video paths to send for inference", default=None, nargs="*")
    parser.add_argument("--endpoint", type=str, help="Endpoint to send for inference", default="http://0.0.0.0:8800/v1", nargs="?")
    parser.add_argument("-p", "--parallel", type=int, help="Number of parallel requests to send", default=1)
    args = parser.parse_args()

    messages = []
    if args.images:
        for image in args.images:
            with open(image, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                mime_type = mimetypes.guess_type(image)[0]
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe the image in detail."
                        },
                        {
                            "type": "image",
                            "image": f"data:{mime_type};base64,{data}"
                        }
                    ]
                })
    elif args.videos:
        for video in args.videos:
            # Parse video path to check for ?nframes=x query parameter
            # Only extract nframes if video starts with "/" (local file path)
            if video.startswith('/') and '?' in video:
                video_path, query_string = video.split('?', 1)
                query_params = parse_qs(query_string)

                video_content = {
                    "type": "video",
                    "video": video_path
                }

                # Add nframes parameter if present
                if 'nframes' in query_params:
                    try:
                        nframes_value = int(query_params['nframes'][0])
                        video_content["nframes"] = nframes_value
                    except (ValueError, IndexError):
                        pass  # Ignore invalid nframes values
            else:
                video_content = {
                    "type": "video",
                    "video": video
                }

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the video in detail."
                    },
                    video_content
                ]
            })
    else:
        raise ValueError("No images or videos provided")

    client = OpenAI(base_url=args.endpoint, api_key="not-used")

    def make_request(request_id):
        """Make a single request to the API"""
        start_time = time.time()
        chat_response = client.chat.completions.create(
            model="nvidia/cosmos",
            messages=messages,
            max_tokens=512,
            stream=False
        )
        infer_time = time.time() - start_time
        return request_id, infer_time, chat_response.choices[0].message

    # Send requests in parallel
    overall_start_time = time.time()
    if args.parallel == 1:
        # Single request mode
        request_id, infer_time, assistant_message = make_request(0)
        print(f"Inference time: {infer_time:.4f} seconds")
        print(assistant_message)
    else:
        # Parallel requests mode
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(make_request, i) for i in range(args.parallel)]

            results = []
            for future in as_completed(futures):
                request_id, infer_time, assistant_message = future.result()
                results.append((request_id, infer_time, assistant_message))
                print(f"Request {request_id} completed in {infer_time:.4f} seconds")

            # Sort results by request_id for consistent output
            results.sort(key=lambda x: x[0])

            overall_time = time.time() - overall_start_time
            avg_time = sum(r[1] for r in results) / len(results)
            print(f"\n--- Summary ---")
            print(f"Total requests: {args.parallel}")
            print(f"Overall time: {overall_time:.4f} seconds")
            print(f"Average inference time: {avg_time:.4f} seconds")
            print(f"Throughput: {args.parallel / overall_time:.2f} requests/second")

            # Print all responses
            print(f"\n--- Responses ---")
            for req_id, _, assistant_message in results:
                print(f"\nResponse (Request {req_id}):")
                print(assistant_message)
