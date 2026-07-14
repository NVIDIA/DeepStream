#!/bin/bash
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

# Test parameters
BROKER="localhost"      # Change if the broker is on a different machine or IP address
PORT=1883              # Default MQTT port for Mosquitto
TOPIC="test/topic"     # The topic to test
MESSAGE="Hello from Mosquitto test!"  # The test message


# Subscribe to the topic in the background
echo "Subscribing to topic '$TOPIC'..."
mosquitto_sub -h $BROKER -p $PORT -t $TOPIC & 
SUB_PID=$!

# Give the subscriber some time to start up
sleep 1

# Publish a message to the topic
echo "Publishing message to topic '$TOPIC'..."
mosquitto_pub -h $BROKER -p $PORT -t $TOPIC -m "$MESSAGE"

# Wait a few seconds to allow the subscriber to receive the message
sleep 2

# Stop the mosquitto_sub process
kill $SUB_PID
wait $SUB_PID

# The script will automatically exit, and the message should be received by the subscriber.
echo "Test completed."
