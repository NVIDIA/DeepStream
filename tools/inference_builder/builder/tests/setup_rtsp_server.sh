#!/bin/bash

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


# RTSP Server Setup Script for Testing
# This script sets up an RTSP server using cvlc to stream MP4 videos for testing
# Supports both MP4 files directly and JPEG images (transcoded to MP4)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
RTSP_PORT=8554
RTSP_MOUNT_POINT="video-stream"
INPUT_FILE=""
CACHE_TIME=1500
FRAME_RATE=1
DURATION=5
TEMP_DIR="/tmp/rtsp_server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/rtsp_server.pid"
LOG_FILE="${SCRIPT_DIR}/logs/rtsp_server.log"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <input-file>

Options:
    -p, --port PORT           RTSP port (default: 8554)
    -m, --mount MOUNT_POINT   RTSP mount point (default: video-stream)
    -c, --cache CACHE_TIME    Network caching time in ms (default: 1500)
    -f, --fps FRAME_RATE      Frame rate for JPEG transcoding (default: 1)
    -d, --duration DURATION   Duration per frame in seconds for JPEG (default: 5)
    --daemon                  Run as daemon in background
    -k, --kill                Kill existing RTSP server
    -s, --status              Check RTSP server status
    -h, --help                Show this help message

Examples:
    $0 sample_video.mp4
    $0 sample_image.jpg
    $0 -p 8555 -m test-stream sample_image.jpg
    $0 --daemon -f 2 -d 3 sample_image.jpg
    $0 --kill
    $0 --status

Environment Variables:
    RTSP_PORT         Default RTSP port
    RTSP_MOUNT_POINT  Default mount point
    INPUT_FILE        Default input file path
    FRAME_RATE        Default frame rate for JPEG transcoding
    DURATION          Default duration per frame for JPEG

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if cvlc is installed
    if ! command -v cvlc &> /dev/null; then
        print_error "cvlc (VLC command line) is not installed"
        print_info "Install VLC: sudo apt-get install vlc"
        exit 1
    fi

    # Check if ffmpeg is installed (for JPEG transcoding)
    if ! command -v ffmpeg &> /dev/null; then
        print_error "ffmpeg is not installed"
        print_info "Install ffmpeg: sudo apt-get install ffmpeg"
        exit 1
    fi

    # Check if port is available
    if netstat -tuln 2>/dev/null | grep -q ":$RTSP_PORT "; then
        print_warning "Port $RTSP_PORT is already in use"
        print_info "Use --kill to stop existing server or specify different port"
    fi

    # Create temp directory
    mkdir -p "$TEMP_DIR"

    print_success "Prerequisites check passed"
}

# Function to transcode JPEG to MP4
transcode_jpeg_to_mp4() {
    local jpeg_file="$1"
    local output_mp4="$2"
    local frame_rate="$3"
    local duration="$4"

    print_info "Transcoding JPEG to MP4..."
    print_info "Input: $jpeg_file"
    print_info "Output: $output_mp4"
    print_info "Frame rate: $frame_rate fps"
    print_info "Duration per frame: $duration seconds"

    # Create a video from JPEG image with specified frame rate and duration
    ffmpeg -loop 1 -i "$jpeg_file" -c:v libx264 -tune stillimage -crf 18 -preset ultrafast \
           -r "$frame_rate" -t "$duration" -y "$output_mp4" > /dev/null 2>&1

    if [ $? -eq 0 ] && [ -f "$output_mp4" ]; then
        print_success "JPEG transcoded to MP4 successfully"
        return 0
    else
        print_error "Failed to transcode JPEG to MP4"
        return 1
    fi
}

# Function to get file type
get_file_type() {
    local file="$1"
    local file_info=$(file "$file" 2>/dev/null)

    if echo "$file_info" | grep -q "JPEG\|JFIF"; then
        echo "jpeg"
    elif echo "$file_info" | grep -q "MP4\|MPEG\|AVI\|MOV\|WebM\|video"; then
        echo "video"
    else
        echo "unknown"
    fi
}

# Function to kill existing RTSP server
kill_rtsp_server() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_info "Killing existing RTSP server (PID: $pid)"
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "Server still running, force killing..."
                kill -9 "$pid"
            fi
            print_success "RTSP server stopped"
        else
            print_warning "PID file exists but process not running"
        fi
        rm -f "$PID_FILE"
    else
        print_info "No RTSP server PID file found"
    fi
}

# Function to check server status
check_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_success "RTSP server is running (PID: $pid)"
            print_info "RTSP URL: rtsp://localhost:$RTSP_PORT/$RTSP_MOUNT_POINT"
            return 0
        else
            print_warning "PID file exists but process not running"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        print_info "RTSP server is not running"
        return 1
    fi
}

# Function to cleanup temp files
cleanup_temp_files() {
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        print_info "Cleaned up temporary files"
    fi
}

# Function to start RTSP server
start_rtsp_server() {
    local input_file="$1"
    local daemon_mode="$2"

    # Validate input file
    if [ ! -f "$input_file" ]; then
        print_error "Input file not found: $input_file"
        exit 1
    fi

    # Get file type
    local file_type=$(get_file_type "$input_file")
    local video_file="$input_file"

    if [ "$file_type" = "jpeg" ]; then
        print_info "JPEG file detected, transcoding to MP4..."
        local output_mp4="$TEMP_DIR/transcoded_$(basename "$input_file" .jpg).mp4"

        if ! transcode_jpeg_to_mp4 "$input_file" "$output_mp4" "$FRAME_RATE" "$DURATION"; then
            exit 1
        fi

        video_file="$output_mp4"
        print_info "Using transcoded MP4: $video_file"
    elif [ "$file_type" = "video" ]; then
        print_info "Video file detected, using directly"
    else
        print_warning "Unknown file type: $input_file"
        print_info "Attempting to use as video file..."
    fi

    # Kill existing server
    kill_rtsp_server

    # Create cvlc command for video streaming
    # Use loop to continuously stream the video
    # Add headless flags for Docker/CI environments
    local cvlc_cmd="vlc-wrapper -I dummy --no-video-title-show --no-xlib --loop \"$video_file\" \":sout=#gather:rtp{sdp=rtsp://:$RTSP_PORT/$RTSP_MOUNT_POINT}\" :network-caching=$CACHE_TIME :sout-all :sout-keep"

    print_info "Starting RTSP server..."
    print_info "Input file: $input_file"
    print_info "Video file: $video_file"
    print_info "RTSP URL: rtsp://localhost:$RTSP_PORT/$RTSP_MOUNT_POINT"
    print_info "Command: $cvlc_cmd"

    if [ "$daemon_mode" = "true" ]; then
        # Run as daemon
        print_info "Running in daemon mode"
        nohup bash -c "$cvlc_cmd" > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        sleep 2

        if kill -0 "$pid" 2>/dev/null; then
            print_success "RTSP server started in background (PID: $pid)"
            print_info "Log file: $LOG_FILE"
            print_info "PID file: $PID_FILE"
        else
            print_error "Failed to start RTSP server"
            if [ -f "$LOG_FILE" ]; then
                print_error "Log file contents:"
                cat "$LOG_FILE"
            fi
            exit 1
        fi
    else
        # Run in foreground
        print_info "Running in foreground mode (Ctrl+C to stop)"
        print_info "RTSP server will be available at: rtsp://localhost:$RTSP_PORT/$RTSP_MOUNT_POINT"
        echo "$$" > "$PID_FILE"

        # Set up cleanup on exit
        trap cleanup_temp_files EXIT

        exec bash -c "$cvlc_cmd"
    fi
}

# Parse command line arguments
DAEMON_MODE="false"
KILL_SERVER="false"
CHECK_STATUS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            RTSP_PORT="$2"
            shift 2
            ;;
        -m|--mount)
            RTSP_MOUNT_POINT="$2"
            shift 2
            ;;
        -c|--cache)
            CACHE_TIME="$2"
            shift 2
            ;;
        -f|--fps)
            FRAME_RATE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        --daemon)
            DAEMON_MODE="true"
            shift
            ;;
        -k|--kill)
            KILL_SERVER="true"
            shift
            ;;
        -s|--status)
            CHECK_STATUS="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                print_error "Multiple input files specified"
                exit 1
            fi
            shift
            ;;
    esac
done

# Handle special commands
if [ "$KILL_SERVER" = "true" ]; then
    kill_rtsp_server
    cleanup_temp_files
    exit 0
fi

if [ "$CHECK_STATUS" = "true" ]; then
    check_status
    exit $?
fi

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    print_error "Input file (JPEG or MP4) is required"
    show_usage
    exit 1
fi

# Check prerequisites
check_prerequisites

# Start RTSP server
start_rtsp_server "$INPUT_FILE" "$DAEMON_MODE"