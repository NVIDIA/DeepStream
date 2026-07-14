# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License


# additional components the user can self install
apt-get update
apt-get install -y gstreamer1.0-libav
# ubuntu 22.04 / updated to Ubuntu 24.04
apt-get install --reinstall -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  libswresample-dev libavutil-dev libavutil58 libavcodec-dev libavcodec60 libavformat-dev libavformat60 libde265-dev libde265-0 libx265-199 libavfilter9 libmpeg2encpp-2.1-0t64 libmpeg2-4 libmpg123-0t64
# new since DS 7.1
apt-get install --reinstall -y libflac12t64 libmp3lame0 libxvidcore4 ffmpeg libfaad2 mjpegtools libvo-aacenc0 libdca0 libdvdnav4 libdvdread8t64
# new for dev-main 8.0 ubuntu 24.04
apt-get install --reinstall -y libvpx9 libavfilter-dev libflac++10 libmjpegutils-2.1-0t64 libopenh264-7 libx264-164 libjbig0 libmplex2-2.1-0t64 libmfx1


# gstreamer1.0-plugins-good fix
cp /tmp99/libgstrtpmanager.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstrtpmanager.so
cp /tmp99/libgstrtsp.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstrtsp.so 
cp /tmp99/libgstvideoparsersbad.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvideoparsersbad.so

echo "Deleting GStreamer cache"
rm -rf ~/.cache/gstreamer-1.0/
