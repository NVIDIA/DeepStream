#!/usr/bin/env bash
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

set -e

if [ $(whoami) != root ]; then
    echo Error: Run this script as a root user
    exit 1
fi

clkfile=/tmp/defclocks.conf
pwrfile=/tmp/defpower.conf

# Xavier
if [ -e /sys/devices/platform/13e10000.host1x/15340000.vic ]; then
    vicctrl=/sys/devices/platform/13e10000.host1x/15340000.vic
    vicfreqctrl=$vicctrl/devfreq/15340000.vic
# Orin?
elif [ -e /sys/devices/platform/bus@0/13e00000.host1x/15340000.vic ]; then
    vicctrl=/sys/devices/platform/bus@0/13e00000.host1x/15340000.vic
    vicfreqctrl=$vicctrl/devfreq/15340000.vic
fi

maxclocks()
{
    if [ ! -e $clkfile ]; then
        jetson_clocks --store $clkfile
        if [ -n "$vicctrl" ]; then
            echo "$vicfreqctrl/governor:$(cat $vicfreqctrl/governor)" >> $clkfile
            echo "$vicfreqctrl/max_freq:$(cat $vicfreqctrl/max_freq)" >> $clkfile
            echo "$vicctrl/power/control:$(cat $vicctrl/power/control)" >> $clkfile
        fi
    fi

    if [ ! -e $pwrfile ]; then
        echo $(nvpmodel -q | tail -n1) > $pwrfile
    fi

    nvpmodel -m 0

    jetson_clocks --fan
    jetson_clocks

    if [ -n "$vicctrl" ]; then
        echo on > $vicctrl/power/control
        echo userspace > $vicfreqctrl/governor
        sleep 1
        maxfreq=$(cat $vicfreqctrl/available_frequencies | rev | cut -f1 -d' ' | rev)
        echo $maxfreq > $vicfreqctrl/max_freq
        echo $maxfreq > $vicfreqctrl/userspace/set_freq
    fi
}

restore()
{
    if [ -e $clkfile ]; then
        jetson_clocks --restore $clkfile > /dev/null 2>&1
    fi

    if [ -e $pwrfile ]; then
        nvpmodel -m $(cat $pwrfile)
    fi
}

action="$1"

case "$action" in
    --restore)
        restore
        ;;
    --max)
        maxclocks
        ;;
    *)
        echo "Unknown option '$action'."
        echo "Usage: $(basename $0) <--max|--restore>"
        exit 1
        ;;
esac
