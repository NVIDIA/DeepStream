#! /bin/bash

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

# usage: sudo ./setup_nvds_logger.sh [log name]
# eg:    sudo ./setup_nvds_logger.sh ds.log
# log name is optional, if not provided, it will default to ds.log
# Log file will be found in /var/log/nvds/[log_name]
# Note: user can set logging severity level to enable log filtering as mentioned below

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

nvdslogpath="/var/log/nvds/"
nvdslogfilepath="/var/log/nvds/ds.log"

if [ "$1" != "" ]; then
    nvdslogfilepath="$nvdslogpath$1"
fi

echo "Using logging location: $nvdslogfilepath"
rm -rf /run/rsyslogd.pid

if [ ! -d $nvdslogpath ]; then
    echo "Creating logging location: $nvdslogpath"
    mkdir $nvdslogpath
    if  [ ! -d $nvdslogpath ]; then
      echo "Unable to create directory at the given path; please check permissions"
      exit 1
    fi
fi

chown root $nvdslogpath
chgrp syslog $nvdslogpath
chmod g+w $nvdslogpath
touch 11-nvds.conf

# Modify log severity level as required and rerun this script
#              0       Emergency: system is unusable
#              1       Alert: action must be taken immediately
#              2       Critical: critical conditions
#              3       Error: error conditions
#              4       Warning: warning conditions
#              5       Notice: normal but significant condition
#              6       Informational: informational messages
#              7       Debug: debug-level messages
# refer https://tools.ietf.org/html/rfc5424.html for more information

# Currently log level is set at INFO (6).
echo "if (\$msg contains 'DSLOG') and (\$syslogseverity <= 6) then $nvdslogfilepath" >> 11-nvds.conf
echo ":msg, contains, \"DSLOG\" ~"  >> 11-nvds.conf
echo "& ~" >> 11-nvds.conf
rm -rf /etc/rsyslog.g/*-nvds.conf

cp 11-nvds.conf /etc/rsyslog.d/
rm 11-nvds.conf
#    sudo touch  /etc/rsyslog.d/10-mwx.conf
chgrp syslog $nvdslogpath
service rsyslog  restart
echo "nvds logging setup. Logging to $nvdslogfilepath"

