#! /bin/bash

####################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
####################################################################################################

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

