/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//This source file presents some common functions/utils used by adapter libraries

#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <string.h>
#include <openssl/sha.h>
#include <glib.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include "nvds_msgapi.h"
#include "nvds_logger.h"
#include "nvds_utils.h"

using namespace std;

#define CONFIG_GROUP_MSG_BROKER "message-broker"

/*
 * Release memory allocated for gobjects
 */
void free_gobjs(GKeyFile *gcfg_file, GError *error, gchar **keys,  gchar *key_name) {
    if(keys != NULL)
        g_strfreev(keys);
    if(error != NULL)
        g_error_free (error);
    if(gcfg_file != NULL)
        g_key_file_free(gcfg_file);
    if(key_name != NULL)
        g_free(key_name);
}

/*
 * Internal function to verify if the config string value is a quoted string
 * config key = "value"
 * If so, strip beginning and ending quote
 */
NvDsMsgApiErrorType strip_quote(char *cfg_value, const char *cfg_key, const char *log_cat) {
    //If value is not empty
    if(strcmp(cfg_value , "")) {
            size_t conflen = strlen(cfg_value);

            //remove "". (string length needs to be at least 2)
            //Could use g_shell_unquote but it might have other side effects
            if ((conflen < 3) || (cfg_value[0] != '"') || (cfg_value[conflen-1] != '"')) {
                nvds_log(log_cat, LOG_ERR,  "invalid key=value format. Must Start and end with \"\"\n", cfg_key);
                return NVDS_MSGAPI_ERR;
            }
            else {
                string res(cfg_value);
                res.erase(0,1); //erase first "
                res.erase(res.length()-1);      //erase last "
                strcpy(cfg_value, res.c_str());
                nvds_log(log_cat, LOG_INFO,  "cfg setting %s = %s\n", cfg_key,cfg_value);
            }
    }
    return NVDS_MSGAPI_OK;
}

/*
 *Function to open a file and search for a config value for a config key
 *If found, place the result in cfg_val
 */
NvDsMsgApiErrorType fetch_config_value(char *config_path, const char *cfg_key, char *cfg_val, int len, const char *log_cat) {
    //iterate over the config params to set one by one
    GKeyFile *key_file = g_key_file_new ();
    gchar **keys = NULL;
    gchar **key = NULL;
    GError *error = NULL;

    if(!config_path) {
        nvds_log(log_cat, LOG_ERR,  "Error parsing config file. config file path pointer cant be NULL");
        return NVDS_MSGAPI_ERR;
    }

    if (!g_key_file_load_from_file (key_file, config_path, G_KEY_FILE_NONE,&error)) {
        nvds_log(log_cat, LOG_ERR,  "unable to load config file at path[%s]; error message = %s\n",config_path, error->message);
        free_gobjs(key_file, error, keys, NULL);
        return NVDS_MSGAPI_ERR;
    }

    keys = g_key_file_get_keys(key_file, CONFIG_GROUP_MSG_BROKER, NULL, &error);
    if (error) {
        nvds_log(log_cat, LOG_ERR,  "config group[%s] in cfg not found. Error:%s\n", CONFIG_GROUP_MSG_BROKER, error->message);
        free_gobjs(key_file, error, keys, NULL);
        return NVDS_MSGAPI_ERR;
    }
    gchar *str=NULL;
    for (key = keys; *key; key++) {
        if (!g_strcmp0(*key, cfg_key)) {
             str = g_key_file_get_string (key_file, CONFIG_GROUP_MSG_BROKER, cfg_key, &error);

            if (error) {
                nvds_log(log_cat, LOG_ERR,  "Error parsing config file. %s\n", error->message);
                free_gobjs(key_file, error, keys, str);
                return NVDS_MSGAPI_ERR;
            }
            if(len < (int) strlen(str) + 1) {
                nvds_log(log_cat, LOG_ERR, "Output string capacity is not sufficient to hold config value. Should be atleast %d", strlen(str) + 1);
                free_gobjs(key_file, error, keys, str);
                return NVDS_MSGAPI_ERR;
            }
            strcpy(cfg_val , str);
            break;
        }
    }
    free_gobjs(key_file, error, keys, str);
    return NVDS_MSGAPI_OK;
}

/*
 *Generate hash value of a string using SHA256
 */
string generate_sha256_hash(string str) {
    unsigned char hashval[SHA256_DIGEST_LENGTH];
    int len = SHA256_DIGEST_LENGTH * 2 + 1;
    char res[len];
    SHA256(reinterpret_cast<const unsigned char*>(str.c_str()), str.length(), hashval);
    for(int i=0; i<SHA256_DIGEST_LENGTH; i++) {
        sprintf(res + (i * 2), "%02x", hashval[i]);
    }
    return string(res);
}

/*
 * remove leading & trailing whitespaces from a string
 */
string trim(string str) {
    auto low = str.find_first_not_of(" \t");
    if(low == string::npos)
        return "";
    auto high = str.find_last_not_of(" \t");
    return str.substr(low , high - low + 1);
}

/*
 * Sort string of format key=value;key1=value1
 * ex: str = "pq=89;xyz=33;abc=123;"
 * output  = "abc=123;pq=89;xyz=33"
 */
string sort_key_value_pairs(string str) {
    map<string, string> mymap;
    istringstream iss(str);
    string kv_pair;
    while(getline(iss, kv_pair, ';')) {
        auto pos = kv_pair.find('=');
        string key = trim(kv_pair.substr(0,pos));
        string val = trim(kv_pair.substr(pos + 1));
        if(key != "")    mymap[key] = val;
    }
    string res = "";
    for(auto i:mymap) {
        res += i.first + "=" + i.second;
    }
    return res;
}

/*
 * Get hostname with IP address or Kubernetes pod information
 * Returns a newly allocated string containing either:
 *   - Kubernetes: "pod-name (pod-dns)"
 *   - Regular: "hostname (IP)"
 * Caller must free with g_free().
 */
char* nvds_get_hostname_with_ip(void) {
    char hostname[256];
    char identifier[512] = "unknown";

    // Check if running in Kubernetes environment
    const char *pod_name = getenv("POD_NAME");
    const char *pod_namespace = getenv("POD_NAMESPACE");
    const char *k8s_service = getenv("KUBERNETES_SERVICE_HOST");

    // If Kubernetes environment variables are set, use pod information
    if (k8s_service != NULL) {
        // Get pod name (either from POD_NAME or HOSTNAME)
        if (pod_name != NULL) {
            g_snprintf(hostname, sizeof(hostname), "%s", pod_name);
        } else if (gethostname(hostname, sizeof(hostname)) != 0) {
            g_snprintf(hostname, sizeof(hostname), "unknown-pod");
        }

        // Construct pod DNS name
        if (pod_namespace != NULL) {
            // Full pod DNS: pod-name.namespace.svc.cluster.local
            g_snprintf(identifier, sizeof(identifier), "%s.%s.svc.cluster.local",
                      hostname, pod_namespace);
        } else {
            // Partial pod DNS without namespace
            g_snprintf(identifier, sizeof(identifier), "%s.pod", hostname);
        }

        // Format as "pod-name (pod-dns)"
        return g_strdup_printf("%s (%s)", hostname, identifier);
    }

    // Non-Kubernetes environment: use traditional hostname and IP
    char ip_address[INET_ADDRSTRLEN] = "unknown";

    // Get hostname
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        g_snprintf(hostname, sizeof(hostname), "localhost");
    }

    // Get IP address
    struct ifaddrs *ifaddr, *ifa;
    if (getifaddrs(&ifaddr) == 0) {
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == NULL) continue;

            // Check for IPv4 address and skip loopback
            if (ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
                // Skip loopback interface (127.0.0.1)
                if (ntohl(addr->sin_addr.s_addr) != INADDR_LOOPBACK) {
                    inet_ntop(AF_INET, &addr->sin_addr, ip_address, INET_ADDRSTRLEN);
                    break;
                }
            }
        }
        freeifaddrs(ifaddr);
    }

    // Format as "hostname (IP)"
    return g_strdup_printf("%s (%s)", hostname, ip_address);
}
