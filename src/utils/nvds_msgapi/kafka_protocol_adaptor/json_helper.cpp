/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jansson.h>
#include "nvds_logger.h"
#include <glib.h>

#define KAFKA_JSON_PARSER "KAKFA_JSON_PARSE"

#define FREE_AND_RETURN(v,p) json_decref(p); return v

int json_get_key_value(const char*, int, const char*, char*, int);

/*
   Returns 0 if key was not found in json.
   If key is found then returns length of the key and populates into the *value location.
   key is in dotted notation based on json schema of message
 */
int json_get_key_value(const char *msg, int msglen, const char *path, char *value, int nbuf)
{
    json_t *root;
    json_error_t error;

    root = json_loadb(msg, msglen, 0, &error);

    if (!root)
    {
      nvds_log(KAFKA_JSON_PARSER, LOG_ERR, "json error on line %d: %s\n", error.line, error.text);
      return 0;
    }

   json_t *jvalue;
   json_t *subroot = root;
   char const *temp;
   const char *remstr = path;
   const char *dotptr;
   char subpath[256]; /* stores remaining part of path to be processed at any time */

   nvds_log(KAFKA_JSON_PARSER, LOG_DEBUG, "finding kafka key of %s within json message\n", path);

   /* parse down the tree based on successive elements in the key*/
   while ((dotptr = strchr(remstr, '.')) != NULL) {
     unsigned int subpath_len = ((dotptr - remstr) < 0)? 0 : (unsigned int)(dotptr - remstr) ;

     /* maximum size of each key is 256 */
     if (subpath_len > sizeof(subpath)) {
       nvds_log(KAFKA_JSON_PARSER, LOG_ERR, "provided json sub key length > 255. \
                                        Error finding kafka key value.\n");
       FREE_AND_RETURN(0,root);
     }

     memcpy(subpath, remstr, subpath_len);

     subpath[subpath_len - 1] = '\0';
     
     jvalue = json_object_get(subroot, subpath);
     remstr = dotptr + 1;
     if (remstr > (path + strlen(path))) { //nothing beyond the next .
       nvds_log(KAFKA_JSON_PARSER, LOG_ERR, "kafka key not found; nothing beyond trailing .\n");
       FREE_AND_RETURN(0,root);
     }
     if(json_is_object(jvalue)) {
       subroot = jvalue;
     } else {
       nvds_log(KAFKA_JSON_PARSER, LOG_ERR, "provided path is not valid\n");
       FREE_AND_RETURN(0,root);
     }
     
     
   }

   // by the time we reach here we are at the last level object
   json_t *idvalue = json_object_get(subroot, remstr);
   if(json_is_string(idvalue)) {
     temp = json_string_value(idvalue);
     g_strlcpy(value, temp, nbuf);
     nvds_log(KAFKA_JSON_PARSER, LOG_DEBUG, "json value for id = %s\n", value);
     FREE_AND_RETURN(strlen(value),root);
   } else {
     nvds_log(KAFKA_JSON_PARSER, LOG_ERR, "json entry corresponding to path \
                               is not string or not found\n");
     FREE_AND_RETURN(strlen(value),root);
   }
    
}
