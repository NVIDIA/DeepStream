/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_test4_yml_parse.h"
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>

guint ds_test4_parse_meta_type(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group]) {

      if (docs[i][group]["msg2p-newapi"]) {
          val= docs[i][group]["msg2p-newapi"].as<guint>();
          return val;
      }
    }
  }

  return 0;
}
