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

#include "post_processor.h"

using namespace std;

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
NvDsPostProcessStatus
ModelPostProcessor::parseLabelsFile(const std::string& labelsFilePath)
{
  std::ifstream labels_file(labelsFilePath);
  std::string delim{';'};
  if (!labels_file.is_open())
  {
    printError("Could not open labels file:%s", safeStr(labelsFilePath));
    return NVDSPOSTPROCESS_CONFIG_FAILED;
  }
  while (labels_file.good() && !labels_file.eof())
  {
    std::string line, word;
    std::vector<std::string> l;
    size_t pos = 0, oldpos = 0;

    std::getline(labels_file, line, '\n');
    if (line.empty())
      continue;

    while ((pos = line.find(delim, oldpos)) != std::string::npos)
    {
      word = line.substr(oldpos, pos - oldpos);
      l.push_back(word);
      oldpos = pos + delim.length();
    }
    l.push_back(line.substr(oldpos));
    m_Labels.push_back(l);
  }

  if (labels_file.bad())
  {
    printError("Failed to parse labels file:%s, iostate:%d",
        safeStr(labelsFilePath), (int)labels_file.rdstate());
    return NVDSPOSTPROCESS_CONFIG_FAILED;
  }
  return NVDSPOSTPROCESS_SUCCESS;
}


NvDsPostProcessStatus
ModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{

    if (!string_empty(initParams.labelsFilePath))
    {
      if (NVDSPOSTPROCESS_SUCCESS != parseLabelsFile(initParams.labelsFilePath)){
          printError("parse label file:%s failed", initParams.labelsFilePath);
      }
    }
    return NVDSPOSTPROCESS_SUCCESS;
}

