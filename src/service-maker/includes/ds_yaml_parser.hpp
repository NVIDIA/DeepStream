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

#include <iostream>
#include <typeinfo>
#include "yaml-cpp/yaml.h"
#include <string.h>

using namespace std;

class Link_To_Element {
	// Class Used to Pass PAD Template Data
	public:
		std:: string target;
		std:: string source_element_src_pad_template;
		std:: string target_element_sink_pad_template;

		Link_To_Element (std:: string target,  std:: string source_element_src_pad_template, std:: string target_element_sink_pad_template) {
			this->target = target;
			this->source_element_src_pad_template = source_element_src_pad_template;
			this->target_element_sink_pad_template = target_element_sink_pad_template;
		}
};

// TODO: Make required members private instead of public
class YAML_DSLink {
public:
	std::string source;
	std::string target;
	YAML_DSLink(std::string src, std::string dest) {
		source.assign(src);
		target.assign(dest);
	}
};

// TODO: Make required members private instead of public
class YAML_DSElement {
public:
	std::string m_elementName;
	std::string m_linkName;
	// Used to pass linking data
	std::vector< Link_To_Element > m_linkToElement;
	std::vector<YAML_DSElement> m_subConfigs;
	// properties
	YAML::Node properties_;

	~YAML_DSElement() {
		m_subConfigs.clear();
	}

	YAML_DSElement(std::string name): m_elementName(name) {
		m_subConfigs.clear();
	}

};

class YAML_DSConfig {
public:
	YAML_DSConfig(std::string ds_yaml_file) {
		m_yamlName.assign(ds_yaml_file);
		m_yamlNode = YAML::LoadFile(ds_yaml_file);
	}
	const std::vector<YAML_DSElement> & getElementsVector() {
		return m_dsElement;
	}
	void Parse();
	void PrintElements();

private:
	YAML::Node m_yamlNode;
	std::string m_yamlName;
	std::vector<YAML_DSElement> m_dsElement;
	std::vector<YAML_DSLink*> m_dsLinks_vec;

	void CreateElementFromNode (const YAML::Node &node);
	void CreateElementsLink (const YAML::Node &node);
	void ParseDSYAML(const YAML::Node &node);
};