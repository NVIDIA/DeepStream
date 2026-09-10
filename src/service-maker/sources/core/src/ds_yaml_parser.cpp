/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ds_yaml_parser.hpp"

void YAML_DSConfig::CreateElementFromNode (const YAML::Node &node)
{
	// This section parses the nodes under "nodes" in YAML
	// These nodes represents individual element in DS pipeline

	for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {

		const YAML::Node& elementNode = *it;

		if (!elementNode["type"]) {
			cout<<"type Value for element is required \n";
			return;
		}

		if(!elementNode["name"]) {
			cout<<"name value for element is required \n";
			return;
		}

		std::string type = elementNode["type"].as<std::string>();
		std::string name = elementNode["name"].as<std::string>();

		if (elementNode["enable"])
		{
			bool enable = elementNode["enable"].as<bool>();
			if (!enable)
				type = "identity";
		}

		YAML_DSElement *element = new YAML_DSElement(type);
		element->m_linkName.assign(name);

		if (elementNode["properties"]){
			element->properties_ = elementNode["properties"];
		}

		m_dsElement.push_back(*element);
		delete element;
	}
}

void YAML_DSConfig::CreateElementsLink (const YAML::Node &node)
{
	for(YAML::const_iterator it=node.begin();it != node.end();++it)
	{
		std::string src = it->first.as<std::string>();
		if (it->second.IsSequence()) {
			auto list_node = it->second;
			for (YAML::const_iterator it2=list_node.begin();it2 != list_node.end();++it2) {
				std::string dest = it2->as<std::string>();
				YAML_DSLink *link = new YAML_DSLink(src, dest);
				m_dsLinks_vec.push_back(link);
			}

		} else {
			std::string dest = it->second.as<std::string>();
			YAML_DSLink *link = new YAML_DSLink(src, dest);
			m_dsLinks_vec.push_back(link);
		}

	}
}

void YAML_DSConfig::ParseDSYAML(const YAML::Node &node)
{
	for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
	{
		std::string key = it->first.as<std::string>();
		cout << "\nNode -- " << key << " : "<< endl;
		try
		{
			if (key.find("deepstream") != std::string::npos)
			{
				ParseDSYAML(it->second);
			}
			if (key.find("nodes") != std::string::npos)
			{

				CreateElementFromNode(it->second);
			}
			else if (key.find("edges") != std::string::npos)
			{
				CreateElementsLink(it->second);
			}
		}
		catch (const exception &e)
		{
			cout << "\t Exception Occured " << e.what() << endl;
			exit(-1);
		}
	}
}

void YAML_DSConfig::Parse()
{
	const YAML::Node &node = m_yamlNode;
	ParseDSYAML(node);

	// Update the m_linkToElement values after YAML parsing
	vector<YAML_DSLink*>::iterator it;
	for (it=m_dsLinks_vec.begin(); it<m_dsLinks_vec.end(); ++it)
	{
		string source = (*it)->source;
		string target = (*it)->target;

		// Element name has " . " in it, for eg: nvurisrcbin.asrc_%u
		// This signifies user has provided a specific pad template that needs to be used during linking 
		// In the above example element: nvurisrcbin, pad_template: asrc_%u

		// Checks if source/target element has pad template attached to it
		// We divide the string into element name and pad template 
		string source_element_src_pad_template = "";
		if (source.find(".") != std::string::npos) {
			source_element_src_pad_template = source.substr (source.find(".")+1, source.length());
			source = source.substr (0, source.find ("."));
		}

		string target_element_sink_pad_template = "";
		if (target.find(".") != std::string::npos) {
			target_element_sink_pad_template = target.substr (target.find(".")+1, target.length());
			target = target.substr (0, target.find ("."));
		}

		// we get only the element name for source and target to create linking data along with user provided pad template if any
		vector<YAML_DSElement>::iterator element_vec;
		for (element_vec=m_dsElement.begin(); element_vec<m_dsElement.end(); ++element_vec)
		{
			if (source.compare(element_vec->m_linkName) == 0)
			{
				element_vec->m_linkToElement.push_back(Link_To_Element(target, source_element_src_pad_template, target_element_sink_pad_template));
				break;
			}
		}
	}
}

void YAML_DSConfig::PrintElements()
{
	vector<YAML_DSElement>::iterator it;
	for (it=m_dsElement.begin(); it<m_dsElement.end(); ++it)
	{
		cout << "Element ---- " << it->m_elementName
				<< " RefName " << it->m_linkName << endl;
		if (it->properties_.IsNull()) continue;
		for (auto prop : it->properties_) {
			if (prop.second.IsSequence()) {
				for (const auto& item : prop.second) {
					cout << prop.first << ":" << item.as<string>() << endl;
				}
			} else {
				cout << prop.first << ":" << prop.second.as<string>() << endl;
			}
		}
	}
}