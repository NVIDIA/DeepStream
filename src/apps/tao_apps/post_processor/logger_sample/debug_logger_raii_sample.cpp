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

#include "../debug_logger_raii.hpp"
#include <vector>

// Sample function demonstrating debug logger usage
void processData(const std::vector<float>& data) {
    // Start debug section
    {
        DEBUG_DUMP_SECTION();
        
        DEBUG_DUMP("Processing %zu elements", data.size());
        
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i] > 0.5f) {
                DEBUG_DUMP("Element %zu: %f exceeds threshold", i, data[i]);
            }
        }
        
        DEBUG_DUMP("Processing complete");
    } // Debug logger automatically closes here
}

// Sample main function for testing
int main() {
    std::vector<float> test_data = {0.1f, 0.6f, 0.3f, 0.8f, 0.2f};
    // expose DEBUG=1
    // unset DEBUG
    processData(test_data);
    return 0;
}
