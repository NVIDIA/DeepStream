
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
#include <Yolo.h>
#include <vector>
#include <numeric>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <argsParser.h>


std::vector<std::string> parse_img_paths(argsParser& cmdLine) {
    return cmdLine.ParseStringList("img");
}

std::string parse_model_path(argsParser& cmdLine) {
    const char* engine_path_str = cmdLine.ParseString("engine");
    std::string engine_path;
    if (engine_path_str) engine_path = std::string(engine_path_str);
    return engine_path;
}

std::string parse_yolov_version(argsParser& cmdLine) {
    const char* version_str = cmdLine.ParseString("version");
    std::string version;
    if (version_str) version = std::string(version_str);
    return version;
}

bool print_help() {
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("---------------------------- yolo images detector ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--engine=yolo.engine' Load yolo trt-engine  \n");
    printf(" '--version=v7' Run yolov7/v8/v9, default v7  \n");
    printf(" '--img=img1,jpg,img2.jpg,img3.jpg' specify the path of the images, split by `,`\n");
    return true;
}


int main(int argc, char** argv){

    argsParser cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if(cmdLine.ParseFlag("help")) { print_help(); return 0; }

    std::string engine_path = parse_model_path(cmdLine);
    std::vector<std::string> img_paths = parse_img_paths(cmdLine);
    std::string yolo_version = parse_yolov_version(cmdLine);
    bool isYolov7 = true;
    if(yolo_version == "v8" || yolo_version == "v9"){
        isYolov7 = false;
    }
    else{
        isYolov7 = true;
    }
    // print img paths
    std::cout<<"input "<<img_paths.size()<<" images, paths: ";
    for(int i = 0;i < img_paths.size();i++) {
        std::cout<< img_paths[i]<<", ";
    }
    std::cout<<std::endl;
    
    Yolo yolo(engine_path);

    std::vector<cv::Mat> bgr_imgs;
    for(int i = 0; i< img_paths.size();i++){
        bgr_imgs.push_back(cv::imread(img_paths[i]));
    }
    
    std::cout<<"preprocess start"<<std::endl;

    yolo.preProcess(bgr_imgs);
    
    std::cout<<"inference start"<<std::endl;

    yolo.infer();

    std::cout<<"postprocessing start"<<std::endl;
    std::vector<std::vector<std::vector<float>>> nmsresults;
    if(isYolov7) // yolov7
        nmsresults = yolo.PostProcess(0.45f, 0.25f, isYolov7);
    else
        nmsresults = yolo.PostProcess(0.75f, 0.25f, isYolov7);
    for(int j =0; j < nmsresults.size();j++){
        Yolo::DrawBoxesonGraph(bgr_imgs[j],nmsresults[j]);
        std::string output_path = img_paths[j] + "detect" + std::to_string(j)+".jpg";      
        cv::imwrite(output_path, bgr_imgs[j]);
        std::cout<<"detectec image written to: "<<output_path<<std::endl;
    }
    
    std::cout<<"Done..."<<std::endl;
    
}
