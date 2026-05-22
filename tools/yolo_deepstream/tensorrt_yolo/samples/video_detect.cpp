
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
#include <iostream>


std::string parse_video_path(argsParser& cmdLine) {
    const char* video_path_str = cmdLine.ParseString("video");
    std::string video_path;
    if (video_path_str) video_path = std::string(video_path_str);
    return video_path;
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
    printf(" '--video=video.mp4' specify the path of the video \n");
    printf(" '--version=v7' Run yolov7/v8/v9, default v7  \n");
    return true;
}


int main(int argc, char** argv){

    argsParser cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if(cmdLine.ParseFlag("help")) { print_help(); return 0; }

    std::string engine_path = parse_model_path(cmdLine);
    std::string video_path = parse_video_path(cmdLine);
    std::string yolo_version = parse_yolov_version(cmdLine);
    bool isYolov7 = true;
    if(yolo_version == "v8" || yolo_version == "v9"){
        isYolov7 = false;
    }
    else{
        isYolov7 = true;
    }
    Yolo yolo(engine_path);

    cv::VideoCapture capture;
    cv::Mat frame;
    frame= capture.open(video_path);
    if(!capture.isOpened())
    {
        printf("can not open ... please check whether your opencv has installed with ffmpeg..\n");
        return -1;
    }
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open(std::string(video_path+".detect.mp4"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);
    std::vector<cv::Mat> framev;
    std::vector<std::vector<std::vector<float>>> nmsresults;
    int total_frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int i = 0;
    while (capture.read(frame)){
        framev.push_back(frame);
        yolo.preProcess(framev);
        yolo.infer();
        nmsresults = yolo.PostProcess(0.45f, 0.25f, isYolov7);
        Yolo::DrawBoxesonGraph(frame,nmsresults[0]);
        writer.write(frame);
        framev.clear();
        i++;
        printf("\r%d / %d", i, total_frame_count);
        fflush(stdout);
    }
    capture.release();
    std::cout<<"Done..."<<std::endl;
    
}
