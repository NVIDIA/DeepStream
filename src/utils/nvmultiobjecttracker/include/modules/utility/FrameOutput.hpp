/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OCV_TRACKING_MODEXEC_FRAMEOUTPUT_HPP_
#define OCV_TRACKING_MODEXEC_FRAMEOUTPUT_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <unordered_map>

/** @brief Writes tracker visualization frames to image files or a video stream. */
class FrameOutput
{
public:
	/** @brief Initialize output state with video output disabled. */
	FrameOutput()
	{
		mOutputVideo = nullptr;
		mIsVideoOutputEnabled = false;
	}

	/** @brief Release the video writer and its output stream. */
	virtual ~FrameOutput()
	{
		if(mOutputVideo != nullptr)
		{
			if(mOutputVideo->isOpened())
			{
				mOutputVideo->release();
			}
		}
	}

	/**
	 * @brief Report whether image output is enabled for a directory key.
	 * @param[in] outputDirectoryKey Output directory key to query.
	 */
	bool isImageOutputEnabled(const uint outputDirectoryKey = 0)
	{
		if(mIsImageOutputEnabled.find(outputDirectoryKey) != mIsImageOutputEnabled.end())
		{
			return mIsImageOutputEnabled.at(outputDirectoryKey);
		}
		else
		{
			return false;
		}
	}

	/** @brief Report whether video output is enabled. */
	bool isVideoOutputEnabled() { return mIsVideoOutputEnabled; }

	/**
	 * @brief Create an AVI writer and enable video output.
	 * @param[in] outputVideoDirectory Directory in which the video is written.
	 * @param[in] outputVideoName Base name of the output video file.
	 * @param[in] videoSize Output video frame size.
	 * @param[in] fps Output video frame rate.
	 * @return True when the output video is opened successfully.
	 */
	bool enableVideoOutput(
		const std::string outputVideoDirectory, const std::string outputVideoName,
		cv::Size videoSize, const double fps = 5.f
	)
	{
		std::string mkdirCmd = std::string("mkdir -p " + outputVideoDirectory);
		if(!system(mkdirCmd.c_str()));

		mkdirCmd = std::string("rm -f " + outputVideoDirectory + "/*.avi");
		if(!system(mkdirCmd.c_str()));

		int codecType = cv::VideoWriter::fourcc('X','V','I','D');
		std::string fullOutputVideoFilename = outputVideoDirectory + "/" + outputVideoName + ".avi";

		if(mOutputVideo != nullptr)
			delete mOutputVideo;

		mOutputVideo = new cv::VideoWriter(fullOutputVideoFilename, codecType, fps, videoSize, true);

		if(!mOutputVideo->isOpened())
		{
			std::cout << "[ERROR] failed to open output video: " << fullOutputVideoFilename << std::endl;
			mIsVideoOutputEnabled = false;
			return false;
		}
		else
		{
			mIsVideoOutputEnabled = true;
			return true;
		}
	}

	/**
	 * @brief Configure a keyed directory for JPEG image output.
	 * @param[in] outputImageDirectory Directory in which images are written.
	 * @param[in] compressionLevel JPEG quality level.
	 * @param[in] outputDirectoryKey Key used to select the output directory.
	 */
	void enableImageOutput(
		const std::string outputImageDirectory, const int compressionLevel = 95, const uint outputDirectoryKey = 0
	)
	{
		assert(outputImageDirectory.size() > 0);

		std::string mkdirCmd = std::string("mkdir -p " + outputImageDirectory);
		if(!system(mkdirCmd.c_str()));

		/// Set the JPEG image quality.
		compression_params.clear();

		compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
		compression_params.push_back(compressionLevel);

		compression_params.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
		compression_params.push_back(1);

		mOutputImageDirectory[outputDirectoryKey] = outputImageDirectory;
		mIsImageOutputEnabled[outputDirectoryKey] = true;
	}

	/**
	 * @brief Append an image to the output video.
	 * @param[in] img Image to write as the next video frame.
	 */
	void writeVideoOutput(const cv::Mat &img)
	{
		assert(mIsVideoOutputEnabled);
		assert(mOutputVideo != nullptr);

		*mOutputVideo << img;
	}

	/**
	 * @brief Write an image to a keyed output directory.
	 * @param[in] img Image to write.
	 * @param[in] imgFilename Base name of the output image file.
	 * @param[in] outputDirectoryKey Key used to select the output directory.
	 * @param[in] subFolderName Optional subdirectory below the keyed output directory.
	 */
	void writeImageOutput(
		const cv::Mat &img, const std::string imgFilename,
		const uint outputDirectoryKey = 0, const std::string subFolderName = ""
	)
	{
		std::stringstream fullOutFilePath;

		fullOutFilePath << mOutputImageDirectory[outputDirectoryKey];

		if(subFolderName.size() > 0)
		{
			fullOutFilePath << "/" << subFolderName;
		}

		/// Create the output directory if it does not exist.
		std::unique_ptr<FILE, decltype(&pclose)> pipe(
			popen(std::string("mkdir -p " + fullOutFilePath.str()).c_str(), "r"), pclose
		);
		if (!pipe) {
			throw std::runtime_error("popen() failed!");
		}

		fullOutFilePath << "/" << imgFilename << ".jpg";

		if(isImageOutputEnabled(outputDirectoryKey) == true && mOutputImageDirectory[outputDirectoryKey].size() > 0)
		{
			cv::imwrite(fullOutFilePath.str(), img, compression_params);
		}
		else
		{
			std::string message = "mOutputImageDirectory[" + std::to_string(outputDirectoryKey) + "] doesn't exist !!";
			fprintf(stdout, "~~ CLOG[%s, %s() @line %d]: %s\n", __FILE__, __func__, __LINE__,  message.c_str());
		}
	}

protected:
	cv::VideoWriter *mOutputVideo; //!< Video writer used when video output is enabled.
	bool mIsVideoOutputEnabled; //!< Indicates whether video output is enabled.
	std::vector<int> compression_params; //!< OpenCV image-encoding parameters.
	std::unordered_map<uint, std::string> mOutputImageDirectory; //!< Image output directories keyed by caller-defined values.
	std::unordered_map<uint, bool> mIsImageOutputEnabled; //!< Image-output enabled state for each directory key.
};

#endif // OCV_TRACKING_MODEXEC_FRAMEOUTPUT_HPP_
