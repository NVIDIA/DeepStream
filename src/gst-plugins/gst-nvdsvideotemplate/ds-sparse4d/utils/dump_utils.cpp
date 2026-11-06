/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dump_utils.h"

// Helper macro for CUDA error checking
inline bool CHECK_(int e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
      std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit (-1);
        return false;
    }
    return true;
}
#define ck(call) CHECK_(call, __LINE__, __FILE__)

// Helper function to create directory if it doesn't exist
static bool createDumpDirectory(const std::string& dirPath) {
    struct stat st = {0};

    // Check if directory exists
    if (stat(dirPath.c_str(), &st) == -1) {
        // Create directory with permissions 0755 (rwxr-xr-x)
        int result = mkdir(dirPath.c_str(), 0755);
        if (result != 0) {
            std::cerr << "Error creating directory: " << dirPath << std::endl;
            return false;
        }
    }

    return true;
}

// Helper function to determine optimal grid layout for tiling images
static std::pair<int, int> getOptimalGridSize(int batchSize) {
    // Find factors of batchSize to determine rows and columns
    int rows = 1;
    int cols = batchSize;

    // Try to find a more square-like layout
    for (int i = 2; i <= sqrt(batchSize); i++) {
        if (batchSize % i == 0) {
            int potential_rows = i;
            int potential_cols = batchSize / i;

            // Choose the configuration that's closest to a square
            if (abs(potential_rows - potential_cols) < abs(rows - cols)) {
                rows = potential_rows;
                cols = potential_cols;
            }
        }
    }

    return {rows, cols};
}

void DumpNvBufSurface(NvBufSurface *in_surface,
                     NvDsBatchMeta *batch_meta,
                     const std::string& image_name,
                     guint frame_num,
                     GstVideoFormat video_fmt,
                     void*& scratch_nvbuf_surface,
                     bool batched_mode,
                     PrettyLogger* logger)
{
  // Create dump directory if it doesn't exist
  std::string dumpDir = "dumps";
  if (!createDumpDirectory(dumpDir)) {
    logger->warning("Failed to create dump directory, will use current directory");
    dumpDir = ".";
  }
  guint numFilled = in_surface->numFilled;
  guint source_id = 0;
  guint i = 0;
  std::ofstream outfile;
  void *input_data = NULL;
  guint size = 0;

  // For batched mode, we'll need to allocate a larger buffer to hold all frames
  if (batched_mode && numFilled > 1) {
    // Compute optimal grid layout
    auto [rows, cols] = getOptimalGridSize(numFilled);

    // Calculate the total dimensions of the tiled image
    int single_width = in_surface->surfaceList[0].width;
    int single_height = in_surface->surfaceList[0].height;
    int total_width = single_width * cols;
    int total_height = single_height * rows;

    // Create a filename that includes the tiled dimensions
    std::string fname = dumpDir + "/" + image_name + "_tiled_" + std::to_string(total_width)
                      + "x" + std::to_string(total_height) + "_"
                      + "grid_" + std::to_string(rows) + "x" + std::to_string(cols) + "_"
                      + "frame-" + std::to_string(frame_num);

    switch (video_fmt) {
    case GST_VIDEO_FORMAT_NV12:
    {
      fname.append(".nv12");

      // Calculate size for the entire tiled image
      size_t pixel_size = (3 * single_width * single_height) / 2;
      size_t total_size = numFilled * pixel_size;

      // Allocate a larger buffer for the tiled image
      void* tiled_buffer = nullptr;
      ck(cudaMallocHost(&tiled_buffer, total_size));

      // For NV12, tiling is more complex due to the planar format
      // This is a simplified implementation - proper tiling would require
      // careful handling of the Y and UV planes

      // For now, we'll just concatenate the frames sequentially
      for (i = 0; i < numFilled; i++) {
        input_data = in_surface->surfaceList[i].dataPtr;
        void* target_ptr = static_cast<uint8_t*>(tiled_buffer) + (i * pixel_size);

        cudaMemcpy2D(target_ptr, single_width,
                    input_data, single_width,
                    single_width, (single_height * 3) / 2,
                    cudaMemcpyDeviceToHost);
      }

      // Write the tiled image to file
      outfile.open(fname, std::ofstream::out);
      outfile.write(reinterpret_cast<char*>(tiled_buffer), total_size);
      outfile.close();

      // Free the tiled buffer
      cudaFreeHost(tiled_buffer);

      logger->info("Dumped Batched Input Frame: " + fname);
    }
    break;

    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRx:
    {
      if (video_fmt == GST_VIDEO_FORMAT_RGBA)
        fname.append(".rgba");
      else if (video_fmt == GST_VIDEO_FORMAT_BGRx)
        fname.append(".bgrx");

      // Calculate the bytes per pixel (4 for RGBA/BGRx)
      int bytes_per_pixel = 4;

      // Allocate buffer for the entire tiled image
      size_t total_size = total_width * total_height * bytes_per_pixel;
      void* tiled_buffer = nullptr;
      ck(cudaMallocHost(&tiled_buffer, total_size));

      // Initialize buffer to zero (black)
      memset(tiled_buffer, 0, total_size);

      // Copy each frame into its position in the grid
      for (i = 0; i < numFilled; i++) {
        input_data = in_surface->surfaceList[i].dataPtr;

        // Calculate position in the grid
        int row = i / cols;
        int col = i % cols;

        // Calculate offset in the tiled buffer
        size_t dst_offset = (row * total_width * single_height + col * single_width) * bytes_per_pixel;
        uint8_t* dst_ptr = static_cast<uint8_t*>(tiled_buffer) + dst_offset;

        // Copy the frame data into the tiled buffer row by row
        for (int h = 0; h < single_height; h++) {
          cudaMemcpy2D(dst_ptr + h * total_width * bytes_per_pixel,
                      total_width * bytes_per_pixel,
                      static_cast<uint8_t*>(input_data) + h * single_width * bytes_per_pixel,
                      single_width * bytes_per_pixel,
                      single_width * bytes_per_pixel,
                      1, cudaMemcpyDeviceToHost);
        }
      }

      // Write the tiled image to file
      outfile.open(fname, std::ofstream::out);
      outfile.write(reinterpret_cast<char*>(tiled_buffer), total_size);
      outfile.close();

      // Free the tiled buffer
      cudaFreeHost(tiled_buffer);

      logger->info("Dumped Batched Input Frame: " + fname);
    }
    break;

    default:
      logger->warning("ERROR: NOT SUPPORTED FORMAT for batched dumping in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      break;
    }

    // Return after batched dump is complete
    return;
  }

  // Original individual file dumping logic (non-batched mode)
  for (i=0; i < numFilled; i++)
  {
      if (batch_meta)
      {
	    NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
	    source_id = frame_meta->pad_index;
      }

	  std::string fname = dumpDir + "/" + image_name + "_" + std::to_string(in_surface->surfaceList[i].width)
	  + "x" + std::to_string(in_surface->surfaceList[i].height) + "_" +
      "frame-" + std::to_string(frame_num) + "_" +
      "batch-" + std::to_string(source_id);

	  input_data = in_surface->surfaceList[i].dataPtr;

	  switch (video_fmt)
	  {
      case GST_VIDEO_FORMAT_NV12:
      {
        fname.append(".nv12");

        size = (in_surface->surfaceList[i].width * in_surface->surfaceList[i].height * 3) / 2;

        if (scratch_nvbuf_surface == NULL)
        {
          ck (cudaMallocHost (&scratch_nvbuf_surface, size));
          outfile.open (fname, std::ofstream::out);
        } else {
          outfile.open (fname, std::ofstream::out | std::ofstream::app);
        }

        cudaMemcpy2D (scratch_nvbuf_surface,
            in_surface->surfaceList[i].width, input_data, in_surface->surfaceList[i].width,
            in_surface->surfaceList[i].width,
            (in_surface->surfaceList[i].height*3)/2, cudaMemcpyDeviceToHost);

        outfile.write(reinterpret_cast<char*>(scratch_nvbuf_surface), size);
        outfile.close();
      }
      break;

      case GST_VIDEO_FORMAT_RGBA:
      case GST_VIDEO_FORMAT_BGRx:
      {
        if (video_fmt == GST_VIDEO_FORMAT_RGBA)
          fname.append(".rgba");
        else if (video_fmt == GST_VIDEO_FORMAT_BGRx)
          fname.append(".bgrx");

        size = in_surface->surfaceList[i].width *in_surface->surfaceList[i].height*4;

        if (scratch_nvbuf_surface == NULL)
        {
          ck (cudaMallocHost (&scratch_nvbuf_surface, size));
          outfile.open (fname, std::ofstream::out);
        }
        else
        {
          outfile.open (fname, std::ofstream::out | std::ofstream::app);
        }

        cudaMemcpy2D (scratch_nvbuf_surface,
            in_surface->surfaceList[i].width*4, input_data,
            in_surface->surfaceList[i].width*4, in_surface->surfaceList[i].width*4,
            in_surface->surfaceList[i].height, cudaMemcpyDeviceToHost);

        outfile.write(reinterpret_cast<char*>(scratch_nvbuf_surface), size);
        outfile.close();
      }
      break;

      default:
        logger->warning("ERROR: NOT SUPPORTED FORMAT for dumping in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        break;
      }
    logger->info("Dumped Input Frame: " + fname);
  }
}


bool dump_tensor(const Tensor& tensor,
                const std::string& image_name,
                bool batched_mode,
                PrettyLogger* logger) {
    // Create dump directory if it doesn't exist
    std::string dumpDir = "dumps";
    if (!createDumpDirectory(dumpDir)) {
      logger->warning("Failed to create dump directory, will use current directory");
      dumpDir = ".";
    }
    // Allocate host memory
    void* host_buffer = nullptr;
    cudaError_t cuda_err = cudaMallocHost(&host_buffer, tensor.size);
    if (cuda_err != cudaSuccess) {
      logger->warning("Failed to allocate host memory for preprocessed frame dump: " + std::string(cudaGetErrorString(cuda_err)) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      return false;
    }

    bool success = false;

    // Use try-catch to ensure host_buffer is always freed even if an exception occurs
    try {
        // Copy from device to host
        cuda_err = cudaMemcpy(host_buffer, tensor.buffer, tensor.size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            logger->warning("Failed to copy data from device to host for image dump: " + std::string(cudaGetErrorString(cuda_err)) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
            throw std::runtime_error("CUDA memcpy failed");
        }

        // Process each batch in the tensor
        success = true;

        int batch_size = 1, channels = 1, height = 1, width = 1;

        // Extract dimensions based on shape size
        if (tensor.shape.size() == 5) {
            // BNCHW format
            batch_size = tensor.shape[1];
            channels = tensor.shape[2];
            height = tensor.shape[3];
            width = tensor.shape[4];
        } else if (tensor.shape.size() == 4) {
            // NCHW format
            batch_size = tensor.shape[0];
            channels = tensor.shape[1];
            height = tensor.shape[2];
            width = tensor.shape[3];
        } else if (tensor.shape.size() == 3) {
            // CHW format (single batch)
            channels = tensor.shape[0];
            height = tensor.shape[1];
            width = tensor.shape[2];
        } else if (tensor.shape.size() == 2) {
            // HW format (single batch, single channel)
            height = tensor.shape[0];
            width = tensor.shape[1];
        } else {
            logger->warning("Unsupported tensor shape size: " + std::to_string(tensor.shape.size()) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
            return false;
        }

        // Log the detected tensor dimensions for debugging
        std::string shape_str = "Tensor shape: [";
        for (size_t i = 0; i < tensor.shape.size(); i++) {
            shape_str += std::to_string(tensor.shape[i]);
            if (i < tensor.shape.size() - 1) shape_str += ",";
        }
        shape_str += "]";
        logger->info(shape_str);

        logger->info("Using dimensions: batch_size=" + std::to_string(batch_size) + 
                    ", channels=" + std::to_string(channels) + 
                    ", height=" + std::to_string(height) + 
                    ", width=" + std::to_string(width));

        // Batched mode for tensors
        if (batched_mode && batch_size > 1) {
            // Compute optimal grid layout
            auto [rows, cols] = getOptimalGridSize(batch_size);

            // Calculate total dimensions for the tiled tensor
            int total_width = width * cols;
            int total_height = height * rows;

            // Format filename to include tiled dimensions
            std::string fname = dumpDir + "/" + image_name + "_tiled_" + std::to_string(total_width) 
                              + "x" + std::to_string(total_height) + "_" 
                              + "grid_" + std::to_string(rows) + "x" + std::to_string(cols) + "_"
                              + "frame-" + std::to_string(tensor.frame_num);

            // Determine file extension based on tensor data type and channel count
            if (tensor.data_type == NvDsDataType_FP32) {
                if (channels == 3) {
                    fname.append("_fp32.rgb"); // 3-channel float data (RGB)
                } else if (channels == 4) {
                    fname.append("_fp32.rgba"); // 4-channel float data (RGBA)
                } else {
                    fname.append("_fp32.tensor"); // Generic tensor float data
                }
            } else if (tensor.data_type == NvDsDataType_UINT8) {
                if (channels == 3) {
                    fname.append("_uint8.rgb"); // 3-channel byte data (RGB)
                } else if (channels == 4) {
                    fname.append("_uint8.rgba"); // 4-channel byte data (RGBA)
                } else {
                    fname.append("_uint8.tensor"); // Generic tensor byte data
                }
            } else {
                fname.append(".raw"); // Generic raw data
            }

            // Calculate sizes
            size_t element_size = (tensor.data_type == NvDsDataType_FP32) ? sizeof(float) : sizeof(uint8_t);

            // Calculate strides for BNCHW format
            // For BNCHW, we need to calculate the correct strides for each dimension
            size_t stride_w = element_size;
            size_t stride_h = width * stride_w;
            size_t stride_c = height * stride_h;
            size_t stride_n = channels * stride_c;

            // Calculate total size for the tiled output
            size_t total_size = total_width * total_height * channels * element_size;

            // Create a tiled buffer
            void* tiled_buffer = nullptr;
            cudaError_t tiled_err = cudaMallocHost(&tiled_buffer, total_size);
            if (tiled_err != cudaSuccess) {
                logger->warning("Failed to allocate tiled buffer for tensor dump: " +
                                std::string(cudaGetErrorString(tiled_err)));
                throw std::runtime_error("CUDA tiled buffer allocation failed");
            }

            // Initialize buffer to zeros
            memset(tiled_buffer, 0, total_size);

            // For BNCHW format, handle specially since B=1 can be ignored
            // This implementation assumes planar data for each channel (NCHW layout)
            for (int n = 0; n < batch_size; n++) {
                // Calculate position in the grid
                int row = n / cols;
                int col = n % cols;

                // Calculate destination base position in tiled image
                size_t dst_base = (row * height * total_width + col * width) * channels * element_size;

                // For each channel
                for (int c = 0; c < channels; c++) {
                    // Process row by row
                    for (int h = 0; h < height; h++) {
                        // Using pre-calculated strides for BNCHW
                        // We skip B since it's always 1 and can be ignored
                        size_t src_offset = 0 + (n * stride_n) + (c * stride_c) + (h * stride_h);

                        // Calculate destination offset for this row
                        size_t dst_row_offset = h * total_width * channels * element_size;

                        // For each pixel in the row
                        for (int w = 0; w < width; w++) {
                            // Copy each pixel from the source to the appropriate place in the destination
                            // For destination, we need to convert planar to interleaved
                            size_t dst_pixel_offset = w * channels * element_size;
                            memcpy(
                                static_cast<uint8_t*>(tiled_buffer) + dst_base + dst_row_offset + dst_pixel_offset + (c * element_size),
                                static_cast<uint8_t*>(host_buffer) + src_offset + (w * stride_w),
                                element_size
                            );
                        }
                    }
                }
            }

            // Write the tiled tensor to file
            std::ofstream outfile(fname, std::ios::binary);
            if (!outfile.is_open()) {
                logger->warning("Could not open file " + fname + " for writing in " +
                                std::string(__FILE__) + ":" + std::to_string(__LINE__));
                cudaFreeHost(tiled_buffer);
                return false;
            }

            outfile.write(static_cast<const char*>(tiled_buffer), total_size);
            outfile.close();

            // Free tiled buffer
            cudaFreeHost(tiled_buffer);

            logger->info("Dumped Batched Tensor: " + fname);

            // Return success after batched dump is complete
            return true;
        }

        // Original individual file dumping logic (non-batched mode)
        for (int b = 0; b < batch_size; b++) {
            // Create filename using the same convention as DumpNvBufSurface
            std::string fname = dumpDir + "/" + image_name + "_" + std::to_string(width) + "x" +
                               std::to_string(height) + "_" +
                               "frame-" + std::to_string(tensor.frame_num) + "_" +
                               "batch-" + std::to_string(b);

            // Determine file extension based on tensor data type and channel count
            if (tensor.data_type == NvDsDataType_FP32) {
                if (channels == 3) {
                    fname.append("_fp32.rgb"); // 3-channel float data (RGB)
                } else if (channels == 4) {
                    fname.append("_fp32.rgba"); // 4-channel float data (RGBA)
                } else {
                    fname.append("_fp32.tensor"); // Generic tensor float data
                }
            } else if (tensor.data_type == NvDsDataType_UINT8) {
                if (channels == 3) {
                    fname.append("_uint8.rgb"); // 3-channel byte data (RGB)
                } else if (channels == 4) {
                    fname.append("_uint8.rgba"); // 4-channel byte data (RGBA)
                } else {
                    fname.append("_uint8.tensor"); // Generic tensor byte data
                }
            } else {
                fname.append(".raw"); // Generic raw data
            }

            // Calculate offset for this batch item in the tensor
            size_t batch_offset = b * channels * height * width *
                                (tensor.data_type == NvDsDataType_FP32 ? sizeof(float) : sizeof(uint8_t));

            // Calculate size of one batch item
            size_t batch_item_size = channels * height * width *
                                    (tensor.data_type == NvDsDataType_FP32 ? sizeof(float) : sizeof(uint8_t));

            // Open file for binary write
            std::ofstream outfile(fname, std::ios::binary);
            if (!outfile.is_open()) {
                logger->warning("Could not open file " + fname + " for writing in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
                success = false;
                continue;
            }

            // Write the raw data for this batch item
            outfile.write(static_cast<const char*>(host_buffer) + batch_offset, batch_item_size);
            outfile.close();

            logger->info("Dumped Preprocessed Frame: " + fname);
        }
    }
    catch (const std::exception& e) {
      logger->warning("Exception during preprocessed frame dump: " + std::string(e.what()) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      success = false;
    }

    // Free the host buffer
    if (host_buffer) {
      cudaFreeHost(host_buffer);
    }

    return success;
}
