/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
* @file nv_aisle_csvparser.hpp
* @brief NVAISLE CSV File Parser library to be used by applications / libraries
*/

#ifndef _NV_AISLE_CSVPARSER_HPP_
#define _NV_AISLE_CSVPARSER_HPP_


#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;

namespace nvaisle_csv {
/** Data structure contaning all the parameters specified in one row
 * of an Aisle CSV file */
typedef struct _NvAisleCsvFields
{
  uint32_t cameraId;        /**< Serial number for each Aisle View Camera Entry. 1st column in CSV file */
  uint32_t rowId;           /**< Row entry number in the CSV file */
  string level;             /**< Parking Level. 7th column in CSV file */
  string cameraIDString;    /**< Camera ID String. 4th column in CSV file */

  string aisleIdStr;        /**< Aisle ID string. 5th column in CSV file */
  uint32_t aisleId;         /**< Set to 0/1 corrsponding to A0/A1 present in "aisleIdStr" */

  /**Top & Bottom Field of View Angles, Viewing parameter : Pith, Yaw, Roll in degrees */
  double dewarpTopAngle, dewarpBottomAngle, dewarpPitch, dewarpYaw, dewarpRoll;
  /** Number of ROI co-ordinates */
  uint32_t numROIpoints;
  /** ROI Co-ordinates */
  float ROI_x0, ROI_y0, ROI_x1, ROI_y1, ROI_x2, ROI_y2, ROI_x3, ROI_y3,
    ROI_x4, ROI_y4, ROI_x5, ROI_y5, ROI_x6, ROI_y6, ROI_x7, ROI_y7;
  /** Global and Camera Co-ordinates */
  float gx0, gy0, gx1, gy1, gx2, gy2, gx3, gy3, cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3;

  int entry;    /**< entry - 0 indicates no entry ROI, 1 indicates valid entry ROI */
  int exit;     /**< exit - 0 indicates no exit ROI, 1 indicates valid exit ROI */
  /** Entry ROI Co-ordinates */
  float entry_ROI_x0, entry_ROI_y0, entry_ROI_x1, entry_ROI_y1, entry_ROI_x2,
        entry_ROI_y2, entry_ROI_x3, entry_ROI_y3;
  /** Exit ROI Co-ordinates */
  float exit_ROI_x0, exit_ROI_y0, exit_ROI_x1, exit_ROI_y1, exit_ROI_x2,
        exit_ROI_y2, exit_ROI_x3, exit_ROI_y3;
  string sensorId;          /**< Sensor ID String. 2nd column in CSV file */
  float dewarpFocalLength;  /**< Focal Lenght of camera lens, in pixels per radian */
  uint32_t dewarpWidth;     /**< dewarped surface width */
  uint32_t dewarpHeight;    /**< dewarped surface height */
  /** Perspective Transformation Matrix matrix for camera points to global points transformation*/
  double h0, h1, h2, h3, h4, h5, h6, h7, h8;

} NvAisleCsvFields;

/** std::map<camera_id, NvAisleCsvFields> */
typedef std::map<uint32_t, NvAisleCsvFields> _AisleIndex_Map;

/** std::map<camera_id, _AisleIndex_Map> */
typedef std::map<uint32_t, _AisleIndex_Map> _AisleCSVMap;

/** std::map<camera-ipaddress, camera-id> */
typedef std::map<string, uint32_t> _AisleCameraMap;

/** std::map<camera-id, num_aisle_views> */
typedef std::map<uint32_t, uint32_t> _AisleCameraViews;

typedef std::pair<_AisleIndex_Map::iterator, bool>  _AisleIndexMap_Result;
typedef std::pair<_AisleCSVMap::iterator, bool>  _AisleCSVMap_Result;
typedef std::pair<_AisleCameraMap::iterator, bool> _AisleCameraMap_Result;
typedef std::pair<_AisleCameraViews::iterator, bool> _AisleCameraView_Result;
/**
 * @brief Class for parsing of Aisle CSV data
 */
class AisleCSVParser
{
  private:
    _AisleCSVMap CSVMap;
    _AisleCameraViews CameraViews;

    string csvFileName;

    /** Function to read all data from CSV file. */
    void LoadCSVData();
    void DestroyCSVParser();

    /** Collect all the aisle views for a "cam_id" in  "vector_surface_index"
     * @return Number of views
    */
    uint32_t prepareAisleCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index);

  public:
    ~AisleCSVParser();

    /** Get all the fields for a particular "cam_id" and "aisle_id" in "fields"
     * @return 0 if successful. -1 if failed.
    */
    int getNvAisleCSVFields (uint32_t cam_id, uint32_t aisle_id, NvAisleCsvFields *fields);
    /** Function to get all the aisle views for a "cam_id" in "vector_surface_index"*
     * @return Number of aisle views */
    uint32_t getNvAisleCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index);

    /** Get all the parsed data in "csvAisleData" */
    void getNvAisleCSVData(std::vector<NvAisleCsvFields> &csvAisleData);

    /** Print all the data */
    void printNvAisleCSVData(void);
    /** Print all the data for a particular "cam_id" and "aisle_id" */
    void printNvAisleCSVData(uint32_t cam_id, uint32_t aisle_id);
    /** Print all the values in "val" */
    void printAisleCSVFields(NvAisleCsvFields val);

    AisleCSVParser(string CSVFileName);
};

}
#endif
