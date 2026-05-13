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
* @file nv_spot_csvparser.hpp
* @brief NVSPOT CSV File Parser library to be used by applications / libraries
*/

#ifndef _NV_SPOT_CSVPARSER_HPP_
#define _NV_SPOT_CSVPARSER_HPP_

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;

namespace nvspot_csv {
/** Data structure contaning all the parameters specified in one row
 * of a Spot CSV file */
typedef struct _NvSpotCsvFields
{
  uint32_t cameraId;        /**< Serial number for each Spot View Camera Entry. 1st column in CSV file */
  uint32_t rowId;           /**< Row entry number in the CSV file */
  string cameraIdString;    /**< Camera ID String. 4th column in CSV file */
  uint32_t surfaceid;
  uint32_t spot_index;      /**< Surface Index 0,1 corrosponding to Spot View. 8th column. */
  string level;
  string spotId;            /**< Unique Spot ID. 5th column in CSV file */

  double dewarpTopAngle, dewarpBottomAngle, dewarpPitch, dewarpYaw, dewarpRoll;
  uint32_t vertical_left;
  uint32_t vertical_right;
  /** Spot View horizontal line co-ordinates */
  uint32_t Horizon_x1, Horizon_y1, Horizon_x2, Horizon_y2;
  /** Spot View ROI co-ordinates */
  uint32_t spot_roi_x1, spot_roi_y1, spot_roi_x2, spot_roi_y2;

  /** Global Co-ordinates */
  float x0, y0, x1, y1, x2, y2, x3, y3;
  float lng0, lat0, lng1, lat1, lng2, lat2, lng3, lat3;

  string sensorId; /**< Sensor ID String. 2nd column in CSV file */

  float dewarpFocalLength; /**< Focal Lenght of camera lens, in pixels per radian */
  uint32_t dewarpWidth;   /**< dewarped surface width */
  uint32_t dewarpHeight;/**< dewarped surface height */
  uint32_t num_views;
} NvSpotCsvFields;

/** std::map<std::pair<surface_id, spot_index>, NvSpotCsvFields> */
typedef std::map<std::pair<uint32_t, uint32_t>, NvSpotCsvFields> _SpotIndex_Map;
typedef std::map<uint32_t, _SpotIndex_Map> _SpotCSVMap;

/** std::map<camera-ipaddress, camera-id> */
typedef std::map<string, uint32_t> _SpotCameraMap;

/** std::map<camera-id, num_spot_views> */
typedef std::map<uint32_t, uint32_t> _SpotCameraViews;

typedef std::pair<_SpotIndex_Map::iterator, bool>  _SpotIndexMap_Result;
typedef std::pair<_SpotCSVMap::iterator, bool>  _SpotCSVMap_Result;
typedef std::pair<_SpotCameraMap::iterator, bool> _SpotCameraMap_Result;
typedef std::pair<_SpotCameraViews::iterator, bool> _SpotCameraView_Result;

/**
 * @brief Class for parsing of Spot CSV data
 */
class SpotCSVParser
{
  private:
    _SpotCSVMap CSVMap;
    _SpotCameraViews CameraViews;

    string csvFileName;

    void LoadCSVData();
    void DestroyCSVParser();
    uint32_t prepareSpotCSVMaxViews (uint32_t cam_id, vector<int>* vector_surface_index);

  public:
    ~SpotCSVParser();

    /** Get all the fields for a particular "cam_id", "surface_id" and "spot_id" in "fields"
     * @return 0 if successful. -1 if failed.
    */
    int getNvSpotCSVFields (uint32_t cam_id, uint32_t surface_id, uint32_t spot_id, NvSpotCsvFields *fields);
    /** Function to get all the spot views for a "cam_id" in "array_surface_index"*
     * @return Number of spot views */
    uint32_t getNvSpotCSVMaxViews (uint32_t cam_id, vector<int>* array_surface_index);
    /** Get all the parsed data in "csvSpotData" */
    void getNvSpotCSVData(std::vector<NvSpotCsvFields> &csvSpotData);
    /** Print all the data */
    void printNvSpotCSVData(void);
    /** Print all the data for a particular "cam_id", "surface_id" and "spot_id" */
    void printNvSpotCSVData(uint32_t cam_id, uint32_t surface_id, uint32_t spot_id);
    /** Print all the values in "val" */
    void printSpotCSVFields(NvSpotCsvFields val);

    SpotCSVParser(string CSVFileName);
};

}

#endif
