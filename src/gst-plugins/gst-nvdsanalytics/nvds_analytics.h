/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef _NVDS_ANALYTICS_H_
#define _NVDS_ANALYTICS_H_
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef __cplusplus
extern "C"
{
#endif


enum class eMode {
 balanced,
 strict,
 loose
};

enum class eModeDir {
 use_dir,
 pos_to_neg,
 neg_to_pos
};

constexpr int LAST_N_FRAMES=10;
constexpr int TIME_OUT_MSEC=8000;
constexpr uint32_t MED_FILT_MSEC=1000;
typedef struct {
  bool enable;
  std::vector<std::pair<int, int>> roi_pts;
  std::string roi_label;
  bool inverse_roi;
  std::vector<int> operate_on_class;
  int stream_id;
}ROIInfo;

typedef struct {
  bool enable;
  std::vector<std::pair<int, int>> roi_pts;
  std::string oc_label;
  std::vector<int> operate_on_class;
  int stream_id;
  int time_threshold_in_ms;
  int object_threshold;
}OverCrowdingInfo;


typedef struct {
  bool enable;
  bool extended;
  std::string lc_label;
  std::pair <double, double> lc_dir;
  std::vector <double> lc_info;
  std::vector<std::pair<int, int>> lcdir_pts;
  std::vector<int> operate_on_class;
  int stream_id;
  enum eMode mode;
  enum eModeDir mode_dir;
}LineCrossingInfo;

typedef struct {
  bool enable;
  std::string dir_label;
  std::pair <double, double> dir_data;
  std::pair<int, int> x1y1;
  std::pair<int, int> x2y2;
  std::vector<int> operate_on_class;
  int stream_id;
  enum eMode mode;
}DirectionInfo;

typedef struct {
  std::vector <ROIInfo> roi_info;
  std::vector <OverCrowdingInfo> overcrowding_info;
  std::vector <LineCrossingInfo> linecrossing_info;
  std::vector <DirectionInfo> direction_info;
  int config_width;
  int config_height;

}StreamInfo;
/*
enum DSANALYTICS_STATUS {
  eDSANALYTICS_STATUS_NO_EVENT = 0,
  eDSANALYTICS_STATUS_INSIDE_ROI = 1,
  eDSANALYTICS_STATUS_ROI_OVERCROWDING = 2,
  eDSANALYTICS_STATUS_DIRECTION_FOLLOWED = 3,
  eDSANALYTICS_STATUS_LINE_CROSSED = 4
};*/

typedef struct {
  uint32_t left;   /**< Holds left coordinate of the box in pixels. */
  uint32_t top;    /**< Holds top coordinate of the box in pixels. */
  uint32_t width;  /**< Holds width of the box in pixels. */
  uint32_t height; /**< Holds height of the box in pixels. */
  uint64_t object_id;
  int32_t class_id;
//  std::unordered_map <std::string, enum DSANALYTICS_STATUS >obj_status;
  std::string str_obj_status;
  std::vector <std::string> roiStatus;
  std::vector <std::string> ocStatus;
  std::vector <std::string> lcStatus;
  std::string dirStatus;

}ObjInf;

typedef struct {
  bool overCrowding;
  uint32_t  overCrowdingCount;

}OverCrowdStatus;

typedef struct {
  std::vector <ObjInf> objList;
  std::unordered_map<std::string, OverCrowdStatus> ocStatus;
  std::unordered_map<int, uint32_t> objCnt;
  std::unordered_map<std::string, uint32_t> objInROIcnt;
  std::unordered_map<std::string, uint64_t> objLCCumCnt;
  std::unordered_map<std::string, uint64_t> objLCCurrCnt;
  uint64_t frmPts {0};
  int32_t srcId;

}NvDsAnalyticProcessParams;


class NvDsAnalyticCtx
{
public:
  static std::unique_ptr<NvDsAnalyticCtx> create(StreamInfo &stream_info,
    int32_t src_id, int32_t width=1920,
    int32_t height=1080, uint32_t filtTime=MED_FILT_MSEC, uint32_t timeOut=300, uint32_t hist=50);
  void destroy();
  virtual void processSource(NvDsAnalyticProcessParams &process_params) = 0;
  NvDsAnalyticCtx(){

  };
 virtual ~NvDsAnalyticCtx() {};
private:
};

using NvDsAnalyticCtxUptr = std::unique_ptr<NvDsAnalyticCtx>;
#ifdef __cplusplus
}
#endif

#endif
