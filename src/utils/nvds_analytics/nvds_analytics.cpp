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
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <map>
#include "nvds_analytics.h"

using namespace std;
static bool
NvDsAnalytics_CheckObjInROI (vector<pair<int, int>> &roi_pts,
    int cgx, int cgy);
bool check_if_intersection_on_segment (double xp, double yp,
                                          double xc, double yc,
                                          vector<double> &lc,
                                          vector<pair<int, int>> &pts);

template  <class T>
static bool
CheckDirection(pair <T, T> &dir_data, int px, int py, int cx, int cy, float dir_sim);

template  <class T>
static bool
CheckLineCrossing(pair <T, T> &lc_dir,
        vector <T> &lc_info, int px, int py, int cx, int cy, float dir_sim);
static inline bool
CheckValidClass (std::vector<int32_t> const  &vec_class, int32_t const class_id);

/* Information for each object */
class NvDsAnalyticInfo {

  public:
    int firstCntrX {0};
    int firstCntrY {0};
    uint64_t intgrlCntrX {0};
    uint64_t intgrlCntrY {0};
    uint64_t SqIntgrlCntrX {0};
    uint64_t SqIntgrlCntrY {0};
    uint64_t intgrlW {0};
    uint64_t intgrlH {0};
    uint64_t currPts {0};       /* Current Time stamp  */
    uint64_t firstPts{0};      /* Time stamp of first frame */
    uint64_t lastFrmSeen{0};   /* Frame number when it was last seen */
    uint64_t strtTimeRoiDet{0};/* Start time of ROI determination */
    uint64_t stpTimeRoiDet{0}; /* Stop time of ROI determination */
    uint64_t frmCnt {0};                   /* Frame count, since object detected */
    uint64_t trackingId {0xFFFFFFFFFFFFFFFF};              /* Object Tracking id  */
    bool prvROIDetStatus {false};
    bool objPresence {true};          /* Prescence of object in current frame */

    /* Index for circular buffers */
    int frmIdx {0};
    /* circular buffer to store m_hist
     * coordinates
     */
    int clsId {-1};
    unordered_map <string, bool > lcStatus;

    /* Previous location for direction determination
     * prev_x, prev_y
     */
    vector <int> prvX; //m_hist
    vector <int> prvY;
    vector <int> prvW;
    vector <int> prvH;

    float stdX {0.0}; /* Standard deviation in X */
    float stdY {0.0}; /* Standard Deviation in Y */
    float mnX  {0.0};  /* Mean X */
    float mnY  {0.0};  /* Mean Y */
    NvDsAnalyticInfo(int32_t hist=LAST_N_FRAMES);
    ~NvDsAnalyticInfo();
};


NvDsAnalyticInfo::NvDsAnalyticInfo (int32_t hist){
  /*objPresence = true;  firstCntrX  = 0;  firstCntrY  = 0;  intgrlCntrX  = 0;
  intgrlCntrY = 0;     intgrlW     = 0;  intgrlH     = 0;  SqIntgrlCntrX = 0;
  stdX        = 0.0;   stdY        = 0.0;mnX         = 0.0;        mnY  = 0.0;
  trackingId = -1;     frmIdx       = 0;  frmCnt      = 0; prvROIDetStatus = false;
  strtTimeRoiDet = 0;  stpTimeRoiDet = 0;
*/
  prvX.resize (hist);
  prvY.resize (hist);
  prvW.resize (hist);
  prvH.resize (hist);
  //FIXME: Set unreasonable number
}

NvDsAnalyticInfo::~NvDsAnalyticInfo (){
}

using Histogram = unordered_map<uint32_t, uint32_t>;

typedef struct ObjTypeHist{
    //sliding window PDF of count
    Histogram cntPdf;
    //History of count using pts
    unordered_map <uint64_t, uint32_t> pastCnt;
    //Last pts
    uint64_t lastUpdate;
} ObjTypeHist;

/* Class for each instance  */
class NvDsAnalyticCtxImpl: public NvDsAnalyticCtx {

    int32_t m_srcId; /* unique identifier of source */
    int32_t m_timeOut {TIME_OUT_MSEC};
    uint32_t m_width {1920};
    uint32_t m_height {1080};
    uint64_t m_frmNm {0}; /* current frame number */
    uint32_t m_hist {LAST_N_FRAMES};
    StreamInfo m_stream_info;
    uint32_t m_filtTime {MED_FILT_MSEC}; /* current frame number */
    unordered_map <uint64_t, NvDsAnalyticInfo> m_nvanalyticInf;
    unordered_map<string, uint64_t> m_objLCcnt;
    //class id based keys
    unordered_map <uint32_t, ObjTypeHist> m_classPdf;
    void clnUpObj();
    void incrFrmNum();
  public:
    /* All objects that are analyzed in the frame */
    NvDsAnalyticCtxImpl (StreamInfo &stream_info, int32_t src_id,
        int32_t width=1920, int32_t height=1080, uint32_t timeOut=TIME_OUT_MSEC, uint32_t hist=LAST_N_FRAMES,
        uint32_t filtTime=MED_FILT_MSEC);
    ~NvDsAnalyticCtxImpl ();

    void processSource (NvDsAnalyticProcessParams &process_params);
    uint32_t getSmoothCnt (uint32_t classId, uint32_t objCnt, uint64_t pts);
    void setSrcId(int srcId);
    void setTimeOut(unsigned int timeOut);
    unsigned int  getTimeOut();
    int  getSrcId();
    int  getFrmNum();


};

NvDsAnalyticCtxUptr NvDsAnalyticCtx::create(StreamInfo &stream_info, int32_t src_id, int32_t width,
    int32_t height, uint32_t filtTime, uint32_t timeOut, uint32_t hist){
    return (NvDsAnalyticCtxUptr) new NvDsAnalyticCtxImpl (stream_info, src_id,
            width, height, timeOut, hist, filtTime);
}

void NvDsAnalyticCtx::destroy(){
  delete (NvDsAnalyticCtxImpl*) this;
}

void NvDsAnalyticCtx::processSource(NvDsAnalyticProcessParams &process_params){
  ((NvDsAnalyticCtxImpl*)this)->processSource(process_params);
};

template <class T>
   T check_pt_on_line (vector<T> &labc, T xp, T yp){
     //ax+by+c
     return (labc[0]*xp + labc[1]*yp + labc[2]);
 }
template <class T>
  pair<double, double> get_dir_vec (T x1, T y1, T x2, T y2) {
    double magxy = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    magxy = sqrt(magxy);

    // Handle division by zero case
    if (magxy < 1e-9) {
      return make_pair(0.0, 0.0);
    }

    double xval = ((double)(x2 - x1)) / magxy;
    double yval = ((double)(y2 - y1)) / magxy;
    return make_pair(xval, yval);
  };

template <class T>
   vector<double> get_line_abc (T x1, T y1, T x2, T y2) {
       //ax+by+c = 0
      vector <double> vabc;
      double a = (y1 - y2);
      double b = (x2 - x1);
      double c = (x1*y2 - x2*y1);
      double mag = sqrt(a*a + b*b);
      vabc.push_back(a/mag);
      vabc.push_back(b/mag);
      vabc.push_back(c/mag);
      return vabc;
   };

template <class T>
   bool get_intersection_point  (vector<T> &labc1,
                                 vector<T> &labc2, double &xi, double &yi) {
     //ax+by+c = 0
     double mag1 = sqrt(labc1[0] * labc1[0] + labc1[1] * labc1[1]);
     double mag2 = sqrt(labc2[0] * labc2[0] + labc2[1] * labc2[1]);
     for_each(begin(labc1), end(labc1), [&](double &v1) { v1 /= mag1; });
     for_each(begin(labc2), end(labc2), [&](double &v2) { v2 /= mag2; });
     double det = (labc1[0] * labc2[1] - labc1[1] * labc2[0]);

     //Improper Direction for given line
     if (abs(det) < 1e-5)
       return false;

     //det = (a1b2 – a2b1)
     //(a1b2 – a2b1) x = c1b2 – c2b1
     //xc = (c2b1 - c1b2)/determinant
     //yc = (a2c1 - a1c2)/determinant
     //  xi = (cl*bd - cd*bl)/det;
     //  yi = (al*cd - ad*cl)/det;
     xi = (-labc1[2] * labc2[1] + labc1[1] * labc2[2]) / det;
     yi = (-labc1[0] * labc2[2] + labc1[2] * labc2[0]) / det;
     return true;
   };

template <class T>
   T get_proj_data (pair<T, T> &dir1, pair<T, T> &dir2) {
     return dir1.first * dir2.first + dir1.second * dir2.second;
   };

   bool check_if_intersection_on_segment (double xp, double yp,
                                          double xc, double yc,
                                          vector<double> &lc,
                                          vector<pair<int, int>> &pts)
   {
     vector<double> obj_dir = get_line_abc(xp, yp, xc, yc);
     double xi, yi;
     if (false == get_intersection_point(obj_dir, lc, xi, yi))
       return false;

     // check xi,yi is between  and b
     double xd, yd;
     pair<double, double> vec1;
     pair<double, double> vec2;
     xd = xi - pts[2].first;
     yd = yi - pts[2].second;
     vec1 = make_pair(xd, yd);
     xd = pts[3].first - pts[2].first;
     yd = pts[3].second - pts[2].second;
     vec2 = make_pair(xd, yd);
     // A----------C-----------B
     if ((get_proj_data(vec1, vec2)) > 0 &&
         (get_proj_data(vec2, vec2) > get_proj_data(vec1, vec1)))
       return true;

      return false;
   };

   NvDsAnalyticCtxImpl::NvDsAnalyticCtxImpl(StreamInfo &stream_info, int32_t src_id,
                                            int32_t width, int32_t height, uint32_t timeOut, uint32_t hist, uint32_t filtTime) :
                                            m_srcId(src_id), m_frmNm(0), m_hist(hist), m_stream_info(stream_info), m_filtTime(filtTime){

     //if (m_stream_info.config_width != width || m_stream_info.config_height != height)
     {
       float scalefx = (float)width / (float)m_stream_info.config_width;
       float scalefy = (float)height / (float)m_stream_info.config_height;

       for (auto &roi : m_stream_info.roi_info)
       {
         for (pair<int, int> &roi_pts : roi.roi_pts)
         {
           roi_pts.first = roi_pts.first * scalefx;
           roi_pts.second = roi_pts.second * scalefy;
         }
       }
       for (auto &roi : m_stream_info.overcrowding_info)
       {
         for (pair<int, int> &roi_pts : roi.roi_pts)
         {
           roi_pts.first = roi_pts.first * scalefx;
           roi_pts.second = roi_pts.second * scalefy;
         }
       }
       for (auto &dir_info : m_stream_info.direction_info)
       {
         int32_t x1 = dir_info.x1y1.first * scalefx;
         int32_t y1 = dir_info.x1y1.second * scalefy;
         int32_t x2 = dir_info.x2y2.first * scalefx;
         int32_t y2 = dir_info.x2y2.second * scalefy;

         dir_info.x1y1 = make_pair(x1, y1);
         dir_info.x2y2 = make_pair(x2, y2);
         //TODO add warning in config file reading
         if (((x1 - x2) == 0) && ((y1 - y2) == 0))
           dir_info.dir_data = make_pair(0.0f, 0.0f);
         else
           dir_info.dir_data = get_dir_vec(x1, y1, x2, y2);
       }
       for (auto &lc_info : m_stream_info.linecrossing_info)
       {
         double xd1, yd1, xd2, yd2, xl1, yl1, xl2, yl2;
         double xi, yi;
         vector<double> dir_abc;
         vector<double> lc_abc;

         for (auto &lc_pt : lc_info.lcdir_pts)
         {
           lc_pt.first = lc_pt.first * scalefx;
           lc_pt.second = lc_pt.second * scalefy;
         }

         xd1 = lc_info.lcdir_pts[0].first;
         yd1 = lc_info.lcdir_pts[0].second;
         xd2 = lc_info.lcdir_pts[1].first;
         yd2 = lc_info.lcdir_pts[1].second;

         //FIXME: Warning during config file reading
         if (((xd1 - xd2) == 0) && ((yd1 - yd2) == 0))
           lc_info.lc_dir = make_pair(0.0, 0.0);
         else
           lc_info.lc_dir = get_dir_vec(xd1, yd1, xd2, yd2);

         // LC vector
         xl1 = lc_info.lcdir_pts[2].first;
         yl1 = lc_info.lcdir_pts[2].second;
         xl2 = lc_info.lcdir_pts[3].first;
         yl2 = lc_info.lcdir_pts[3].second;

         /* (y1-y2)x + (x2-x1)y + (x1y2-x2y1) = 0 */
         lc_info.lc_info = get_line_abc(xl1, yl1, xl2, yl2);
         lc_info.lcdir_pts.push_back(make_pair(xl1, yl1));
         lc_info.lcdir_pts.push_back(make_pair(xl2, yl2));

         dir_abc = get_line_abc(xd1, yd1, xd2, yd2);
         lc_abc = get_line_abc(xl1, yl1, xl2, yl2);

         if (get_intersection_point(dir_abc, lc_abc, xi, yi))
         {
           pair<double, double> dir1 = get_dir_vec(xd1, yd1, xi, yi);
           pair<double, double> dir2 = get_dir_vec(xd2, yd2, xi, yi);
           double mag1 = get_proj_data(dir1, lc_info.lc_dir);
           double mag2 = get_proj_data(dir2, lc_info.lc_dir);
           double val1 = check_pt_on_line(lc_abc, xd1, yd1);
           double val2 = check_pt_on_line(lc_abc, xd2, yd2);//lc_abc[0] * xd2 + lc_abc[1] * yd2 + lc_abc[2];
           lc_info.mode_dir = eModeDir::use_dir;

           if ((mag1 > 0.99 && val1 > 1e-5) || (mag1 < -0.99 && val1 < 1e-5))
             lc_info.mode_dir = eModeDir::pos_to_neg;
           else if ((mag1 < -0.99 && val1 > 1e-5) || (mag1 > 0.99 && val1 < 1e-5))
             lc_info.mode_dir = eModeDir::neg_to_pos;
           else if ((mag2 > 0.99 && val2 > 0) || (mag2 < -0.99 && val2 < 0))
             lc_info.mode_dir = eModeDir::pos_to_neg;
           else if ((mag2 < -0.99 && val2 > 0) || (mag2 > 0.99 && val2 < 0))
             lc_info.mode_dir = eModeDir::neg_to_pos;
         }
         else
         {
           printf("Improper direction configured");
         }
       }
       m_stream_info.config_width = width;
       m_stream_info.config_height = height;

       stream_info = m_stream_info;
     }

 }


NvDsAnalyticCtxImpl::~NvDsAnalyticCtxImpl (){
  m_nvanalyticInf.clear();
}


void NvDsAnalyticCtxImpl::setSrcId (int srcId){
  this->m_srcId = srcId;
}

int NvDsAnalyticCtxImpl::getSrcId (){
  return m_srcId;
}

void NvDsAnalyticCtxImpl::incrFrmNum (){
  this->m_frmNm++;
}

int NvDsAnalyticCtxImpl::getFrmNum (){
  return m_frmNm;
}

void NvDsAnalyticCtxImpl::setTimeOut (unsigned int timeOut){
  m_timeOut = timeOut;
}

unsigned int NvDsAnalyticCtxImpl::getTimeOut (){
  return m_timeOut;
}


uint32_t NvDsAnalyticCtxImpl::getSmoothCnt (uint32_t classId, uint32_t objCnt, uint64_t pts){
    ObjTypeHist &cls_hist = m_classPdf[classId];
    std::vector<uint64_t> delete_past;
    Histogram::iterator got_cnt =  cls_hist.cntPdf.find (objCnt);

    //update PDF with new info
    if (got_cnt != cls_hist.cntPdf.end()){
        got_cnt->second += 1;
    }
    else{
        cls_hist.cntPdf[objCnt] = 1;
    }

    for (auto& past_frm : cls_hist.pastCnt){
        //Check for expired objects
        if ((pts - past_frm.first)/1e6 > m_filtTime){
            delete_past.push_back(past_frm.first);
            cls_hist.cntPdf[past_frm.second] -= 1;
        }
    }

    for_each(begin(delete_past),end(delete_past),
                                    [&](uint64_t n){cls_hist.pastCnt.erase(n);});
    got_cnt = std::max_element(begin(cls_hist.cntPdf), end(cls_hist.cntPdf),[]
            (std::pair<uint32_t, uint32_t> const & elem1, std::pair<uint32_t, uint32_t> const & elem2)
            {
            return (elem1.second <  elem2.second);});
    /*
    for_each(begin(cls_hist.cntPdf),end(cls_hist.cntPdf),[classId]
             (std::pair<uint32_t,uint32_t>const & it){
            std::cout << "class "<< classId << " " << it.first <<" is" << it.second<<std::endl;
     });
*/
    cls_hist.pastCnt[pts] = objCnt;
    cls_hist.lastUpdate = pts;

    return got_cnt->first;
}

/* Initialize analytic context */
void NvDsAnalyticCtxImpl::processSource (NvDsAnalyticProcessParams &process_params)
{
  vector<ObjInf> &pstObjInf = process_params.objList;
  int curr_x = 0, curr_y = 0;
  int curr_ht = 0;
  int curr_wdth = 0;
  vector <uint64_t> remove_obj;

  string objStatus;
  uint64_t frm_pts =  process_params.frmPts;
  int  frm_win = 0;

  incrFrmNum();

  for (auto& n : m_nvanalyticInf) {
    NvDsAnalyticInfo & obj_info =  n.second;
    /* Get the time since object is not visible in ms*/
    long long last_seen = (frm_pts - obj_info.currPts);
    /* total object life since first detected */
    long long time_diff = (frm_pts - obj_info.firstPts);

    int prev_idx = obj_info.frmIdx - 1;

    /* Invalidate the object by default */
    obj_info.objPresence = false;

    /* If end of circular buffer use the last frame which is at the end of
     * the buffer
     */
    if (prev_idx < 0) {
      prev_idx = m_hist - 1;
    }

    time_diff /= 1e6;
    /* Get the time since object is not visible in ms*/
    last_seen /= 1e6;

    /* If not seen for 4000 ms remove the object*/
    if ((last_seen > m_timeOut)) remove_obj.push_back(n.first);
  }

  /* Clear lost objects */
  for_each(begin(remove_obj),end(remove_obj),[&](uint64_t n){m_nvanalyticInf.erase(n);});
  /*for (auto& n : remove_obj)
    m_nvanalyticInf.erase (n);
  */
  /* clear as work is done */
  remove_obj.clear();

  //cout << "***************Total Objects IN " <<  getFrmNum() << " " << pstObjInf.size() << endl;
  /* Find correspondence for all prev objects */
  for (auto &each_obj : pstObjInf) {
    /* update the global counting */
    process_params.objCnt[each_obj.class_id]++;

    /* Search the object location*/
    unordered_map<uint64_t, NvDsAnalyticInfo>::const_iterator got_obj =
      m_nvanalyticInf.find (each_obj.object_id);
    int32_t prev_x = 0, prev_y = 0, prev_h = 0, prev_w = 0;

    /* Get current statistics */
    curr_x = each_obj.left+each_obj.width/2;
    curr_y = each_obj.top+each_obj.height/2;
    curr_ht = each_obj.height;
    curr_wdth = each_obj.width;

    /* If its an old object update the information */
    if (got_obj != m_nvanalyticInf.end()){
      NvDsAnalyticInfo & obj_info = (NvDsAnalyticInfo &)got_obj->second;

      /* update its presence and timestamps  */
      obj_info.objPresence = true;
      obj_info.lastFrmSeen = frm_pts;
      obj_info.currPts = frm_pts;
    }
    /* if new object insert the same in the map */
    else {
      NvDsAnalyticInfo new_obj_info(m_hist);
      new_obj_info.trackingId  = each_obj.object_id;
      new_obj_info.firstPts    = frm_pts;
      new_obj_info.currPts     = frm_pts;
      new_obj_info.lastFrmSeen = frm_pts;
      new_obj_info.objPresence = true;
      m_nvanalyticInf[new_obj_info.trackingId] = std::move(new_obj_info);
    }

    NvDsAnalyticInfo & obj_info = m_nvanalyticInf[each_obj.object_id];

    /* default object status null */
    objStatus.clear();
    /* Check if enough frames to replace the oldest
     * cgs with the new one using a circular buffer
     */
    int32_t frm_idx = obj_info.frmIdx;
    if (obj_info.frmCnt >= m_hist) {
      /* Remove the oldest sample from the short term
       * buffer for intgrl and sq intgrl both x,y
       */
      obj_info.intgrlCntrX   -=  obj_info.prvX[frm_idx];
      obj_info.intgrlCntrY   -=  obj_info.prvY[frm_idx];
      obj_info.SqIntgrlCntrX -= (obj_info.prvX[frm_idx]*obj_info.prvX[frm_idx]);
      obj_info.SqIntgrlCntrY -= (obj_info.prvY[frm_idx]*obj_info.prvY[frm_idx]);
      obj_info.intgrlW       -=  obj_info.prvW[frm_idx];
      obj_info.intgrlH       -=  obj_info.prvH[frm_idx];
    }

    /* Add the current sample in short term circular buffer */
    obj_info.prvX[frm_idx] = curr_x;
    obj_info.prvY[frm_idx] = curr_y;
    obj_info.prvW[frm_idx] = each_obj.width;
    obj_info.prvH[frm_idx] = each_obj.height;

    /* move to next buffer index */
    frm_idx++;
    /* Make it ciruclar to overwrite on the oldest sample*/
    obj_info.frmIdx = frm_idx % m_hist;

    /* update the intgrl and sqr intgrl with new sample */
    obj_info.intgrlCntrX   += curr_x;
    obj_info.intgrlCntrY   += curr_y;
    obj_info.intgrlW       += each_obj.width;
    obj_info.intgrlH       += each_obj.height;
    obj_info.SqIntgrlCntrX += (curr_x * curr_x);
    obj_info.SqIntgrlCntrY += (curr_y * curr_y);

    /* Size of window for analysis*/
    frm_win = m_hist;

    /* If not enough frames update the window size accordingly */
    if (obj_info.frmCnt < m_hist) frm_win = obj_info.frmCnt + 1;

    float mean_h = ((float)obj_info.intgrlH) / (frm_win);
    /* Get mean x and mean y and std sqr*/
    obj_info.mnX  = (float)obj_info.intgrlCntrX / frm_win;
    obj_info.mnY  = (float)obj_info.intgrlCntrY / frm_win;
    obj_info.stdX = (float)obj_info.SqIntgrlCntrX / frm_win;
    obj_info.stdY = (float)obj_info.SqIntgrlCntrY / frm_win;

    /* Subtract to get variance */
    obj_info.stdX -= (obj_info.mnX * obj_info.mnX);
    obj_info.stdY -= (obj_info.mnY * obj_info.mnY);

    obj_info.frmCnt++;
    //  printf ("Frm idx %d Frm Cnt %d\n", obj_info.frm_idx, obj_info.frame_count);
    /* ROI Filtering */
    for (auto &roi: m_stream_info.roi_info){
      bool roi_status = false;

      if (roi.enable == false ||
         //(roi.operate_on_class !=-1 && roi.operate_on_class != each_obj.class_id)
         !CheckValidClass(roi.operate_on_class, each_obj.class_id)
         )
        continue;

      roi_status = NvDsAnalytics_CheckObjInROI(roi.roi_pts, curr_x,
                                                    curr_y + (int)mean_h/2);

      if ((roi_status==true && roi.inverse_roi == false) ||
          (roi_status==false && roi.inverse_roi == true)){
        //each_obj.obj_status[roi.roi_label] = eDSANALYTICS_STATUS_INSIDE_ROI;
        each_obj.str_obj_status += " ROI:";
        each_obj.str_obj_status += roi.roi_label;
        process_params.objInROIcnt[roi.roi_label] += 1;
        each_obj.roiStatus.push_back(roi.roi_label);
        //cout << each_obj.str_obj_status << " " <<  getFrmNum() <<" INSIDE " << each_obj.object_id << endl;
      }
    }
    for (auto &roi: m_stream_info.overcrowding_info){
      bool roi_status = false;

      if (roi.enable == false ||
         !CheckValidClass(roi.operate_on_class, each_obj.class_id) )
        // (roi.operate_on_class !=-1 && roi.operate_on_class != each_obj.class_id))
        continue;

      roi_status = NvDsAnalytics_CheckObjInROI(roi.roi_pts, curr_x,
                                                    curr_y + (int)mean_h/2);

      //FIXME: Add time threshold check
      if (roi_status==true){
        //each_obj.obj_status[roi.oc_label] = eDSANALYTICS_STATUS_INSIDE_ROI;
        process_params.ocStatus[roi.oc_label].overCrowdingCount += 1;

        each_obj.ocStatus.push_back(roi.oc_label);

        if (process_params.ocStatus[roi.oc_label].overCrowdingCount >=
                                        (uint32_t)roi.object_threshold){
          process_params.ocStatus[roi.oc_label].overCrowding = true;
        }
      }
    }

    if (obj_info.frmCnt > 3) {
        /* Choose LAST_N_FRAMEth previous point  */

        if (obj_info.frmCnt < m_hist){
            prev_x=
                (obj_info.prvX[0] +
                 obj_info.prvX[1])/2;
            prev_y =
                (obj_info.prvY[0] +
                 obj_info.prvY[1])/2;
            /* Get displacement in x and y */
            prev_h = (obj_info.prvH[0] +
                    obj_info.prvH[1] )/2;
            prev_w = (obj_info.prvW[0] +
                      obj_info.prvW[1])/2;
        }
        else{
            prev_x = (obj_info.prvX[(obj_info.frmIdx+0)% m_hist]
                    + obj_info.prvX[(obj_info.frmIdx+1)% m_hist]
                    + obj_info.prvX[(obj_info.frmIdx+2)% m_hist]
                    + obj_info.prvX[(obj_info.frmIdx+3)% m_hist]) / 4;
            prev_y = (obj_info.prvY[(obj_info.frmIdx+0)% m_hist]
                    + obj_info.prvY[(obj_info.frmIdx+1)% m_hist]
                    + obj_info.prvY[(obj_info.frmIdx+2)% m_hist]
                    + obj_info.prvY[(obj_info.frmIdx+3)% m_hist]) / 4;
            /* Get displacement in x and y */
            prev_h = (obj_info.prvH[(obj_info.frmIdx+0)% m_hist]
                    + obj_info.prvH[(obj_info.frmIdx+1)% m_hist]
                    + obj_info.prvH[(obj_info.frmIdx+2)% m_hist]
                    + obj_info.prvH[(obj_info.frmIdx+3)% m_hist]) / 4;
            prev_w = (obj_info.prvW[(obj_info.frmIdx+0)% m_hist]
                    + obj_info.prvW[(obj_info.frmIdx+1)% m_hist]
                    + obj_info.prvW[(obj_info.frmIdx+2)% m_hist]
                    + obj_info.prvW[(obj_info.frmIdx+3)% m_hist]) / 4;
        }
    }
    each_obj.dirStatus = "";
    for (auto &dir: m_stream_info.direction_info){
        if (dir.enable == false ||
            !CheckValidClass(dir.operate_on_class, each_obj.class_id))
//           (dir.operate_on_class !=-1 &&
//           dir.operate_on_class != each_obj.class_id))
            continue;
        //FIXME: Get it from meta header
        if (0xFFFFFFFFFFFFFFFF == each_obj.object_id){
            cout << "NVDSANALYTICS:Cant get direction information without"
                " enabling tracker, disabling direction" << endl;
            dir.enable = false;
        }
        if ((obj_info.frmCnt > m_hist) &&
            (obj_info.stdX  <= 400) && ((obj_info.stdY/mean_h) <= 0.01)) {
            each_obj.str_obj_status += " STOPPED ";
            break;
        }

        /* Get displacement in x and y */
        float diff_x = curr_x - prev_x;
        float diff_y = (curr_y + curr_ht/2) - (prev_y + prev_h/2);
        int max_disp = diff_x * diff_x + diff_y * diff_y;
        bool dir_status = false;

        /* Get L2 Displacement  */
        /* check if enough displacement FIXME: heuristics!*/
        if (max_disp > 3000 && obj_info.frmCnt > 10) {
           dir_status =
               CheckDirection(dir.dir_data, prev_x, prev_y+(int32_t)mean_h/2,
                           curr_x, curr_y +(int32_t)mean_h/2, 0.5f);
           if (true == dir_status){
               each_obj.str_obj_status += " DIR:" + dir.dir_label + " ";
               each_obj.dirStatus += " DIR:" + dir.dir_label + " ";
               //cout << each_obj.object_id <<" is moving " << dir.dir_label << endl;
           }

#if 0
                /* Get the angle using inverse tan */
            double angle = atan2(diff_y, diff_x);
            angle = (angle * 180 / M_PI);
            /* depending upon angle update the object direction*/
            // if (angle <= 45 && angle > -45 && diff_x > 0)
            if (angle < 22.5 && angle > -22.5 && diff_x > 0){
                objStatus.append("East");
            }
            else if (angle >= 22.5 && angle <= 67.5 && diff_x > 0 && diff_y > 0){
                objStatus.append ("South East");
            }
            else if (angle > 67.5 && angle < 112.5 && diff_y > 0){
                //                            else if (angle >=45 && angle < 135 && diff_y > 0)
                objStatus.append ("South East");
            }
            else if (angle >=112.5 && angle <= 157.5 && diff_x < 0 && diff_y > 0){
                objStatus.append ("South West");
            }
            else if ((angle > 157.5 || angle < -157.5) && diff_x < 0){
                // else if ((angle >= 135 || angle < -135) && diff_x < 0)
                objStatus.append ("West");
            }
            else if ((angle >= -157.5 &&  angle <= -112.5) && diff_x < 0 && diff_y < 0){
                objStatus.append ("North West");
            }
            else if (angle > -112.5 && angle < -67.5 && diff_y < 0){
                //                            else if (angle >= -135 && angle <= -45 && diff_y < 0)
                objStatus.append ("North");
            }
            else if (angle >= -67.5 && angle <= -22.5 && diff_x > 0 && diff_y < 0){
                objStatus.append ("North East");
            }
#endif
        }
    }

    for (auto &lc: m_stream_info.linecrossing_info){
        if (lc.enable == false || obj_info.lcStatus[lc.lc_label] == true  ||
            !CheckValidClass(lc.operate_on_class, each_obj.class_id)
          // (lc.operate_on_class !=-1 && lc.operate_on_class != each_obj.class_id)
           )
            continue;
        //FIXME: Get it from meta header
        if (0xFFFFFFFFFFFFFFFF == each_obj.object_id){
            std::cout << "NVDSANALYTICS:Cant get direction information without"
                " enabling tracker, disabling direction" << std::endl;
            lc.enable = false;
        }

        /* Get displacement in x and y */
        float diff_x = curr_x - prev_x;
        float diff_y = (curr_y + curr_ht/2) - (prev_y + prev_h/2);
        int max_disp = diff_x * diff_x + diff_y * diff_y;
        bool dir_status = false;
        int disp_threshold = 3000;
        float dir_sim = 0.5;
        uint32_t frm_threshold = 10;

        if (lc.mode == eMode::strict){
          disp_threshold = 6000;
          dir_sim = 0.8;
          frm_threshold = 20;
        }
        else if (lc.mode == eMode::loose){
          disp_threshold = 1000;
          dir_sim = 0.1;
          frm_threshold = 3;
        }
        /* Get L2 Displacement  */
        /* check if enough displacement FIXME: heuristics!*/
        if (max_disp > disp_threshold && obj_info.frmCnt > frm_threshold)
         {
           if (eModeDir::use_dir == lc.mode_dir ||
               eMode::loose != lc.mode){
            dir_status = CheckLineCrossing(lc.lc_dir,
                    lc.lc_info, prev_x, prev_y + (int32_t)mean_h/2,
                       curr_x, curr_y + (int32_t)curr_ht/2, dir_sim);
           }
           else {
            double magc = check_pt_on_line(lc.lc_info, static_cast<double>(curr_x),
               static_cast<double>(curr_y+curr_ht/2));
            // lc.lc_info[0]*curr_x + lc.lc_info[1]*(curr_y+curr_ht/2) + lc.lc_info[2];
            double magp = check_pt_on_line(lc.lc_info, static_cast<double>(prev_x),
               static_cast<double>(prev_y+mean_h/2));
             //lc.lc_info[0]*prev_x + lc.lc_info[1]*(prev_y+mean_h/2) + lc.lc_info[2];
            if ((magc >= 0 && eModeDir::neg_to_pos == lc.mode_dir && magp < 0) ||
                (magc <= 0 && eModeDir::pos_to_neg == lc.mode_dir && magp > 0))
                    dir_status = true;
           }
           if (lc.extended == false && dir_status == true){
             dir_status = false;
             //double xi, yi;
             double xp = prev_x, yp = prev_y + mean_h/2.0,
                    xc = curr_x, yc = curr_y + curr_ht/2.0;

             if (check_if_intersection_on_segment(xp, yp, xc, yc, lc.lc_info, lc.lcdir_pts) ||
                 check_if_intersection_on_segment(xp+prev_w/2, yp, xc+curr_wdth/2, yc, lc.lc_info, lc.lcdir_pts) ||
                 check_if_intersection_on_segment(xp-prev_w/2, yp, xc-curr_wdth/2, yc, lc.lc_info, lc.lcdir_pts))
             {
               dir_status = true;
             }
                                              /*
             std::vector<double>obj_dir = get_line_abc(xp, yp, xc, yc);
             if (get_intersection_point(obj_dir, lc.lc_info, xi,yi)){
               // check xi,yi is between  and b
               double xd, yd;
               double mag1;
               std::pair<double,double> vec1;
               std::pair<double,double> vec2;
               xd = xi - lc.lcdir_pts[2].first;
               yd = yi - lc.lcdir_pts[2].second;
               vec1 = std::make_pair(xd,yd);
               xd = lc.lcdir_pts[3].first - lc.lcdir_pts[2].first;
               yd = lc.lcdir_pts[3].second - lc.lcdir_pts[2].second;
               vec2 = std::make_pair(xd,yd);
               if ((get_cos_data(vec1, vec2)) > 0 &&
                   (get_cos_data(vec2,vec2) > get_cos_data(vec1,vec1))){
                 dir_status = true;
               }
             }*/
           }
           if (true == dir_status){
                each_obj.str_obj_status += " LC:" + lc.lc_label + " ";
                obj_info.lcStatus[lc.lc_label] = true;
                m_objLCcnt[lc.lc_label] += 1;
                process_params.objLCCurrCnt[lc.lc_label]+= 1;
                each_obj.lcStatus.push_back(lc.lc_label);
            }
        }
    }
  }
  process_params.objLCCumCnt = m_objLCcnt;

  if (m_filtTime){
      for (auto &each_obj : process_params.objCnt) {
          each_obj.second = getSmoothCnt(each_obj.first, each_obj.second,frm_pts);
      }

      for (auto &each_pdf : m_classPdf){
          if (each_pdf.second.lastUpdate != frm_pts){
              process_params.objCnt[each_pdf.first]=
                  getSmoothCnt(each_pdf.first, 0,frm_pts);
          }
      }
  }

}

template  <class T>
static bool
CheckDirection(pair <T, T> &dir_data, int px, int py, int cx, int cy, float dir_sim)
{
    int x = cx - px, y = cy - py;
    T f_dpval = dir_data.first*x + dir_data.second*y;
    T f_normval = sqrt(x*x + y*y);

    if (x == 0 && y ==0)
        return false;

    f_dpval /= f_normval;
    if (f_dpval < dir_sim)
        return false;

    return true;
}

static bool
NvDsAnalytics_CheckObjInROI (vector<pair<int, int>> &roi_pts, int cgx, int cgy)
{
  uint32_t total_pts_in_roi = 0;
  uint32_t icnt = 0;
  int32_t prev_x = 0, prev_y = 0, frst_x = 0;
  int32_t curr_x = 0, curr_y = 0, frst_y = 0;
  bool in_roi = false;

  /* Check for bbox */
  frst_x = prev_x = roi_pts[0].first;
  frst_y = prev_y = roi_pts[0].second;

  for (icnt = 0; icnt < roi_pts.size(); icnt++)
  {
    int aval = 0, bval = 0, cval = 0, xicp  =0;
    if (icnt < roi_pts.size()-1){
      curr_x = roi_pts[icnt+1].first;
      curr_y = roi_pts[icnt+1].second;
    }
    else {
      curr_x = frst_x;
      curr_y = frst_y;
    }
    /* Check y intercept */
    if (((cgy >= prev_y) && (cgy < curr_y)) ||
        ((cgy < prev_y) && (cgy >= curr_y))) {
      /* Get line parameters */
      aval = prev_y - curr_y;
      bval = curr_x - prev_x;
      cval = curr_y*prev_x - curr_x*prev_y;

      if (aval !=0)  {
        xicp = -1 * (bval*cgy + cval)/aval;
        if ((xicp >=0) && (xicp <= cgx))
          total_pts_in_roi ++;
      }
    }
    prev_x = curr_x;
    prev_y = curr_y;
  }

  if (total_pts_in_roi%2)    in_roi = true;

  return in_roi;
}

static inline bool
CheckValidClass (std::vector<int32_t> const  &vec_class, int32_t const class_id)
{
  std::vector<int32_t>::const_iterator it;

  //Default for all classes -1, do it on all obj
  if (vec_class.size() == 0)
    return true;

 it = std::find (begin(vec_class), end(vec_class), -1);
 if (it != end(vec_class))
    return true;

 it = std::find (begin(vec_class), end(vec_class), class_id);
 if (it != end(vec_class))
    return true;

  return false;
}

template  <class T>
static bool
CheckLineCrossing(pair <T, T> &lc_dir,
        vector <T> &lc_info, int px, int py, int cx, int cy, float dir_sim)
{
    int32_t chk_curr = 0, chk_prev = 0;

    if (CheckDirection(lc_dir, px, py, cx, cy, dir_sim)){
        chk_prev = lc_info[0]*px + lc_info[1]*py + lc_info[2];
        chk_curr = lc_info[0]*cx + lc_info[1]*cy + lc_info[2];

        if ((chk_curr >0 && chk_prev < 0) ||
            (chk_curr <0 && chk_prev > 0))
            return true;
    }

    return false;
}

#if 0
int Match_Object (ObjInf *pstObjInf, AnalyInfo *panal_inf_obj) {

  int prev_idx =
      panal_inf_obj->frm_idx == 0 ?
          LAST_N_FRAMES - 1 : (panal_inf_obj->frm_idx - 1);
  int prev_w = panal_inf_obj->prev_w[prev_idx];
  int prev_h = panal_inf_obj->prev_h[prev_idx];
  int prev_x = panal_inf_obj->prev_x[prev_idx] - prev_w / 2;
  int prev_y = panal_inf_obj->prev_y[prev_idx] - prev_h / 2;
  int curr_x = pstObjInf->strt_x;
  int curr_y = pstObjInf->strt_y;
  int curr_w = pstObjInf->width;
  int curr_h = pstObjInf->height;
  int overlap = 0;
  int overlap_width = 0;
  int overlap_height = 0;

  if (curr_x < (prev_x + prev_w - 1) && curr_x >= prev_x) {

    overlap = 1;
    overlap_width = prev_x + prev_w - curr_x;
    if (curr_y < (prev_y + prev_h - 1) && curr_y >= prev_y) {
      overlap_height = prev_y + prev_h - curr_y;

    } else if (prev_y < (curr_y + curr_h - 1) && prev_y >= curr_y) {
      overlap_height = curr_y + curr_h - prev_y;
    } else {
      overlap = 0;
    }
  } else if (prev_x < (curr_x + curr_w - 1) && prev_x >= curr_x) {

    overlap = 1;
    overlap_width = curr_x + curr_w - prev_x;
    if (curr_y < (prev_y + prev_h - 1) && curr_y >= prev_y) {
      overlap_height = prev_y + prev_h - curr_y;

    } else if (prev_y < (curr_y + curr_h - 1) && prev_y >= curr_y) {
      overlap_height = curr_y + curr_h - prev_y;
    } else {
      overlap = 0;
    }
  }

  if (overlap){
    float overlap_area = overlap_height*overlap_width;

    float overlap_ratio = overlap_area/(prev_w*prev_h + curr_w*curr_h);
    //printf ("Overlap of %f\n",overlap_ratio);
    if (overlap_ratio > 0.25){
        overlap = 1;
     }
    else{
      overlap = 0;
    }
  }
  return overlap;
}
#endif
