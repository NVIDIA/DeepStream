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

#ifndef CV_FUNCTIONS_HPP
#define CV_FUNCTIONS_HPP

#include <sys/stat.h>
#include <stdexcept>
#include <iostream>
#include <vector>

#if defined(VIZ_DEBUG) || defined(VIZ_DEBUG_SEG)
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "modules/utility/dataTypes.hpp"

namespace CV_FUNCS {

/**
 * @brief Convert a bounding box to an estimator measurement vector.
 * @param[in] bbox Bounding box to convert.
 * @param[in] useAspc Selects aspect ratio instead of width in the measurement vector.
 */
Eigen::VectorXd convertBboxToZ(const NvMotDataTypes::Rect2f &bbox, const bool useAspc=false);

/**
 * @brief Compute the convex hull of double-precision points.
 * @param[in] points Input points.
 * @param[out] hull Convex-hull vertices in traversal order.
 */
void convexHull(const std::vector<Eigen::Vector2d> &points, std::vector<Eigen::Vector2d> &hull);

/**
 * @brief Compute the convex hull of single-precision points.
 * @param[in] points Input points.
 * @param[out] hull Convex-hull vertices in traversal order.
 */
void convexHull(const std::vector<Eigen::Vector2f> &points, std::vector<Eigen::Vector2f> &hull);

/**
 * @brief Compute the axis-aligned bounding rectangle of a point set.
 * @param[in] points Input points.
 */
NvMotDataTypes::Rect2f boundingRect(const std::vector<Eigen::Vector2d> &points);

/**
 * @brief Compute the bounding rectangle of nonzero mask values.
 * @param[in] mask Row-major mask values.
 * @param[in] width Mask width.
 * @param[in] height Mask height.
 */
NvMotDataTypes::Rect2f boundingRect(float *mask, const int width, const int height);

#ifdef VIZ_DEBUG
/**
 * @brief Convert an OpenCV point to an Eigen point.
 * @param[in] pt_ocv OpenCV point to convert.
 */
inline Eigen::Vector2f convertOcvPtToEigenPt(const cv::Point2d pt_ocv)
{
	Eigen::Vector2f pt_e;
	pt_e << pt_ocv.x, pt_ocv.y;
	return pt_e;
}
#endif

/**
 * @brief Construct an axis-aligned rectangle from its origin and size.
 * @param[in] minX Left coordinate.
 * @param[in] minY Top coordinate.
 * @param[in] width Rectangle width.
 * @param[in] height Rectangle height.
 */
inline NvMotDataTypes::Rect2f constructRect2f(
	const float minX=0, const float minY=0, const float width=0, const float height=0
)
{
	Eigen::Vector2f min(minX, minY);
	Eigen::Vector2f max(minX + width, minY + height);

	return NvMotDataTypes::Rect2f(min, max);
}

/**
 * @brief Translate a bounding box by horizontal and vertical displacements.
 * @param[in] bbox Bounding box to translate.
 * @param[in] displacement_x Horizontal displacement.
 * @param[in] displacement_y Vertical displacement.
 */
inline NvMotDataTypes::Rect2f displaceBbox(
	const NvMotDataTypes::Rect2f &bbox, const float displacement_x, const float displacement_y
)
{
	/// Update the target center from the displacement in correlation response.
	Eigen::Vector2f min = bbox.min();
	Eigen::Vector2f max = bbox.max();
	min(0) += displacement_x;
	max(0) += displacement_x;
	min(1) += displacement_y;
	max(1) += displacement_y;

	return NvMotDataTypes::Rect2f(min, max);
}

/**
 * @brief Compute the center point of a bounding box.
 * @param[in] bbox Bounding box whose center is requested.
 */
inline Eigen::Vector2f getBboxCenterPt(const NvMotDataTypes::Rect2f &bbox)
{
	const Eigen::Vector2f min = bbox.min();
	const Eigen::Vector2f size = bbox.sizes();

	return Eigen::Vector2f(min(0) + size(0)/2, min(1) + size(1)/2);
}

/**
 * @brief Format an Eigen bounding box as a readable string.
 * @param[in] bbox Bounding box to format.
 */
inline const std::string getEigenRectString(const NvMotDataTypes::Rect2f &bbox)
{
	std::stringstream out;
	out << "[" << bbox.sizes()(0) << " x " << bbox.sizes()(1) << " from (" << bbox.min()(0) << ", " << bbox.min()(1) << ")]";
	return out.str();
}

/**
 * @brief Compare two bounding boxes for exact equality.
 * @param[in] bbox1 First bounding box.
 * @param[in] bbox2 Second bounding box.
 */
inline bool isBboxEqual(const NvMotDataTypes::Rect2f &bbox1, const NvMotDataTypes::Rect2f &bbox2)
{
	return bbox1.min() == bbox2.min() && bbox1.max() == bbox2.max();
}

/**
 * @brief Compute a nonnegative modulo result for positive divisors.
 * @param[in] a Dividend.
 * @param[in] b Divisor.
 */
inline int modul(int a, int b)
{
	return ((a % b) + b) % b;
}

/**
 * @brief Adjust the width of the bbox according to the aspect ratio specified.
 * @param[in] bbox Bounding box whose width is adjusted to the given aspect ratio.
 * @param[in] aspectRatio Width divided by height used to resize the box.
 * @return Adjusted bounding box.
 */
inline NvMotDataTypes::Rect2f getBoxWithAspectRatio(const NvMotDataTypes::Rect2f &bbox, const float aspectRatio)
{
    Eigen::Vector2f min_new = bbox.min();
    Eigen::Vector2f max_new = bbox.max();

    const float width_prev = max_new(0) - min_new(0);
    const float height = max_new(1) - min_new(1);
    const float width_new = height * aspectRatio;
    min_new(0) -= (width_new - width_prev) / 2.0;
    max_new(0) += (width_new - width_prev) / 2.0;

    return NvMotDataTypes::Rect2f(min_new, max_new);
}

#if defined(VIZ_DEBUG) || defined(VIZ_DEBUG_SEG)
/**
 * @brief Convert an OpenCV rectangle to the tracker rectangle type.
 * @param[in] bbox OpenCV rectangle to convert.
 */
inline NvMotDataTypes::Rect2f convertCvRect2f2MotRect2f(const cv::Rect2f &bbox)
{
	Eigen::Vector2f min; min << bbox.x, bbox.y;
	Eigen::Vector2f max; max << bbox.x + bbox.width, bbox.y + bbox.height;

	return NvMotDataTypes::Rect2f(min, max);
}

/**
 * @brief Convert a tracker rectangle to the OpenCV rectangle type.
 * @param[in] bbox Tracker rectangle to convert.
 */
inline cv::Rect2f convertMotRect2f2CvRect2f(const NvMotDataTypes::Rect2f &bbox)
{
	Eigen::Vector2f min = bbox.min();

	return cv::Rect2f(min(0), min(1), bbox.sizes()(0), bbox.sizes()(1));
}
#endif

/**
 * @brief Convert an estimator state vector to a bounding box.
 * @param[in] x Estimator state vector.
 * @param[in] useAspc Selects aspect ratio instead of width in the state vector.
 */
inline NvMotDataTypes::Rect2f convertXToBbox(const Eigen::VectorXd &x, const bool useAspc=false)
{
	assert(x.rows() >= 4 || x.cols() >= 4);

	double h = x(3);

	/// Derive the width from the aspect ratio when the state uses one.
	double w = useAspc ? x(2) * h : x(2);

	return constructRect2f(x(0)-w/2, x(1)-h/2, w, h);
}

/**
 * @brief Generate a separable two-dimensional Hann window.
 * @param[in] width Window width.
 * @param[in] height Window height.
 * @param[in] offsetFactor_y Vertical offset as a fraction of window height.
 */
inline NvMotDataTypes::RowMajorMatrixXf getHanningWindow(const int width, const int height, float offsetFactor_y = 0.f)
{
	Eigen::RowVectorXf hann_rows(height);
	Eigen::RowVectorXf hann_cols(width);
    int NN = height - 1;

    int offset_y =  NN * offsetFactor_y;

    if(NN != 0)
    {
        for (int i = 0; i < height; ++i)
        {
			int i_p = i-offset_y > height ? 0 : i-offset_y;
            hann_rows(i) = (float)(1.0/2.0 * (1.0 - cos(2*M_PI*(i_p)/NN)));
        }
    }
    NN = width - 1;
    if(NN != 0)
    {
        for (int i = 0; i < width; ++i)
        {
            hann_cols(i) = (float)(1.0/2.0 * (1.0 - cos(2*M_PI*i/NN)));
        }
    }

    NvMotDataTypes::RowMajorMatrixXf hann_window = hann_rows.transpose() * hann_cols;

	return hann_window;
}

/**
 * @brief Compute a symmetric area-ratio score between two bounding boxes.
 * @param[in] currBox Current bounding box.
 * @param[in] prevBox Previous bounding box.
 */
inline float computeSizeScore(const NvMotDataTypes::Rect2f &currBox, const NvMotDataTypes::Rect2f &prevBox)
{
	const float areaPrev = prevBox.volume();
	const float areaCurr = currBox.volume();

	if(prevBox.isEmpty() || areaPrev < std::numeric_limits<float>::epsilon()
			|| currBox.isEmpty() || areaCurr < std::numeric_limits<float>::epsilon())
		return 0;

    float sizeRatio = areaCurr / areaPrev;

    if(sizeRatio > 1)
		sizeRatio = 1.f / sizeRatio;

    return sizeRatio;
}

/**
 * @brief Circularly shift a matrix along both axes.
 * @param[in] matrix Matrix to shift.
 * @param[in] dx Horizontal shift.
 * @param[in] dy Vertical shift.
 */
inline Eigen::MatrixXf circshift(const Eigen::MatrixXf &matrix, int dx, int dy)
{
	Eigen::MatrixXf matrix_out(matrix.rows(), matrix.cols());
	const int rows = matrix.rows();
	const int cols = matrix.cols();
    int idx_y = 0;
    int idx_x = 0;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            idx_y = modul(i+dy, rows);
            idx_x = modul(j+dx, cols);
            matrix_out(idx_y, idx_x) = matrix(i,j);
        }
    }
    return matrix_out;
}

/**
 * @brief Create a two-dimensional Gaussian response map.
 * @param[in] sigma Gaussian standard deviation.
 * @param[in] w Response-map width.
 * @param[in] h Response-map height.
 * @param[in] wrap_kernel Circularly shifts the peak to the origin when true.
 */
inline Eigen::MatrixXf get2dGaussianPdf(const float sigma, const int w, const int h, const bool wrap_kernel = true)
{
	Eigen::MatrixXf y(h, w);
    float w2 = static_cast<float>(int(w/2));
    float h2 = static_cast<float>(int(h/2));

    /// Calculate for each pixel separately.
    for(int i=0; i<h; i++) {
        for(int j=0; j<w; j++) {
            y(i,j) = (float)exp((-0.5 / pow(sigma, 2)) * (pow((i-h2), 2) + pow((j-w2), 2)));
        }
    }

    /// Wrap-around with the circular shifting.
    if(wrap_kernel)
    {
		y = circshift(y, int(w/2), int(h/2));
    }

    return y;
}

/**
 * @brief Recover bounding-box extent clipped by an image border using the previous box size.
 * @param[in] tBbox_new Bounding box that may be clipped by the image border.
 * @param[in] srcImageBox Source image rectangle whose size defines the clipping border.
 * @param[in] tBbox_prev Previous bounding box whose width or height is restored on a clipped side.
 * @return Unclipped bounding box.
 */
inline const NvMotDataTypes::Rect2f unclipBbox(
	const NvMotDataTypes::Rect2f tBbox_new,
	const NvMotDataTypes::Rect2f srcImageBox, const NvMotDataTypes::Rect2f tBbox_prev
)
{
	Eigen::Vector2f min_new = tBbox_new.min();
	Eigen::Vector2f max_new = tBbox_new.max();
	const float w_new = tBbox_new.sizes()(0);
	const float h_new = tBbox_new.sizes()(1);

	const float w_prev = tBbox_prev.sizes()(0);
	const float h_prev = tBbox_prev.sizes()(1);

	const float w_src = srcImageBox.max()(0);
	const float h_src = srcImageBox.max()(1);

	/// Recover a left-clipped box.
	if(w_prev > w_new && min_new(0) <= 1)
	{
		min_new(0) = max_new(0) - w_prev;
	}

	/// Recover a right-clipped box.
	if(w_prev > w_new && max_new(0) > w_src - 1)
	{
		max_new(0) = min_new(0) + w_prev;
	}

	/// Recover a top-clipped box.
	if(h_prev > h_new && min_new(1) <= 1)
	{
		min_new(1) = max_new(1) - h_prev;
	}

	/// Recover a bottom-clipped box.
	if(h_prev > h_new && max_new(1) > h_src - 1)
	{
		max_new(1) = min_new(1) + h_prev;
	}

	return NvMotDataTypes::Rect2f(min_new, max_new);
}

/**
 * @brief Expand a bounding box into a square search region.
 * @param[in] bbox_new Bounding box around which the search region is centered.
 * @param[in] paddingScale Padding scale applied to the geometric mean box size.
 */
inline const NvMotDataTypes::Rect2f computeSearchRegion(
	const NvMotDataTypes::Rect2f &bbox_new, const float paddingScale
)
{
	const Eigen::Vector2f min_new = bbox_new.min();
	const float newW = bbox_new.sizes()(0);
	const float newH = bbox_new.sizes()(1);

	Eigen::Vector2f centerPt;

	centerPt << min_new(0) + newW / 2.0f, min_new(1) + newH / 2.0f;

	float searchRegionSize_w = static_cast<float>(floorf(newW + paddingScale*sqrt(newW * newH)));
	float searchRegionSize_h = static_cast<float>(floorf(newH + paddingScale*sqrt(newW * newH)));
	searchRegionSize_w = (searchRegionSize_w + searchRegionSize_h) / 2.0f;
	searchRegionSize_h = searchRegionSize_w;

	/// Define search region for the update.
	NvMotDataTypes::Rect2f searchRegion = constructRect2f(
		centerPt(0) - searchRegionSize_w/2.f, centerPt(1) - searchRegionSize_h/2.f,
		searchRegionSize_w, searchRegionSize_h
	);

	return searchRegion;
}

/**
 * @brief Compute intersection over union for two bounding boxes.
 * @param[in] currBox Current bounding box.
 * @param[in] prevBox Previous bounding box.
 */
inline float computeIOU(const NvMotDataTypes::Rect2f currBox, const NvMotDataTypes::Rect2f prevBox)
{
	const float areaPrev = prevBox.volume();
	const float areaCurr = currBox.volume();

	if(prevBox.isEmpty() || areaPrev < std::numeric_limits<float>::epsilon()
			|| currBox.isEmpty() || areaCurr < std::numeric_limits<float>::epsilon())
		return 0;

	if(currBox.intersects(prevBox))
	{
		float intersection = currBox.intersection(prevBox).volume();
		float union_area = areaCurr + areaPrev - intersection;

		if(union_area < std::numeric_limits<float>::epsilon() || intersection < std::numeric_limits<float>::epsilon())
		{
			return 0;
		}
		else
		{
			return intersection / union_area;
		}
	}
	else
	{
		return 0;
	}
}

/**
 * @brief Compute the fraction of an object bounding box visible inside an image box.
 * @param[in] objBbox Object bounding box.
 * @param[in] imgBbox Image bounding box.
 */
inline float computeVisibility(const NvMotDataTypes::Rect2f &objBbox, const NvMotDataTypes::Rect2f &imgBbox)
{
	if(imgBbox.isEmpty() || imgBbox.volume() < std::numeric_limits<float>::epsilon()
			|| objBbox.isEmpty() || objBbox.volume() < std::numeric_limits<float>::epsilon())
		return 0;

	if(objBbox.intersects(imgBbox))
	{
		float intersection = objBbox.intersection(imgBbox).volume();

		if(intersection < std::numeric_limits<float>::epsilon())
		{
			return 0;
		}
		else
		{
			float iou = intersection / objBbox.volume();

			return iou;
		}
	}
	else
	{
		return 0;
	}
}

/**
 * @brief Classify the orientation of three double-precision points.
 * @param[in] p First point.
 * @param[in] q Second point.
 * @param[in] r Third point.
 * @return Zero for collinear points, one for clockwise order, or two for counterclockwise order.
 */
inline int orientation(const Eigen::Vector2d& p, const Eigen::Vector2d& q, const Eigen::Vector2d& r)
{
    float val = (q(1) - p(1)) * (r(0) - q(0)) -
              (q(0) - p(0)) * (r(1) - q(1));

    /// Return the collinear classification.
    if (val == 0.) return 0;
    /// Classify the turn as clockwise or counterclockwise.
    return (val > 0) ? 1 : 2;
}

/**
 * @brief Classify the orientation of three single-precision points.
 * @param[in] p First point.
 * @param[in] q Second point.
 * @param[in] r Third point.
 * @return Zero for collinear points, one for clockwise order, or two for counterclockwise order.
 */
inline int orientation(const Eigen::Vector2f& p, const Eigen::Vector2f& q, const Eigen::Vector2f& r)
{
    float val = (q(1) - p(1)) * (r(0) - q(0)) -
              (q(0) - p(0)) * (r(1) - q(1));

    /// Return the collinear classification.
    if (val == 0.) return 0;
    /// Classify the turn as clockwise or counterclockwise.
    return (val > 0) ? 1 : 2;
}

#ifdef VIZ_DEBUG

/**
 * @brief Circularly shift an OpenCV matrix along both axes.
 * @param[in] matrix Matrix to shift.
 * @param[in] dx Horizontal shift.
 * @param[in] dy Vertical shift.
 */
inline cv::Mat circshift(cv::Mat matrix, int dx, int dy)
{
	cv::Mat matrix_out = matrix.clone();
    int idx_y = 0;
    int idx_x = 0;
    for(int i=0; i<matrix.rows; i++) {
        for(int j=0; j<matrix.cols; j++) {
            idx_y = modul(i+dy, matrix.rows);
            idx_x = modul(j+dx, matrix.cols);
            matrix_out.at<float>(idx_y, idx_x) = matrix.at<float>(i,j);
        }
    }
    return matrix_out;
}

/**
 * @brief Blend a foreground image over a background using per-pixel opacity.
 * @param[in] background Background image.
 * @param[in] foreground Foreground image.
 * @param[out] output Blended output image.
 * @param[in] location Foreground origin in the background image.
 * @param[in] foregroundDensityMap Optional per-pixel density map.
 * @param[in] opacity_min Minimum density required for blending.
 * @param[in] opacity_offset Offset applied to scaled density.
 * @param[in] opacity_scale Scale applied to density.
 */
inline void OverlayImage(
    const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output,
    cv::Point2i location = cv::Point(0,0), const cv::Mat &foregroundDensityMap = cv::Mat(),
    float opacity_min = 0, float opacity_offset = 1, float opacity_scale = 1
)
{
    if(background.cols != output.cols || background.rows != output.rows || background.type() != output.type() || background.channels() != output.channels())
        background.copyTo(output);

    /// Start at the row indicated by location, or at row 0 if location.y is negative.
    for(int y = std::max(location.y , 0); y < background.rows; ++y)
    {
        /// Convert the background row to a foreground row.
        int fY = y - location.y;

        /// Skip rows beyond the foreground image.
        if(fY >= foreground.rows)
            continue;

        /// Start at the requested column or the background's left edge.
        for(int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            /// Convert the background column to a foreground column.
            int fX = x - location.x;

            {
                bool isFilled = false;
                for(int c = 0; c < output.channels(); ++c)
                    isFilled |= foreground.data[fY*foreground.step + fX*foreground.channels() + c];
                if(!isFilled)
                    continue;
            }

            /// Skip columns beyond the foreground image.
            if(fX >= foreground.cols)
                continue;

			/// Compute foreground opacity from transformed-image pixel density.
			double opacity_fg = 0;

            if(foregroundDensityMap.rows > 0 && foregroundDensityMap.at<float>(fY,fX) > opacity_min)
            {
                opacity_fg = std::max(
                    double(0),
                    std::min(double(1.), double(foregroundDensityMap.at<float>(fY,fX)*opacity_scale + opacity_offset))
                );
            }
			unsigned char foregroundPx, backgroundPx;
            for(int c = 0; opacity_fg > 0 && c < output.channels(); ++c)
            {
                foregroundPx = foreground.data[fY*foreground.step + fX*foreground.channels() + c];
                backgroundPx = background.data[y*background.step + x*background.channels() + c];

                output.data[y*output.step + output.channels()*x + c] = backgroundPx*(1.-opacity_fg) + foregroundPx*opacity_fg;
            }
        }
    }
}

/**
 * @brief Project a point between image planes with a homography.
 * @param[in] pt Point in the source plane.
 * @param[in] projMat_A2B Homography from the source plane to the destination plane.
 */
inline cv::Point2f ConverPtBtwPlanes(const cv::Point2f &pt, const cv::Mat &projMat_A2B)
{
    cv::Mat pt_homo = cv::Mat(3,1,projMat_A2B.type());
    pt_homo.at<double>(0,0) = pt.x;
    pt_homo.at<double>(1,0) = pt.y;
    pt_homo.at<double>(2,0) = 1.;

    cv::Mat temp = projMat_A2B * pt_homo;

    cv::Point2f pt_transformed(temp.at<double>(0,0)/temp.at<double>(2,0), temp.at<double>(1,0)/temp.at<double>(2,0));

	return pt_transformed;
}

/**
 * @brief Leave the output unchanged because density-to-opacity conversion is not implemented.
 * @param[in] input Input RGBA image placeholder.
 * @param[in] output Output RGBA image placeholder that remains unchanged.
 */
inline void Density2Opacity(const cv::Mat &input, cv::Mat &output)
{

}

/**
 * @brief Warp an image with nearest-neighbor sampling under a homography.
 * @param[in] inImg Source image.
 * @param[out] outImg Warped output image.
 * @param[in] projMat_In2out Homography from input to output coordinates.
 * @return True after the output image is populated.
 */
inline bool warpPerspective_custom(const cv::Mat &inImg, cv::Mat &outImg, const cv::Mat &projMat_In2out)
{
    outImg = cv::Mat(inImg.cols, inImg.rows, inImg.type(), cv::Scalar(0));

    /// Invert the homography to map output coordinates back to the input image.
    cv::Mat projMat_Out2In = projMat_In2out.inv();

    for(int y = 0; y < outImg.rows; ++y)
    {
        for(int x = 0; x < outImg.cols; ++x)
        {
            cv::Point2f pt_in = ConverPtBtwPlanes(cv::Point2f(x,y), projMat_Out2In);

            /// Skip samples outside the input image.
            if(pt_in.x >= inImg.cols || pt_in.y >= inImg.rows || pt_in.x < 0 || pt_in.y < 0)
                continue;

            for(int c = 0; c < outImg.channels(); ++c)
            {
                outImg.data[y*outImg.step + x*outImg.channels() + c] = inImg.data[int(pt_in.y+0.5)*inImg.step + int(pt_in.x+0.5)*inImg.channels() + c];
            }
        }
    }

    return true;
}

/**
 * @brief Estimate normalized pixel density after a projective transform.
 * @param[in] inImg Source image used to size the density map.
 * @param[out] densityImg Normalized single-channel density map.
 * @param[in] projMat_In2out Homography from input to output coordinates.
 * @return True after the density map is populated.
 */
inline bool getTransformedPixelDensity(const cv::Mat &inImg, cv::Mat &densityImg, const cv::Mat &projMat_In2out)
{
    /// Initialize a single-channel density map.
    densityImg = cv::Mat(inImg.rows, inImg.cols, CV_32F, cv::Scalar(0));

    uint STEP = 10;

    for(uint y = 0; y < densityImg.rows-STEP; y += STEP)
    {
        for(uint x = 0; x < densityImg.cols-STEP; x += STEP)
        {
            cv::Point2f pt_in_1 = ConverPtBtwPlanes(cv::Point2f(x,y), projMat_In2out);
            cv::Point2f pt_in_2 = ConverPtBtwPlanes(cv::Point2f(x+1,y+1), projMat_In2out);
            float dist_sq = pow(pt_in_1.x - pt_in_2.x,2) + pow(pt_in_1.y - pt_in_2.y,2);
            /// Shorter pixel-to-pixel distance would mean higher density.
            float density = 1./dist_sq;

            for(uint m=0; m<STEP; m++)
                for(uint n=0; n<STEP; n++)
                    densityImg.at<float>(y+m,x+n) = density;
        }
    }
    cv::normalize(densityImg, densityImg, 0, 1, cv::NORM_MINMAX, CV_32F);

    return true;
}

/**
 * @brief Check whether a filesystem path exists.
 * @param[in] filename Path to check.
 * @return True when the path exists.
 */
inline bool isFileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

/**
 * @brief Convert a single-channel NV12 image to BGR.
 * @param[in] nv12_img Single-channel NV12 image.
 */
inline cv::Mat NV12_to_BGR(const cv::Mat nv12_img)
{
	assert(nv12_img.channels() == 1);

    cv::Mat bgr_img;
    cv::cvtColor(nv12_img, bgr_img, cv::COLOR_YUV2BGR_NV12);

    return bgr_img;
}

/**
 * @brief Convert raw single-channel NV12 data to BGR.
 * @param[in] yuv_data Pointer to NV12 image data.
 * @param[in] width Image width.
 * @param[in] height Image height.
 */
inline cv::Mat NV12_to_BGR(unsigned char* yuv_data, unsigned int width, unsigned int height)
{
    cv::Mat nv12_img = cv::Mat(height * 3/2, width, CV_8UC1, yuv_data);
    cv::Mat bgr_img;
    cv::cvtColor(nv12_img, bgr_img, cv::COLOR_YUV2BGR_NV12);

    return bgr_img;
}

/**
 * @brief Convert a three-channel BGR image to single-channel NV12 storage.
 * @param[in] src Source BGR image.
 */
inline cv::Mat BGR_to_NV12(const cv::Mat src)
{
	assert(src.type() == CV_8UC3);

	/// Allocate single-channel storage for the Y and interleaved UV planes.
	cv::Mat dst = cv::Mat::zeros(cv::Size(src.size().width, src.size().height*1.5), CV_8UC1);

	/// Populate the Y plane.
	for(int y=0; y<src.rows; y++)
	{
		for(int x=0; x<src.cols; x++)
		{
			const cv::Vec3b bgr = src.at<cv::Vec3b>(cv::Point(x,y));
			const uchar B = bgr.val[0];
			const uchar G = bgr.val[1];
			const uchar R = bgr.val[2];

			uchar &Y = dst.at<uchar>(cv::Point(x,y));

			Y = 0.257*R + 0.504*G + 0.098*B + 16 + 0.5f;
		}
	}

	for(int y=0; y<src.rows; y+=2)
	{
		for(int x=0; x<src.cols; x+=2)
		{
			const cv::Vec3b bgr = src.at<cv::Vec3b>(cv::Point(x,y));
			const uchar B = bgr.val[0];
			const uchar G = bgr.val[1];
			const uchar R = bgr.val[2];

			uchar &Cb = dst.at<uchar>(cv::Point(x+0,y/2+src.rows));
			uchar &Cr = dst.at<uchar>(cv::Point(x+1,y/2+src.rows));

			Cb = -0.148*R - 0.291*G + 0.439*B + 128;
			Cr =  0.439*R - 0.368*G - 0.071*B + 128;
		}
	}

	return dst;
}

/**
 * @brief Convert a three-channel BGR image to two-channel NV12 storage.
 * @param[in] src Source BGR image.
 */
inline cv::Mat BGR_to_NV12_2ch(const cv::Mat src)
{
	assert(src.type() == CV_8UC3);

	/// Create two single-channel images.
	std::vector<cv::Mat> dst_per_ch(2);
	dst_per_ch[0] = cv::Mat::zeros(src.size(), CV_8UC1);
	dst_per_ch[1] = cv::Mat::zeros(src.size(), CV_8UC1);

	/// Populate the Y plane.
	for(int y=0; y<src.rows; y++)
	{
		for(int x=0; x<src.cols; x++)
		{
			const cv::Vec3b bgr = src.at<cv::Vec3b>(cv::Point(x,y));
			const uchar B = bgr.val[0];
			const uchar G = bgr.val[1];
			const uchar R = bgr.val[2];

			uchar &Y = dst_per_ch[0].at<uchar>(cv::Point(x,y));

			Y = 0.257*R + 0.504*G + 0.098*B + 16 + 0.5f;
		}
	}

	for(int y=0; y<src.rows; y+=2)
	{
		for(int x=0; x<src.cols; x+=2)
		{
			const cv::Vec3b bgr = src.at<cv::Vec3b>(cv::Point(x,y));
			const uchar B = bgr.val[0];
			const uchar G = bgr.val[1];
			const uchar R = bgr.val[2];

			uchar &Cb = dst_per_ch[1].at<uchar>(cv::Point(x+0,y/2));
			uchar &Cr = dst_per_ch[1].at<uchar>(cv::Point(x+1,y/2));

			Cb = -0.148*R - 0.291*G + 0.439*B + 128;
			Cr =  0.439*R - 0.368*G - 0.071*B + 128;
		}
	}

	cv::Mat dst;

	/// Merge the single-channel planes into a two-channel NV12 image.
	cv::merge(dst_per_ch, dst);

	return dst;
}

/**
 * @brief Convert a two-channel NV12 image to BGR.
 * @param[in] src Source two-channel NV12 image.
 */
inline cv::Mat NV12_2ch_to_BGR(cv::Mat src)
{
	assert(src.channels() == 2);

	std::vector<cv::Mat> nv12_per_ch;

	/// Split the NV12 image into single-channel Y and UV images.
	cv::split(src, nv12_per_ch);

    cv::Mat nv12_img = cv::Mat::zeros(cv::Size(src.size().width, src.size().height*3/2), CV_8UC1);

	/// Copy the Y component into the destination NV12 image.
    nv12_per_ch[0].copyTo(nv12_img( cv::Rect(0,0,src.size().width,src.size().height) ));

	/// Copy the UV component into the destination NV12 image.
    nv12_per_ch[1](
			cv::Rect(0,0,src.size().width,src.size().height*0.5)
			).copyTo(nv12_img( cv::Rect(0,src.size().height,src.size().width,src.size().height*0.5) ));

    cv::Mat bgr_img;
    cv::cvtColor(nv12_img, bgr_img, cv::COLOR_YUV2BGR_NV12);

    return bgr_img;
}

#endif

} // end of CV_FUNCS

#endif // CV_FUNCTIONS_HPP
