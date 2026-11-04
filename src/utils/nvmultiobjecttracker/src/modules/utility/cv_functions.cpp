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

#include "modules/utility/cv_functions.hpp"

namespace CV_FUNCS {

/**
 * @brief Convert a bounding box to an estimator measurement vector.
 * @param[in] bbox Bounding box to convert.
 * @param[in] useAspc Selects aspect ratio instead of width in the measurement vector.
 */
Eigen::VectorXd convertBboxToZ(const NvMotDataTypes::Rect2f &bbox, const bool useAspc)
{

	Eigen::Vector2f min = bbox.min();
	Eigen::Vector2f max = bbox.max();

	Eigen::Vector2f loc; loc << (min(0) + max(0))/2.f, (min(1) + max(1))/2.f;
	double w = max(0) - min(0);
	double h = max(1) - min(1);

	/// Type should match with the type used in the state estimator.
	Eigen::VectorXd z(4);

	z << (double)loc(0), (double)loc(1), (useAspc ? w/h : w), h;

	return z;
}

/**
 * @brief Compute the convex hull of double-precision points.
 * @param[in] points Input points.
 * @param[out] hull Convex-hull vertices in traversal order.
 */
void convexHull(const std::vector<Eigen::Vector2d> &points, std::vector<Eigen::Vector2d> &hull)
{
	/// There must be at least 3 points.
	if (points.size() < 3)
	{
		hull = points;
		return;
	}

	hull.clear();
	/// Find the leftmost point.
	int l = 0;
	for (uint i = 1; i < points.size(); i++)
	{
		if (points.at(i)(0) < points.at(l)(0))
		{
			l = i;
		}
	}

	/// Traverse counterclockwise from the leftmost point in O(h) output-sensitive iterations.
	int p = l, q;
	do
	{
		/// Add current point to result.
		hull.push_back(points.at(p));
		/// Initialize the next hull point before searching for a more counterclockwise candidate.
		q = (p+1) % points.size();
		for (uint i = 0; i < points.size(); i++)
		{
			/// Replace the candidate when the point is more counterclockwise.
			if (orientation(points.at(p), points.at(i), points.at(q)) == 2)
			{
				q = i;
			}
		}
		/// Set p to q, the most counterclockwise point with respect to p, so that q is added to the result hull.
		p = q;
	} while (p != l); // continue until the walk returns to the first point
}

/**
 * @brief Compute the convex hull of single-precision points.
 * @param[in] points Input points.
 * @param[out] hull Convex-hull vertices in traversal order.
 */
void convexHull(const std::vector<Eigen::Vector2f> &points, std::vector<Eigen::Vector2f> &hull)
{
	/// There must be at least 3 points.
	if (points.size() < 3)
	{
		hull = points;
		return;
	}

	hull.clear();
	/// Find the leftmost point.
	int l = 0;
	for (uint i = 1; i < points.size(); i++)
	{
		if (points.at(i)(0) < points.at(l)(0))
		{
			l = i;
		}
	}

	/// Traverse counterclockwise from the leftmost point in O(h) output-sensitive iterations.
	int p = l, q;
	do
	{
		/// Add current point to result.
		hull.push_back(points.at(p));
		/// Initialize the next hull point before searching for a more counterclockwise candidate.
		q = (p+1) % points.size();
		for (uint i = 0; i < points.size(); i++)
		{
			/// Replace the candidate when the point is more counterclockwise.
			if (orientation(points.at(p), points.at(i), points.at(q)) == 2)
			{
				q = i;
			}
		}
		/// Set p to q, the most counterclockwise point with respect to p, so that q is added to the result hull.
		p = q;
	} while (p != l); // continue until the walk returns to the first point
}

/**
 * @brief Compute the axis-aligned bounding rectangle of a point set.
 * @param[in] points Input points.
 */
NvMotDataTypes::Rect2f boundingRect(const std::vector<Eigen::Vector2d> &points)
{
	if (points.empty())
	{
		return constructRect2f(0, 0, 0, 0);
	}

	std::vector<float> x_vec, y_vec;
	for (uint i=0; i < points.size(); i++)
	{
		x_vec.push_back(points.at(i)(0));
		y_vec.push_back(points.at(i)(1));
	}

	auto x_minmax = std::minmax_element(x_vec.begin(), x_vec.end());
	auto y_minmax = std::minmax_element(y_vec.begin(), y_vec.end());
	return constructRect2f(
		*x_minmax.first, *y_minmax.first, *x_minmax.second - *x_minmax.first, *y_minmax.second - *y_minmax.first
	);

}

/**
 * @brief Compute the bounding rectangle of nonzero mask values.
 * @param[in] mask Row-major mask values.
 * @param[in] width Mask width.
 * @param[in] height Mask height.
 */
NvMotDataTypes::Rect2f boundingRect(float *mask, const int width, const int height)
{
	if (width <= 0 || height <= 0)
	{
		return constructRect2f(0, 0, 0, 0);
	}

	std::vector<int> sumReduceX(height,0), sumReduceY(width,0);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			sumReduceX[y] += int(mask[y*width + x] > 0);
			sumReduceY[x] += int(mask[y*width + x] > 0);
		}
	}

	int minX = 0, minY = 0, maxX = 0, maxY = 0;
	for (int x = 0; x < width; x++)
	{
		if (sumReduceY[x] > 0)
		{
			minX = x;
			break;
		}
	}
	for (int x = width-1; x >= 0; x--)
	{
		if (sumReduceY[x] > 0)
		{
			maxX = x;
			break;
		}
	}

	for (int y = 0; y < height; y++)
	{
		if (sumReduceX[y] > 0)
		{
			minY = y;
			break;
		}
	}
	for (int y = height-1; y >= 0; y--)
	{
		if (sumReduceX[y] > 0)
		{
			maxY = y;
			break;
		}
	}

	return constructRect2f(minX, minY, maxX-minX, maxY-minY);
}

} // end of CV_FUNCS
