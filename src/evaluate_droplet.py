#     droplet fitting program for magnetoactive surfaces
#     Copyright (C) 2022  Raphael Kriegl

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Droplet eval function
from __future__ import annotations

from math import cos, sin, pi,degrees

import numpy as np
import cv2

from fit_classes import EllipseFit, RobustPolarPolyFit, SplineFit, Side
from droplet import Droplet
from contour import Contour, ContourError

def evaluate_droplet(img, y_base, mask: tuple[int,int,int,int] = None, divider: int = None, fit_type: str = "ellipse", contour_lim: tuple[int,int] = (0,200), cutoff_top: int = 0, weighted_fit: bool = False) -> Droplet:
    """ 
    Analyze an image for a droplet and determine the contact angles

    ---
    :param img: the image to be evaluated as np.ndarray
    :param y_base: the y coordinate of the surface the droplet sits on
    :param mask: a 4-tuple which marks the corners of a rectagle to use as blocking mask for contour detection (needle)
    :param divider: x coordinate where to split detected contour, if None, uses mean of contour
    :param fit_type: choose fitting algorithm, ell_splt and polar are recommended
        "ellipse". approcimates droplet with single ellipse 
        "ell_splt": uses 2 ellipses to fit each side 
        "spline": uses cubic splines to approximate contour 
        "polar": polar polynomial fit for each side 
    :param contour_lim: sets the start point and length of the used contour, used to limit the points used for fitting
    :param cutoff_top: start y coordinate to cut top of image
    :param weighted_fit: whether to use a weighted fit, not supported by all methods. Points closer to contact line are weighted more
    :returns: a Droplet() object
    """

    # determine fit type
    fit_class = None
    if fit_type == "ell_splt" or fit_type == "ellipse":
        fit_class = EllipseFit
    elif fit_type == "spline":
        fit_class = SplineFit
    elif fit_type == "polar":
        fit_class = RobustPolarPolyFit
    elif fit_type == "contour": # only detect contour
        fit_class = None
    else:
        raise ValueError(f"Invalid fit type: {fit_type}")

    drplt = Droplet()

    # crop img from baseline down (contains no useful information)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    if divider is None: divider = "auto"
    crop_img = np.copy(img[cutoff_top:y_base,:])
    bw_edges = edge_detection(crop_img)

    # apply mask, used for blocking needle
    if (not mask is None):
        x,y,w,h = mask
        mask_mat = np.ones([y_base-cutoff_top, width], dtype="uint8")
        mask_mat[:, x:x+w] = 0
        
        bw_edges = cv2.bitwise_and(bw_edges, bw_edges, mask=mask_mat)
        masked = True
        divider = x+w//2
    else:
        masked = False

    # edge detection
    edge, left_edge, right_edge, divider = find_contour(bw_edges, masked, divider)

    # corrections if image top was cut
    edge.offset(0,cutoff_top)
    left_edge.offset(0,cutoff_top)
    right_edge.offset(0,cutoff_top)

    # setting first available data to droplet object
    drplt.contour = edge
    drplt.image = np.copy(img)
    drplt.is_valid = True
    drplt.has_contour = True

    # if normal ellipse fit is used, whole droplet contour is fit
    if fit_type == "ellipse":
        edge = edge.shorten(y_base, contour_lim)

        # FIT
        fit = fit_class.fit_contour(edge, weighted=weighted_fit)

        drplt.fit_curve = [fit]
        drplt.fit_type = fit_type
        drplt.has_fit = True

        # calc residuals
        r2 = fit_class.coefficient_of_determination([fit])
        drplt.r2 = r2

        # calc slope and angle of tangent at intersections
        (m_t_l,point_int_l), (m_t_r,point_int_r) = fit.slope_at_intersect((0,y_base))

    # all others will use droplet halves separately
    elif fit_type:
        left_edge = left_edge.shorten(y_base, contour_lim)
        right_edge = right_edge.shorten(y_base, contour_lim)

        # FIT
        left_fit = fit_class.fit_contour(left_edge, weighted=weighted_fit, side=Side.LEFT)
        right_fit = fit_class.fit_contour(right_edge, weighted=weighted_fit, side=Side.RIGHT)

        drplt.fit_curve = [left_fit,right_fit]
        drplt.fit_type = fit_type
        drplt.has_fit = True

        # calc residuals
        r2 = fit_class.coefficient_of_determination([left_fit, right_fit])
        drplt.r2 = r2

        # calc slope and angle of tangent at intersections
        m_t_l,point_int_l = left_fit.slope_at_intersect((0,y_base))[0]
        m_t_r,point_int_r = right_fit.slope_at_intersect((0,y_base))[-1]


    if fit_type:
        # calc angle from inclination of tangents
        angle_l = (pi - np.arctan2(m_t_l,1)) % pi
        angle_r = (np.arctan2(m_t_r,1) + pi) % pi


        #write values to droplet object
        drplt.angle_l = degrees(angle_l)
        drplt.angle_r = degrees(angle_r)
        drplt.has_angle = True
        
        drplt.divider = divider
        drplt.tan_l_m = m_t_l
        drplt.tan_r_m = m_t_r
        line_length = 100
        drplt.line_l = (*point_int_l, point_int_l[0] + line_length*cos(angle_l), point_int_l[1] - line_length*sin(angle_l))
        drplt.line_r = (*point_int_r, point_int_r[0] - line_length*cos(angle_r), point_int_l[1] - line_length*sin(angle_r))
        drplt.int_l = point_int_l
        drplt.int_r = point_int_r
        drplt.base_diam = point_int_r[0] - point_int_l[0]
        
    drplt.is_valid = True

    return drplt

def edge_detection(img):
    # based on opendrop/features/conan.py > extract_contact_angle_features
    # https://github.com/jdber1/opendrop/blob/master/opendrop/features/conan.py
    # Copyright OpenDrop Contributors GNU GPL v3

    thresh = 0.2

    blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
    dx = cv2.Scharr(blur, cv2.CV_16S, dx=1, dy=0)
    dy = cv2.Scharr(blur, cv2.CV_16S, dx=0, dy=1)

    # Use magnitude of gradient squared to get sharper edges.
    mask = (dx.astype(float)**2 + dy.astype(float)**2)
    mask = np.sqrt(mask)
    mask = (mask/mask.max() * (2**8 - 1)).astype(np.uint8)

    # Ignore weak gradients.
    mask[mask < thresh * mask.max()] = 0

    cv2.adaptiveThreshold(
        mask,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=5,
        C=0,
        dst=mask
    )

    # Hack: Thin edges using cv2.Canny()
    grad_max = (abs(dx) + abs(dy)).max()
    edges = cv2.Canny(mask*dx, mask*dy, grad_max*thresh/2, grad_max*thresh)

    return edges


def find_contour(img: np.ndarray, is_masked: bool, divider):
    """searches for contours and returns the ones with largest bounding rect

    :param img: grayscale or bw image
    :param is_masked: if image was masked
    :type is_masked: bool
    :raises ContourError: if no contours are detected
    :return: if not is_masked: contour with largest bounding rect, split by divider

            else: the two contours with largest bounding rect merged, split by divider
    """
    # find all contours in image, https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html 
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    length = len(contours)
    if length == 0:
        raise ContourError('No contours found!')

    contours = np.array(contours, dtype=object)

    # if only one contour found: use it
    if length == 1:
        contour = Contour(np.array(contours[0], dtype=np.int32))
    # if multiple contours found: evaluate which one is droplet
    else:
        area_list = np.zeros(length, dtype=float)
        rect_list = np.zeros([length,4], dtype=float)

        for idx, cont in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cont)
            # store contour, area of bounding rect and bounding rect in array
            area_list[idx] = w*h
            rect_list[idx] = np.asarray([x,y,w,h])

        # sort contours by bounding rect area
        sorted_indizes = area_list.argsort()
        area_list = area_list[sorted_indizes]
        contours_sorted = contours[sorted_indizes]
        rect_list = rect_list[sorted_indizes]
        
        if is_masked and len(rect_list) > 1:
            # select largest 2 non overlapping contours, assumes mask splits largest contour in the middle
            # check if second largest contour is not from inside the droplet by checking overlap of bounding rects
            BR = rect_list[-1] # biggest rect
            b_x, b_y, b_w, b_h = BR
            SR = rect_list[-2] # slightly smaller rect
            s_x, s_y, s_w, s_h = SR
            # check if smaller rect overlaps with larger rect
            if ((b_w+b_x < s_x) or (b_x > s_w+s_x) or (b_y > s_h+s_y) or (b_h+b_y < s_y)):
                # if not, both rects are valid droplet contours, merge them
                contour = Contour(np.concatenate((contours_sorted[-2], contours_sorted[-1])))
            else:
                # else only biggest is valid droplet contour
                contour = Contour(contours_sorted[-1])
        else:
            # contour with largest area
            contour = Contour(contours_sorted[-1])

    
    if divider is None:
        return Contour(contour)
    else:
        # divide contour for double ellipse fit
        if divider == "auto":               
            #find the min y value, assume thats the middle of the droplet and use that as divider
            x_vals = contour.get_x_values()
            y_vals = contour.get_y_values()
            divider = np.mean(x_vals[np.where(y_vals == np.amin(y_vals))])
        left, right = [], []
        # split contour in left and right
        for elem in contour.points:
            if elem[0] < divider:
                left.append(elem)
            else: 
                right.append(elem)
        # readjust shape of arrays to fit opencv contour style
        left = np.array(left)
        right = np.array(right)
        return contour, Contour(left, True), Contour(right,True), divider
