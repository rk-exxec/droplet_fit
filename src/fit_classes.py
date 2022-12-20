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


from __future__ import annotations
from abc import ABC, abstractmethod
import enum
import logging
from math import  cos, sin, sqrt, radians, degrees
from typing import Callable
from PySide6.QtGui import QPainter, QPen, Qt, QPolygonF
from PySide6.QtCore import QPointF

import numba as nb
from numpy.polynomial import Polynomial
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import root, OptimizeResult
from skimage.measure import ransac
from skimage.measure import EllipseModel
from skimage.transform import AffineTransform
import scipy.linalg as sl
import cv2

from contour import Contour, ContourError
from precompiled import calc_residuals

class Side(enum.Enum):
    LEFT = 0,
    RIGHT = 1,
    BOTH = 2

class IFitCurve(ABC):
    @abstractmethod
    def __init__(self, contour=None):
        if contour is not None:
            if not isinstance(contour, Contour):
                contour = Contour(contour)
        self.contour:Contour = contour

    @abstractmethod
    def slope_at_intersect(self, int_line):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def fit_contour(cls, contour: Contour, weighted = False, side=None) -> IFitCurve:
        raise NotImplementedError()

    @staticmethod
    def coefficient_of_determination(fits: list[IFitCurve], contours=None):
        """
        calculates the coefficient of determination R²!
        https://en.wikipedia.org/wiki/Coefficient_of_determination

        -----
        :param polys: list of PolyFit elements
        :return: R²: 0 worst, 1 best, other: wrong fit model!
        """
        residuals = 0
        totals = 0
        if contours:
            for fit,cont in zip(fits,contours):
                residuals += fit.square_sum_of_residuals(cont)
                totals += fit.total_sum_of_squares(cont)
        else:
            for fit in fits:
                residuals += fit.square_sum_of_residuals()
                totals += fit.total_sum_of_squares()

        r2 = 1 - (residuals/totals) if residuals <= totals else 0.0

        return r2

    def draw(self):
        """ This function is used to draw the fit function in a Qt paint event handler"""
        raise NotImplementedError()

class SplineFit(IFitCurve):
    num_fit_points = 10
    def __init__(self, ppoly, contour):
        self.ppoly : PPoly = ppoly
        super().__init__(contour=contour)

        self._pen = QPen(Qt.magenta,1.5)
        self._pen.setCosmetic(True)
        self._pen_contour = QPen(Qt.yellow,1)
        self._pen_contour.setCosmetic(False)

    def __getitem__(self, key):
        raise IndexError(f"'{key}' is not a valid index")

    def _intersect_line(self, line_m, line_t) -> list[float]:
        if line_m == 0:
            return [int(self.ppoly(line_t)), line_t]

    def _slope(self, point) -> float:
        """
        calculates slope of tangent at the point

        :param point: point where the slope will be calculated
        :returns: the slope of the tangent
        """
        m_tan = self.ppoly.derivative(1)(point[1])
        # invert because spline its inverse to actual curve
        return 1/m_tan

    def slope_at_intersect(self, int_line):
        """
        calculates slope of tangent at the intersection with the line

        :param int_line: intersecting line y=mx+t (m,t)
        :returns: the slope of the tangent and intersecting point as tuple (m_tan, [x,y])
        """
        point_int = self._intersect_line(*int_line)
        return [(self._slope(point_int),point_int)]
    
    def draw(self, painter:QPainter, img_size):
        """ This function is used to draw the fit function in a Qt paint event handler"""
        self.contour.draw(painter, img_size, self._pen_contour)
        painter.setPen(self._pen)
        points = [QPointF(self.ppoly(i), i) for i in range(3, img_size[1],3)]
        # for i in range(3, img_size[1],3):
        #     painter.drawLine(int(self.ppoly(i-3)), i-3, int(self.ppoly(i)), i)
        painter.drawPolyline(QPolygonF.fromList(points))


    @classmethod
    def fit_contour(cls, contour: Contour, weighted = False, side=None):
        """fit a cubic spline to the given contour

        :param contour: the contour to interpolate with spline
        :type contour: Contour
        :raises ValueError: if fit fails or produces invalid ellipse
        :return: Spline object
        """
        if not isinstance(contour, Contour):
            contour = Contour(contour)
        
        data = contour.points.T
        y_data = data[1]
        unique_data, sorted_idx = np.unique(y_data, return_index=True)
        # fitting x over y, because there will be duplicate x values
        selection = np.linspace(0, len(unique_data)-1, cls.num_fit_points, dtype=int, endpoint=True) # select 30 evenly spaced points from contour
        new_data = contour.points[sorted_idx][selection].T
        ppoly = CubicSpline(new_data[1], new_data[0])

        return SplineFit(ppoly, contour)

    def square_sum_of_residuals(self, contour=None) -> float:
        """return residuals from fit

        :return: [description]
        :rtype: [type]
        """
        residuals = 0.0
        if not contour: contour = self.contour
        for point in contour.points:
            residuals += (point[0] - self.ppoly(point[1]))**2
        return residuals

    def total_sum_of_squares(self, contour=None) -> float:
        """calculates the total sum of squares for the contour
        for all points of the contour
        Σ (d - m)²

        """
        if not contour: contour = self.contour
        distances = contour.points.T[0]
        mean_dist = np.mean(distances)

        return np.sum((distances - mean_dist)**2)


class EllipseFit(IFitCurve):
    def __init__(self, a=1, b=1, x0 = 0, y0 = 0, theta = 0, contour=None):
        self.x0 = x0
        self.y0 = y0
        self.origin = (x0,y0)
        self.a = a
        self.b = b
        self.maj = a*2
        self.min = b*2
        self.theta = theta
        self.tilt_deg = degrees(theta)
        super().__init__(contour=contour)
        self.aff_transform = AffineTransform(rotation=self.theta, translation=self.origin)
        self._pen = QPen(Qt.magenta,1.5)
        self._pen.setCosmetic(True)
        self._pen_contour = QPen(Qt.yellow,1)
        self._pen_contour.setCosmetic(False)

    def __getitem__(self, key):
        if key == 0:
            return self.x0
        elif key == 1:
            return self.y0
        elif key == 2:
            return self.a
        elif key == 3:
            return self.b
        elif key == 4:
            return self.theta
        else:
            raise IndexError(f"'{key}' is not a valid index")

    def get_normalized_contour(self, contour=None):
        """ transforms contour into elipse coordinate system, so origin is 0,0 and axes are parralel to x and y
        i.e. translation and rotation
        needed for calculations
        """
        if not contour: contour = self.contour
        if contour == None:
            raise ValueError("Ellipse: cannot get normalized contour of none type")
        data = contour.points
        rot_mat = np.array( [
                    [np.cos(self.theta)    ,   np.sin(self.theta)], 
                    [-np.sin(self.theta)   ,   np.cos(self.theta)] 
                    ])
        # subtract origin
        data_trans = data - np.array([self.x0, self.y0])
        # rotate around origin 
        data_trans = rot_mat.dot(data_trans.T).T
        return Contour(data_trans)

    def _point_to_ellipse_coords(self, point):
        """ transforms point to elipse coordinate system, so ellipse origin is 0,0 and axes are parralel to x and y
        i.e. translation and rotation
        needed for calculations
        """
        data = np.array(point)
        rot_mat = np.array( [
                    [np.cos(self.theta)    ,   np.sin(self.theta)], 
                    [-np.sin(self.theta)   ,   np.cos(self.theta)] 
                    ])
        # subtract origin
        data_trans = data - np.array([self.x0, self.y0])
        # rotate around origin 
        data_trans = rot_mat.dot(data_trans.T).T
        return data_trans

    def _point_to_outer_coords(self,point):
        """ transforms point from elipse coordinate system, so ellipse origin is 0,0 and axes are parralel to x and y
        i.e. translation and rotation
        needed for calculations
        """
        data = np.array(point)
        rot_mat = np.array( [
                    [np.cos(self.theta)   ,  -np.sin(self.theta)], 
                    [np.sin(self.theta)   ,   np.cos(self.theta)] 
                    ])
        # rotate around origin 
        data_trans = rot_mat.dot(data.T).T
        # add offset
        data_trans = data + np.array([self.x0, self.y0])
        return data_trans

    def _intersect_line(self, line_m, line_t) -> list[list[float]]:
        """
        calculates intersection(s) of the ellipse with a line

        :param line_m, line_t: m is the slope and t is intercept of the intersecting line
        :returns: intersection points, first is the left one, second the right one
        """

        # transform line to ellipse origin
        m = line_m
        t = line_t + line_m*self.x0 - self.y0
        # precalc cos and sin
        cos_t = cos(self.theta)
        sin_t = sin(self.theta)
        # ## -->> http://quickcalcbasic.com/ellipse%20line%20intersection.pdf
        # intersect sloped line with rotated ellipse
        a = self.b**2 * (cos_t**2 + 2*m*cos_t*sin_t + m**2 * sin_t**2) + self.a**2 * (m**2*cos_t**2 - 2*m*cos_t*sin_t + sin_t**2)
        b = 2*self.b**2 * t * (cos_t * sin_t + m*sin_t**2) + 2*self.a**2 * t * (m*cos_t**2 - cos_t*sin_t)
        c = t**2 * (self.b**2 * sin_t**2 + self.a**2 * cos_t**2) - self.b**2 * self.a**2

        det = b**2 - 4*a*c

        # solving quadratic equation for intersections ax^2 + bx + c = 0
        # and calculating y coordinate from line equation, then translating back from ellipse coordinates
        if det > 0:
            x1: float = (-b - np.sqrt(det))/(2*a) + self.x0
            x2: float = (-b + np.sqrt(det))/(2*a) + self.x0
            y1 = m*x1 + t + self.y0
            y2 = m*x2 + t + self.y0

            if x1 < x2:
                return [[x1,y1], [x2,y2]]
            else:
                return [[x2,y2], [x1,y1]]
        # only one solution 
        elif det == 0:
            x: float = (-b / (2*a)) + self.x0
            y = m*x + t + self.y0
            # transform back
            return [[x,y]]
        # no valid intersection in real plane
        else: return None

    def _slope(self, point) -> float:
        """
        calculates slope of tangent at the point, the point needs to be on the ellipse!

        :param point: point where the slope will be calculated
        :returns: the slope of the tangent
        """
        # transform to non-rotated ellipse centered to origin
        x_rot, y_rot = self._point_to_ellipse_coords(point)
        # general line equation for tangent Ax + By = C
        # general ellipse eqn x²/a² + y²/b² = 1
        # if C = 1, then Ax + By = x²/a² + y²/b²
        # => A = x/a², B = y/b², C = 1
        tan_a = x_rot/self.a**2
        tan_b = y_rot/self.b**2
        # tan_c = 1
        #rotate tangent line back to angle of the rotated ellipse
        tan_a_r = tan_a*np.cos(self.theta) - tan_b*np.sin(self.theta)
        tan_b_r = tan_b*np.cos(self.theta) + tan_a*np.sin(self.theta)
        #calc slope of tangent m = -A/B
        m_tan = - (tan_a_r / tan_b_r)

        return m_tan

    def slope_at_intersect(self, int_line):
        """
        calculates slope of tangent at the intersection with the line

        :param int_line: intersecting line y=mx+t (m,t)
        :returns: the slope of the tangent and intersecting point as tuple (m_tan, [x,y])
        """
        points = self._intersect_line(*int_line)

        if points is None:
            raise ContourError('No valid intersections found')

        slope_point_pairs = [(self._slope(p),p) for p in points]

        return slope_point_pairs

    @classmethod
    def fit_contour(cls, contour: Contour, weighted = False, side=None):
        """fit an ellipse to the given contour

        :param contour:
        :type contour: Contour
        :raises ValueError: if fit fails or produces invalid ellipse
        :return: fitted parameters stored in an Ellipse object
        """
        if weighted: logging.warn("Ellipse fit does not support weights.")
        if not isinstance(contour, Contour):
            contour = Contour(contour)

        # apply ellipse fitting algorithm to droplet
        (x0,y0), (maj_ax,min_ax), phi_deg = cv2.fitEllipseDirect(contour.cv2)
        phi = radians(phi_deg)
        a = maj_ax/2
        b = min_ax/2

        if a == 0 or b == 0:
            raise ValueError('Malformed ellipse fit! Axis = 0')
        return EllipseFit(a,b,x0,y0,phi,contour)

    def square_sum_of_residuals(self, contour=None) -> float:
        """return residuals from fit

        :return: [description]
        :rtype: [type]
        """
        if not contour: contour = self.contour
        return np.sum(calc_residuals(self.get_normalized_contour(contour).points, self.a, self.b))

    def total_sum_of_squares(self, contour=None) -> float:
        """calculates the total sum of squares for the contour assosiated with this ellipse
        for all points of the contour
        Σ (d - m)²
        d: distance of point to origin
        m: mean distance to origin

        """
        if not contour: contour = self.contour
        ncont = self.get_normalized_contour(contour)
        distances = np.linalg.norm(ncont.points, axis=1)
        #distances = np.sqrt((ncont.get_x_values())**2 + (ncont.get_y_values())**2)
        mean_dist = np.mean(distances)

        return np.sum((distances - mean_dist)**2)

    def draw(self, painter: QPainter, img_size):
        """ This function is used to draw the fit function in a Qt paint event handler"""
        self.contour.draw(painter, img_size, self._pen_contour)

        painter.setPen(self._pen)
        # move origin to ellipse origin
        painter.translate(*self.origin)

        # draw diagnostics
        # db_painter.setPen(pen_fine)
        # #  lines parallel to coordinate axes
        # db_painter.drawLine(0,0,20*scale_x,0)s
        # db_painter.drawLine(0,0,0,20*scale_y)
        # # angle arc
        # db_painter.drawArc(-5*scale_x, -5*scale_y, 10*scale_x, 10*scale_y, 0, -ell.tilt_deg*16)

        # rotate coordinates to ellipse tilt
        painter.rotate(self.tilt_deg)

        # draw ellipse
        # db_painter.setPen(pen)
        
        painter.drawEllipse(-self.a, -self.b, self.maj, self.min)
        
        # # major and minor axis for diagnostics
        # db_painter.drawLine(0, 0, self._droplet.maj/2, 0)
        # db_painter.drawLine(0, 0, 0, self._droplet.min/2)

        #undo transformation
        painter.rotate(-self.tilt_deg)
        painter.translate(-self.x0, -self.y0)


class RobustPolarPolyFit(IFitCurve):
    degree = 3
    def __init__(self, poly, contour, offset=(0,0), side:Side=Side.BOTH):
        self.poly: Polynomial = poly
        super().__init__(contour=contour)
        self.offset = offset
        self.side:Side = side

        self._pen = QPen(Qt.magenta,1.5)
        self._pen.setCosmetic(True)
        self._pen_contour = QPen(Qt.yellow,1)
        self._pen_contour.setCosmetic(False)

    def cart2pol(self, x, y) -> tuple[float,float]:
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (phi, rho)

    def pol2cart(self, phi, rho) -> tuple[float,float]:
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def __getitem__(self, key):
        raise IndexError(f"'{key}' is not a valid index")

    def get_offset_poly(self) -> Polynomial:
        offphi, offr = self.cart2pol(*self.offset)
        shifted_poly = self.poly(Polynomial([-offphi,1])) - [offr]
        return shifted_poly

    def _intersect_line(self, line_m, line_t) -> list[float]:
        # y = mx + t 
        # => https://math.stackexchange.com/questions/2736228/polar-coordinate-function-of-a-straight-line
        # r = - t / ( m cos(phi) - sin(phi))
        line_t -= self.offset[1]

        # function for finding intersection of polynomial and base line
        def f(polar):
            phi, rho = polar
            # solve set of equations
            # eq 1: r = p(phi), where p is the fitted polynomial, 0 = r - p(phi)
            # eq 2: r = - t / (m*cos(phi) - sin(phi)) line equation in polar coords, 0 = r + t / (m*cos(phi) - sin(phi)) 
            z = np.array([rho - self.poly(phi), rho + (line_t / ( line_m*cos(phi) - sin(phi)))])
            return z

        def find_nearest_point(array, point):
            array = np.asarray(array)
            # converting back to  cartesian to calc distance between array points and reference point
            xa, ya = self.pol2cart(array.T[0], array.T[1])
            xp, yp = self.pol2cart(*point)
            dist_array = np.sqrt((xa-xp)**2 + (ya-yp)**2)
            idx = dist_array.argmin()
            return idx

        # first initial guess is contour point closest to baseline
        point = np.asarray(self.cart2pol(*self.contour.get_contact_point()))
        res: OptimizeResult = root(f, point)
        res1: OptimizeResult = root(f, [3, 500.0])
        res2: OptimizeResult = root(f, [0.01, 200.0])

        points = np.vstack([res.x, res1.x, res2.x])
        if not (res.success or res1.success or res2.success):
            raise RuntimeError("Polar Poly Fit: no intersections found")

        # select appropriate intersection according to which side the of the droplet the curve was fitted to
        contr_rhos = self.contour.polar.T[1] # list of rho values
        # search closest intersection point to end of fitted contour (max radius/rho)
        end_of_contour = self.contour.polar[np.argmax(contr_rhos)]
        point = find_nearest_point(points, end_of_contour)
        # if none found, use point with smallest angle

        if point.size == 0:
            point = np.argmin(points.T[0])
            logging.warning(f"PolarFit: Did not find intersection! Found: {str(points)}")

        # if multiple points found per intersection, use index where phi is closest to origin
        elif np.asarray(point).ndim > 0:
            if self.side == Side.LEFT:
                point = np.argmax(points[point].T[0])
            elif self.side == Side.RIGHT:
                point = np.argmin(points[point].T[0])

        return points[point].reshape(1,-1)

    def _slope(self, point) -> float:
        """
        calculates slope of tangent at the point

        :param point: point where the slope will be calculated
        :returns: the slope of the tangent
        """

        phi,_ = point
        r_d = self.poly.deriv(1)(phi)
        r = self.poly(phi)
        m_tan = (r_d*sin(phi) + r*cos(phi))/(r_d*cos(phi) - r*sin(phi))

        return m_tan

    def slope_at_intersect(self, int_line):
        """
        calculates slope of tangent at the intersection with the line

        :param int_line: intersecting line y=mx+t (m,t)
        :returns: the slope of the tangent and intersecting point as tuple (m_tan, [x,y])
        """
        points_int = self._intersect_line(*int_line)
        solutions = []
        for point_int in points_int:
            slope = self._slope(point_int)
            point = self.pol2cart(*point_int)
            trpointx = point[0] + self.offset[0]
            trpointy = point[1] + self.offset[1]
            solutions.append((slope, [trpointx, trpointy]))
        return solutions


    @classmethod
    def fit_contour(cls, contour: Contour, weighted = False, side=None):
        """fit a polynomial to the given contour

        :param contour: the contour to fit
        :type contour: Contour
        :raises ValueError: if fit fails or produces invalid ellipse
        :return: Spline object
        """
        if not isinstance(contour, Contour):
            contour = Contour(contour)
        offset = contour.get_apex()
        
        contour.offset(offset[0] * -1, offset[1] * -1)

        # remove small values
        rho_data = contour.polar.T[1]
        clip_data_idx = np.argwhere(rho_data > 20).squeeze()
        clipped_data = Contour(contour.points[clip_data_idx],is_simple=True)

        # sort by phi coord, remove duplicates
        phi_data = clipped_data.polar.T[0]
        _, sorted_idx = np.unique(phi_data, return_index=True)
        data:np.ndarray = clipped_data.polar[sorted_idx].T
        # the farther from origin the more weight, corresponds to increased weight at droplet base
        min_r = data[1].min()
        weights = data[1]/min_r if weighted else None

        # fitting several orders and compare R2, select best fit
        polys = []
        coeffs = []
        for deg in [3,4,5]:
            fpoly = Polynomial.fit(data[0], data[1], deg=deg, w=weights)
            p = RobustPolarPolyFit(fpoly, clipped_data, offset=offset, side=side)
            polys.append(p)
            coeffs.append(cls.coefficient_of_determination([p]))

        poly = polys[np.argmax(coeffs)]
        return poly

    def square_sum_of_residuals(self, contour=None) -> float:
        """return residuals from fit

        :return: [description]
        :rtype: [type]
        """
        if not contour: contour = self.contour
        phi_data = contour.polar.T[0]
        r_data = contour.polar.T[1]
        p_data = self.poly(phi_data)
        sqr = np.sum((r_data - p_data)**2)
        return sqr


    def total_sum_of_squares(self, contour=None) -> float:
        """calculates the total sum of squares
        for all points of the contour
        Σ (d - m)²

        """
        if not contour: contour = self.contour
        distances = contour.polar.T[1]
        mean_dist = np.mean(distances)

        return np.sum((distances - mean_dist)**2)

    def draw(self, painter:QPainter, img_size):
        """ This function is used to draw the fit function in a Qt paint event handler"""
        painter.translate(*self.offset)
        self.contour.draw(painter, img_size, self._pen_contour)
        painter.setPen(self._pen)
        if self.side == Side.LEFT:
            points = [QPointF(*self.pol2cart(i, self.poly(i))) for i in np.linspace(np.pi/2, np.pi,100)]
        elif self.side == Side.RIGHT:
            points = [QPointF(*self.pol2cart(i, self.poly(i))) for i in np.linspace(0, np.pi/2,100)]
        else:
            points = [QPointF(*self.pol2cart(i, self.poly(i))) for i in np.linspace(0, np.pi,100)]

        painter.drawPolyline(QPolygonF.fromList(points))
