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
from PySide6.QtGui import QPainter, QPen, Qt
import numpy as np

import cv2

class ContourError(Exception):
    pass

class BoundingRect():
    """Stores a bounding rect, origin, size, and each point in clockwise order
    """
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.a = (x,y)
        self.b = (x+w,y)
        self.c = (x,y+h)
        self.d = (x+w,y+h)
        self.points = (self.a, self.b, self.c, self.d)

class Contour():
    """ Represents the set of points that describe the contour af an object
    
    This class gives multiple utility functions to ease the working with cv2 contours and integrate them into Qt applications
    """
    def __init__(self, contour: np.ndarray, is_simple=False):
        """initialize contour object with array of points

        :param contour: the array of points, in opencv format (shape (*,1,2)), if is_simple is True in shape (*,2)
        :type contour: ndarray with shape (*,1,2) or (*,2)
        :param is_simple: [description], defaults to False
        :type is_simple: bool, optional
        """
        self.cv2_contour :np.ndarray = None
        self.contour_points :np.ndarray = None
        if not is_simple:
            self.cv2_contour = contour
            self.contour_points = contour.reshape(-1,2)
        else:
            self.cv2_contour = contour.reshape(-1,1,2)
            self.contour_points = contour

        self._pen = QPen(Qt.blue, 1)
        self._pen.setCosmetic(True)
        self._bounding_rect: BoundingRect = None

    def __getitem__(self,key):
        return self.points[key]

    def __add__(self, o):
        return self.append(o)

    @property
    def bounding_rect(self):
        if self._bounding_rect == None:
            self._bounding_rect = BoundingRect(*cv2.boundingRect(self.cv2_contour))
        return self._bounding_rect

    @property
    def cv2(self):
        """contour as np.ndarray in the correct format for opencv 
        every element of type [[int,int]]
        """
        return self.cv2_contour

    @cv2.setter
    def cv2(self, value: np.ndarray):
        if np.shape(value[0]) != (1,2):
            raise ValueError("Contour.cv2 only accepts arrays in the shape of (:,1,2)!")
        self.cv2_contour = value
        self.contour_points = value.reshape(-1,2)

    @property
    def points(self) -> np.ndarray:
        """contour as list of points
        every element of type [int,int]
        """
        return self.contour_points

    @points.setter
    def points(self, value: np.ndarray):
        if np.shape(value[0]) != (2,):
            raise ValueError("Contour.points only accepts arrays in the shape of (:,2)!")
        self.contour_points = value
        self.cv2_contour = value.reshape(-1,1,2)

    @property
    def polar(self) -> np.ndarray:
        x = self.get_x_values()
        y = self.get_y_values()
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        pol_coords = np.array([phi,rho]).T
        return pol_coords

    def get_apex(self) -> tuple[float,float]:
        """returns point, where contour is lowest, which equals the droplet apex
        """
        return self.points[np.argmin(self.points.T[1])].copy()

    def get_contact_point(self) -> tuple[float,float]:
        """returns point, where contour is closest to baseline
        """
        return self.points[np.argmax(self.points.T[1])].copy()

    def get_x_values(self) -> np.ndarray:
        """return array of all x values of the contour
        """
        return self.points.T[0]

    def get_y_values(self) -> np.ndarray:
        """return array of all y values of the contour
        """
        return self.points.T[1]

    def shorten(self, baseline:int, limits: tuple(int,int)):
        """cuts the contour: keep only points that are within the limits 
        :param limits: interval of distances from max y value of contour (the baseline of the droplet) as tuple

        :returns: shortened contour
        """
        y_vals = self.contour_points.T[1]
        short_contour = self.contour_points[np.where(np.logical_and(y_vals > (baseline - limits[1]), y_vals < (baseline - limits[0])))]
        return Contour(short_contour,True)

    def append(self, contour: np.ndarray):
        """appends contour to another by merging the underlying numpy arrays with concatenation and then creates new Contour object from that data

        :param contour: contour to append, in opencv format (*,1,2)
        :type contour: np.ndarray
        """
        return Contour(np.concatenate(self.cv2, contour))

    def offset(self, x = 0, y = 0):
        self.points += np.array([x,y])

    def simplify(self):
        """this function tries to remove any contour atifacts that are not part of the droplet
        by finding the triple point by its sharp corners and cutting them off

        :return: contour without artifacts
        """
        # approximate contour with polygon
        poly:np.ndarray	 = cv2.approxPolyDP(self.cv2, 4, False)
        poly = poly.reshape(-1,2)
        poly = poly[poly[:,0].argsort()]
        cutoff = []
        # find sharp bends in polygon, indicating droplet contact points to surface
        for i in range(len(poly)-2):
            x1,y1 = poly[i]
            x2,y2 = poly[i+1]
            x3,y3 = poly[i+2]
            vec1 = [x2-x1,y2-y1]
            vec2 = [x3-x2,y3-y2]
            uvec1 = vec1 / np.linalg.norm(vec1)
            uvec2 = vec2 / np.linalg.norm(vec2)
            dot_product = np.dot(uvec1, uvec2)
            angle = np.arccos(dot_product)
            if abs(angle) < 3*np.pi/4:
                cutoff.append(poly[i+1])
                break

        # cut contour at the bend points, leaving only the actual droplet contour
        idx = []
        for cutoff_point in cutoff:
            for j in range(len(self.points)):
                if self.points[j][0] == cutoff_point[0] and self.points[j][1] == cutoff_point[1]:
                    idx.append(j)
                    break
        parts = np.split(self.cv2, idx, axis=0)
        final_contour = Contour(max(parts, key=len))
        return final_contour,poly

    def draw(self, painter: QPainter, size, pen:QPen=None):
        """ This function is used to draw the contour in a Qt paint event handler"""
        painter.setPen(self._pen if pen is None else pen)
        for point in self.points:
            painter.drawPoint(*point)
