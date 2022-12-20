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

from math import degrees
from PySide6.QtCore import QSettings
import logging
import inspect

from numpy.lib.function_base import angle

from contour import Contour
from fit_classes import IFitCurve

class Droplet:
    """
    provides a singleton storage structure for droplet information and conversion functions  

    saves droplet scale with QSettings

    Has the following attributes:

    - **is_valid**: whether contained data is valid
    - **r2**: the coefficient tof determination of the droplet fit
    - **angle_l**, **angle_r**: the left and right tangent angles
    - **has_fit**: indicates whether the fit was successfull
    - **has_angle**: indicates whether a contact angle could be calculated from fit
    - **has_contour**: indicates whether droplet contour was found
    - **fit_curve**: the IFitCurve objects that were used to fit droplet
    - **fit_type**: name of the fitting curve type
    - **fit_class**: class of the fitting type
    - **contour**: the detected droplet contour
    - **divider**: if droplet halves were fit separetely, this indicates the x value where it was split
    - **tan_l_m**, **tan_r_m**: slope of left and right tangent
    - **int_l**, **int_r**: left and right intersections of ellipse with baseline
    - **line_l**, **line_r**: left and right tangent as 4-Tuple (x1,y1,x2,y2)
    - **base_diam**: diameter of the contact surface of droplet
    - **scale_px_to_mm**: scale to convert between px and mm, is loaded from storage on startup
    """
    def __init__(self):
        self.is_valid       : bool                  = False
        self.r2             : float                 = 0
        self.angle_l        : float                 = 0.0
        self.angle_r        : float                 = 0.0
        self.has_angle                              = False
        self.fit_curve      : list[IFitCurve]       = None
        self.fit_type       : str                   = ""
        self.fit_class      : IFitCurve             = None
        self.has_fit                                = False
        self.divider        : int                   = 0
        self.tan_l_m        : int                   = 0
        self.int_l          : tuple[int,int]        = (0,0)
        self.line_l         : tuple[int,int,int,int] = (0,0,0,0)
        self.tan_r_m        : int                   = 0
        self.int_r          : tuple[int,int]        = (0,0)
        self.line_r         : tuple[int,int,int,int] = (0,0,0,0)
        self.base_diam      : int                   = 0
        self.contour        : Contour               = None          
        self.has_contour                            = False
        settings                                    = QSettings()
        var_scale = settings.value("droplet/scale_px_to_mm", 0.0)
        self.scale_px_to_mm : float                 = float(var_scale) if var_scale else None # try to load from persistent storage
        self.error          :str                    = ""

    def __str__(self) -> str:
        if self.is_valid:
            # if scalefactor is present, display in metric, else in pixles
            if self.scale_px_to_mm is None or self.scale_px_to_mm <= 0:
                ret = inspect.cleandoc(f'''
                    Angle Left:
                    {round(self.angle_l,1):.1f}°
                    Angle Right:
                    {round(self.angle_r,1):.1f}°
                    Surface Diam:
                    {round(self.base_diam):.2f} px
                    R²:
                    {round(self.r2,5)}
                    '''
                )
            else:
                ret =  inspect.cleandoc(f'''
                    Angle Left:
                    {round(self.angle_l,1):.1f}°
                    Angle Right:
                    {round(self.angle_r,1):.1f}°
                    Surface Diam:
                    {round(self.base_diam_mm,2):.2f} mm
                    R²:
                    {round(self.r2,5)}
                    '''   
                )
        else:
            ret = 'No droplet!'
            if self.error:
                ret = f'{ret}\n{self.error}'
        
        return ret

    @staticmethod
    def delete_scale():
        settings = QSettings()
        settings.remove("droplet")

    @property
    def base_diam_mm(self):
        """ droplet contact surface width in mm

        .. seealso:: :meth:`set_scale` 
        """
        if self.scale_px_to_mm == 0: raise ValueError("no pixel scale set")
        return self.base_diam * self.scale_px_to_mm

    @staticmethod
    def set_scale(scale):
        """ set and store a scalefactor to calculate mm from pixels

        :param scale: the scalefactor to calculate mm from px
        
        .. seealso:: :meth:`camera_control.CameraControl.calib_size` 
        """
        logging.info(f"droplet: set scale to {scale}")
        # save in persistent storage
        settings = QSettings()
        settings.setValue("droplet/scale_px_to_mm", scale)