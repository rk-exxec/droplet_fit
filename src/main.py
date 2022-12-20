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

################################
# adjust these values
filename = 'test_polar.png'
baseline = 650
mask = [674,-78, 120, 1049]
contour_lim = (4,400)
fit_type = "polar"

################################
# GUI code
import sys
import numpy as np
import cv2

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QApplication
from PySide6.QtGui import QPixmap, QPaintEvent, QPainter, QImage, Qt, QBrush, QPen
from droplet import Droplet

from evaluate_droplet import evaluate_droplet

class Window(QMainWindow):
    def __init__(self, image, droplet):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.setMinimumWidth(640)
        self.setMinimumHeight(480)
        h,w = image.shape[:2]
        self.image = QImage(image, w, h, w, QImage.Format_Grayscale8)
        self.scaled = self.image.scaled(self.size(), aspectMode=Qt.KeepAspectRatio, mode=Qt.SmoothTransformation)
        self._pixmap = QPixmap.fromImage(self.scaled)
        self.drop = droplet
        self._double_buffer = None

    
    def get_from_image_transform(self):
        """ 
        Gets the scale and offset for a Image to Widget coordinate transform
        """
        pw, ph = self.scaled.size().toTuple()              # scaled image size
        iw, ih = self.image.size().toTuple()                # original size of image
        cw, ch = self.size().toTuple()                      # display container size
        scale_x = float(pw) / float(iw)
        offset_x = abs(pw - cw)/2.0
        scale_y = float(ph) / float(ih)
        offset_y = abs(ph -  ch)/2.0
        return scale_x, scale_y, offset_x, offset_y

    def paintEvent(self, event: QPaintEvent):
        """
        custom paint event to 
        draw camera stream and droplet approximation if available

        uses double buffering to avoid flicker
        """
        # completely override super.paintEvent() to use double buffering
        painter = QPainter(self)
        
        buf = self.doubleBufferPaint(self._double_buffer)
        # painting the buffer pixmap to screen
        painter.drawImage(0, 0, buf)
        painter.end()

    def doubleBufferPaint(self, buffer=None):
        if buffer is None:
            buffer = QImage(self.width(), self.height(), QImage.Format_RGB888)
        buffer.fill(Qt.black)
        # calculate offset and scale of droplet image pixmap
        scale_x, scale_y, offset_x, offset_y = self.get_from_image_transform()

        db_painter = QPainter(buffer)
        db_painter.setRenderHints(QPainter.Antialiasing)
        db_painter.setBackground(QBrush(Qt.black))
        db_painter.setPen(QPen(Qt.black,0))
        db_painter.drawPixmap(offset_x, offset_y, self._pixmap)
        pen = QPen(Qt.magenta,2)
        pen_contour = QPen(Qt.blue,1)
        pen.setCosmetic(True)
        drp = self.drop
        db_painter.setPen(pen)
        # draw droplet outline and tangent only if evaluate_droplet was successful
        if drp.is_valid:
            try:
                # transforming true image coordinates to scaled pixmap coordinates
                db_painter.translate(offset_x, offset_y)
                db_painter.scale(scale_x, scale_y)

                # draw contour
                if drp.has_contour:
                    drp.contour.draw(db_painter,self.size().toTuple(), pen_contour)
          
                db_painter.setPen(pen)

                if drp.has_angle:
                    # drawing tangents and baseline
                    db_painter.drawLine(*drp.line_l)
                    db_painter.drawLine(*drp.line_r)
                    db_painter.drawLine(*drp.int_l, *drp.int_r)

                if drp.has_fit:
                    if len(drp.fit_curve) == 1:
                        drp.fit_curve[0].draw(db_painter, self.size().toTuple())
                    else:
                        # if more than one fit, use clipping rect to prevent overlapping
                        for i, crv in enumerate(drp.fit_curve):
                            db_painter.save()
                            db_painter.setClipRect(i*drp.divider, 0, (i+1)*drp.divider, self.height()/scale_y)
                            # draw contour which the ellipse fit used
                            
                            crv.draw(db_painter, self.size().toTuple())
                            db_painter.restore()

                        db_painter.setClipRect(0, 0, self.width(), self.height())
                
            except Exception as ex:
                print(ex)
        db_painter.end()
        return buffer

    
# main function:
if __name__ == "__main__":
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im = np.reshape(im, im.shape + (1,) )
    (h,w,d) = np.shape(im)
    drp = Droplet()
    try:
        drp = evaluate_droplet(im, baseline, mask=mask, contour_lim=contour_lim, fit_type=fit_type, weighted_fit=False)
        print(drp)
    except Exception as ex:
        print(ex)

    app = QApplication()
    window = Window(im, drp)
    window.show()
    sys.exit(app.exec())

