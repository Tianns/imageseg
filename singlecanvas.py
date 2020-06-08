import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from shape import Shape
from lib import distance
from PIL import Image,ImageDraw
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT   = Qt.PointingHandCursor
CURSOR_DRAW    = Qt.CrossCursor
CURSOR_MOVE    = Qt.ClosedHandCursor
CURSOR_GRAB    = Qt.OpenHandCursor
DEFAULT_BACK_COLOR = QColor(0, 0, 0)
DEFAULT_FORE_COLOR = QColor(255, 255, 255)
DEFAULT_VIEW_COLOR = QColor(0, 255, 0)
DEFAULT_EDIT_COLOR = QColor(255, 255, 0)

class Canvas(QWidget):
    doubleClickRequest=pyqtSignal(int) #双击改变模式的信号
    zoomRequest=pyqtSignal(int) #浏览模式滚轮缩放image信号
    scrollRequest=pyqtSignal(int,int,bool)#编辑模式滚轮滑动image信号
    brushResizeRequest=pyqtSignal(int) #Ctrl+滚轮缩放brushSize信号
    #drawingSingal=pyqtSignal()#正在画
    drawoutSingal=pyqtSignal()#画完了
    movingSingal=pyqtSignal()#cursor following 、update canvas
    undoSingal=pyqtSignal()#check undo state
    VIEW,EDITFORE,EDITBACK,EDIT=0,1,2,3 #三种模式，浏览、前景编辑、背景编辑
    shapes=[]
    undoshapes=[]
    scale=1.0
    point_size=8
    current=None
    cursorPoint=Shape()
    cursorStyle=CURSOR_DEFAULT
    def __init__(self):
        super(Canvas,self).__init__()
        self.mode=self.VIEW
        self.colors={
            self.EDITFORE:DEFAULT_FORE_COLOR,
            self.EDITBACK:DEFAULT_BACK_COLOR,
            self.VIEW:DEFAULT_VIEW_COLOR,
            self.EDIT: DEFAULT_EDIT_COLOR

        }
        self.mousePressed=False
        self.isLeftPressed = False
        self.isRightPressed = False
        self.pressedPt=QPoint()
        self.image=None
        self.pixmap = QPixmap()##移植
        self._painter = QPainter()##移植
        self.mpixmap = QPixmap()

        self.setMouseTracking(True)
    def enterEvent(self, ev):
        self.overrideCursor(CURSOR_DRAW)
    def leaveEvent(self, ev):
        self.restoreCursor()
        self.cursorPoint.clear()
        self.movingSingal.emit()

    def mouseDoubleClickEvent(self, event):
        '''【双击】
           浏览模式下，左键双击编辑前景，右键双击编辑背景；
           编辑模式下，双击结束
           '''
        if self.mode==self.VIEW:
            self.mode=self.EDIT
        else:
            self.mode=self.VIEW
        #logging.info("mode:%s;button:%s"%(self.mode,event.button()))
        self.doubleClickRequest.emit(self.mode)

    #需要设置sizeHint，从而保证滚动视图正常显示
    # These two, along with a call to adjustSize are required for the scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()
    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        mods = ev.modifiers()  # 判断按下了哪些修饰键（Shift,Ctrl , Alt,等等）
        delta = ev.angleDelta()  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        #logging.info(delta.y())
        if Qt.ControlModifier == int(mods):  # with Ctrl/Command key
            #brush Resize
            self.brushResizeRequest.emit(delta.y())
        else:
            self.zoomRequest.emit(delta.y())
            # if self.mode==self.VIEW:
            #     #zoom
            #     self.zoomRequest.emit(delta.y())
            # else:
            #     # scroll
            #     self.scrollRequest.emit(delta.x(), Qt.Horizontal,False)
            #     self.scrollRequest.emit(delta.y(), Qt.Vertical,False)

        ev.accept()
        
    def mousePressEvent(self, ev):
        self.mousePressed=True
        self.pressedPt=self.transformPos(ev.pos())
        if ev.buttons() == Qt.LeftButton:
            self.isLeftPressed = True
        else:
            self.isRightPressed = True
    def mouseReleaseEvent(self, ev):
        self.mousePressed=False
        self.isLeftPressed = False
        self.isRightPressed = False
        if self.current:
            self.shapes.append(self.current)

            Canvas.current=None
            self.undoshapes.clear()
            self.drawoutSingal.emit()
            self.undoSingal.emit()
    def mouseMoveEvent(self, ev):
        pos=self.transformPos(ev.pos())
        Canvas.cursorPoint.clear()
        try:
            if self.mousePressed:##单击
                if self.mode==self.VIEW:##浏览
                    delta=self.pressedPt-pos##滑动，相当于滚动条
                    if delta.x()!=0:
                        self.scrollRequest.emit(delta.x(), Qt.Horizontal,True)
                    if delta.y()!=0:
                        self.scrollRequest.emit(delta.y(), Qt.Vertical,True)
                else:
                    if self.isLeftPressed:
                        self.mode = self.EDITFORE
                    else:
                        self.mode = self.EDITBACK

                    if self.outOfPixmap(pos):
                        return
                    else:
                        self.drawing(self.pressedPt)##编辑模式下，划线

                self.pressedPt = pos
            else:
                Canvas.cursorPoint.addPoint(pos)
                Canvas.cursorPoint.setSize(self.point_size)
                Canvas.cursorPoint.setColor(self.colors[self.mode])
            self.movingSingal.emit()
        except Exception as e:
            print(e)

    def drawing(self,p):

        pos=p
        Canvas.current = self.current if self.current else Shape(self.colors[self.mode], self.point_size)
        #pos_ = self.current[-1] if len(self.current) else QPoint(self.pixmap.width()/2,self.pixmap.height()/2)

        try:
            if self.outOfPixmap(pos):##超出图片的部分无效
                pos = self.intersectionPoint(self.current[-1], pos)##求与界面的交点

            self.current.addPoint(pos)##把交点位置加进来
        except Exception as e:
            print(e, file=sys.stderr)
            return


    # 这是怎么求得交点~相当厉害了
    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        point_size = self.point_size / (2.0 * self.scale)
        size = self.pixmap.size()
        # points = [(0,0),
        #           (size.width(), 0),
        #           (size.width(), size.height()),
        #           (0, size.height())]
        points = [(point_size, point_size),
                  (size.width()-point_size, point_size),
                  (size.width()-point_size, size.height()-point_size),
                  (point_size, size.height()-point_size)]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i+1)%4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else: # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)
    def intersectingEdges(self, point1, point2, points):
        """For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen."""
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i+1) % 4]
            denom = (y4-y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)
            nub = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4)/2, (y3 + y4)/2)
                d = distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        point_size=self.point_size/(2.0*self.scale)
        return not ( point_size<= p.x()<= w-point_size and point_size<= p.y()<= h-point_size)

    def loadPixmap(self, pixmap):
        self.resetState()
        self.pixmap = pixmap

    def loadMaskPixmap(self, pixmap):
        self.resetState()
        self.mpixmap = pixmap


        return self.mpixmap
    def modeChanged(self):
        if Canvas.current:
            Canvas.shapes.append(Canvas.current)
            Canvas.current=None
        Canvas.cursorPoint.clear()
    def resetState(self):
        '''
        清空画笔痕迹
        '''
        self.restoreCursor()
        self.shapes.clear()
        self.undoshapes.clear()
        Canvas.current=None
        Canvas.cursorPoint.clear()
        self.undoSingal.emit()
        #self.update()

    def paintEvent(self, event):##继承重写，渲染

        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)
        p = self._painter##画笔

        #logging.info(p.device())
        #logging.info(self.image)
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(0, 0, self.pixmap)##加载图片
        p.drawPixmap(0, 0, self.mpixmap)

        # p1.begin(self)
        # p1.setRenderHint(QPainter.Antialiasing)
        # p1.setRenderHint(QPainter.HighQualityAntialiasing)
        # p1.setRenderHint(QPainter.SmoothPixmapTransform)
        # p1.scale(self.scale, self.scale)
        # p1.translate(self.offsetToCenter())
        # p1.drawPixmap(0, 0, self.mpixmap)  ##加载图片
        #p.drawImage(0,0,self.image)
        Shape.scale = self.scale
        for shape in self.shapes:##轮廓
            shape.paint(p)

        if self.current:
            Canvas.current.paint(p)
        if Canvas.cursorPoint:
            Canvas.cursorPoint.paint(p)
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw-w)/(2*s) if aw > w else 0
        y = (ah-h)/(2*s) if ah > h else 0
        return QPointF(x, y)

    def overrideCursor(self, cursor):
        self.restoreCursor()
        Canvas.cursorStyle = cursor
        QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    #先把前景和背景的痕迹画到一个全黑图上，找到改动的范围getbbox，扩大成200*200的整数倍
    #依次
    def drawshapes(self,image,shapes,channel=1):##把痕迹和图片结合
        draw = ImageDraw.Draw(image)##imagedraw是pil里的，image也要是pil格式
        for shape in shapes:##shapes记录画笔痕迹
            c=shape.color.getRgb()[0]
            color = 1 if channel==1 else (c,c,c)#'red' if shape.color==DEFAULT_FORE_COLOR else 'blue'
            #logging.info(color)
            width = shape.point_size
            points=shape.points
            r=width/2.0
            for i in range(len(points)-1):
                draw.ellipse((points[i].x()-r, points[i].y()-r,points[i].x()+r, points[i].y()+r),fill=color)
                line= [points[i].x(), points[i].y(), points[i + 1].x(), points[i + 1].y()]
                draw.line(line, fill=color,width=width)
            draw.ellipse((points[-1].x() - r, points[-1].y() - r, points[-1].x() + r, points[-1].y() + r), fill=color)
    def mask2image(self):##
        mask= Image.fromqimage(self.image)##把self.image转化为pil格式，在mask上drawimage
        self.drawshapes(mask, self.shapes,3)
        lbl=np.zeros(mask.size,np.uint8)
        lbl=Image.fromarray(lbl)
        self.drawshapes(lbl, self.shapes,1)
        return mask,lbl