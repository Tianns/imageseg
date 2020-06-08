from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Shape(object):
    P_ROUND=1
    point_type=P_ROUND
    point_size = 8
    scale = 1.0
    def __init__(self,color=None,size=1):
        self.points=[]
        self.color = color
        self.point_size=size
    def addPoint(self,point):
        self.points.append(point)
    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None
    def clear(self):
        self.points.clear()
    def setSize(self,size):
        self.point_size=size
    def setColor(self,color):
        self.color=color
    def paint(self,painter):
        if self.points:
            pen=QPen(self.color)
            pen.setWidth(max(1, int(round(self.point_size / self.scale))))
            pen.setCapStyle(Qt.RoundCap)
            #pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            path=QPainterPath()
            path.moveTo(self.points[0])
            if len(self.points)==1:
                self.addToPath(path, 0)
                painter.fillPath(path,self.color)
            else:
                for i,p in enumerate(self.points):
                    path.lineTo(p)
                painter.drawPath(path)

    def addToPath(self,path,i):
        d=self.point_size/self.scale
        type=self.point_type
        point=self.points[i]
        if type==self.P_ROUND:
            path.addEllipse(point,d/2.0,d/2.0)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value