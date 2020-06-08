# -*- coding: utf-8 -*-

import sys,re

import res_rc

from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5.QtCore import  *

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *

# from canvas import Canvas
from singlecanvas import Canvas

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import

class QmySingleColumn(QMainWindow):
    editDone = pyqtSignal()  # 编辑完成
    colorSingal = pyqtSignal()#更改颜色
    ifsave = pyqtSignal(bool)  # 是否保存分割结果
    FIT_WINDOW, MANUAL_ZOOM = 0, 1  # 适应窗口，缩放默认
    CANVAS, SEGCANVAS = 1, 2  # 代表两个窗口
    VIEW, EDITFORE, EDITBACK = 0, 1, 2  # 三种模式，浏览、前景编辑、背景编辑
    def __init__(self,img_path="color.jpg"):
        super().__init__()  # 调用父类构造函数，创建窗体
        self.color = str()
        self.newcolor = str()
        self.transparency = None
        self.is_color_change = False
        self.read_all_parameters()
        self.value = int(self.transparency)
        self.precolor = self.color

        self.mImage = QImage()
        self.colorSingal.connect(self.paintMask)
        self.initUI()
        # self.brushResizeWidget.valueChanged.connect(self.valueChange)
        # self.reload.clicked.connect(self.reload)
        self.curcolor.mousePressEvent = self.get_color

        self.canvas = Canvas()
        self.segCanvas = Canvas()
        self.filename = img_path
        self.savepath = img_path
        self.canvasList = {
            self.CANVAS: self.canvas,
            self.SEGCANVAS: self.segCanvas
        }
        # canvas添加滚动条
        scroll = self.addScroll(self.canvas)
        scroll_ = self.addScroll(self.segCanvas)
        self.scrollBars = {
            Qt.Vertical: (scroll.verticalScrollBar(), scroll_.verticalScrollBar()),
            Qt.Horizontal: (scroll.horizontalScrollBar(), scroll_.horizontalScrollBar())
        }
        # 绑定信号槽，保证两个canvas的scroller动作一致
        self.scrollBars[Qt.Vertical][0].valueChanged.connect(self.scrollBars[Qt.Vertical][1].setValue)
        self.scrollBars[Qt.Horizontal][0].valueChanged.connect(self.scrollBars[Qt.Horizontal][1].setValue)
        self.scrollBars[Qt.Vertical][1].valueChanged.connect(self.scrollBars[Qt.Vertical][0].setValue)
        self.scrollBars[Qt.Horizontal][1].valueChanged.connect(self.scrollBars[Qt.Horizontal][0].setValue)
        # 添加停靠组件
        self.dock = self.addDock(scroll, 'origin')
        self.dock_ = self.addDock(scroll_, 'segmentation')
        self.dock_.hide()
        # 两个dock的顶部不显示关闭选项
        self.dock.setFeatures(self.dock.features() ^ QDockWidget.DockWidgetClosable)
        self.dock_.setFeatures(self.dock_.features() ^ QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_)

        # callbacks
        self.canvas.doubleClickRequest.connect(self.toggleMode)
        self.segCanvas.doubleClickRequest.connect(self.toggleMode)
        self.canvas.brushResizeRequest.connect(self.brushResizeRequest)
        self.segCanvas.brushResizeRequest.connect(self.brushResizeRequest)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.segCanvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.segCanvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.undoSingal.connect(self.toggleActions)
        self.segCanvas.undoSingal.connect(self.toggleActions)
        # self.canvas.drawingSingal.connect(self.paintCanvas)
        # self.segCanvas.drawingSingal.connect(self.paintCanvas)
        self.canvas.movingSingal.connect(self.paintCanvas)
        self.segCanvas.movingSingal.connect(self.paintCanvas)
        self.canvas.drawoutSingal.connect(self.setDirty)
        self.segCanvas.drawoutSingal.connect(self.setDirty)
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.brushResizeWidget.valueChanged.connect(self.paintCanvas)
        self.spinbox_op.valueChanged.connect(self.paintMask)
        self.reload.clicked.connect(self.clearall)
        self.Zoom_In.triggered.connect(self.plusZoom)
        self.Zoom_Out.triggered.connect(self.subZoom)
        self.cancel.clicked.connect(self.closeEvent)
        self.action_undo.triggered.connect(self.undo)
        self.save.clicked.connect(self.saveFile)

        # state
        self.mode = self.VIEW  # 浏览模式
        self.dirty = False  # Whether we need to save or not.分割后或分割结果有变化是为真

        self.zoomMode = self.MANUAL_ZOOM  # 缩放模式为默认
        self.scalers = {  # 两种尺度，适应窗口和默认
            self.FIT_WINDOW: self.scaleFitWindow,
            self.MANUAL_ZOOM: lambda: 1,  # Set to one to scale to 100% when loading files.
        }
        self.begin()  # 初始化

    def initUI(self):
        icon = QIcon(":/icons/images/app.ico")
        self.setWindowIcon(icon)
        self.statusBar().showMessage('modify in single column mode',5000)
        self.setGeometry(300, 300, 1200,600)
        self.setWindowTitle('Human Modify')
        # self.setWindowIcon(QIcon(":/icons/images/app.png"))
        self.toolBar = QToolBar()
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)


        #toolbar items
        self.label = QLabel()
        self.label.setText("Paint Size")
        self.label.setAlignment(Qt.AlignHCenter)

        self.label_op = QLabel()
        self.label_op.setAlignment(Qt.AlignHCenter)
        self.label_op.setText("Transparency")

        self.label_color = QLabel()
        self.label_color.setText("Object Color")
        self.label_color.setAlignment(Qt.AlignHCenter)

        self.object_color = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        self.curcolor = QLabel()
        layout.addWidget(self.curcolor)
        # self.layout.addWidget(self.curcolor)
        # self.layout.setAlignment(Qt.AlignCenter)
        self.curcolor.setFixedSize(30, 30)
        self.curcolor.setStyleSheet("QWidget{background-color:" + self.color + "}")

        self.curcolor.setAlignment(Qt.AlignHCenter)
        self.object_color.setLayout(layout)

        self.brushResizeWidget = QSlider(Qt.Horizontal)
        self.brushResizeWidget.setFocusPolicy(Qt.NoFocus)
        self.brushResizeWidget.setRange(1, 20)
        self.brushResizeWidget.setValue(8)

        self.action_undo = QAction(QIcon(':/icons/images/undo.png'), 'undo', self)
        self.action_undo.setShortcut('Ctrl+Z')
        self.action_undo.setEnabled(False)
        ##blank between buttons

        self.blank1 = QLabel()
        self.blank1.setFixedHeight(20)
        self.blank1.setVisible(True)
        self.blank2 = QLabel()
        self.blank2.setFixedHeight(20)
        self.blank2.setVisible(True)

        # reload
        self.reload = QPushButton()
        self.reload.setText('Reload')
        qssStyle1 = '''
                            QPushButton{
                                background-color: #EEEE00
                            }
                            '''
        self.reload.setStyleSheet(qssStyle1)
        self.reload.setGeometry(QRect(60, 30, 41, 16))

        self.save = QPushButton()
        self.save.setText('Save')
        qssStyle2 = '''
                                    QPushButton{
                                        background-color: YellowGreen
                                    }
                                    '''
        self.save.setStyleSheet(qssStyle2)


        self.cancel = QPushButton()
        self.cancel.setText('Cancel')
        qssStyle3 = '''
                                        QPushButton{
                                            background-color: IndianRed
                                        }
                                        '''
        self.cancel.setStyleSheet(qssStyle3)


        self.Zoom_In = QAction(QIcon(':/icons/images/zoom-in.png'), 'Zoom In', self)
        self.Zoom_In.setShortcut('Ctrl+=')

        self.spinbox_op = QSpinBox()
        self.spinbox_op.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spinbox_op.setSuffix('%')
        self.spinbox_op.setRange(0, 101)
        self.spinbox_op.setValue(self.value)
        self.spinbox_op.setAlignment(Qt.AlignCenter)

        self.zoomWidget = QSpinBox()
        self.zoomWidget.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.zoomWidget.setSuffix('%')
        self.zoomWidget.setRange(1, 1000)
        self.zoomWidget.setValue(100)
        self.zoomWidget.setAlignment(Qt.AlignCenter)


        self.Zoom_Out = QAction(QIcon(':/icons/images/zoom-out.png'), 'Zoom Out', self)
        self.Zoom_Out.setShortcut('Ctrl+-')

        self.toolBar.addWidget(self.label)
        self.toolBar.addWidget(self.brushResizeWidget)
        self.toolBar.addAction(self.action_undo)
        self.toolBar.addWidget(self.reload)
        self.toolBar.addWidget(self.blank1)
        self.toolBar.addWidget(self.save)
        self.toolBar.addWidget(self.blank2)
        self.toolBar.addWidget(self.cancel)
        self.toolBar.addWidget(self.object_color)
        self.toolBar.addWidget(self.label_color)
        self.toolBar.addWidget(self.spinbox_op)
        self.toolBar.addWidget(self.label_op)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.Zoom_In)
        self.toolBar.addWidget(self.zoomWidget)
        self.toolBar.addAction(self.Zoom_Out)

        self.addToolBar(Qt.LeftToolBarArea, self.toolBar)


##  ==============自定义功能函数========================
    def read_all_parameters(self):
        try:
            self.setting_file = open("./config.txt", "r")
        except Exception as err:
            print("open file error:{}".format(err))
        try:
            self.size_of_input = self.setting_file.readline()[:-1]
            self.overlap_size = self.setting_file.readline()[:-1]
            self.batch_size = self.setting_file.readline()[:-1]
            gpu_idx = self.setting_file.readline()[:-1]
            self.mean = self.setting_file.readline()[:-1]
            self.std = self.setting_file.readline()[:-1]
            tta = self.setting_file.readline()[:-1]

            post = self.setting_file.readline()[:-1]

            self.unet_path = self.setting_file.readline()[:-1]
            self.wpunet_path = self.setting_file.readline()[:-1]
            self.color = self.setting_file.readline()[:-1]
            self.transparency = self.setting_file.readline()[:-1]

            self.aug_type = int(tta)
            self.use_post_process = int(post)
            self.gpu_idx = int(gpu_idx)

        except Exception as err:
            print("Parameters error0".format(err))
        self.setting_file.close()
    def initstatus(self,color, transparency):
        self.color = color

        self.transparency = transparency
        self.spinbox_op.setValue(int(transparency))
        self.curcolor.setStyleSheet("QWidget{background-color:" + self.color + "}")

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0 # So that no scrollbars are generated.
        w1 = self.dock.width() - e
        h1 = self.dock.height() - e
        a1 = w1/ h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        #logging.info('dock_w:%d h:%d \n canvas_w:%d h:%d'%(w1,h1,w2,h2))
        return w1 / w2 if a2 >= a1 else h1 / h2
    def loadFile(self,filename): #显示原图
        pixmap0 = QPixmap()
        pixmap0.load(filename)

        self.canvas.image = pixmap0.toImage()
        self.canvas.loadPixmap(pixmap0)

    def loadResult(self,img): #显示分割图
        self.segCanvas.image = img
        self.mImage = img
        pix = QPixmap.fromImage(img)

        self.segCanvas.loadPixmap(pix)

    def loadPath(self,filename): #输出路径
        self.savepath = filename

    def loadOrigin(self,img): #显示原图
        self.canvas.image = img
        pix = QPixmap.fromImage(img)

        self.canvas.loadPixmap(pix)
    def loadMask(self,img): #显示原图

        pix = QPixmap.fromImage(img)
        # p1 = QPainter(pix)
        # p1.setCompositionMode(QPainter.CompositionMode_Source)
        # p1.drawPixmap(0, 0, self.maskPixmap)
        self.canvas.loadMaskPixmap(pix)
    def begin(self):
        self.setClean()
        self.mode=self.VIEW
        self.canvas.mode=self.segCanvas.mode=self.mode
        self.segCanvas.setEnabled(False)
        self.save.setEnabled(False)
        self.loadFile(self.filename)
        self.dock_.hide()
        im = QImage()
        im.load(self.filename)
        self.loadResult(im)
    def setClean(self):
        self.dirty=False
        self.save.setEnabled(False)
    def addDock(self,widget, title):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        return dock
    def addScroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll
    def addZoom(self, increment):  #滚轮缩放
        v1 = self.zoomWidget.value() + increment
        self.zoomWidget.setValue(v1)
    def subZoom(self):  #使用工具栏按钮缩小
        v2 = self.zoomWidget.value() - 10
        self.zoomWidget.setValue(v2)
    def plusZoom(self): #使用工具栏按钮放大
        v3 = self.zoomWidget.value() + 10
        self.zoomWidget.setValue(v3)
    def mayContinue(self):
        # canvas是否有未保存变化，是否保存
        return not (self.dirty and self.action_undo.isEnabled() and not self.discardChangesDialog())
    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = 'Do you want to quit modification without saving result?'
        return yes == QMessageBox.warning(self, 'Attention', msg, yes | no)

##  ==============event处理函数==========================
# 利用对话框获取颜色
    def get_color(self, e):
        colordialog = QColorDialog.getColor()
        if colordialog.isValid():
            self.newcolor = colordialog.name()

            self.curcolor.setStyleSheet("QWidget{background-color:" + self.newcolor + "}")
            self.is_color_change = True
            self.colorSingal.emit()
    ##关闭窗口
    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        else:
            self.editDone.emit()

##  ==========由connectSlotsByName()自动连接的槽函数============


##  =============自定义槽函数===============================
    def scrollRequest(self, delta, orientation,drag=False):
        #logging.info("scrollrequest %d" %delta)
        bars = self.scrollBars[orientation]
        if not drag:
            units = - delta * 0.01  # natural scroll
            for bar in bars:
                bar.setValue(bar.value() + bar.singleStep()*units)
        else:
            units=delta*0.25 #drag Scroll
            for bar in bars:
                bar.setValue(bar.value()+units)



    def toggleMode(self, value=None):#改变操作模式

        self.mode = value if value is not None else self.mode  # 更新mode
        self.canvas.mode = self.mode
        self.segCanvas.mode = self.mode

        self.canvas.modeChanged()
        self.paintCanvas()

    def brushResizeRequest(self, delta):
        units = self.brushResizeWidget.value() + delta * 0.025
        self.brushResizeWidget.setValue(units)

    def toggleActions(self):#undo状态
        if Canvas.shapes:
            self.action_undo.setEnabled(True)
        else:
            self.action_undo.setEnabled(False)
            self.save.setEnabled(False)

    def paintCanvas(self):
        # logging.info('zoom %d'%self.zoomWidget.value())
        Canvas.scale = 0.01 * self.zoomWidget.value()
        Canvas.point_size = self.brushResizeWidget.value()
        for canvas in self.canvasList.values():
            # assert not canvas_.image.isNull(), "cannot paint null image"
            canvas.adjustSize()
            canvas.update()
    def paintMask(self):

        # logging.info('zoom %d'%self.zoomWidget.value())
        try:
            a = (1-self.spinbox_op.value() * 0.01) * 255

            if self.is_color_change:
                text = self.newcolor.replace('#', '')
                textArr = re.findall(r'.{2}', text)

                r = int(textArr[0], 16)
                g = int(textArr[1], 16)
                b = int(textArr[2], 16)

                self.qcolor = QColor(r, g, b, a)
                self.precolor = self.newcolor

            else:
                text = self.precolor.replace('#', '')
                textArr = re.findall(r'.{2}', text)

                r = int(textArr[0], 16)
                g = int(textArr[1], 16)
                b = int(textArr[2], 16)
                self.qcolor = QColor(r, g, b, a)

            width = self.mImage.width()
            height = self.mImage.height()
            image2 = self.mImage.convertToFormat(QImage.Format_ARGB32)

            for x in range(width):
                for y in range(height):
                    if (image2.pixel(x, y) == 0xFF000000):
                        image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                    else:
                        image2.setPixelColor(x, y, self.qcolor)
                    #     image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                    # else:
                    #     image2.setPixelColor(x, y, self.qcolor)
            pix = QPixmap.fromImage(image2)

            self.is_color_change = False
            # p1 = QPainter(pix)
            # p1.setCompositionMode(QPainter.CompositionMode_Source)
            # p1.drawPixmap(0, 0, self.maskPixmap)
            self.canvas.loadMaskPixmap(pix)
        except Exception as e:
            print(e)
        # for canvas in self.canvasList.values():
        #     # assert not canvas_.image.isNull(), "cannot paint null image"
        #     # canvas.adjustSize()
        #     canvas.update()
    def zoomRequest(self, delta):
        # delta是滚轮滚动角度，正负之分
        units = delta * 0.1
        self.addZoom(units)
    def setDirty(self):
        self.dirty = True
        self.save.setEnabled(True)

    def undo(self): #撤销
        if Canvas.shapes:
            shape=Canvas.shapes.pop()
            Canvas.undoshapes.append(shape)
            self.toggleActions()
            self.paintCanvas()
    def clearall(self): #reload,清空所有痕迹
        if not self.action_undo.isEnabled():
            self.zoomWidget.setValue(100)
        else:
            while self.action_undo.isEnabled():
                self.undo()
            self.zoomWidget.setValue(100)

    def saveFile(self):
        # if self.segCanvas.image and filename:

        mask, lbl = self.segCanvas.mask2image()  # mask痕迹转化为image；lbl为全黑背景仅保留痕迹的图片，用于提取finetunedata
        mask.save(self.savepath)
        self.statusBar().showMessage('Successfully saved in' + " " + self.savepath,5000)
        self.setClean()
        self.ifsave.emit(True)


##  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QmySingleColumn()  # 创建窗体
    form.show()
    sys.exit(app.exec_())


