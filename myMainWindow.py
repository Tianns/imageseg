# -*- coding: utf-8 -*-

import logging,sys
import re
import skimage
from PIL import Image

from PyQt5.QtGui import QImage,QPixmap,QColor
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox,QFileDialog,QListWidgetItem
from PyQt5.QtCore import  pyqtSlot, Qt, QDir, QFileInfo

from skimage import filters
from Unet import *
from clahe import clahe
# from numpy import *
from process import *
from doubleThreshold import double_thresold
from myDoubleColumn import QmyDoubleColumn
from mySingleColumn import QmySingleColumn
from setting import setting
from ui_MainWindow import Ui_MainWindow

logging.basicConfig(level=logging.INFO)

class QmyMainWindow(QMainWindow):
    pathChanged = pyqtSignal(str) #DoubleThreshold传入原图路径
    fileTrans = pyqtSignal(str)  # modify传入原图路径
    labelTrans = pyqtSignal(QImage)  #传输分割图
    maskTrans = pyqtSignal(QImage)  #传输遮罩图
    originTrans = pyqtSignal(QImage)  # 修改后的原图
    outdirTrans = pyqtSignal(str)  # 传入输出路径
    initStatus = pyqtSignal(str,str)  # 传入color,transparency
    def __init__(self, parent=None):
        super().__init__(parent)   # 调用父类构造函数，创建窗体
        self.ui=Ui_MainWindow()    # 创建UI对象
        self.ui.setupUi(self)      # 构造UI界面

        self.setWindowTitle("Image Segmentation based on deep learning")
        self.showMaximized()
        self.ui.statusBar.showMessage("system started",5000)
        self.curPixmap = QPixmap()   #图片
        self.topPixmap = QPixmap()  # 遮罩图片
        self.scaled = QPixmap()
        
        self.seg = QImage()#分割图
        self.maskimg = QImage()  # 遮罩图
        self.new_origin = QImage()#修改后的原图
        self.is_segment_img = False #是否分割
        self.is_origin_change = False #原图是否修改
        self.is_save_segment = False #分割图是否保存
        self.is_save_origin = False #修改后的原图是否保存
        self.is_enter_modify = False #进入人工标注
        self.is_multi = 0 #多图模式
        self.stuple = (self.is_segment_img,self.is_origin_change,self.is_save_segment,self.is_save_origin)#初始状态
        self.inittuple = self.stuple
        self.pixRatio=1            #显示比例
        self.zdict = {}            #文件路径
        self.sdict = {}            #状态字典
        self.savedict = {}         #修改原图保存路径
        self.saveseg = {}           #分割图保存路径
        self.multipath = None
        #doublethreshold
        self.dt = double_thresold()
        self.dt.button1_3.clicked.connect(self.dt_apply)

        self.pathChanged.connect(self.dt.histpath)
        #doublecolumn
        self.d_col = QmyDoubleColumn()
        self.fileTrans.connect(self.d_col.loadFile)
        self.labelTrans.connect(self.d_col.loadResult)
        self.originTrans.connect(self.d_col.loadOrigin)
        self.outdirTrans.connect(self.d_col.loadPath)
        self.d_col.editDone.connect(self.reshow)
        self.d_col.ifsave.connect(self.seg_save_status)

        #输出目录
        self.outdir = None
        self.edit_labelpath = None
        self.is_outdir_change = False
        #均衡直方图
        self.clahe = clahe()
        self.clahe.btn_apply.clicked.connect(self.clahe_apply)
        self.__Flag = (QtCore.Qt.ItemIsSelectable| QtCore.Qt.ItemIsEnabled )  # 节点标志
        # self.itemFlags = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        #                   | Qt.ItemIsEnabled | Qt.ItemIsAutoTristate)  # 节点标志
        #参数设定
        self.set = setting()
        self.is_parameters_change = False  # 是否更改参数
        self.input_size = None
        self.overlap_size = None
        self.batch_size = None
        self.gpu_idx = 0
        self.aug_type = 0
        self.mean = None
        self.std = None
        self.use_post_process = 0
        self.unet_path = None
        self.wpunet_path = None
        self.color = None
        self.transparency = None
        # self.pregress = QmyDialog()

        self.read_all_parameters()  # 读取初始参数
        # singlecolumn
        self.origincolor = self.color
        self.s_col = QmySingleColumn()
        self.fileTrans.connect(self.s_col.loadFile)
        self.labelTrans.connect(self.s_col.loadResult)
        self.maskTrans.connect(self.s_col.loadMask)
        self.originTrans.connect(self.s_col.loadOrigin)
        self.outdirTrans.connect(self.s_col.loadPath)
        self.initStatus.connect(self.s_col.initstatus)
        self.s_col.editDone.connect(self.reshow)
        self.s_col.ifsave.connect(self.seg_save_status)
        self.s_col.color = self.color
        self.s_col.transparency = self.transparency
        self.qcolor = None
        self.set.paraChanged.connect(self.netstatus)
##  ==============自定义功能函数========================

    def __enableButtons(self):   ##工具栏按钮判断
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        if count == 1:

            self.ui.actionImage_Segmentation.setEnabled(True)

            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            # self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        elif count > 1:
            self.ui.actionNext_Image.setEnabled(True)
            self.ui.actionPrev_Image.setEnabled(True)
            self.ui.actionImage_Segmentation.setEnabled(True)
            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            # self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        else:
            return

    def __enableSegModify(self):  ##分割相关工具栏按钮判断

        if self.is_segment_img :
            self.ui.actionHuman_Modify.setEnabled(True)
            self.ui.actionDouble_Column.setEnabled(True)
            self.ui.actionSingle_Column.setEnabled(True)
            self.ui.actionInverse_Value.setEnabled(True)
            self.ui.actionSave.setEnabled(True)
            self.ui.actionSave_Label_as.setEnabled(True)
            self.ui.actionDelete_Label_File.setEnabled(True)
        else:
            self.ui.actionHuman_Modify.setEnabled(False)
            self.ui.actionDouble_Column.setEnabled(False)
            self.ui.actionSingle_Column.setEnabled(False)
            self.ui.actionInverse_Value.setEnabled(False)
            self.ui.actionSave.setEnabled(False)
            self.ui.actionSave_Label_as.setEnabled(False)
            self.ui.actionDelete_Label_File.setEnabled(False)

    def __enableOriginModify(self):  ##原图相关工具栏按钮判断
        if self.is_origin_change :
            self.ui.actionSave_Origin_as.setEnabled(True)
            self.ui.actionSave.setEnabled(True)
        else:
            self.ui.actionSave_Origin_as.setEnabled(False)
            self.ui.actionSave.setEnabled(False)
    def resetstatus(self,file): #重置分割状态
        tup = self.sdict[file]
        self.is_segment_img = tup[0]  # 是否分割
        self.is_origin_change = tup[1]  # 原图是否修改
        self.is_save_segment = tup[2]  # 分割图是否保存
        self.is_save_origin = tup[3]  # 修改后的原图是否保存
        segtext = file.split(".")[0] + "_label.png"
        dirname = self.zdict[file]
        fullFileName = os.path.join(dirname, segtext)  # 带路径文件名
        if os.path.exists(fullFileName):
            self.is_segment_img = True
            self.is_save_segment = True
            self.saveseg[file] = fullFileName
        if not self.is_save_segment :
            self.is_segment_img = False
        if not self.is_save_origin :
            self.is_origin_change = False
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.sdict[file] = self.stuple
        self.__enableOriginModify()
        self.__enableSegModify()

    def cleartop(self):  ##清空遮罩
        pix1 = QPixmap(self.topPixmap.size())
        pix1.fill(QtCore.Qt.transparent)
        pix0 = pix1.scaled(self.scaled.size())
        self.topPixmap = pix0
        self.ui.LabPicture.setPixmap(self.topPixmap)

    def __getpicPath(self, text):  ##图片完整路径
        dirname = self.zdict[text]
        fullFileName = os.path.join(dirname, text)  # 带路径文件名
        return fullFileName

    def __displayPic(self,text):  ##显示图片

        fullFileName = self.__getpicPath(text)
        self.curPixmap.load(fullFileName)  # 原始图片
        self.ui.statusBar.showMessage("Loaded" + " " + text, 5000)
        self.on_actZoomRealSize_triggered()  # 适合原图大小显示


    def is_gray(self,img_path): #判断是否是灰度图
        try:
            img = skimage.io.imread(img_path)
            if img.ndim == 2:
                return True

            return False

        except Exception as err:
            print("judge gray error:{}".format(err))
            return False
    def enhance_img(self,cla): #显示修改后的原图
        fname = self.ui.listWidget.currentItem().text()
        
        cla = cla*255
        cla1 = cla.astype(np.uint8)
        # print(cla1.dtype)
        shrink = cv2.cvtColor(cla1, cv2.COLOR_BGR2RGB)
        self.new_origin = QImage(shrink.data,
                          shrink.shape[1],
                          shrink.shape[0],
                          QImage.Format_RGB888)
        image3 = self.new_origin

        self.curPixmap = QPixmap.fromImage(image3)
        self.cleartop()
        self.is_origin_change = True
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
        self.sdict[fname] = self.stuple
        self.__enableOriginModify()
        pix0 = self.curPixmap.scaled(self.scaled.size())
        self.ui.LabB.setPixmap(pix0)



##  ==============event处理函数==========================
    def wheelEvent(self,event):
        angle=event.angleDelta() / 8
        angleY = angle.y()
        #判断鼠标位置在scrollarea
        labelrect = QRect(self.ui.scrollArea.pos() + self.ui.centralWidget.pos(),self.ui.scrollArea.size())
        if labelrect.contains(event.pos()):
            if angleY > 0:#  滚轮上滚
                self.on_actZoomIn_triggered()
            else:  # 滚轮下滚
                self.on_actZoomOut_triggered()
        else:
            event.ignore()

    def closeEvent(self, event):
        # self.processbar.close()
        # self.processbar.close()
        if self.is_segment_img and not self.is_save_segment and not self.is_enter_modify:
            yes, no = QMessageBox.Yes, QMessageBox.No
            msg = 'Do you want to quit system without saving result?'
            if QMessageBox.warning(self, 'Attention', msg, yes | no) == no :
                event.ignore()
            else:
                return
        else:
            return



##  ==========由connectSlotsByName()自动连接的槽函数============

    @pyqtSlot()  ##适应窗口显示图片
    def on_actZoomFitWin_triggered(self):
        H = self.ui.scrollArea.height()  # 得到scrollArea的高度
        realH = self.curPixmap.height()  # 原始图片的实际高度
        pixRatio1 = float(H) / realH  # 当前显示比例，必须转换为浮点数

        W = self.ui.scrollArea.width() - 20
        realW = self.curPixmap.width()
        pixRatio2 = float(W) / realW
        self.pixRatio = min([pixRatio1,pixRatio2])
        pix1 = self.curPixmap.scaled(W - 30,H - 30)
        self.scaled = pix1

        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap == None :
            return
        else:
            pix2 = self.topPixmap.scaled(W - 30,H - 30)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##适应宽度显示
    def on_actZoomFitW_triggered(self):
        W = self.ui.scrollArea.width() - 20
        realW = self.curPixmap.width()
        self.pixRatio = float(W) / realW
        pix1 = self.curPixmap.scaledToWidth(W - 30)
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap == None :
            return
        else:
            pix2 = self.topPixmap.scaledToWidth(W - 30)
            self.ui.LabPicture.setPixmap(pix2)



    @pyqtSlot()  ##实际大小
    def on_actZoomRealSize_triggered(self):
        self.pixRatio = 1  # 恢复显示比例为1
        self.scaled = self.curPixmap
        self.ui.LabB.setPixmap(self.curPixmap)
        if self.topPixmap == None :

            return
        else:
            self.ui.LabPicture.setPixmap(self.topPixmap)

    @pyqtSlot()  ##放大显示
    def on_actZoomIn_triggered(self):
        self.pixRatio = self.pixRatio * 1.2
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        pix1 = self.curPixmap.scaled(W, H)  # 图片缩放到指定高度和宽度，保持长宽比例
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)
        if self.topPixmap == None :
            return
        else:
            W0 = self.pixRatio * self.topPixmap.width()
            H0 = self.pixRatio * self.topPixmap.height()
            pix2 = self.topPixmap.scaled(W0, H0)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##缩小显示
    def on_actZoomOut_triggered(self):
        self.pixRatio = self.pixRatio * 0.8
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        pix1 = self.curPixmap.scaled(W, H)  # 图片缩放到指定高度和宽度，保持长宽比例
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)
        if self.topPixmap == None :
            return
        else:
            W0 = self.pixRatio * self.topPixmap.width()
            H0 = self.pixRatio * self.topPixmap.height()
            pix2 = self.topPixmap.scaled(W0, H0)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##打开图片
    def on_actionOpen_triggered(self):

        try:
            fileList, flt = QFileDialog.getOpenFileNames(self, "choose an image", "",
                                                         "Images(*.jpg *.bmp *.jpeg *.png)")
            if len(fileList)<1:
                return
            else:
                self.cleartop()
                fullFileName = fileList[0]  # 带路径文件名
                aItem = QListWidgetItem()
                aItem.setText(fullFileName.split("/")[-1])
                new1 = fullFileName.split("/")[-1]
                if new1 in self.zdict:
                    dlgTitle = "Attention"
                    strInfo = "A file with the same name exists in the file list\nplease use different name."
                    QMessageBox.information(self, dlgTitle, strInfo)
                    return
                else:
                    self.zdict[fullFileName.split("/")[-1]] = os.path.dirname(fullFileName).replace('/', '\\')
                    self.savedict[fullFileName.split("/")[-1]] = os.path.dirname(fullFileName).replace('/', '\\')
                    self.sdict[fullFileName.split("/")[-1]] = self.inittuple
                    aItem.setCheckState(QtCore.Qt.Unchecked)
                    Flag = self.__Flag
                    aItem.setFlags(Flag)
                    self.resetstatus(fullFileName.split("/")[-1])
                    self.ui.listWidget.addItem(aItem)
                    self.ui.listWidget.setCurrentItem(aItem)
                    self.curPixmap.load(fullFileName)  # 原始图片
                    self.ui.statusBar.showMessage("Loaded" + " " + fullFileName.split("/")[-1], 5000)
                    self.on_actZoomRealSize_triggered()  # 原图大小显示
                    self.__enableButtons()
        except Exception as e:
            print(e)

    @pyqtSlot()  ##打开目录
    def on_actionOpen_Dir_triggered(self):
        aList = []
        curDir = QDir.currentPath()
        dirStr = QFileDialog.getExistingDirectory(self, "open directory", curDir, QFileDialog.ShowDirsOnly)
        try:
            if dirStr.strip() == '':
                return
            else:
                self.ui.listWidget.blockSignals(True)
                self.ui.listWidget.clear()
                self.ui.listWidget.blockSignals(False)
                self.cleartop()
                self.zdict.clear()
                self.savedict.clear()
                aList.append(dirStr)

                # # QFileDialog.
                # for root, dirs, files in os.walk(dirStr):
                #     for name in dirs:
                #         aList.append(os.path.join(root, name))
                #
                # for allfile in aList:
                #     filedir = allfile.replace('/', '\\')
                #     dirObj = QDir(allfile)  # QDir对象
                #     strList = dirObj.entryList(QDir.Files)
                #     try:
                #         for line in strList:
                #             if line.endswith('jpg') or line.endswith('bmp') or line.endswith('jpeg') or line.endswith(
                #                     'png') or line.endswith('JPG') or line.endswith('BMP') or line.endswith(
                #                     'JPEG') or line.endswith('PNG'):
                #                 if line in self.zdict:
                #                     dlgTitle = "Attention"
                #                     strInfo = "A file with the same name exists in the file list\nplease use different name."
                #                     QMessageBox.information(self, dlgTitle, strInfo)
                #                     return
                #                 else:
                #                     self.zdict[line] = filedir
                #
                #                     self.savedict[line] = filedir
                #                     self.sdict[line] = self.inittuple
                #                     aItem = QListWidgetItem()
                #                     aItem.setText(line)
                #                     aItem.setCheckState(QtCore.Qt.Unchecked)
                #                     Flag = self.__Flag
                #                     aItem.setFlags(Flag)
                #                     self.ui.listWidget.addItem(aItem)
                dirObj = QDir(dirStr)  # QDir对象
                filedir = dirStr.replace('/', '\\')

                strList = dirObj.entryList(QDir.Files)
                try:
                    for line in strList:

                        if line.split("_")[-1] != "label.png" and (line.endswith('jpg') or line.endswith('bmp') or line.endswith('jpeg') or line.endswith(
                                'png') or line.endswith('JPG') or line.endswith('BMP') or line.endswith(
                                'JPEG') or line.endswith('PNG')):
                            if line in self.zdict:
                                dlgTitle = "Attention"
                                strInfo = "A file with the same name exists in the file list\nplease use different name."
                                QMessageBox.information(self, dlgTitle, strInfo)
                                return
                            else:
                                self.zdict[line] = filedir

                                self.savedict[line] = filedir
                                self.sdict[line] = self.inittuple
                                aItem = QListWidgetItem()
                                aItem.setText(line)
                                aItem.setCheckState(QtCore.Qt.Unchecked)
                                Flag = self.__Flag
                                aItem.setFlags(Flag)
                                self.ui.listWidget.addItem(aItem)
                except Exception as e:
                    dlgTitle = "Attention"
                    strInfo = "There are no qualified files in the current directory\nplease check the suffix, the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’"
                    QMessageBox.about(self, dlgTitle, strInfo)
                    print(e)
                try:

                    self.ui.listWidget.blockSignals(True)
                    self.ui.listWidget.setCurrentRow(0)
                    self.ui.listWidget.blockSignals(False)
                    filetext = self.ui.listWidget.item(0).text()
                    self.resetstatus(filetext)
                    if self.is_save_segment:
                        spath = self.saveseg[filetext]
                        self.qcolor = self.text2rgb(self.color)
                        im = QImage(spath)
                        image2 = im.convertToFormat(QImage.Format_ARGB32)
                        self.seg = im
                        width = im.width()
                        height = im.height()
                        for x in range(width):
                            for y in range(height):
                                if (image2.pixel(x, y) == 0xFF000000):
                                    image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                else:
                                    image2.setPixelColor(x, y, self.qcolor)
                        self.maskimg = image2
                        self.topPixmap = QPixmap.fromImage(image2)
                        self.labelpath = spath

                        pix0 = self.topPixmap.scaled(self.scaled.size())
                        self.ui.LabPicture.setPixmap(pix0)
                    self.__displayPic(filetext)
                    self.__enableButtons()
                except Exception as e:
                    dlgTitle = "Attention"
                    strInfo = "There are no qualified files in the current directory\nplease check the suffix, the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’"
                    QMessageBox.information(self, dlgTitle, strInfo)
                    print(e)
        except Exception as e:
            print(e)

    ##切换图片
    def on_listWidget_currentItemChanged(self, current, previous):

        try:
            filetext = current.text()

            if self.ui.actionMulti_Image_Process.isChecked() and self.is_save_segment:
                self.cleartop()
                img_path = self.__getpicPath(filetext)  # 带路径文件名
                self.qcolor = self.text2rgb(self.color)
                self.__displayPic(filetext)
                if self.is_multi == 2:
                    multipath = self.Omultipath
                else:
                    multipath = self.Umultipath
                try:
                    spath = os.path.splitext(img_path)[0] + '_label.png'
                    outdir = multipath.replace('/', '\\')
                    spath = os.path.join(outdir, spath.split("\\")[-1])

                    im = QImage(spath)
                    image2 = im.convertToFormat(QImage.Format_ARGB32)
                    self.seg = im
                    width = im.width()
                    height = im.height()
                    for x in range(width):
                        for y in range(height):
                            if (image2.pixel(x, y) == 0xFF000000):
                                image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                            else:
                                image2.setPixelColor(x, y, self.qcolor)
                    self.maskimg = image2
                    self.topPixmap = QPixmap.fromImage(image2)
                    self.labelpath = spath

                    pix0 = self.topPixmap.scaled(self.scaled.size())
                    self.ui.LabPicture.setPixmap(pix0)
                    self.resetstatus(filetext)
                except Exception as e:
                    print(e)
            else:
                if self.is_segment_img:

                    if self.is_save_segment:
                        self.resetstatus(filetext)
                        self.cleartop()
                        self.__displayPic(filetext)

                        if self.is_save_segment:

                            spath = self.saveseg[filetext]

                            im = QImage(spath)
                            image2 = im.convertToFormat(QImage.Format_ARGB32)
                            self.seg = im
                            width = im.width()
                            height = im.height()
                            self.qcolor = self.text2rgb(self.color)
                            for x in range(width):
                                for y in range(height):
                                    if (image2.pixel(x, y) == 0xFF000000):
                                        image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                    else:
                                        image2.setPixelColor(x, y, self.qcolor)
                            self.maskimg = image2
                            self.topPixmap = QPixmap.fromImage(image2)
                            self.labelpath = spath

                            pix0 = self.topPixmap.scaled(self.scaled.size())
                            self.ui.LabPicture.setPixmap(pix0)

                    else:

                        yes, no = QMessageBox.Yes, QMessageBox.No
                        msg = 'Do you want to leave current image without saving result?'
                        if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes):

                            self.cleartop()

                            self.__displayPic(filetext)
                            self.resetstatus(filetext)
                            if self.is_save_segment:
                                spath = self.saveseg[filetext]
                                self.qcolor = self.text2rgb(self.color)
                                im = QImage(spath)
                                image2 = im.convertToFormat(QImage.Format_ARGB32)
                                self.seg = im
                                width = im.width()
                                height = im.height()
                                for x in range(width):
                                    for y in range(height):
                                        if (image2.pixel(x, y) == 0xFF000000):
                                            image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                        else:
                                            image2.setPixelColor(x, y, self.qcolor)
                                self.maskimg = image2
                                self.topPixmap = QPixmap.fromImage(image2)
                                self.labelpath = spath

                                pix0 = self.topPixmap.scaled(self.scaled.size())
                                self.ui.LabPicture.setPixmap(pix0)

                        else:
                            self.ui.listWidget.blockSignals(True)
                            if previous:
                                self.ui.listWidget.setCurrentItem(previous)

                            else:
                                self.ui.listWidget.setCurrentRow(0)
                        self.ui.listWidget.blockSignals(False)
                else:

                    self.__displayPic(filetext)


                    self.resetstatus(filetext)
                    if self.is_save_segment:
                        spath = self.saveseg[filetext]
                        self.qcolor = self.text2rgb(self.color)
                        im = QImage(spath)
                        image2 = im.convertToFormat(QImage.Format_ARGB32)
                        self.seg = im
                        width = im.width()
                        height = im.height()
                        for x in range(width):
                            for y in range(height):
                                if (image2.pixel(x, y) == 0xFF000000):
                                    image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                else:
                                    image2.setPixelColor(x, y, self.qcolor)
                        self.maskimg = image2
                        self.topPixmap = QPixmap.fromImage(image2)
                        self.labelpath = spath

                        pix0 = self.topPixmap.scaled(self.scaled.size())
                        self.ui.LabPicture.setPixmap(pix0)
        except Exception as e:
            print(e)


    @pyqtSlot()  ##下一张
    def on_actionNext_Image_triggered(self):
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        count = count - 1
        currentRow = self.ui.listWidget.currentRow()

        try:
            if currentRow == count:
                # text = self.ui.listWidget.item(currentRow).text()
                # self.__displayPic(text)
                return

            elif currentRow < count:

                self.ui.listWidget.setCurrentRow(currentRow + 1)
        except Exception as e:
            print(e)
            return

    @pyqtSlot()  ##上一张
    def on_actionPrev_Image_triggered(self):
        currentRow = self.ui.listWidget.currentRow()
        try:
            if currentRow == 0:
                # text = self.ui.listWidget.item(currentRow).text()
                # self.__displayPic(text)
                return

            elif currentRow > 0:
                # nexttext = self.ui.listWidget.item(currentRow - 1).text()
                # self.__displayPic(nexttext)
                self.ui.listWidget.setCurrentRow(currentRow - 1)
        except Exception as e:
            print(e)
            return

    @pyqtSlot()  ##文件列表搜索
    def on_search_editingFinished(self):
        list = []
        first = 0
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        for i1 in range(count):
            list.append(self.ui.listWidget.item(i1).text())
        word = self.ui.search.text()

        if word.strip() != '':
            for i2 in range(count):
                if word in list[i2]:
                    t1 = self.ui.listWidget.item(first).text()
                    t2 = self.ui.listWidget.item(i2).text()
                    cItem = self.ui.listWidget.item(first)
                    bItem1 = self.ui.listWidget.item(i2)
                    bItem1.setText(t1)
                    cItem.setText(t2)
                    cItem.setCheckState(QtCore.Qt.Checked)
                    first = first + 1
                else:
                    bItem2 = self.ui.listWidget.item(i2)
                    bItem2.setCheckState(QtCore.Qt.Unchecked)
        else:
            for i3 in range(count):
                dItem = self.ui.listWidget.item(i3)
                dItem.setCheckState(QtCore.Qt.Unchecked)



    @pyqtSlot()  ##OTSU
    def on_actionOTSU_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        self.cleartop()
        self.qcolor = self.text2rgb(self.color)
        try:
            if self.ui.actionMulti_Image_Process.isChecked():
                self.is_multi = 2
                curDir = QDir.currentPath()
                outdir = QFileDialog.getExistingDirectory(self, "save all OTSU results in directory", curDir,
                                                          QFileDialog.ShowDirsOnly)
                self.Omultipath = outdir
                if outdir.strip() == '':
                    return
                else:

                    self.MultiThread = MultiThread(self.is_multi, self.zdict, self.multiindex, self.HEfinish,
                                                   self.OTSUfinish, outdir,self.savedict)
                    self.MultiThread.start()
                    self.MultiThread.selfstop.connect(self.grayWarning)
                    self.processbar = Processbar()
                    self.processbar.show()
                    self.processbar.setValue(1)
                    self.processbar.stop.connect(self.MultiThread.stop)
            else:
                if self.is_gray(img_path):
                     # if self.is_origin_change
                    im = QImage(img_path)
                    width = im.width()
                    height = im.height()
                    if self.is_origin_change == False:
                        image = skimage.io.imread(img_path)

                    else:
                        img = Image.fromqimage(self.new_origin)
                        image1 = img.convert('L')
                        image = np.asarray(image1)
                    thresh = skimage.filters.threshold_otsu(image)
                    dst = (image >= thresh) * 1.0

                    dst *= 255
                    d_img = dst.astype(np.uint8)
                    shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

                    self.seg = QImage(shrink.data,
                                      shrink.shape[1],
                                      shrink.shape[0],
                                      QImage.Format_RGB888)
                    image2 = self.seg.convertToFormat(QImage.Format_ARGB32)


                    for x in range(width):
                        for y in range(height):
                            if(image2.pixel(x,y) == 0xFF000000):
                                image2.setPixelColor(x,y,QColor(0, 0, 0,0))
                            else:
                                image2.setPixelColor(x, y, self.qcolor)
                    self.maskimg = image2
                    self.topPixmap = QPixmap.fromImage(image2)
                    self.is_segment_img = True
                    self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
                    self.sdict[fname] = self.stuple
                    print(self.sdict[fname])
                    pix0 = self.topPixmap.scaled(self.scaled.size())
                    self.ui.LabPicture.setPixmap(pix0)
                    self.__enableSegModify()
                else:
                    QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")
        except Exception as err:
            print("otsu Error:{0}".format(err))

    @pyqtSlot()  ##double threshold
    def  on_actionDouble_Threshold_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名

        if self.is_gray(img_path):

            self.pathChanged.emit(img_path)


            self.dt.show()
        else:
            QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")

    @pyqtSlot()  ##Inverse_Value
    def  on_actionInverse_Value_triggered(self):

        # self.seg = self.seg.invertPixels(QImage.InvertRgb)
        width = self.seg.width()
        height = self.seg.height()

        image2 = self.seg.convertToFormat(QImage.Format_ARGB32)
        image2.invertPixels(QImage.InvertRgb)
        self.seg = image2.convertToFormat(QImage.Format_RGB888)

        position = 255 * self.transparency
        for x in range(width):
            for y in range(height):
                if (image2.pixel(x, y) == 0xFF000000):
                    image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                else:
                    image2.setPixelColor(x, y, self.qcolor)
        self.maskimg = image2
        self.topPixmap = QPixmap.fromImage(image2)

        pix0 = self.topPixmap.scaled(self.scaled.size())
        self.ui.LabPicture.setPixmap(pix0)

    @pyqtSlot()  ##UNet
    def on_actionUnet_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)
        self.qcolor = self.text2rgb(self.color)
        pth_address = self.unet_path

        if self.ui.actionMulti_Image_Process.isChecked():
            self.is_multi = 3
            curDir = QDir.currentPath()
            outdir = QFileDialog.getExistingDirectory(self, "save all Unet results in directory", curDir,
                                                      QFileDialog.ShowDirsOnly)
            self.Umultipath = outdir
            if outdir.strip() == '':
                return
            else:

                self.segThread = SegThread(int(self.input_size), pth_address, int(self.overlap_size),
                                           int(self.batch_size), self.aug_type, float(self.mean), float(self.std),
                                           self.use_post_process, self.multiindex,
                                           self.cropbegin, self.segbegin, self.seging, self.transbegin,
                                           self.complete, self.OTSUfinish,img_path,outdir,self.zdict,self.savedict)
                self.segThread.start()
                self.segThread.grayError.connect(self.grayWarning)
                self.processbar = Processbar()
                self.processbar.show()
                self.processbar.setValue(0)
                self.processbar.stop.connect(self.segThread.stop)
        else:
            try:

                if self.is_gray(img_path):
                    self.processbar = Processbar()
                    self.processbar.show()

                    im = QImage(img_path)
                    width = im.width()
                    height = im.height()


                    if self.is_origin_change == False:

                        try:
                            self.segThread = SegThread(int(self.input_size), pth_address, int(self.overlap_size),
                                                       int(self.batch_size), self.aug_type, float(self.mean), float(self.std),
                                                       self.use_post_process,self.multiindex,
                                                       self.cropbegin, self.segbegin, self.seging, self.transbegin,
                                                       self.complete,self.OTSUfinish,img_path)

                            self.segThread.start()
                            self.processbar.stop.connect(self.segThread.stop)
                            self.segThread.runtimeSignal.connect(self.runtimeWarning)
                            self.segThread.grayError.connect(self.grayWarning)
                        except Exception as e:
                            QMessageBox.information(self, "Error", str(e))
                            print("unet Error1:{0}".format(e))
                    else:

                        img = Image.fromqimage(self.new_origin)
                        image1 = img.convert('L')
                        img = np.asarray(image1)
                        self.segThread = SegThread(int(self.input_size), pth_address, int(self.overlap_size),
                                                   int(self.batch_size), self.aug_type, float(self.mean), float(self.std),
                                                   self.use_post_process,self.multiindex,
                                                   self.cropbegin, self.segbegin, self.seging, self.transbegin,
                                                   self.complete,self.OTSUfinish, img_path,img)

                        self.segThread.start()
                        self.processbar.stop.connect(self.segThread.stop)
                        self.segThread.runtimeSignal.connect(self.runtimeWarning)

                else:
                    QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")
            except Exception as err:
                print("unet Error:{0}".format(err))


    @pyqtSlot()  ##Human_Modify
    def on_actionHuman_Modify_triggered(self):
        self.on_actionSingle_Column_triggered()

    @pyqtSlot()  ##Image_Segmentation
    def on_actionImage_Segmentation_triggered(self):
        self.on_actionUnet_triggered()

    @pyqtSlot()  ##single_column_Modify
    def on_actionSingle_Column_triggered(self):
        self.is_enter_modify = True
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        if self.is_save_segment:
            outdir = self.labelpath
        elif self.is_outdir_change:
            newpath = os.path.join(self.outdir, fname)
            outdir = os.path.splitext(newpath)[0] + '_label.png'
        else:
            default_resultfile_name = os.path.splitext(img_path)[0] + '_label.png'
            outdir = default_resultfile_name
        if self.is_origin_change:
            self.originTrans.emit(self.new_origin)
        else:
            self.fileTrans.emit(img_path)
        self.labelTrans.emit(self.seg)
        self.maskTrans.emit(self.maskimg)
        self.outdirTrans.emit(outdir)
        self.edit_labelpath = outdir
        self.s_col.show()
        self.close()

    @pyqtSlot()  ##double_column_Modify
    def on_actionDouble_Column_triggered(self):
        self.is_enter_modify = True
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        if self.is_save_segment :
            outdir = self.labelpath
        elif self.is_outdir_change:
            newpath = os.path.join(self.outdir, fname)
            outdir = os.path.splitext(newpath)[0] + '_label.png'
        else:
            default_resultfile_name = os.path.splitext(img_path)[0] + '_label.png'
            outdir = default_resultfile_name
        if self.is_origin_change:
            self.originTrans.emit(self.new_origin)
        else:
            self.fileTrans.emit(img_path)
        self.labelTrans.emit(self.seg)
        self.outdirTrans.emit(outdir)
        self.edit_labelpath = outdir
        self.d_col.show()
        self.close()

    @pyqtSlot()  ##改变输出目录
    def on_actionChange_Output_Dir_triggered(self):
        curDir = QDir.currentPath()

        self.outdir = QFileDialog.getExistingDirectory(self, "save labels in directory", curDir, QFileDialog.ShowDirsOnly)

        if self.outdir.strip() == '':
            return
        else:
            self.ui.statusBar.showMessage("Change output directory.Labels will be saved in" + " " + self.outdir, 5000)
            self.is_outdir_change = True

    @pyqtSlot()  ##清空分割图
    def on_actionDelete_Label_File_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = 'You are about to permanently delete this label file, proceed anyway?'
        if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes ):
            if self.is_save_segment:
                os.remove(self.labelpath)
                del self.saveseg[fname]
                self.cleartop()
            else:
                self.cleartop()

            self.is_segment_img = False
            self.is_save_segment = False
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
            self.__enableSegModify()
        else:
            return

    @pyqtSlot()  ##另存分割图
    def on_actionSave_Label_as_triggered(self):

        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)
        default_resultfile_name = os.path.splitext(fullname)[0] + '_label.png'
        filters = "Image (*.png)"
        # basename = os.path.splitext(self.resultname)[0]
        # default_resultfile_name = self.resultname  # os.path.join(self.currentPath(),basename + '.jpg')
        if self.is_outdir_change:
            newpath = os.path.join(self.outdir,fname)
            resultfile_name = os.path.splitext(newpath)[0] + '_label.png'
        else:
            resultfile_name = default_resultfile_name
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'Save Label' , resultfile_name,
            filters)
        if filename:
            self.labelpath = str(filename)
            self.saveseg[fname] = self.labelpath
            self.seg.save(self.labelpath,"PNG")
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##另存修改后的原图
    def on_actionSave_Origin_as_triggered(self):

        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)
        default_resultfile_name = os.path.splitext(fullname)[0] + '_enhance.png'
        filters = "Image (*.png)"
        # basename = os.path.splitext(self.resultname)[0]
        # default_resultfile_name = self.resultname  # os.path.join(self.currentPath(),basename + '.jpg')
        if self.is_outdir_change:
            newpath = os.path.join(self.outdir, fname)
            resultfile_name = os.path.splitext(newpath)[0] + '_enhance.png'
        else:
            resultfile_name = default_resultfile_name
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'Save Origin', resultfile_name, filters)
        if filename:
            self.outpath = str(filename)
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(self.outpath)
            updir = os.path.split(self.outpath)

            self.savedict[fname] = updir[0].replace('/','\\')
            del self.savedict[fname]
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##检测原图是否修改和是否分割图并保存
    def on_actionSave_triggered(self):

        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)

        if self.is_origin_change and not self.is_save_origin and not self.is_segment_img:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
            else:
                origin_name = default_oringin_name
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(origin_name)
            updir = os.path.split(origin_name)
            self.savedict[fname] = updir[0].replace('/', '\\')
            del self.savedict[fname]
            self.statusBar().showMessage('Enhanced image is saved in' + " " + origin_name, 4000)
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and not self.is_save_segment and not self.is_origin_change:
            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                seg_name = default_seg_name
            self.labelpath = seg_name
            print(seg_name)
            self.seg.save(seg_name, "PNG")
            self.saveseg[fname] = self.labelpath
            self.statusBar().showMessage('Label is saved in' + " " + seg_name, 4000)
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and self.is_origin_change and not self.is_save_segment and not self.is_save_origin:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'

            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                origin_name = default_oringin_name
                seg_name = default_seg_name
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(origin_name)
            updir = os.path.split(origin_name)
            self.savedict[fname] = updir[0].replace('/', '\\')
            del self.savedict[fname]
            self.labelpath = seg_name
            self.seg.save(seg_name, "PNG")

            self.saveseg[fname] = self.labelpath
            self.statusBar().showMessage('Enhanced image and label are saved separately in' + " " + origin_name+ " " + seg_name, 4000)
            self.is_save_origin = True
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##均衡直方图
    def on_actionCLAHE_triggered(self):

        self.clahe.show()

    @pyqtSlot()  ##直方图
    def on_actionHE_triggered(self):

        if self.ui.actionMulti_Image_Process.isChecked():
            self.is_multi = 1
            curDir = QDir.currentPath()
            outdir = QFileDialog.getExistingDirectory(self, "save all HE results in directory", curDir,
                                                           QFileDialog.ShowDirsOnly)
            if outdir.strip() == '':
                return
            else:
                self.Hmultipath = outdir
                self.processbar = Processbar()
                self.processbar.show()
                self.processbar.setValue(1)
                self.MultiThread = MultiThread(self.is_multi,self.zdict,self.multiindex,self.HEfinish,self.OTSUfinish,outdir)
                self.MultiThread.start()
                self.processbar.stop.connect(self.MultiThread.stop)

        else:
            fname = self.ui.listWidget.currentItem().text()
            img_path = self.__getpicPath(fname)
            try:
                img = skimage.io.imread(img_path)
                img = skimage.exposure.equalize_hist(img)
                self.enhance_img(img)
            except Exception as err:
                print("histogram equalize error:{}".format(err))

    @pyqtSlot()  ##参数设置
    def on_actionMethod_Setting_triggered(self):
        self.set.show()




    ##  =============自定义槽函数===============================
    # double_threshold按下按钮时触发，对图片进行处理，先处理大于Min的部分，然后处理小于Max的部分，然后取交集
    def dt_apply(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        self.cleartop()
        self.qcolor = self.text2rgb(self.color)
        try:
            if self.dt.minValue <= self.dt.maxValue :

                im = QImage(img_path)
                width = im.width()
                height = im.height()
                if self.is_origin_change == False:
                    self.img = skimage.io.imread(img_path)

                else:
                    img = Image.fromqimage(self.new_origin)
                    image1 = img.convert('L')
                    self.img = np.asarray(image1)

                dt = np.zeros_like(self.img)
                dt[(self.img <= self.dt.minValue) | (self.img >= self.dt.maxValue)] = 1.0

                dt *= 255

                d_img = np.asarray(dt)

                shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

                self.seg = QImage(shrink.data,
                                    shrink.shape[1],
                                    shrink.shape[0],
                                    QImage.Format_RGB888)

                image3 = self.seg.convertToFormat(QImage.Format_ARGB32)
                for x in range(width):
                    for y in range(height):
                        if (image3.pixel(x, y) == 0xFF000000):
                            image3.setPixelColor(x, y, QColor(0, 0, 0, 0))
                        else:
                            image3.setPixelColor(x, y, self.qcolor)
                self.maskimg = image3
                self.topPixmap = QPixmap.fromImage(image3)
                self.is_segment_img = True
                self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
                self.sdict[fname] = self.stuple

                pix0 = self.topPixmap.scaled(self.scaled.size())
                self.ui.LabPicture.setPixmap(pix0)
                self.__enableSegModify()
            else:
                return
        except Exception as err:
            print("apply error:{}".format(err))



    def cropbegin(self):  #unet单图开始overtile的信号
        self.processbar.setText('Cropping..')
        self.processbar.setValue(3)

    def segbegin(self,val): #unet单图开始分割的信号
        self.processSum=val
        self.processbar.setText('Segmenting..')
        self.processbar.setValue(5)
    def seging(self,val):   #unet单图正在分割的信号
        step = int(5 + val / self.processSum * 90)

        self.processbar.setValue(step)
    def transbegin(self):   #unet单图分割完类型转换的信号
        self.processbar.setText('Processing..')
        self.processbar.setValue(97)

    def complete(self,out_img): #unet单图分割总流程完成

        if type(out_img) is np.ndarray and out_img.size != 0:
            self.processbar.setValue(100)
            fname = self.ui.listWidget.currentItem().text()
            img_path = self.__getpicPath(fname)
            im = QImage(img_path)
            width = im.width()
            height = im.height()
            un = out_img * 255

            d_img = un.astype(np.uint8)

            shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
            self.seg = QImage(shrink.data,
                              shrink.shape[1],
                              shrink.shape[0],
                              QImage.Format_RGB888)
            image2 = self.seg.convertToFormat(QImage.Format_ARGB32)
            for x in range(width):
                for y in range(height):
                    if (image2.pixel(x, y) == 0xFF000000):
                        image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                    else:
                        image2.setPixelColor(x, y, self.qcolor)
            self.maskimg = image2
            self.topPixmap = QPixmap.fromImage(image2)
            self.is_segment_img = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple

            pix0 = self.topPixmap.scaled(self.scaled.size())
            self.ui.LabPicture.setPixmap(pix0)
            self.__enableSegModify()
            self.processbar.close()
        else:

            return
    def multiindex(self,val):   #多图分割过程中改变进度条的值
        self.processbar.setText('Please Wait..')
        self.processbar.setValue(1)
        step = int(val / self.ui.listWidget.count() * 90)

        self.processbar.setValue(step)
    def HEfinish(self): #多图HE完成
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)
        li = []
        lis = []
        self.processbar.setValue(100)
        for ke in self.sdict.keys():
            li.append(ke)
        for k in self.zdict.keys():
            lis.append(k)
        for i in range(len(li)):
            idx = li[i]
            tu = list(self.sdict[idx])
            tu[1] = True
            tu[3] = True
            tu = tuple(tu)
            self.sdict[idx] = tu
        self.is_save_origin = True
        self.is_origin_change = True
        for i in range(len(lis)):
            text = lis[i]
            dirname = self.zdict[text]
            im_path = os.path.join(dirname, text)
            path = os.path.splitext(im_path)[0] + '_enhance.png'
            outdir = self.Hmultipath.replace('/', '\\')
            path = path.split("\\")[-1]
            self.savedict[path] = outdir
            del self.savedict[text]
        try:
            path = os.path.splitext(img_path)[0] + '_enhance.png'

            outdir = self.Hmultipath.replace('/', '\\')
            spath = os.path.join(outdir, path.split("\\")[-1])

            im = QImage(spath)
            self.curPixmap = QPixmap.fromImage(im)
            self.cleartop()
            pix0 = self.curPixmap.scaled(self.scaled.size())
            self.ui.LabB.setPixmap(pix0)
            self.__enableOriginModify()
            self.processbar.close()
        except Exception as err:
            print("histogram equalize finish error:{}".format(err))

    def OTSUfinish(self):   #多图OTSU和UNet分割完成
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        li = []

        self.processbar.setValue(100)
        for ke in self.sdict.keys():
            li.append(ke)

        for i in range(len(li)):
            idx = li[i]
            tu = list(self.sdict[idx])
            tu[0] = True
            tu[2] = True
            tu = tuple(tu)
            self.sdict[idx] = tu
        self.is_save_segment = True
        self.is_segment_img = True
        self.qcolor = self.text2rgb(self.color)
        if self.is_multi == 2:
            multipath = self.Omultipath
        else:
            multipath = self.Umultipath
        try:
            spath = os.path.splitext(img_path)[0] + '_label.png'
            outdir = multipath.replace('/', '\\')
            spath = os.path.join(outdir, spath.split("\\")[-1])
            im = QImage(spath)
            image2 = im.convertToFormat(QImage.Format_ARGB32)
            width = im.width()
            height = im.height()
            for x in range(width):
                for y in range(height):
                    if (image2.pixel(x, y) == 0xFF000000):
                        image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                    else:
                        image2.setPixelColor(x, y, self.qcolor)
            self.maskimg = image2
            self.topPixmap = QPixmap.fromImage(image2)
            self.labelpath = spath
            pix0 = self.topPixmap.scaled(self.scaled.size())
            self.ui.LabPicture.setPixmap(pix0)
            self.__enableSegModify()

            self.processbar.close()
        except Exception as err:
            print("otsufinish Error:{0}".format(err))
    def runtimeWarning(self):   #unet内存溢出保存
        self.processbar.close()
        QMessageBox.information(self, "Error", "Memory is out of use.")

    def grayWarning(self):  #多图分割时图片不是灰度图报错

        self.processbar.close()
        QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")

    def clahe_apply(self):  #clahe
        # 读取参数值
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        self.ker = self.clahe.le_ker.text()

        self.clip = self.clahe.le_clip.text()
        
        self.img = skimage.io.imread(img_path)
        # 设置flag，为真时取默认值
        ker_def = False
        clip_def = False
        if self.ker == "default" or self.ker == "":
            ker_def = True
        if self.clip == "default" or self.clip == "":
            clip_def = True
        # 四种情况的判定
        if ker_def and clip_def:
            try:
                enhance_img = skimage.exposure.equalize_adapthist(self.img)
                print(type(enhance_img))
                self.enhance_img(enhance_img)
            except Exception as err:
                print("default error:{}".format(err))
            return
            # 类型转换
        if not ker_def:
            try:
                self.ker = int(self.ker)
                if not self.ker > 1:
                    QMessageBox.information(self, "Warning", "Invalid input")
                    return
            except:
                try:
                    self.ker = self.ker.split(',')
                    self.ker = list(map(int, self.ker))
                    # 全为1时等同于整数1
                    if len(set(self.ker)) == 1 and self.ker[0] == 1:
                        QMessageBox.information(self, "Warning", "Invalid input")
                        return
                    # 检查是否存在-1等非法输入
                    for i in set(self.ker):
                        if i <= 0:
                            QMessageBox.information(self, "Warning", "Invalid input")
                            return
                except Exception as err:
                    QMessageBox.information(self, "Warning", "Invalid input")
                    return
        if not clip_def:
            try:
                self.clip = float(self.clip)
                if self.clip < 0 or self.clip > 1:
                    QMessageBox.information(self, "Warning", "Invalid input")
                    return
            except Exception as err:
                QMessageBox.information(self, "Warning", "Invalid input")
                return

        if ker_def and not clip_def:
            try:
                enhance_img = skimage.exposure.equalize_adapthist(self.img, clip_limit=self.clip)
                # print(np.max(enhance_img)+";"+ np.min(enhance_img))
                self.enhance_img(enhance_img)
            except Exception as err:
                print("clip input error:{}".format(err))
            return
        if not ker_def and clip_def:
            try:
                enhance_img = skimage.exposure.equalize_adapthist(self.img, kernel_size=self.ker)
                self.enhance_img(enhance_img)
            except Exception as err:
                print("kernel input error:{}".format(err))
            return
        try:
            enhance_img = skimage.exposure.equalize_adapthist(self.img, self.ker, self.clip)
            self.enhance_img(enhance_img)

        except Exception as err:
            print("both input error:{}".format(err))
            return
        return

    def reshow(self):   #退出人工标注主窗口重现
        self.showMaximized()
        self.is_enter_modify = False
    def seg_save_status(self,s):    #在人工标注中保存图片
        self.is_save_segment = s
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.labelpath = self.edit_labelpath
        fname = self.ui.listWidget.currentItem().text()
        self.sdict[fname] = self.stuple
    def netstatus(self):    #改变颜色和透明度

        self.is_parameters_change == True
        self.read_all_parameters()
        self.initStatus.emit(self.color, self.transparency)
        self.s_col.color = self.color
        self.s_col.transparency = self.transparency
        # self.qcolor = self.text2rgb(self.color)

    def text2rgb(self,text):    #从config的颜色值得到qcolor
        text= text.replace('#','')
        textArr = re.findall(r'.{2}', text)

        r = int(textArr[0],16)
        g = int(textArr[1], 16)
        b = int(textArr[2], 16)
        # print(r)
        
        a =(1-int(self.transparency)*0.01)*255

        return QColor(r,g,b,a)

    # 按序读取参数
    def read_all_parameters(self):
        try:
            self.setting_file = open("./config.txt", "r")
        except Exception as err:
            print("open file error:{}".format(err))
        try:
            self.input_size = self.setting_file.readline()[:-1]
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
            self.use_post_process = not bool(int(post))

            self.gpu_idx = int(gpu_idx)

        except Exception as err:
            print("Parameters error0".format(err))
        self.setting_file.close()
##多图HE,OTSU
class MultiThread(QThread):

    indexTrans = pyqtSignal(int)

    HEfinished = pyqtSignal()

    OTSUfinished = pyqtSignal()
    selfstop = pyqtSignal()

    def __init__(self,mode,adict,multiindex,HEfinish,OTSUfinish,dir,savedict={}):
        super(MultiThread, self).__init__()

        self.mode = mode
        self.filedict = adict
        self.savedict = savedict
        self.flag = 1
        self.outdir = dir
        # processbar singal
        self.HEfinished.connect(HEfinish)
        self.indexTrans.connect(multiindex)
        self.OTSUfinished.connect(OTSUfinish)
        # logging.info(self.filename)
    def run(self):
        try:
            if self.flag != 1:
                return
            else:
                if self.mode == 1:
                    self.HE()
                if self.mode == 2:
                    self.OTSU()
        except Exception as e:
            self.selfstop.emit()
            self.stop()
            print(e)

    def stop(self):
        self.flag = 0
        
    def HE(self):
        li = []
        for ke in self.filedict.keys():
            li.append(ke)
        for i in range(len(li)):
            self.indexTrans.emit(i)
            if self.flag == 1:
                text = li[i]
                dirname = self.filedict[text]
                img_path = os.path.join(dirname, text)
                try:
                    img = skimage.io.imread(img_path)
                    img = skimage.exposure.equalize_hist(img)
                    cla = img * 255
                    cla1 = cla.astype(np.uint8)
                    
                    shrink = cv2.cvtColor(cla1, cv2.COLOR_BGR2RGB)
                    image3 = QImage(shrink.data,
                                             shrink.shape[1],
                                             shrink.shape[0],
                                             QImage.Format_RGB888)
                    img = Image.fromqimage(image3)
                    image = img.convert('L')
                    file_name = os.path.splitext(img_path)[0] + '_enhance.png'
                    outdir = self.outdir.replace('/', '\\')
                    file_name = os.path.join(outdir,file_name.split("\\")[-1])
                    image.save(file_name)

                except Exception as err:
                    print("histogram equalize error:{}".format(err))
            else:
                print('stop')
                return
        self.HEfinished.emit()

    def OTSU(self):

        lis = []
        if self.savedict:
            adict = self.savedict
        else:
            adict = self.filedict

        for ke in adict.keys():
            lis.append(ke)

        for i in range(len(lis)):

            self.indexTrans.emit(i)
            if self.flag == 1:
                text = lis[i]
                dirname = adict[text]
                ori_path = os.path.join(dirname, text)
                image = skimage.io.imread(ori_path)

                try:
                    # if image.ndim == 2:
                    assert image.ndim == 2,"The input image should be gray scale"
                    thresh = skimage.filters.threshold_otsu(image)
                    dst = (image >= thresh) * 1.0

                    dst *= 255
                    d_img = dst.astype(np.uint8)
                    shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

                    image2 = QImage(shrink.data,
                                      shrink.shape[1],
                                      shrink.shape[0],
                                      QImage.Format_RGB888)
                    file_name = os.path.splitext(ori_path)[0] + '_label.png'
                    name = file_name.split("\\")[-1]

                    outdir = self.outdir.replace('/', '\\')
                    file_name = os.path.join(outdir, name)
                    image2.save(file_name, "PNG")

                except Exception as err:
                    raise(err)
                    print("ostu error:{}".format(err))
            else:
                print('stop')
                return
        self.OTSUfinished.emit()
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
