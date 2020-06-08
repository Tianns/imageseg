from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import re
import skimage
from PIL import Image
from skimage import filters
from Unet_ import *
from clahe import clahe
# from numpy import *
from process import *
from doubleThreshold import double_thresold, MyFigure
from myDoubleColumn import QmyDoubleColumn
from mySingleColumn import QmySingleColumn
from setting import setting
from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow):
    pathChanged = pyqtSignal(str)  # 传入原图路径
    fileTrans = pyqtSignal(str)  # 传入原图路径
    labelTrans = pyqtSignal(QImage)  # 传输分割图
    maskTrans = pyqtSignal(QImage)  # 传输遮罩图
    originTrans = pyqtSignal(QImage)  # 修改后的原图
    outdirTrans = pyqtSignal(str)  # 传入输出路径
    initStatus = pyqtSignal(str, str)  # 传入color,transparency

    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_MainWindow()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面

        self.setWindowTitle("Image Segmentation based on deep learning")
        self.showMaximized()
        self.ui.statusBar.showMessage("system started", 5000)
        self.curPixmap = QPixmap()  # 图片
        self.topPixmap = QPixmap()  # 图片
        self.scaled = QPixmap()

        self.seg = QImage()  # 分割图
        self.maskimg = QImage()  # 遮罩图
        self.new_origin = QImage()  # 修改后的原图
        self.is_segment_img = False  # 是否分割
        self.is_origin_change = False  # 原图是否修改
        self.is_save_segment = False  # 分割图是否保存
        self.is_save_origin = False  # 修改后的原图是否保存
        self.is_enter_modify = False  # 进入人工标注

        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.inittuple = self.stuple
        self.pixRatio = 1  # 显示比例
        self.zdict = {}
        self.sdict = {}  # 状态字典

        # doublethreshold
        self.dt = double_thresold()
        self.dt.button1_3.clicked.connect(self.dt_apply)
        self.dt_figure = MyFigure()
        self.pathChanged.connect(self.dt_figure.hist)
        # doublecolumn
        self.d_col = QmyDoubleColumn()
        self.fileTrans.connect(self.d_col.loadFile)
        self.labelTrans.connect(self.d_col.loadResult)
        self.originTrans.connect(self.d_col.loadOrigin)
        self.outdirTrans.connect(self.d_col.loadPath)
        self.d_col.editDone.connect(self.reshow)
        self.d_col.ifsave.connect(self.seg_save_status)

        # 输出目录
        self.outdir = None
        self.is_outdir_change = False
        # 均衡直方图
        self.clahe = clahe()
        self.clahe.btn_apply.clicked.connect(self.clahe_apply)
        self.__Flag = (Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # 节点标志
        # self.itemFlags = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        #                   | Qt.ItemIsEnabled | Qt.ItemIsAutoTristate)  # 节点标志
        # 参数设定
        self.set = setting()
        self.is_parameters_change = False  # 是否更改参数
        self.size_of_input = None
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

    def __enableButtons(self):  ##工具栏按钮判断
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        if count == 1:

            self.ui.actionImage_Segmentation.setEnabled(True)

            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        elif count > 1:
            self.ui.actionNext_Image.setEnabled(True)
            self.ui.actionPrev_Image.setEnabled(True)

            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        else:
            return

    def __enableSegModify(self):  ##分割相关工具栏按钮判断

        if self.is_segment_img:
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
        if self.is_origin_change:
            self.ui.actionSave_Origin_as.setEnabled(True)
            self.ui.actionSave.setEnabled(True)
        else:
            self.ui.actionSave_Origin_as.setEnabled(False)
            self.ui.actionSave.setEnabled(False)

    def resetstatus(self, file):
        tup = self.sdict[file]
        self.is_segment_img = tup[0]  # 是否分割
        self.is_origin_change = tup[1]  # 原图是否修改
        self.is_save_segment = tup[2]  # 分割图是否保存
        self.is_save_origin = tup[3]  # 修改后的原图是否保存
        if not self.is_save_segment:
            self.is_segment_img = False
        if not self.is_save_origin:
            self.is_origin_change = False
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.sdict[file] = self.stuple
        self.__enableOriginModify()
        self.__enableSegModify()

    def cleartop(self):  ##清空遮罩
        pix1 = QPixmap(self.topPixmap.size())
        pix1.fill(Qt.transparent)
        pix0 = pix1.scaled(self.scaled.size())
        self.topPixmap = pix0
        self.ui.LabPicture.setPixmap(self.topPixmap)

    def __getpicPath(self, text):  ##图片完整路径
        dirname = self.zdict[text]
        fullFileName = os.path.join(dirname, text)  # 带路径文件名
        return fullFileName

    def __displayPic(self, text):  ##显示图片

        fullFileName = self.__getpicPath(text)
        self.curPixmap.load(fullFileName)  # 原始图片
        self.ui.statusBar.showMessage("Loaded" + " " + text, 5000)
        self.on_actZoomFitWin_triggered()  # 适合窗口显示

    # def __displaynewPic(self, filetext):  ##显示图片
    #     if self.is_segment_img:
    #         if self.is_save_segment:
    #             self.cleartop()
    #
    #             self.__displayPic(filetext)
    #             self.resetstatus(filetext)
    #         else:
    #
    #             yes, no = QMessageBox.Yes, QMessageBox.No
    #             msg = 'Do you want to leave current image without saving result?'
    #             if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes):
    #
    #                 self.cleartop()
    #
    #                 self.__displayPic(filetext)
    #                 self.resetstatus(filetext)
    #             else:
    #                 return
    #     else:
    #
    #         self.__displayPic(filetext)
    #         self.resetstatus(filetext)
    def is_gray(self, img_path):  # 判断是否是灰度图
        try:
            img = skimage.io.imread(img_path)
            if img.ndim == 2:
                return True

            return False

        except Exception as err:
            print("judge gray error:{}".format(err))
            return False

    def enhance_img(self, cla):  # 显示修改后的原图
        fname = self.ui.listWidget.currentItem().text()

        cla = cla * 255
        cla1 = cla.astype(np.uint8)
        # print(cla1.dtype)
        shrink = cv2.cvtColor(cla1, cv2.COLOR_BGR2RGB)
        self.new_origin = QImage(shrink.data,
                                 shrink.shape[1],
                                 shrink.shape[0],
                                 QImage.Format_RGB888)
        image3 = self.new_origin
        self.curPixmap = QPixmap.fromImage(image3)

        self.is_origin_change = True
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
        self.sdict[fname] = self.stuple
        self.__enableOriginModify()
        pix0 = self.curPixmap.scaled(self.scaled.size())
        self.ui.LabB.setPixmap(pix0)

    ##  ==============event处理函数==========================
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        angleY = angle.y()
        if angleY > 0:  # 滚轮上滚
            self.on_actZoomIn_triggered()
        else:  # 滚轮下滚
            self.on_actZoomOut_triggered()

    def closeEvent(self, event):
        if self.is_segment_img and not self.is_save_segment and not self.is_enter_modify:
            yes, no = QMessageBox.Yes, QMessageBox.No
            msg = 'Do you want to quit system without saving result?'
            if QMessageBox.warning(self, 'Attention', msg, yes | no) == no:
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
        self.pixRatio = min([pixRatio1, pixRatio2])
        pix1 = self.curPixmap.scaled(W - 30, H - 30)
        self.scaled = pix1

        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap == None:
            return
        else:
            pix2 = self.topPixmap.scaled(W - 30, H - 30)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##适应宽度显示
    def on_actZoomFitW_triggered(self):
        W = self.ui.scrollArea.width() - 20
        realW = self.curPixmap.width()
        self.pixRatio = float(W) / realW
        pix1 = self.curPixmap.scaledToWidth(W - 30)
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap == None:
            return
        else:
            pix2 = self.topPixmap.scaledToWidth(W - 30)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##实际大小
    def on_actZoomRealSize_triggered(self):
        self.pixRatio = 1  # 恢复显示比例为1
        self.scaled = self.curPixmap
        self.ui.LabB.setPixmap(self.curPixmap)
        if self.topPixmap == None:

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
        if self.topPixmap == None:
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
        if self.topPixmap == None:
            return
        else:
            W0 = self.pixRatio * self.topPixmap.width()
            H0 = self.pixRatio * self.topPixmap.height()
            pix2 = self.topPixmap.scaled(W0, H0)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()  ##打开图片
    def on_actionOpen_triggered(self):
        fileList, flt = QFileDialog.getOpenFileNames(self, "choose an image", "",
                                                     "Images(*.jpg *.bmp *.jpeg *.png)")
        if len(fileList) < 1:
            return
        else:
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
                self.sdict[fullFileName.split("/")[-1]] = self.inittuple
                aItem.setCheckState(Qt.Unchecked)
                Flag = self.__Flag
                aItem.setFlags(Flag)
                self.ui.listWidget.addItem(aItem)
                self.ui.listWidget.setCurrentItem(aItem)
                self.curPixmap.load(fullFileName)  # 原始图片
                self.ui.statusBar.showMessage("Loaded" + " " + fullFileName.split("/")[-1], 5000)
                self.on_actZoomFitWin_triggered()  # 适合窗口显示
                self.__enableButtons()

    @pyqtSlot()  ##打开目录
    def on_actionOpen_Dir_triggered(self):

        curDir = QDir.currentPath()

        dirStr = QFileDialog.getExistingDirectory(self, "open directory", curDir, QFileDialog.ShowDirsOnly)

        if dirStr.strip() == '':
            return
        else:
            aList = []
            self.ui.listWidget.clear()
            aList.append(dirStr)

            # QFileDialog.
            for root, dirs, files in os.walk(dirStr):
                for name in dirs:
                    aList.append(os.path.join(root, name))

            for allfile in aList:
                filedir = allfile.replace('/', '\\')
                dirObj = QDir(allfile)  # QDir对象
                strList = dirObj.entryList(QDir.Files)
                try:
                    for line in strList:
                        if line.endswith('jpg') or line.endswith('bmp') or line.endswith('jpeg') or line.endswith(
                                'png') or line.endswith('JPG') or line.endswith('BMP') or line.endswith(
                            'JPEG') or line.endswith('PNG'):
                            self.zdict[line] = filedir
                            self.sdict[line] = self.inittuple
                            aItem = QListWidgetItem()
                            aItem.setText(line)
                            aItem.setCheckState(Qt.Unchecked)
                            Flag = self.__Flag
                            aItem.setFlags(Flag)
                            self.ui.listWidget.addItem(aItem)

                except Exception as e:
                    dlgTitle = "Attention"
                    strInfo = "There are no qualified files in the current directory\nplease check the suffix, the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’"
                    QMessageBox.about(self, dlgTitle, strInfo)
                    print(e)
            try:
                filetext = self.ui.listWidget.item(0).text()
                self.__displayPic(filetext)
                self.ui.listWidget.setCurrentRow(0)

                self.__enableButtons()
            except Exception as e:
                dlgTitle = "Attention"
                strInfo = "There are no qualified files in the current directory\nplease check the suffix, the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’"
                QMessageBox.information(self, dlgTitle, strInfo)
                print(e)

    ##切换图片
    def on_listWidget_currentItemChanged(self, current, previous):

        # if(current == None):
        #     current = self.ui.listWidget.item(0)
        #     filetext = current.text()
        #     self.__displayPic(filetext)
        #     print('hz')
        # else:
        filetext = current.text()

        if self.is_segment_img:
            if self.is_save_segment:
                self.cleartop()

                self.__displayPic(filetext)
                self.resetstatus(filetext)
            else:

                yes, no = QMessageBox.Yes, QMessageBox.No
                msg = 'Do you want to leave current image without saving result?'
                if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes):

                    self.cleartop()

                    self.__displayPic(filetext)
                    self.resetstatus(filetext)
                else:
                    self.ui.listWidget.blockSignals(True)
                    self.ui.listWidget.setCurrentItem(previous)
                    # previous.setSelected(True)
                    previous.setFocus()
                self.ui.listWidget.blockSignals(False)
        else:

            self.__displayPic(filetext)
            self.resetstatus(filetext)

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

    @pyqtSlot()  ##搜索
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
                    cItem.setCheckState(Qt.Checked)
                    first = first + 1
                else:
                    bItem2 = self.ui.listWidget.item(i2)
                    bItem2.setCheckState(Qt.Unchecked)
        else:
            for i3 in range(count):
                dItem = self.ui.listWidget.item(i3)
                dItem.setCheckState(Qt.Unchecked)

    @pyqtSlot()  ##OTSU
    def on_actionOTSU_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名

        self.qcolor = self.text2rgb(self.color)
        try:

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
                        if (image2.pixel(x, y) == 0xFF000000):
                            image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                        else:
                            image2.setPixelColor(x, y, self.qcolor)
                self.maskimg = image2
                self.topPixmap = QPixmap.fromImage(image2)
                self.is_segment_img = True
                self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
                self.sdict[fname] = self.stuple
                # pix1 = QPixmap(self.topPixmap.size())
                # pix1.fill(Qt.transparent)
                #
                # p1 = QPainter(pix1)
                # p1.setCompositionMode(QPainter.CompositionMode_Source)
                # p1.drawPixmap(0, 0, self.topPixmap)
                #
                # p1.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                # position = 255 * self.transparency
                # p1.fillRect(pix1.rect(), QColor(0, 0, 0,position))
                #
                # p1.end()
                # self.topPixmap = pix1
                pix0 = self.topPixmap.scaled(self.scaled.size())
                self.ui.LabPicture.setPixmap(pix0)
                self.__enableSegModify()
            else:
                QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")
        except Exception as err:
            print("otsu Error:{0}".format(err))

    @pyqtSlot()  ##double threshold
    def on_actionDouble_Threshold_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名

        if self.is_gray(img_path):

            self.pathChanged.emit(img_path)
            # self.dt_figure.img_path = img_path
            # print(os.path.basename(sys.argv[0]) +self.dt_figure.img_path)
            self.dt.show()
        else:
            QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")

    @pyqtSlot()  ##Inverse_Value
    def on_actionInverse_Value_triggered(self):

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

        try:

            if self.is_gray(img_path):
                # if self.is_origin_change
                im = QImage(img_path)
                width = im.width()
                height = im.height()
                pth_address = self.unet_path

                if self.is_origin_change == False:
                    pth_address = self.unet_path
                    # pth_address = os.path.join(os.getcwd(), "model", "parameters", "zipandunzip.ipynb")
                    try:
                        self.segThread = SegThread(self.input_size, img_path, pth_address, self.overlap_size,
                                                   self.batch_size, self.aug_type, self.mean, self.std,
                                                   self.use_post_process,
                                                   self.cropbegin, self.segbegin, self.seging, self.transbegin,
                                                   self.complete)

                        self.segThread.start()
                        self.processbar = Processbar()
                        self.processbar.show()
                        # print(int(self.size_of_input)+","+int(self.overlap_size)+","+int(self.batch_size)+","+float(self.mean)+","+float(self.std))
                    #     net_inference = NetInference(gpu_detector, input_size=int(self.size_of_input),
                    #                                  overlap_size=int(self.overlap_size),
                    #                                  batch_size=int(self.batch_size), aug_type=self.aug_type,
                    #                                  mean=float(self.mean), std=float(self.std),
                    #                                  pth_address=pth_address, use_post_process=self.use_post_process)
                    except AssertionError as e:
                        QMessageBox.information(self, "Error", str(e))
                        # img_path = os.path.join(os.getcwd(), "data", "iron_dilate", "data_experiment", "images", "295.png")
                        # in_img = load_img(img_path)
                        # try:
                        #     out_img = net_inference._forward_one_image(in_img)
                        #
                        # except AssertionError as e:
                        #     QMessageBox.information(self, "Error", str(e))
                        # except RuntimeError as e:
                        #     QMessageBox.information(self, "Error", "The gpu or cpu memory is out of use, please adjust input size or batch num, or check whether there is other application allocate gpu")
                        # un = out_img * 255
                        #
                        # d_img = un.astype(np.uint8)
                        #
                        # shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
                        # self.seg = QImage(shrink.data,
                        #                   shrink.shape[1],
                        #                   shrink.shape[0],
                        #                   QImage.Format_RGB888)
                        # self.seg.save('./1024.png',"PNG")
                else:

                    img = Image.fromqimage(self.new_origin)
                    image1 = img.convert('L')
                    img = np.asarray(image1)
                    thresh = skimage.filters.threshold_otsu(img)
                    dst = (img >= thresh) * 1.0

                    dst *= 255
                    d_img = dst.astype(np.uint8)
                    # d_img = np.asarray(dst)

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
            else:
                QMessageBox.information(self, "Warning", "The threshold methods can not be applied to RGB image")
        except Exception as err:
            print("unet Error:{0}".format(err))
            # else:
            #     show_size = 512
            #     plt.figure(figsize=(20, 20))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(in_img[0: show_size, 0: show_size], cmap="gray")
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(out_img[0: show_size, 0: show_size], cmap="gray")
            #     plt.show()

    @pyqtSlot()  ##Human_Modify
    def on_actionHuman_Modify_triggered(self):
        self.on_actionSingle_Column_triggered()

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
        self.s_col.show()
        self.close()

    @pyqtSlot()  ##double_column_Modify
    def on_actionDouble_Column_triggered(self):
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
        self.outdirTrans.emit(outdir)
        self.d_col.show()
        self.close()

    @pyqtSlot()  ##改变输出目录
    def on_actionChange_Output_Dir_triggered(self):
        curDir = QDir.currentPath()

        self.outdir = QFileDialog.getExistingDirectory(self, "save labels in directory", curDir,
                                                       QFileDialog.ShowDirsOnly)

        if self.outdir.strip() == '':
            return
        else:
            self.ui.statusBar.showMessage("Change output directory.Labels will be saved in" + " " + self.outdir, 5000)
            self.is_outdir_change = True

    @pyqtSlot()  ##清空分割图
    def on_actionDelete_Label_File_triggered(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = 'You are about to permanently delete this label file, proceed anyway?'
        if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes):
            if self.is_save_segment:
                os.remove(self.labelpath)
                self.cleartop()
            else:
                self.cleartop()
            fname = self.ui.listWidget.currentItem().text()
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
            newpath = os.path.join(self.outdir, fname)
            resultfile_name = os.path.splitext(newpath)[0] + '_label.png'
        else:
            resultfile_name = default_resultfile_name
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'Save Label', resultfile_name,
            filters)
        if filename:
            self.labelpath = str(filename)
            self.seg.save(self.labelpath, "PNG")
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
            self.new_origin.save(self.outpath, "PNG")
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##检测原图是否修改和是否分割图并保存
    def on_actionSave_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)
        if self.is_origin_change and not self.is_segment_img:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
            else:
                origin_name = default_oringin_name
            self.new_origin.save(origin_name, "PNG")
            self.statusBar().showMessage('Enhanced image is saved in' + " " + origin_name, 4000)
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and not self.is_origin_change:
            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                seg_name = default_seg_name
            self.labelpath = seg_name
            self.seg.save(seg_name, "PNG")
            self.statusBar().showMessage('Label is saved in' + " " + seg_name, 4000)
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and self.is_origin_change:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'
            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                origin_name = default_oringin_name
                seg_name = default_seg_name
            self.new_origin.save(origin_name, "PNG")
            self.labelpath = seg_name
            self.seg.save(seg_name, "PNG")
            self.statusBar().showMessage(
                'Enhanced image and label are saved separately in' + " " + origin_name + " " + seg_name, 4000)
            self.is_save_origin = True
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##均衡直方图
    def on_actionCLAHE_triggered(self):
        self.clahe.show()

    @pyqtSlot()  ##直方图
    def on_actionHE_triggered(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)
        try:
            img = skimage.io.imread(img_path)
            img = skimage.exposure.equalize_hist(img)
            self.enhance_img(img)
        except Exception as err:
            print("histogram equalize error:{}".format(err))

    @pyqtSlot()  ##直方图
    def on_actionMethod_Setting_triggered(self):
        self.set.show()

    ##  =============自定义槽函数===============================
    # double_threshold按下按钮时触发，对图片进行处理，先处理大于Min的部分，然后处理小于Max的部分，然后取交集
    def dt_apply(self):
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)  # 带路径文件名
        self.qcolor = self.text2rgb(self.color)
        try:
            if self.dt.minValue <= self.dt.maxValue:

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
                position = 255 * self.transparency
                self.seg = QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  QImage.Format_RGB888)
                # else:
                #     img = Image.fromqimage(self.new_origin)
                #     image1 = img.convert('L')
                #     img = np.asarray(image1)
                #     thresh = skimage.filters.threshold_otsu(img)
                #     dst = (img >= thresh) * 1.0
                #
                #     dst *= 255
                #     d_img = dst.astype(np.uint8)
                #     # d_img = np.asarray(dst)
                #
                #     shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
                #
                #     self.seg = QImage(shrink.data,
                #                       shrink.shape[1],
                #                       shrink.shape[0],
                #                       QImage.Format_RGB888)
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

            # clahe单击按钮事件

    def cropbegin(self):
        self.processbar.setText('Cropping..')
        self.processbar.setValue(3)

    def segbegin(self, val):
        self.processSum = val
        self.processbar.setText('Segmenting..')
        self.processbar.setValue(5)

    def seging(self, val):
        step = int(5 + val / self.processSum * 90)

        self.processbar.setValue(step)

    def transbegin(self):
        self.processbar.setText('Processing..')
        self.processbar.setValue(97)

    def complete(self, out_img):
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

    def clahe_apply(self):
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
                print(np.max(enhance_img) + ";" + np.min(enhance_img))
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

    def reshow(self):
        self.showMaximized()
        self.is_enter_modify = False

    def seg_save_status(self, s):
        self.is_save_segment = s
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        fname = self.ui.listWidget.currentItem().text()
        self.sdict[fname] = self.stuple

    def netstatus(self):

        self.is_parameters_change == True
        self.read_all_parameters()
        self.initStatus.emit(self.color, self.transparency)
        self.s_col.color = self.color
        self.s_col.transparency = self.transparency
        # self.qcolor = self.text2rgb(self.color)

    def text2rgb(self, text):
        text = text.replace('#', '')
        textArr = re.findall(r'.{2}', text)

        r = int(textArr[0], 16)
        g = int(textArr[1], 16)
        b = int(textArr[2], 16)
        # print(r)

        a = (1 - int(self.transparency) * 0.01) * 255

        return QColor(r, g, b, a)

    # 按序读取参数
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
            self.use_post_process = not bool(int(post))

            self.gpu_idx = int(gpu_idx)

        except Exception as err:
            print("Parameters error0".format(err))
        self.setting_file.close()


##  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QmyMainWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())