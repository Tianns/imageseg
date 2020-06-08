import sys, random, cv2, numpy as np, math
from PyQt5.QtWidgets import (QWidget,
                             QPushButton, QMessageBox,QLineEdit, QLabel,
                             QHBoxLayout, QVBoxLayout, QStackedLayout,
                            QSlider)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
import skimage, matplotlib, skimage.io
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def is_gray(img_path):
    try:
        img = skimage.io.imread(img_path)
        if img.ndim == 2:
            return True
        return False
    except Exception as err:
        print("judge gray error:{}".format(err))
        return False


# 集成pyqt5控件和matplotlib绘图功能
class MyFigure(FigureCanvas):
    def __init__(self):
        self.w = 5
        self.h = 4
        self.d = 100
        self.fig = plt.Figure(figsize=(self.w, self.h), dpi=self.d)
        # self.fig = Figure(figsize=(self.w, self.h), dpi=self.d)
        super(MyFigure, self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.left = 100
        self.right = 200

    # 直方图绘制
    def hist(self, img_path="1023.png"):

        img = skimage.io.imread(img_path, as_gray=True)
        histogram, bin_center = skimage.exposure.histogram(img, 256)
        histogram_y = np.array(histogram)
        n = histogram_y.size
        histogram_x = np.array(np.arange(n))
        self.axes.cla()
        self.axes.plot(histogram_x, histogram_y)
        self.draw()
class double_thresold(QWidget):
    from_slider_min = pyqtSignal(str)
    from_slider_max = pyqtSignal(str)
    from_line_min = pyqtSignal(int)
    from_line_max = pyqtSignal(int)
    from_label_min = pyqtSignal(int)
    from_label_max = pyqtSignal(int)

    def __init__(self,img_path='1023.png'):
        super().__init__()
        self.minValue = 100
        self.maxValue = 200
        self.press = 0
        self.now = 0
        self.img_path = img_path

        self.init_ui()
        # self.histpath(self.img_path)
    # 初始化
    def init_ui(self):
        # 设置窗口大小，标题
        self.setFixedSize(750, 300)
        self.setWindowModality(Qt.ApplicationModal)
        self.move(300, 300)
        self.setWindowTitle("Double Threshold")
        # 定义框架，主框架下有左右两个盒式框架，左侧框架中是堆叠框架，实现控件的堆叠，右侧框架（纵向排列）中有两个横向排列盒式框架，存放标签、文本框和滑块
        self.main_box = QHBoxLayout()
        self.left1 = QVBoxLayout()
        self.stack = QStackedLayout()
        self.right1 = QVBoxLayout()
        self.right1_1 = QHBoxLayout()
        self.right1_2 = QHBoxLayout()

        # 定义控件，F为嵌入的matplotlib图像
        try:
            self.F = MyFigure()
            # self.F.hist(self.img_path)
            self.F.setMinimumWidth(500)
            self.F.setMaximumWidth(500)
            self.right1.setSpacing(10)
            self.right1_1.setSpacing(10)
            self.right1_2.setSpacing(10)
            self.label1_1 = QLabel("Min Val:")
            self.lineEdit1_1 = QLineEdit("100")
            self.lineEdit1_1.setMaximumWidth(70)
            self.slider1_1 = QSlider(Qt.Horizontal)
            self.slider1_1.setFocusPolicy(Qt.NoFocus)
            self.slider1_1.setRange(0, 255)
            self.slider1_1.setValue(100)
            self.label1_2 = QLabel("Max Val:")
            self.lineEdit1_2 = QLineEdit("200")
            self.lineEdit1_2.setMaximumWidth(70)
            self.slider1_2 = QSlider(Qt.Horizontal)
            self.slider1_2.setFocusPolicy(Qt.NoFocus)
            self.slider1_2.setRange(0, 255)
            self.slider1_2.setValue(200)
            self.button1_3 = QPushButton("Apply to Image")
            # 用label和F的堆叠实现用蓝色色块标注当前选中区域的功能
            self.mask = QLabel()
            # 堆叠框架只能间接添加框架
            self.mask_layout_widget = QWidget()
            self.mask_layout = QHBoxLayout()
            self.mask_layout.addWidget(self.mask)
            self.mask_layout_widget.setLayout(self.mask_layout)
            # 绑定控件的事件

            self.slider1_1.valueChanged.connect(self.change_sld_min_value)
            self.slider1_2.valueChanged.connect(self.change_sld_max_value)
            self.lineEdit1_1.editingFinished.connect(self.change_line_min_value_finished)
            self.lineEdit1_2.editingFinished.connect(self.change_line_max_value_finished)
            self.mask.mousePressEvent = self.label_mouse_press_event
            self.mask.mouseMoveEvent = self.label_mouse_move_event
        except Exception as err:
            print("set widgets:{0}".format(err))
        # 把控件添加到框架中
        try:
            self.left1.addLayout(self.stack)
            self.stack.addWidget(self.F)
            self.stack.addWidget(self.mask_layout_widget)
            self.right1_1.addWidget(self.label1_1)
            self.right1_1.addWidget(self.lineEdit1_1)
            self.right1_1.addWidget(self.slider1_1)
            self.right1_2.addWidget(self.label1_2)
            self.right1_2.addWidget(self.lineEdit1_2)
            self.right1_2.addWidget(self.slider1_2)
            self.right1.addLayout(self.right1_1)
            self.right1.addLayout(self.right1_2)
            self.right1.addWidget(self.button1_3)
        except Exception as err:
            print(err)
        # 对框架进行设置，并把框架添加到主框架中
        try:
            self.stack.setStackingMode(1)
            self.main_box.addLayout(self.left1)
            self.main_box.addLayout(self.right1)
            self.setLayout(self.main_box)
        except Exception as err:
            print(err)
        # 设置色块label覆盖区域的初始数据，如边距和颜色
        try:
            self.mask_left_margin = 78
            self.mask_up_margin = 33
            self.mask_right_margin = 65
            self.mask_bottom_margin = 30
            self.mask_range = self.F.width() - self.mask_left_margin - self.mask_right_margin
            self.mask.setStyleSheet("QLabel{background-color: rgb(0,0,255,50);}")
            self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(), self.mask_bottom_margin)
        except Exception as err:
            print("mask Error:{0}".format(err))
        # 为设置的信号绑定函数
        try:
            self.from_slider_min.connect(self.change_line_min_value)
            self.from_slider_min.connect(self.change_label_min)
            self.from_slider_max.connect(self.change_line_max_value)
            self.from_slider_max.connect(self.change_label_max)
            self.from_line_min.connect(self.change_sld_min_value)
            self.from_line_min.connect(self.change_label_min)
            self.from_line_max.connect(self.change_sld_max_value)
            self.from_line_max.connect(self.change_label_max)
            self.from_label_max.connect(self.change_line_max_value)
            self.from_label_max.connect(self.change_sld_max_value)
            self.from_label_min.connect(self.change_line_min_value)
            self.from_label_min.connect(self.change_sld_min_value)

        except Exception as err:
            print("Signal Error:{0}".format(err))
    def histpath(self,path):
        self.img_path = path
        self.F.hist(self.img_path)
    # 分解拖动 = 按下 + 移动
    # 按下事件，记录按下时的位置
    def label_mouse_press_event(self, event):
        if event.buttons() == Qt.LeftButton:
            self.press = event.pos().x()

    # 移动事件，仅在左键被按下时进行处理，获取当前的位置，计算和初始位置的差值，把差值传递到移动label的函数
    def label_mouse_move_event(self, event):
        if event.buttons() == Qt.LeftButton:
            self.now = event.pos().x()
            self.differ = self.now - self.press
            self.move_mask(self.differ)

    # 判定文本是否符合要求
    def is_available_int(self, text):
        try:
            if int(text) >= 0 and int(text) <= 255:
                return True
            return False
        except:
            print("except")
            return False

    # 计算label色块的左边界
    def eva_start(self):
        return int(round(self.minValue * self.mask_range / 255.0)) + self.mask_left_margin

    # 计算label色块的右边界
    def eva_end(self):
        return int(round((255 - self.maxValue) * self.mask_range / 255.0)) + self.mask_right_margin

    # 拖动滑块触发的事件，区分是操作控件时触发还是其他控件的信号触发
    def change_sld_min_value(self, value):
        # 其他信号触发
        if type(self.sender()) != QSlider:
            # 屏蔽自身信号，避免死循环
            self.slider1_1.blockSignals(True)
            try:
                self.slider1_1.setValue(value)
            except Exception as err:
                print("err in set slider1_1 value: {0}".format(err))
            self.slider1_1.blockSignals(False)
        # 自身信号触发
        else:
            if value > self.maxValue:
                QMessageBox.information(self, "Warning", "Minimum value cannot be greater than maximum value")
                value = 0
                self.slider1_1.setValue(value)
            else:
                self.from_slider_min.emit(str(value))

    # 拖动滑块触发的事件，区分是操作控件时触发还是其他控件的信号触发
    def change_sld_max_value(self, value):
        # 其他信号触发
        if type(self.sender()) != QSlider:
            # 屏蔽自身信号，避免死循环
            self.slider1_2.blockSignals(True)
            try:
                self.slider1_2.setValue(value)
            except Exception as err:
                print("err in set slider1_2 value: {0}".format(err))
            self.slider1_2.blockSignals(False)
        # 自身信号触发
        else:
            if value < self.minValue:
                QMessageBox.information(self, "Warning", "Maximum value cannot be less than minimum value")
                value = 255
                self.slider1_2.setValue(value)
            else:
                self.from_slider_max.emit(str(value))

    # 更改文本（Min Val）触发的事件，区分是操作控件时触发还是其他控件的信号触发
    def change_line_min_value(self, text):
        # 来自其他信号
        if type(self.sender()) != QLineEdit:
            self.lineEdit1_1.blockSignals(True)
            self.lineEdit1_1.setText(str(text))
            self.lineEdit1_1.blockSignals(False)
        else:
        # 来自自身事件信号
            # 判定文本是否符合要求
            if self.is_available_int(text):
                if int(text) > self.maxValue:
                    QMessageBox.information(self, "Warning", "Minimum value cannot be greater than maximum value")
                    text = "0"
                    self.lineEdit1_1.setText(str(text))
                else:
                    self.from_line_min.emit(int(text))
            else:
                QMessageBox.information(self, "Attention", "Please enter a positive integer between 0 and 255")
                self.lineEdit1_1.blockSignals(True)
                self.lineEdit1_1.setText("0")
                self.lineEdit1_1.blockSignals(False)

    def change_line_min_value_finished(self):
        text = self.lineEdit1_1.text()
        # 来自其他信号
        if type(self.sender()) != QLineEdit:
            self.lineEdit1_1.blockSignals(True)
            self.lineEdit1_1.setText(str(text))
            self.lineEdit1_1.blockSignals(False)
        else:
        # 来自自身事件信号
            # 判定文本是否符合要求
            if self.is_available_int(text):
                if int(text) > self.maxValue:
                    QMessageBox.information(self, "Warning", "Minimum value cannot be greater than maximum value")
                    text = "0"
                    self.lineEdit1_1.setText(str(text))
                else:
                    self.from_line_min.emit(int(text))
            else:
                QMessageBox.information(self, "Attention", "Please enter a positive integer between 0 and 255")
                self.lineEdit1_1.blockSignals(True)
                self.lineEdit1_1.setText("0")
                self.lineEdit1_1.blockSignals(False)

    def change_line_max_value(self, text):
        if type(self.sender()) != QLineEdit:
            # 其他信号触发
            self.lineEdit1_2.blockSignals(True)
            self.lineEdit1_2.setText(str(text))
            self.lineEdit1_2.blockSignals(False)
        else:
            # 来自自身事件信号
            # 判定文本是否符合要求
            if self.is_available_int(text):
                if int(text) < self.minValue:
                    QMessageBox.information(self, "Warning", "Maximum value cannot be less than minimum value")
                    text = "255"
                    self.lineEdit1_2.setText(str(text))
                else:
                    self.from_line_max.emit(int(text))
            else:
                QMessageBox.information(self, "Attention", "Please enter a positive integer between 0 and 255")
                self.lineEdit1_2.blockSignals(True)
                self.lineEdit1_2.setText("255")
                self.lineEdit1_2.blockSignals(False)

    def change_line_max_value_finished(self):
        text = self.lineEdit1_2.text()
        if type(self.sender()) != QLineEdit:
            # 其他信号触发
            self.lineEdit1_2.blockSignals(True)
            self.lineEdit1_2.setText(str(text))
            self.lineEdit1_2.blockSignals(False)
        else:
            # 来自自身事件信号
            # 判定文本是否符合要求
            if self.is_available_int(text):
                if int(text) < self.minValue:
                    QMessageBox.information(self, "Warning", "Maximum value cannot be less than minimum value")
                    text = "255"
                    self.lineEdit1_2.setText(str(text))
                else:
                    self.from_line_max.emit(int(text))
            else:
                QMessageBox.information(self, "Attention", "Please enter a positive integer between 0 and 255")
                self.lineEdit1_2.blockSignals(True)
                self.lineEdit1_2.setText("255")
                self.lineEdit1_2.blockSignals(False)

    # 只会从其他控件处收到信号，更改色块显示的范围
    def change_label_min(self, left):
        if self.is_available_int(left):
            if int(left) <= self.maxValue:
                self.minValue = int(left)
                self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(),
                                                    self.mask_bottom_margin)

    # 只会从其他控件处收到信号，更改色块显示的范围
    def change_label_max(self, right):
        if self.is_available_int(right):
            if int(right) >= self.minValue:
                self.maxValue = int(right)
                self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(),
                                                    self.mask_bottom_margin)

    # 拖动色块时触发，重新计算左右边界，并把信号传递到其他控件
    def move_mask(self, diff):
        if self.is_available_int(self.minValue + diff) and self.is_available_int(self.maxValue + diff):
            self.minValue += diff
            self.maxValue += diff
            self.avail_range = self.maxValue - self.minValue
            self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(),
                                                self.mask_bottom_margin)
            self.from_label_min.emit(self.minValue)
            self.from_label_max.emit(self.maxValue)
        elif self.is_available_int(self.minValue + diff):
            self.maxValue = 255
            self.minValue = 255 - self.avail_range
            self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(),
                                                self.mask_bottom_margin)
            self.from_label_min.emit(self.minValue)
            self.from_label_max.emit(self.maxValue)
        elif self.minValue + diff < 0:
            self.minValue = 0
            self.maxValue = self.avail_range
            self.mask_layout.setContentsMargins(self.eva_start(), self.mask_up_margin, self.eva_end(),
                                                self.mask_bottom_margin)
            self.from_label_min.emit(self.minValue)
            self.from_label_max.emit(self.maxValue)

    # 按下按钮时触发，对图片进行处理，先处理大于Min的部分，然后处理小于Max的部分，然后取交集
    def apply(self):
        try:
            if is_gray(self.img_path):
                self.img = skimage.io.imread(self.img_path)
                # ret, temp_image1 = cv2.threshold(self.img, self.minValue, 255, cv2.THRESH_BINARY)
                # ret, temp_image2 = cv2.threshold(self.img, self.maxValue, 255, cv2.THRESH_BINARY_INV)
                # image = cv2.bitwise_and(temp_image1, temp_image2)
                dt = np.zeros_like(self.img)
                dt[(self.img >= self.minValue) & (self.img <= self.maxValue)] = 1.0
                # cv2.imshow("double threshold", dt * 255)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                dt *= 255
                return dt
        except Exception as err:
            print("apply error:{}".format(err))