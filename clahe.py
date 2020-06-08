import cv2
from PyQt5.QtWidgets import QWidget,QPushButton, QMessageBox,QLineEdit, QLabel,QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal,Qt
import skimage, skimage.io

class clahe(QWidget):

    def __init__(self, img_path="./014-000.png"):
        super().__init__()
        self.initUI()
        self.img_path = img_path

    def initUI(self):
        self.setFixedSize(500, 200)
        self.setWindowModality(Qt.ApplicationModal)
        self.move(300, 300)
        self.setWindowTitle("CLAHE")
        # 设置框架
        self.main_box = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.vbox = QVBoxLayout()
        # 设置控件
        self.tip = QLabel("The kernel size should be a positive integar and the clip limit should be in range[0,1]")
        self.tip.setWordWrap(True)
        self.label_ker = QLabel("Kernel Size: ")
        self.label_clip = QLabel("Clip Limit: ")
        self.le_ker = QLineEdit("default")
        self.le_clip = QLineEdit("default")
        self.btn_apply = QPushButton("Apply to Image")
        # 添加控件
        self.hbox.setSpacing(10)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.label_ker)
        self.hbox.addWidget(self.le_ker)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.label_clip)
        self.hbox.addWidget(self.le_clip)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.btn_apply)
        self.vbox.addStretch(1)
        self.main_box.addWidget(self.tip)
        self.main_box.addStretch(2)
        self.main_box.addLayout(self.hbox)
        self.main_box.addStretch(1)
        self.main_box.addLayout(self.vbox)
        self.main_box.addStretch(1)
        self.main_box.setSpacing(10)
        self.setLayout(self.main_box)
        # 绑定事件
    #     self.btn_apply.clicked.connect(self.apply)
    #
    # # 单击按钮事件
    # def apply(self):
    #     # 读取参数值
    #     self.ker = self.le_ker.text()
    #     self.clip = self.le_clip.text()
    #     self.img = skimage.io.imread(self.img_path)
    #     # 设置flag，为真时取默认值
    #     ker_def = False
    #     clip_def = False
    #     if self.ker == "default" or self.ker == "":
    #         ker_def = True
    #     if self.clip == "default" or self.clip == "":
    #         clip_def = True
    #     # 四种情况的判定
    #     if ker_def and clip_def:
    #         try:
    #             clahe_img = skimage.exposure.equalize_adapthist(self.img)
    #             # self.show_img(clahe_img)
    #         except Exception as err:
    #             print("default error:{}".format(err))
    #         return clahe_img
    #         # 类型转换
    #     if not ker_def:
    #         try:
    #             self.ker = int(self.ker)
    #             if not self.ker > 1:
    #                 QMessageBox.information(self, "Warning", "Invalid input")
    #                 return
    #         except:
    #             try:
    #                 self.ker = self.ker.split(',')
    #                 self.ker = list(map(int, self.ker))
    #                 # 全为1时等同于整数1
    #                 if len(set(self.ker)) == 1 and self.ker[0] == 1:
    #                     QMessageBox.information(self, "Warning", "Invalid input")
    #                     return
    #                 # 检查是否存在-1等非法输入
    #                 for i in set(self.ker):
    #                     if i <= 0:
    #                         QMessageBox.information(self, "Warning", "Invalid input")
    #                         return
    #             except Exception as err:
    #                 QMessageBox.information(self, "Warning", "Invalid input")
    #                 return
    #     if not clip_def:
    #         try:
    #             self.clip = float(self.clip)
    #             if self.clip < 0 or self.clip > 1:
    #                 QMessageBox.information(self, "Warning", "Invalid input")
    #                 return
    #         except Exception as err:
    #             QMessageBox.information(self, "Warning", "Invalid input")
    #             return
    #
    #     if ker_def and not clip_def:
    #         try:
    #             clahe_img = skimage.exposure.equalize_adapthist(self.img, clip_limit=self.clip)
    #             # self.show_img(clahe_img)
    #
    #             return clahe_img
    #         except Exception as err:
    #             print("clip input error:{}".format(err))
    #         return
    #     if not ker_def and clip_def:
    #         try:
    #             clahe_img = skimage.exposure.equalize_adapthist(self.img, kernel_size=self.ker)
    #             # self.show_img(clahe_img)
    #             return clahe_img
    #         except Exception as err:
    #             print("kernel input error:{}".format(err))
    #         return
    #     try:
    #         clahe_img = skimage.exposure.equalize_adapthist(self.img, self.ker, self.clip)
    #         # self.show_img(clahe_img)
    #         return clahe_img
    #     except Exception as err:
    #         print("both input error:{}".format(err))
    #         return

    # # 显示图片
    # def show_img(self, img):
    #     cv2.imshow("clahe", img)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()