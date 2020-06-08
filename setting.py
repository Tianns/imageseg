from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys,torch,os
class setting(QWidget):
    paraChanged = pyqtSignal()
    def __init__(self):
        super().__init__()
        # 参数定义
        self._gpu_number = torch.cuda.device_count()  # if there is np gpu, return 0
        # print(self._gpu_number)
        self.setting_file = open("./config.txt", "r")
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
        self.read_all_parameters()
        self.initUI()

    # 初始化界面
    def initUI(self):
        # 定义大小，不允许拖动放大
        self.setFixedSize(500, 500)

        self.main_box = QVBoxLayout()
        self.setWindowTitle("Setting")

        # 定义控件和框架， 绑定事件
        self.row1 = QLabel("Network Parameters:")

        self.row2 = QHBoxLayout()
        self.row2_1 = QHBoxLayout()
        self.row2_1_1 = QLabel("Size of input:")
        self.row2_1_2 = QLineEdit(self.size_of_input)
        self.row2_1.setSpacing(10)
        self.row2_1.addWidget(self.row2_1_1)
        self.row2_1.addWidget(self.row2_1_2)
        self.row2_2 = QHBoxLayout()
        self.row2_2_1 = QLabel("Overlap Length:")
        self.row2_2_2 = QLineEdit(self.overlap_size)
        self.row2_2.setSpacing(10)
        self.row2_2.addWidget(self.row2_2_1)
        self.row2_2.addWidget(self.row2_2_2)
        self.row2.addLayout(self.row2_1)
        self.row2.addStretch(1)
        self.row2.addLayout(self.row2_2)
        self.row2.addStretch(1)

        self.row3 = QHBoxLayout()
        self.row3_1 = QHBoxLayout()
        self.row3_1_1 = QLabel("Batch Size:")
        self.row3_1_2 = QLineEdit(self.batch_size)
        self.row3_1.setSpacing(10)
        self.row3_1.addWidget(self.row3_1_1)
        self.row3_1.addWidget(self.row3_1_2)
        self.row3_2 = QHBoxLayout()
        self.row3_2_1 = QLabel("GPU Index:")
        self.row3_2_2 = QComboBox()
        if self._gpu_number == 0:
            self.row3_2_2.addItem("")
            self.row3_2_2.setItemText(0, "Not use GPU")
            self.row3_2_2.setCurrentIndex(0)
        else:

            qlist = []
            qlist.append("Not use GPU")
            for i in range(self._gpu_number):
                qlist.append(str(i))
            # print(qlist)

            self.row3_2_2.addItems(qlist)

            self.row3_2_2.setCurrentIndex(self.gpu_idx)

        self.row3_2.setSpacing(10)
        self.row3_2.addWidget(self.row3_2_1)
        self.row3_2.addWidget(self.row3_2_2)
        self.row3.addLayout(self.row3_1)
        self.row3.addStretch(1)
        self.row3.addLayout(self.row3_2)
        self.row3.addStretch(1)

        self.row4 = QHBoxLayout()
        self.row4_1 = QHBoxLayout()
        self.row4_1_1 = QLabel("mean:")
        self.row4_1_2 = QLineEdit(self.mean)
        self.row4_1.setSpacing(10)
        self.row4_1.addWidget(self.row4_1_1)
        self.row4_1.addWidget(self.row4_1_2)
        self.row4_2 = QHBoxLayout()
        self.row4_2_1 = QLabel("std:")
        self.row4_2_2 = QLineEdit(self.std)
        self.row4_2.setSpacing(10)
        self.row4_2.addWidget(self.row4_2_1)
        self.row4_2.addWidget(self.row4_2_2)
        self.row4.addLayout(self.row4_1)
        self.row4.addStretch(1)
        self.row4.addLayout(self.row4_2)

        self.row5 = QHBoxLayout()
        self.row5_1 = QHBoxLayout()
        self.row5_1_1 = QLabel("TTA:")
        self.row5_1_2 = QComboBox()
        self.row5_1_2.addItem("")
        self.row5_1_2.addItem("")
        self.row5_1_2.addItem("")
        self.row5_1_2.setItemText(0,"No TTA")
        self.row5_1_2.setItemText(1, "TTA with 4 variants")
        self.row5_1_2.setItemText(2, "TTA with 8 variants")
        self.row5_1_2.setCurrentIndex(self.aug_type)
        self.row5_1.setSpacing(10)
        self.row5_1.addWidget(self.row5_1_1)
        self.row5_1.addWidget(self.row5_1_2)
        self.row5_2 = QHBoxLayout()
        self.row5_2_1 = QLabel("Post Process:")
        self.row5_2_2 = QComboBox()
        self.row5_2_2.addItem("")
        self.row5_2_2.addItem("")
        self.row5_2_2.setItemText(0, "Yes")
        self.row5_2_2.setItemText(1, "No")
        self.row5_2_2.setCurrentIndex(self.use_post_process)
        self.row5_2.setSpacing(10)
        self.row5_2.addWidget(self.row5_2_1)
        self.row5_2.addWidget(self.row5_2_2)
        self.row5.addLayout(self.row5_1)
        self.row5.addStretch(1)
        self.row5.addLayout(self.row5_2)



        self.row6 = QLabel("Address of Unet model:")

        self.row7 = QHBoxLayout()
        self.row7_1 = QLineEdit(self.unet_path)
        self.row7_2 = QPushButton("Browse..")
        self.row7_2.clicked.connect(self.getUet)
        self.row7.addWidget(self.row7_1)
        self.row7.addWidget(self.row7_2)
        self.row7.setSpacing(10)

        self.row8_1 = QLabel("Address of WPUnet model:")
        self.row9_2 = QPushButton("Browse..")
        self.row9_1 = QLineEdit(self.wpunet_path)
        self.row9_1.setEnabled(False)
        self.row9 = QHBoxLayout()
        self.row9_2.clicked.connect(self.getWPUnet)
        self.row9_2.setEnabled(False)
        self.row9.addWidget(self.row9_1)
        self.row9.addWidget(self.row9_2)
        self.row9.setSpacing(10)

        self.row10 = QLabel("Mask Parameter:")
        self.row11 = QHBoxLayout()
        self.row11_1 = QHBoxLayout()
        self.row11_1_1 = QLabel("Object Color:")
        self.row11_1_2 = QLabel()
        self.row11_1_2.setFixedSize(30, 30)
        self.row11_1_2.setStyleSheet("QWidget{background-color:" + self.color + "}")
        self.row11_1_2.mousePressEvent = self.get_color
        # print("{background-color:" + self.color + "}")
        self.row11_1.setSpacing(10)
        self.row11_1.addWidget(self.row11_1_1)
        self.row11_1.addWidget(self.row11_1_2)
        self.row11_2 = QHBoxLayout()
        self.row11_2_1 = QLabel("Transparency:")
        self.row11_2_2 = QSpinBox()
        self.row11_2_2.setFixedWidth(123)
        self.row11_2_2.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.row11_2_2.setSuffix('%')
        self.row11_2_2.setMaximum(100)
        self.row11_2_2.setValue(int(self.transparency))
        self.row11_2_2.setAlignment(Qt.AlignCenter)
        self.row11_2.setSpacing(10)
        self.row11_2.addWidget(self.row11_2_1)
        self.row11_2.addWidget(self.row11_2_2)
        self.row11.addLayout(self.row11_1)
        self.row11.addStretch(1)
        self.row11.addLayout(self.row11_2)

        self.row12 = QHBoxLayout()
        self.row12_1 = QPushButton("Save")
        self.row12_1.clicked.connect(self.save_all_parameters)
        self.row12.addStretch(1)
        self.row12.addWidget(self.row12_1)

        self.main_box.addWidget(self.row1)
        self.main_box.addLayout(self.row2)
        self.main_box.addLayout(self.row3)
        self.main_box.addLayout(self.row4)
        self.main_box.addLayout(self.row5)
        self.main_box.addWidget(self.row6)
        self.main_box.addLayout(self.row7)
        self.main_box.addWidget(self.row8_1)
        self.main_box.addLayout(self.row9)
        self.main_box.addWidget(self.row10)
        self.main_box.addLayout(self.row11)
        self.main_box.addLayout(self.row12)

        self.setLayout(self.main_box)

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
            self.use_post_process = int(post)
            self.gpu_idx = int(gpu_idx)

        except Exception as err:
            print("Parameters error0".format(err))
        self.setting_file.close()

    # 利用对话框获取颜色
    def get_color(self, e):
        colordialog = QColorDialog.getColor()
        if colordialog.isValid():
            self.color = colordialog.name()
            self.row11_1_2.setStyleSheet("QWidget{background-color:" + self.color + "}")

    # 保存并关闭
    def save_all_parameters(self):
        size_of_input = self.row2_1_2.text()
        overlap_size = self.row2_2_2.text()
        batch_size = self.row3_1_2.text()
        gpu_idx = str(self.row3_2_2.currentIndex())
        mean = self.row4_1_2.text()
        std = self.row4_2_2.text()
        tta = str(self.row5_1_2.currentIndex())
        post = str(self.row5_2_2.currentIndex())
        unet = self.row7_1.text()
        wpunet = self.row9_1.text()
        transparency = str(self.row11_2_2.value())
        if not self.judge(size_of_input, overlap_size, transparency,batch_size,mean,std):
            return
        try:
            self.setting_file = open("./config.txt", "w")
        except Exception as err:
            print("open file error:{}".format(err))
        try:
            self.setting_file.write(size_of_input + "\n")
            self.setting_file.write(overlap_size + "\n")
            self.setting_file.write(batch_size + "\n")
            self.setting_file.write(gpu_idx + "\n")
            self.setting_file.write(mean + "\n")
            self.setting_file.write(std + "\n")
            self.setting_file.write(tta + "\n")
            self.setting_file.write(post + "\n")
            self.setting_file.write(unet + "\n")
            self.setting_file.write(wpunet + "\n")
            self.setting_file.write(self.color + "\n")
            self.setting_file.write(transparency + "\n")
        except Exception as err:
            print("Parameters error1".format(err))

        self.setting_file.close()
        self.close()
        self.paraChanged.emit()

    # 判定数据是否合法
    def judge(self, input_size, overlap_size, transparency,batch_size,mean,std):
        try:
            assert int(input_size) % 16 == 0 and int(input_size)>16, "The input size must be a positive int and can be divided by 16"

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The input size must be int and can be divided by 16")
            return False
        try:
            assert int(overlap_size) > 0, "The overlap size must be int and greater than 0"

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The overlap size must be int and greater than 0")
            return False
        try:
            assert int(input_size) > (2 * int(overlap_size)), "The input size {} must be greater than 2 times overlap size {}".format(int(input_size), int(overlap_size))

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The input size {} must be greater than 2 times overlap size {}".format(int(input_size), int(overlap_size)))
            return False
        try:
            assert int(batch_size) > 0, "The batch num must be int and greater than 0"

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The batch num must be int and greater than 0")
            return False

        try:
            assert float(mean) > 0, "The mean must be float and greater than 0"

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The mean must be float and greater than 0")
            return False
        try:
            assert float(std) > 0, "The std  must be float and greater than 0"

        except (AssertionError, ValueError) as err:
            QMessageBox.information(self, "Error", "The std  must be float and greater than 0")
            return False
        try:
            assert self.row7_1.text(), "Please set pth address"
            a = True
        except Exception as err:
            QMessageBox.information(self, "Error", str(err))
            return False
        if a:
            try:
                os.path.exists(self.row7_1.text())

            except Exception as err:
                QMessageBox.information(self, "Error", "The address does not exist.")
                return False
        try:
            assert torch.cuda.is_available(), "This computer has no gpu"

            assert self.row7_1.text(), "Please set pth address"
            assert 0 <= int(transparency) <= 100,\
                "transparency should be between 0 and 100"

            return True
        except Exception as err:
            #print(err)
            QMessageBox.information(self, "Error", str(err))
            return False

    def getUet(self):
        self.row7_1.setText(self.getPath())

    def getWPUnet(self):
        self.row9_1.setText(self.getPath())

    def getPath(self):
        fname = QFileDialog.getOpenFileName(self, 'Choose file',  "", "Parameters file(*.pth *.pkl)")
        # fname = QFileDialog.getOpenFileName(self, "Choose path", "C:/")
        if len(fname)<1:
            return
        else:

            return fname[0]
        # return ""
##  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = setting()  # 创建窗体
    form.show()
    sys.exit(app.exec_())