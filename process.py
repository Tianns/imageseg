from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtWidgets import QApplication, QProgressBar, QPushButton
from PyQt5.QtCore import pyqtSignal
import time
from ui_process import Ui_Dialog

class Processbar(QtWidgets.QDialog):
    stop = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_Dialog()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面

    def setValue(self,val):
        # self.step=val
        self.ui.progressBar.setValue(val)

    def setText(self,str):
        self.ui.label.setText(str)
    def closeEvent(self, QCloseEvent):
        self.stop.emit()







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Processbar()
    window.show()
    sys.exit(app.exec_())
