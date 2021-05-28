# coding:utf-8

from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtWidgets import MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import image_prediction as ip
import os


class MainWindow_MINIST(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow_MINIST, self).__init__()
        self.ui_setup(self)

        # 初始化值
        self.img_path = None

    def ui_setup(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # 主窗口
        MainWindow.setObjectName('手写字体识别')
        # MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 1000)

        # 打开文件按钮
        self.openfile_button = QtWidgets.QPushButton(MainWindow)
        self.openfile_button.setGeometry(QtCore.QRect(50, 50, 150, 50))
        font1 = QtGui.QFont()
        font1.setPointSize(11)
        font1.setBold(True)
        font1.setWeight(75)
        self.openfile_button.setFont(font1)
        self.openfile_button.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.openfile_button.setAutoFillBackground(True)
        self.openfile_button.setIconSize(QtCore.QSize(20, 20))
        self.openfile_button.setAutoRepeatDelay(300)
        self.openfile_button.setText(_translate("MainWindow", "打开文件"))
        # self.openfile_button.setObjectName("pushButton_")

        # 二值化按钮
        self.bin_button = QtWidgets.QPushButton(MainWindow)
        self.bin_button.setGeometry(QtCore.QRect(50, 810, 150, 50))
        font2 = QtGui.QFont()
        font2.setPointSize(11)
        font2.setBold(True)
        font2.setWeight(75)
        self.bin_button.setFont(font2)
        self.bin_button.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.bin_button.setAutoFillBackground(True)
        self.bin_button.setIconSize(QtCore.QSize(20, 20))
        self.bin_button.setAutoRepeatDelay(300)
        # self.bin_button.setObjectName("pushButton_2")
        self.bin_button.setText(_translate("MainWindow", "二值化"))

        # 显示结果按钮
        self.result_button = QtWidgets.QPushButton(MainWindow)
        self.result_button.setGeometry(QtCore.QRect(50, 900, 150, 50))
        font3 = QtGui.QFont()
        font3.setPointSize(11)
        font3.setBold(True)
        font3.setWeight(75)
        self.result_button.setFont(font3)
        self.result_button.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.result_button.setAutoFillBackground(True)
        self.result_button.setIconSize(QtCore.QSize(20, 20))
        self.result_button.setAutoRepeatDelay(300)
        # self.result_button.setObjectName("pushButton_3")
        self.result_button.setText(_translate("MainWindow", "显示结果"))

        # 展示图片框
        self.pic_label = QtWidgets.QLabel(MainWindow)
        self.pic_label.setGeometry(QtCore.QRect(250, 50, 1200, 900))
        self.pic_label.setStyleSheet("border: 1px solid black;")
        # self.pic_widget.setObjectName("widget")

        # 黑白阈值条 灰度=value*2+29
        self.threshold_bar = QtWidgets.QScrollBar(MainWindow)
        self.threshold_bar.setGeometry(QtCore.QRect(50, 120, 20, 600))
        self.threshold_bar.setOrientation(QtCore.Qt.Vertical)
        self.threshold_bar.setValue(50)  # 初始阈值129
        # self.threshold_bar.setValue(1000)
        # self.threshold_bar.setStyleSheet("border: 1px solid black;")
        # self.threshold_bar.setObjectName("verticalScrollBar")
        self.label_1 = QtWidgets.QLabel(MainWindow)
        self.label_1.setGeometry(QtCore.QRect(40, 750, 40, 20))
        self.label_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_1.setTextFormat(QtCore.Qt.AutoText)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_1.setObjectName("label")
        self.label_1.setText(_translate("MainWindow", "黑白"))

        # 画框最小值条 0~198
        self.min_bar = QtWidgets.QScrollBar(MainWindow)
        self.min_bar.setGeometry(QtCore.QRect(115, 120, 20, 600))
        self.min_bar.setOrientation(QtCore.Qt.Vertical)
        self.min_bar.setValue(25)
        # self.min_bar.setObjectName("verticalScrollBar_2")
        self.label_2 = QtWidgets.QLabel(MainWindow)
        self.label_2.setGeometry(QtCore.QRect(100, 750, 45, 20))
        self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_2.setObjectName("label_2")
        self.label_2.setText(_translate("MainWindow", "最小框"))

        # 画框最大值 4000~5980 size = value*20+4000
        self.max_bar = QtWidgets.QScrollBar(MainWindow)
        self.max_bar.setGeometry(QtCore.QRect(180, 120, 20, 600))
        self.max_bar.setOrientation(QtCore.Qt.Vertical)
        self.max_bar.setValue(50)
        # self.max_bar.setObjectName("verticalScrollBar_3")
        self.label_3 = QtWidgets.QLabel(MainWindow)
        self.label_3.setGeometry(QtCore.QRect(170, 750, 45, 20))
        self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_3.setObjectName("label_3")
        self.label_3.setText(_translate("MainWindow", "最大框"))

        # 设置各种信号槽
        self.openfile_button.clicked.connect(self.open_click)

        self.bin_button.clicked.connect(self.bin_click)

        self.result_button.clicked.connect(self.result_click)

    def open_click(self):  # 打开文件按钮
        # print(self.threshold_bar.value())
        self.img_path, _ = QFileDialog.getOpenFileName(self, '选择文件')
        # print(file_path)

        # 读取，处理并显示图片
        img_ori = cv2.imread(self.img_path)
        img_ori = cv2.resize(img_ori, (1200, 900))  # 分辨率降低，否则太大
        # cv2.imshow('1',self.img_ori)

        shrink = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        # cv 图片转换成 qt图片
        img_ori_show = QtGui.QImage(
            shrink.data, shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3, QtGui.QImage.Format_RGB888)
        # label 控件显示图片
        self.pic_label.setPixmap(QtGui.QPixmap.fromImage(img_ori_show))
        # self.pic_label.show()

    def bin_click(self):
        if (self.img_path is not None):
            # 读取，处理并显示图片
            img_ori = cv2.imread(self.img_path)
            img_ori = cv2.resize(img_ori, (1200, 900))  # 分辨率降低，否则太大

            # 获得阈值
            threshold = self.threshold_bar.value()*2+29

            # 二值化
            img = ip.grayscale_image(img_ori)  # 灰度化
            img = ip.inverse_img(img)  # 灰度反相
            img = ip.binarization_img(img, threshold=threshold)  # 二值化

            # 显示
            shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv 图片转换成 qt图片
            img_show = QtGui.QImage(
                shrink.data, shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3, QtGui.QImage.Format_RGB888)
            # label 控件显示图片
            self.pic_label.setPixmap(QtGui.QPixmap.fromImage(img_show))
            # self.pic_label.show()
        else:
            QMessageBox.critical(MainWindow, '出错啦！', '未找到图片')

    def result_click(self):
        if (self.img_path is not None):
            # 读取，处理并显示图片
            img_ori = cv2.imread(self.img_path)
            img_ori = cv2.resize(img_ori, (1200, 900))  # 分辨率降低，否则太大

            threshold = self.threshold_bar.value()*2+29
            minvalue = self.min_bar.value()*2
            maxvalue = self.max_bar.value()*20+4000

            img = ip.grayscale_image(img_ori)
            img = ip.inverse_img(img)  # 灰度反相
            img = ip.binarization_img(img, threshold=threshold)  # 二值化

            borders = ip.borders_img(img, minsize=minvalue, maxsize=maxvalue)
            img_data = ip.minist_img(img, borders)
            result = ip.prediction(img_data)

            img = ip.draw_borders(img_ori, borders, result)

            # 显示
            shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv 图片转换成 qt图片
            img_show = QtGui.QImage(
                shrink.data, shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3, QtGui.QImage.Format_RGB888)
            # label 控件显示图片
            self.pic_label.setPixmap(QtGui.QPixmap.fromImage(img_show))
        else:
            QMessageBox.critical(MainWindow, '出错啦！', '未找到图片')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindow_MINIST()
    ui.ui_setup(MainWindow)
    MainWindow.show()
    # print(ui.threshold_bar.value())
    sys.exit(app.exec_())

    os.system("pause")
