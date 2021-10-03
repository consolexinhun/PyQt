# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1104, 758)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.sourceImg = ImageView(self.centralwidget)
        self.sourceImg.setGeometry(QtCore.QRect(10, 50, 531, 551))
        self.sourceImg.setObjectName("sourceImg")
        self.msg = QtWidgets.QLabel(self.centralwidget)
        self.msg.setGeometry(QtCore.QRect(30, 620, 741, 21))
        self.msg.setText("")
        self.msg.setObjectName("msg")
        self.targetImg = QtWidgets.QGraphicsView(self.centralwidget)
        self.targetImg.setGeometry(QtCore.QRect(560, 50, 531, 551))
        self.targetImg.setObjectName("targetImg")
        self.msg_width = QtWidgets.QLabel(self.centralwidget)
        self.msg_width.setGeometry(QtCore.QRect(120, 640, 121, 31))
        self.msg_width.setText("")
        self.msg_width.setObjectName("msg_width")
        self.msg_height = QtWidgets.QLabel(self.centralwidget)
        self.msg_height.setGeometry(QtCore.QRect(370, 640, 121, 31))
        self.msg_height.setText("")
        self.msg_height.setObjectName("msg_height")
        self.label_width = QtWidgets.QLabel(self.centralwidget)
        self.label_width.setGeometry(QtCore.QRect(20, 650, 72, 15))
        self.label_width.setObjectName("label_width")
        self.label_height = QtWidgets.QLabel(self.centralwidget)
        self.label_height.setGeometry(QtCore.QRect(270, 650, 72, 15))
        self.label_height.setObjectName("label_height")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(220, 20, 72, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(790, 20, 72, 15))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1104, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionClear = QtWidgets.QAction(MainWindow)
        self.actionClear.setObjectName("actionClear")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionRot90 = QtWidgets.QAction(MainWindow)
        self.actionRot90.setObjectName("actionRot90")
        self.actionThreshold = QtWidgets.QAction(MainWindow)
        self.actionThreshold.setObjectName("actionThreshold")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionCanny = QtWidgets.QAction(MainWindow)
        self.actionCanny.setObjectName("actionCanny")
        self.actionPSP = QtWidgets.QAction(MainWindow)
        self.actionPSP.setObjectName("actionPSP")
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionClear)
        self.menu.addAction(self.actionQuit)
        self.menu_2.addAction(self.actionRot90)
        self.menu_2.addAction(self.actionThreshold)
        self.menu_2.addAction(self.actionGray)
        self.menu_2.addAction(self.actionCanny)
        self.menu_3.addAction(self.actionPSP)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "组织病理图像分割软件"))
        self.label_width.setText(_translate("MainWindow", "图像宽度："))
        self.label_height.setText(_translate("MainWindow", "图像高度："))
        self.label.setText(_translate("MainWindow", "源图像"))
        self.label_2.setText(_translate("MainWindow", "目标图像"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "图像操作"))
        self.menu_3.setTitle(_translate("MainWindow", "图像分割"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpen.setText(_translate("MainWindow", "打开"))
        self.actionSave.setText(_translate("MainWindow", "保存"))
        self.actionClear.setText(_translate("MainWindow", "清空"))
        self.actionQuit.setText(_translate("MainWindow", "退出"))
        self.actionRot90.setText(_translate("MainWindow", "旋转90°"))
        self.actionThreshold.setText(_translate("MainWindow", "二值化"))
        self.actionGray.setText(_translate("MainWindow", "灰度化"))
        self.actionCanny.setText(_translate("MainWindow", "边缘提取"))
        self.actionPSP.setText(_translate("MainWindow", "深度监督PSP"))
from appMain import ImageView
