import sys

import cv2
import numpy as np
from PIL import Image, ImageQt

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, \
     QGraphicsScene, QProgressBar, QProgressDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread

from single_inference import do_inference, do_inference_progress
# from progress import ProgressBarWindow


class MyProgressBar(QMainWindow):
    def __init__(self, parent=None):
        super(MyProgressBar, self).__init__(parent)
        loadUi("progress.ui", self)


def graph_show(graph, img):
    """
    func:
        在 graph 上显示 img
    :param graph: QGraphicsView
    :param img: np
    :return:
    """
    pil_img = Image.fromarray(img)  # np -> PIL
    qt_img = ImageQt.ImageQt(pil_img)  # PIL -> QImage

    source_scene = QGraphicsScene()
    graph_width, graph_height = graph.width()-2, graph.height()-2
    source_scene.addPixmap(QPixmap.fromImage(qt_img).scaled(graph_width, graph_height))
    graph.setScene(source_scene)


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        loadUi("main.ui", self)

        self.source_content = None  # 原图 np 数据

        self.target_content = None

        self.img_name = None  # 图像路径
        # 菜单
        self.actionOpen.triggered.connect(self.openfile)  # 打开
        self.actionSave.triggered.connect(self.savefile)  # 保存图片
        self.actionClear.triggered.connect(self.clear)  # 清空
        self.actionQuit.triggered.connect(QApplication.quit)  # 退出

        # 图像操作
        self.actionRot90.triggered.connect(self.rot90)  # 旋转90度
        self.actionThreshold.triggered.connect(self.threshold)  # 阈值
        self.actionGray.triggered.connect(self.gray)  # 灰度化
        self.actionCanny.triggered.connect(self.canny)  # 边缘提取

        # 图像分割
        self.actionPSP.triggered.connect(self.psp)  # psp 语义分割

    def openfile(self):
        """
        func:
            打开文件

        :return:
        """
        QMessageBox.information(self, "提醒", "请选择图片，并且路径不能包含中文", QMessageBox.Ok)

        self.img_name, img_type = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;*.jpg;")
        img = cv2.imread(rf"{self.img_name}")

        self.msg.setText(f"文件名称为{self.img_name}")

        if img is None:
            self.msg.setText(f"打开文件{self.img_name}错误")
            return

        img_h, img_w = img.shape[:2]
        self.msg_width.setText(f"{img_w}")
        self.msg_height.setText(f"{img_h}")

        self.source_content = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        graph_show(self.sourceImg, self.source_content)

    def savefile(self):
        if self.target_content is None:
            QMessageBox.warning(self, "警告", "请对图像进行操作", QMessageBox.Ok)
            return

        save_name, save_type = QFileDialog.getSaveFileName(self, "保存图片", "")

        if save_name is None or save_name == "":
            return

        if len(self.target_content.shape) == 2:
            cv2.imwrite(save_name, self.target_content)
        else:
            cv2.imwrite(save_name, self.target_content[:, :, ::-1])  # 保存图像必须是 BGR 的

    def rot90(self):
        """
        func:
            将图像旋转 90 度
        :return:
        """
        if self.target_content is None:
            self.target_content = self.source_content

        self.target_content = np.rot90(self.target_content)
        graph_show(self.targetImg, self.target_content)

    def threshold(self):
        """
        func:
            阈值分割
        :return:
        """
        self.target_content = self.source_content
        self.target_content = cv2.cvtColor(self.target_content, cv2.COLOR_RGB2GRAY)
        threshold, self.target_content = cv2.threshold(self.target_content, 0, 255,
                                                       cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        self.msg.setText(f"二值化的阈值为:{threshold}")
        graph_show(self.targetImg, self.target_content)

    def gray(self):
        """
        func:
            灰度化
        :return:
        """
        self.target_content = self.source_content
        self.target_content = cv2.cvtColor(self.target_content, cv2.COLOR_RGB2GRAY)
        graph_show(self.targetImg, self.target_content)

    def canny(self):
        """
        func:
            Canny 边缘提取
        :return:
        """
        self.target_content = self.source_content
        self.target_content = cv2.cvtColor(self.target_content, cv2.COLOR_RGB2GRAY)
        self.target_content = cv2.Canny(self.target_content, 50, 120)
        graph_show(self.targetImg, self.target_content)

    def psp(self):
        """
        func:
            全监督PSP分割
        :return:
        """

        progress = MyProgressBar(self)
        progress.progressBar.setValue(0)
        progress.show()
        QApplication.processEvents()
        # for i in range(1, 101):
        #     progress.progressBar.setValue(i)
        #     QApplication.processEvents()
        self.target_content= do_inference_progress(self.img_name, progress)

        progress.close()
        graph_show(self.targetImg, self.target_content)


    def clear(self):
        """
        func:
            清空左右两边的图和底下的消息
        :return:
        """
        self.sourceImg.setScene(QGraphicsScene())
        self.targetImg.setScene(QGraphicsScene())

        self.msg_width.setText("")
        self.msg_height.setText("")
        self.msg.setText("")

        self.source_content = None  # 原图 np 数据
        self.target_content = None  # 变换后的 np 数据
        self.img_name = None  # 图像路径


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()

    sys.exit(app.exec_())
