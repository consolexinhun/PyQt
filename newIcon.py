import sys

from PyQt5.QtWidgets import QApplication, QAction, QStyle
from PyQt5.QtGui import QIcon, QPixmap

from appMain import MyMainWindow


class IconWindow(MyMainWindow):
    def __init__(self, parent=None):
        super(IconWindow, self).__init__(parent)

        # openAction = QAction(QIcon("icons/file-open.png"), "打开", self)
        openAction = QAction(QIcon( QApplication.style().standardIcon(QStyle.SP_DialogOpenButton) ), "&打开", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openfile)
        self.toolBar.addAction(openAction)

        saveAction = QAction(QIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton)), "&保存", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.savefile)
        self.toolBar.addAction(saveAction)

        clearAction = QAction(QIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload)), "&清空", self)
        clearAction.triggered.connect(self.clear)
        self.toolBar.addAction(clearAction)

        quitAction = QAction(QIcon(QApplication.style().standardIcon(QStyle.SP_LineEditClearButton)), "&退出", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.triggered.connect(QApplication.quit)
        self.toolBar.addAction(quitAction)

        rotAction = QAction(QIcon(QApplication.style().standardIcon(QStyle.SP_ArrowForward)), "&旋转", self)
        rotAction.triggered.connect(self.rot90)
        self.toolBar.addAction(rotAction)

        thresAction = QAction(QIcon("icons/threshold.png"), "&阈值", self)
        thresAction.triggered.connect(self.threshold)
        self.toolBar.addAction(thresAction)

        grayAction = QAction(QIcon("icons/gray.png"), "&灰度", self)
        grayAction.triggered.connect(self.gray)
        self.toolBar.addAction(grayAction)

        cannyAction = QAction(QIcon("icons/canny.png"), "&边缘", self)
        cannyAction.triggered.connect(self.canny)
        self.toolBar.addAction(cannyAction)

        pspAction = QAction(QIcon("icons/segmentation.png"), "&分割", self)
        pspAction.triggered.connect(self.psp)
        self.toolBar.addAction(pspAction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = IconWindow()
    myWin.show()

    sys.exit(app.exec_())
