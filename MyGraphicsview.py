from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, \
     QGraphicsScene, QGraphicsView
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, QIcon, QMouseEvent
from PyQt5.QtCore import Qt, QRectF, QSizeF, QPointF


        # fitInView(graph, QRectF(QPointF(0, 0), QSizeF(qt_img.size())), Qt.KeepAspectRatio)


def add_move_drag(graph, qt_img):
    graph.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    graph.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    # wheelEvent(graph, graph.event)

    # fitInView(graph, QRectF(QPointF(0, 0), QSizeF(qt_img.size())), Qt.KeepAspectRatio)
    # fitInView(graph, QRectF(QPointF(0, 0), QSizeF(qt_img.size())), Qt.KeepAspectRatioByExpanding)
    # fitInView(graph, QRectF(QPointF(0, 0), QSizeF(qt_img.size())), Qt.IgnoreAspectRatio)


def fitInView(graph, rect, flags=Qt.IgnoreAspectRatio):
    viewRect = graph.viewport().rect()
    sceneRect = graph.transform().mapRect(rect)

    print(f"view:{viewRect.size()}")
    print(f"scene:{sceneRect.size()}")

    ratio_x = viewRect.width() / sceneRect.width()
    ratio_y = viewRect.height() / sceneRect.height()

    if flags == Qt.KeepAspectRatio:
        ratio_x = ratio_y = min(ratio_x, ratio_y)
    elif flags == Qt.KeepAspectRatioByExpanding:
        ratio_x = ratio_y = max(ratio_x, ratio_x)
    elif flags == Qt.IgnoreAspectRatio:
        ratio_x, ratio_y = ratio_x, ratio_y

    graph.scale(ratio_x, ratio_y)
    graph.centerOn(rect.center())
#
# def mousePressEvent(graph, event):
#     if event.button() == Qt.LeftButton:
#         graph.middleMouseButtonPress(event)
#     else:
#         super().mousePressEvent(event)
#
# # 判断鼠标松开的类型
# def mouseReleaseEvent(graph, event):
#     if event.button() == Qt.LeftButton:
#         graph.middleMouseButtonRelease(event)
#     else:
#         super().mouseReleaseEvent(event)
#
# # 拖拽功能 - 按下 的实现
# def middleMouseButtonPress(graph, event):
#     # 设置画布拖拽
#     graph.setDragMode(QGraphicsView.ScrollHandDrag)
#     fakeEvent = QMouseEvent(event.type(), event.localPos(),
#                             event.screenPos(),
#                             Qt.LeftButton, event.buttons() | Qt.LeftButton,
#                             event.modifiers())
#     super().mousePressEvent(fakeEvent)
#
# # 拖拽功能 - 松开 的实现
# def middleMouseButtonRelease(graph, event):
#     fakeEvent = QMouseEvent(event.type(), event.localPos(),
#                             event.screenPos(),
#                             Qt.LeftButton, event.buttons() & ~Qt.LeftButton,
#                             event.modifiers())
#     super().mouseReleaseEvent(fakeEvent)
#     # 取消拖拽
#     graph.setDragMode(QGraphicsView.NoDrag)
#
# # 滚轮缩放的实现
# def wheelEvent(graph, event):
#     # 放大触发
#     if event.angleDelta().y() > 0:
#         zoomFactor = 1 + 0.1
#     # 缩小触发
#     else:
#         zoomFactor = 1 - 0.1
#     graph.scale(zoomFactor, zoomFactor)