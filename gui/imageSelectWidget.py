from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QWidget, QScrollArea, QGraphicsOpacityEffect, QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QPoint,QDir
from gui.widgets.notificationWidget import NotificationWidget
from os.path import join, dirname


class ImageSelectWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageSelectWidget, self).__init__(parent)
        self.setObjectName("ImageSelectWidget")
        with open(join(dirname(QDir(QDir().currentPath()).currentPath()), "gui", "qss", "imgSelector.qss"), 'r',
                  encoding='utf-8') as f:  # 样式表
            self.setStyleSheet(f.read())
        self.initUi()

    def initUi(self):
        self.setMinimumSize(1000, 130)
        self.setButtons()
        self.setWidget()
        self.setLayouts()

    def setButtons(self):
        ## 上一张图片按钮
        self.previousLabel = ButtonLabel(objectName="PreviousLabel")
        self.previousLabel.setPixmap(QPixmap.fromImage(QImage(':/image/previous.png')))
        self.previousLabel.setFixedWidth(80)
        self.previousLabel.setAlignment(Qt.AlignCenter)
        self.previousLabel.press.connect(self.previous)

        ## 下一张图片按钮
        self.nextLabel = ButtonLabel(objectName="NextLabel")
        self.nextLabel.setPixmap(QPixmap.fromImage(QImage(':/image/next.png')))
        self.nextLabel.setFixedWidth(80)
        self.nextLabel.setAlignment(Qt.AlignCenter)
        self.nextLabel.press.connect(self.next)

    def setWidget(self):
        ## 图片选择区域
        self.imgWidget = ImageListArea()
        self.imgWidget.setObjectName("ImgWidget")
        self.imgWidget.outRangeSignal.connect(self.outRangeNotification)

    def setLayouts(self):
        # 水平布局
        hLayout = QHBoxLayout()
        hLayout.addWidget(self.previousLabel, Qt.AlignLeft)
        hLayout.addWidget(self.imgWidget, Qt.AlignCenter)
        hLayout.addWidget(self.nextLabel, Qt.AlignRight)
        hLayout.setSpacing(0)
        hLayout.setContentsMargins(3, 15, 3, 15)  # 设置与外边框的距离
        self.setLayout(hLayout)

    def addItem(self, id, imgPath):
        self.imgWidget.addItem(id, imgPath)

    def previous(self):
        # 上一张图片
        self.imgWidget.previous()

    def next(self):
        # 下一张图片
        self.imgWidget.next()

    def outRangeNotification(self, sign):
        ## 点击图片超出范围提示
        if sign == 0:
            NotificationWidget.info('提示', '已经是第一张图片了', pos=(self.notifyPos[0],
                                                            self.notifyPos[1]))
        elif sign == 1:
            NotificationWidget.info('提示', '已经是最后一张图片了', pos=(self.notifyPos[0],
                                                             self.notifyPos[1]))
        elif sign == -1:
            NotificationWidget.info('提示', '还没有添加任何图片', pos=(self.notifyPos[0],
                                                            self.notifyPos[1]))

    def updateNotifyPos(self):
        x = self.parentWidget().mapToGlobal(QPoint(0, 0)).x()  # 获取屏幕坐标
        y = self.parentWidget().mapToGlobal(QPoint(0, 0)).y()
        w = self.parentWidget().size().width()  # 获取高度和宽度
        h = self.parentWidget().size().height()
        self.notifyPos = (x + int(w / 2) - 200, y + int(h / 2) - 100)  # 窗口位置


class ButtonLabel(QLabel):
    press = pyqtSignal()

    def __init__(self, objectName="ButtonLabel"):
        super(ButtonLabel, self).__init__()
        self.setObjectName(objectName)

    def mousePressEvent(self, e):
        self.press.emit()


class ImageListArea(QScrollArea):
    showKeySignal = pyqtSignal(str)
    outRangeSignal = pyqtSignal(int)

    def __init__(self):
        super(ImageListArea, self).__init__()
        self.scrollBar = self.horizontalScrollBar()  # 水平滚动按钮
        self.setFrameShape(self.NoFrame)
        self.setWidgetResizable(True)  # 自适应大小
        self.imgListWidget = QWidget()
        self.setWidget(self.imgListWidget)

        self.gLayout = QGridLayout()  # 网格布局
        self.imgListWidget.setLayout(self.gLayout)

        self.imgItemList = []
        self.curImgId = -1
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def addItem(self, id, imgPath):
        # 添加一张图片
        item = ItemWidget(id, imgPath)
        item.selectImgSignal.connect(self.setSelected)  # 图片点击事件
        self.imgItemList.append(item)
        self.updateLayout()

    def resizeEvent(self, event):
        super(ImageListArea, self).resizeEvent(event)
        self.updateLayout()

    def updateLayout(self):
        ## 更新布局
        self.imgListWidget.setFixedWidth((self.size().height() - 20) * 1.57 * (len(self.imgItemList)))  # 根据滑动栏的宽度设置宽度
        self.imgListWidget.setFixedHeight(self.size().height() - 20)  # 根据滑动栏的高度设置高度
        ## 设置每一个item的高度与宽度
        for i in range(len(self.imgItemList)):
            item = self.imgItemList[i]
            item.setFixedHeight(self.size().height() - 20)
            item.setFixedWidth((self.size().height() - 20) * 1.57)
        ## 删除之前的布局
        for i in range(len(self.imgItemList)):
            self.gLayout.removeWidget(self.imgItemList[i])
        ## 添加新的布局
        for i in range(len(self.imgItemList)):
            self.gLayout.addWidget(self.imgItemList[i], 0, i)
        self.imgListWidget.setLayout(self.gLayout)

    def setSelected(self, id):
        # 设置选中图片
        for i in range(len(self.imgItemList)):
            item = self.imgItemList[i]
            if item.id == id:
                self.curImgId = i
                item.setSelected()
                item.isSelected = True
            else:
                item.setNoSelected()
                item.isSelected = False
        # 发送处理信号
        self.showKeySignal.emit(id)

    def updatePos(self, id):
        # 找到需要更新的位置
        newImgId = 0
        for item in self.imgItemList:
            if item.id == id:
                break
            newImgId += 1
        # 每个item大小
        itemWidth = int(self.imgListWidget.size().width() / len(self.imgItemList))
        # 移动位置
        self.scrollBar.setValue(itemWidth * newImgId - 2 * itemWidth)
        self.curImgId = newImgId

    def next(self):
        # 选择图片和移动位置
        if self.curImgId != -1 and self.curImgId < len(self.imgItemList) - 1:
            self.setSelected(self.imgItemList[self.curImgId + 1].id)
            self.updatePos(self.imgItemList[self.curImgId].id)
        elif self.curImgId == -1:
            self.outRangeSignal.emit(-1)
        else:
            self.outRangeSignal.emit(1)

    def previous(self):
        # 选择图片和移动位置
        if self.curImgId != -1 and self.curImgId > 0:
            self.setSelected(self.imgItemList[self.curImgId - 1].id)
            self.updatePos(self.imgItemList[self.curImgId].id)
        elif self.curImgId == -1:
            self.outRangeSignal.emit(-1)
        else:
            self.outRangeSignal.emit(0)


class ItemWidget(QWidget):
    selectImgSignal = pyqtSignal(str)

    def __init__(self, id, imgPath):
        super(ItemWidget, self).__init__()
        self.id = id
        self.imgPath = imgPath
        vLayout = QVBoxLayout()
        self.imgLabel = QLabel("")
        img = QPixmap.fromImage(QImage(self.imgPath))
        self.imgLabel.setPixmap(img)
        self.imgLabel.setAlignment(Qt.AlignCenter)
        self.imgLabel.setScaledContents(True)
        vLayout.addWidget(self.imgLabel)
        self.setLayout(vLayout)
        self.isSelected = False

    def enterEvent(self, e):
        self.imgLabel.setStyleSheet("border:5px solid #4F8216;")

    def leaveEvent(self, e):
        if self.isSelected == False:
            self.imgLabel.setStyleSheet("")

    def setSelected(self):
        # 设置为不透明
        op = QGraphicsOpacityEffect()
        op.setOpacity(1)
        self.imgLabel.setGraphicsEffect(op)
        self.imgLabel.setAutoFillBackground(True)
        self.imgLabel.setStyleSheet("border:5px solid #4F8216;")

    def setNoSelected(self):
        ## 设置为半透明
        op = QGraphicsOpacityEffect()
        op.setOpacity(0.4)
        self.imgLabel.setGraphicsEffect(op)
        self.imgLabel.setAutoFillBackground(True)
        self.imgLabel.setStyleSheet("")

    def mousePressEvent(self, e):
        self.selectImgSignal.emit(self.id)
