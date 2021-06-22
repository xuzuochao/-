from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QWidget, \
    QBoxLayout, QFrame, QFileDialog, QListWidget, QListWidgetItem, QListView, QScrollArea, QGridLayout, \
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QSpacerItem, QGridLayout, QSpacerItem, QSizePolicy, \
    QGraphicsDropShadowEffect, \
    QListWidgetItem
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap, QPainter, QColor, QPainterPath
from PyQt5.QtCore import Qt, QFile, QSize, pyqtSignal, QPoint, QTimer, QRectF, QThread, QDir
import sys
from os.path import basename, join, dirname
from gui.imageSelectWidget import ImageSelectWidget
from gui.widgets.keyStepWidget import KeyStepWidget
from gui.widgets.detailProcessWidget import detailProcessWidget
import ctypes
from gui.widgets.waitWidget import WaitWidget
from gui.image import image_rc
import hashlib
from card_number_identification.process_one import processOne, loadModel
from copy import deepcopy
import time


class BaseWidget(QWidget):
    """整个GUI界面的基础widget

    """

    def __init__(self):
        super(BaseWidget, self).__init__()
        self.initUi()
        self.imgList = []  # 所有图像列表
        self.curImgId = ""  # 当前识别和显示的图片Id
        self.curShow = ""  # 当前显示的界面

        # 线程1只存储关键结果
        self.imgProcessThread1 = ImgProcessThread()
        self.imgProcessThread1.processOver.connect(self.processOver)
        self.imgProcessThread1.modelLoadSignal.connect(self.modelLoadOver)
        self.imgProcessThread1.start()

        self.judgeThread = JudgeThread()  # 监控是否识别完成的线程
        self.judgeThread.showSignal.connect(self.resultShow)
        self.judgeThread.start()

    def initUi(self):
        ## 窗口初始化
        self.setMinimumSize(1030, 700)  # 最小大小
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("bankCardNumberIdentification")  # 添加进程号
        self.setWindowTitle('银行卡卡号识别')
        self.setWindowIcon(QIcon(":/image/bankCard.png"))

        with open(join(dirname(QDir(QDir().currentPath()).currentPath()), "gui", "qss", "base.qss"), 'r',
                  encoding='utf-8') as f:  # 样式表
            self.setStyleSheet(f.read())
        self.setObjectName('MainWindow')  # 名称

        self.setWidgets()
        self.setButtons()
        self.setLayouts()

    def setWidgets(self):
        self.imageSelectWidget = ImageSelectWidget()
        self.imageSelectWidget.imgWidget.showKeySignal.connect(self.selectOneImg)
        self.keyResultWidget = KeyStepWidget(self, objectName="KeyResultWidget")
        self.detailProcessWidget = detailProcessWidget(self, objectName="DetailProcessWidget")
        self.detailProcessWidget.hide()
        self.waitWidget = WaitWidget(self, objectName="WaitWidget")
        self.waitWidget.hide()

    def setButtons(self):
        # 按钮字体设置
        font = QFont('黑体', 100)
        font.setLetterSpacing(QFont.PercentageSpacing, -2)
        font.setLetterSpacing(QFont.AbsoluteSpacing, -4)

        self.hintButton = QPushButton('正在加载模型...', self, objectName="HintButton")  # 提示按钮
        self.hintButton.setFont(font)
        self.hintButton.setFixedSize(260, 45)  # 固定大小

        self.selectButton = QPushButton("点击添加图片", self, objectName="SelectButton")  # 添加图片按钮
        self.selectButton.setFixedSize(300, 45)
        self.selectButton.setFont(font)
        self.selectButton.clicked.connect(self.selectImg)

        self.keyResultButton = QPushButton("关键步骤结果", self, objectName="KeyResultButton")  # 关键结果按钮
        self.keyResultButton.clicked.connect(self.keyButtonPress)
        self.keyResultButton.setFixedSize(150, 55)

        self.detailProcessButton = QPushButton("详细识别过程", self, objectName="DetailProcessButton")  # 详细过程按钮
        self.detailProcessButton.clicked.connect(self.detailButtonPress)
        self.detailProcessButton.setFixedSize(150, 55)

    def setLayouts(self):
        v = QVBoxLayout()
        ## 设置顶部布局
        topLayout = QHBoxLayout()  # 水平布局
        topLayout.setAlignment(Qt.AlignTop)  # 布局置顶

        topLayout.addWidget(self.hintButton, 2, Qt.AlignLeft)  # 按钮居左
        topLayout.addWidget(self.selectButton, 6, Qt.AlignCenter)  # 按钮居中
        topLayout.addWidget(self.keyResultButton, 1, Qt.AlignRight | Qt.AlignBottom)  # 按钮局右下
        topLayout.addWidget(self.detailProcessButton, 0, Qt.AlignRight | Qt.AlignBottom)  # 按钮居右下
        topLayout.setSpacing(0)  # 按钮间隔为0

        ## 设置中部布局
        midLayout = QHBoxLayout()
        midLayout.addWidget(self.keyResultWidget)
        midLayout.addWidget(self.detailProcessWidget)
        midLayout.addWidget(self.waitWidget)

        ## 设置底部布局
        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.imageSelectWidget)

        v.addLayout(topLayout)
        v.addLayout(midLayout)
        v.addLayout(bottomLayout)
        v.setSpacing(0)
        v.setContentsMargins(0, 10, 0, 0)  # 设置与外边框的距离
        self.setLayout(v)
        v.setStretchFactor(midLayout, 3)
        v.setStretchFactor(bottomLayout, 1)

    def selectImg(self):
        """点击'添加图片'按钮的操作

        :return:
        """
        files, ok = QFileDialog.getOpenFileNames(self,
                                                 "打开多张或单张银行卡",
                                                 "../demo/test_images/",
                                                 "(*.jpg *.png *.bmp *.jpeg *.png *.jfif *.jpe  *.gif *.dib)")
        if files != []:
            old_ids = self.getIds(self.imgList)
            for file in files:
                id = self.getMD5(file)
                if id not in old_ids:
                    self.imgList.append(Img(id, basename(file), file))
                    self.imageSelectWidget.addItem(self.imgList[-1].id, self.imgList[-1].path)
                    self.imgProcessThread1.processQueue.append(self.imgList[-1])  # 将图片加入处理队列
            if len(self.imgList) != len(old_ids):  # 如果添加了图片，重新选择item
                self.imageSelectWidget.imgWidget.setSelected(self.imgList[len(old_ids)].id)  # 设置默认选择item
                self.imageSelectWidget.imgWidget.updatePos(self.imgList[len(old_ids)].id)

    def selectOneImg(self, id):
        """选择一张图片的操作

        :param id: 图像ID
        :return: 空
        """
        self.imgProcessThread1.adjustPriority(id)  # 调整此id的图片优先识别
        self.curImgId = id  # 当前的图片
        self.curShow = "keyStep"  # 当前显示的界面
        self.keyButtonPress()  # 相当于按下关键步骤按钮

    def resultShow(self):
        if self.curShow == 'detailProcess':
            self.detailProcessShow()
        elif self.curShow == "keyStep":
            self.keyStepShow()
        else:
            pass

    def keyStepShow(self):
        """显示关键步骤结果

        :return:
        """
        imgIndex = self.getImgIndex(self.curImgId)
        if imgIndex == -1:  # 如果没有添加图片，什么也不做
            return
        if self.imgList[imgIndex].isRecognition == True:
            self.judgeThread.isJudge = False  # 关闭判断线程
            self.waitWidget.hide()
            self.detailProcessWidget.hide()
            self.keyButtonPressStyle()
            # 主线是关键步骤widget
            self.keyResultWidget.show()
            # 如果识别完成，直接显示结果
            self.keyResultWidget.setName(self.imgList[imgIndex].name)
            self.keyResultWidget.update()
            self.print("北海沐光队")
            self.print(self.imgList[imgIndex].name)
        else:
            # 如果未识别完成
            # 设置50ms的定时器判断识别是否完成
            self.judgeThread.isJudge = True
            # 显示等待框
            self.keyResultWidget.hide()
            self.detailProcessWidget.hide()
            self.keyButtonPressStyle()
            self.waitWidget.show()
            self.waitWidget.movie.start()
            # 提示消息
            self.print("正在识别")

    def detailProcessShow(self):
        imgIndex = self.getImgIndex(self.curImgId)
        if imgIndex == -1:
            return
        if self.imgList[imgIndex].isRecognitionWithDebug == True:
            self.judgeThread.isJudge = False
            self.waitWidget.hide()
            self.keyResultWidget.hide()
            self.detailButtonPressStyle()
            self.detailProcessWidget.show()
            # 如果识别完成，直接显示结果
            self.detailProcessWidget.setPath(self.imgList[imgIndex].path)
            self.detailProcessWidget.update()
            self.print("北海沐光队")
        else:
            # 如果未识别完成
            # 设置50ms的定时器判断处理完成没
            self.judgeThread.judgeType = 'detailProcess'
            self.judgeThread.isJudge = True
            # 显示等待框
            self.keyResultWidget.hide()
            self.detailProcessWidget.hide()
            self.detailButtonPressStyle()
            self.waitWidget.show()
            self.waitWidget.movie.start()
            # 提示消息
            self.print("正在识别")

    def keyButtonPress(self):
        self.curShow = 'keyStep'
        self.keyButtonPressStyle()
        self.resultShow()

    def detailButtonPress(self):
        self.curShow = 'detailProcess'
        self.detailButtonPressStyle()
        self.resultShow()

    def keyButtonPressStyle(self):
        self.keyResultButton.setStyleSheet("QPushButton{background-color:white}"
                                           "QPushButton:hover{border:2px solid #508C87;}"
                                           "QPushButton:pressed {border:3px solid #508C87;}"
                                           )
        self.detailProcessButton.setStyleSheet("QPushButton{background-color:#d4dad7}"
                                               "QPushButton:hover{border:2px solid #508C87;}"
                                               "QPushButton:pressed {border:3px solid #508C87;}")

    def detailButtonPressStyle(self):
        self.detailProcessButton.setStyleSheet("QPushButton{background-color:white}"
                                               "QPushButton:hover{border:2px solid #508C87;}"
                                               "QPushButton:pressed {border:3px solid #508C87;}")
        self.keyResultButton.setStyleSheet("QPushButton{background-color:#d4dad7}"
                                           "QPushButton:hover{border:2px solid #508C87;}"
                                           "QPushButton:pressed {border:3px solid #508C87;}")

    def resizeEvent(self, e):
        # 窗口放大缩小时重新计算提示窗位置
        self.imageSelectWidget.updateNotifyPos()

    def moveEvent(self, e):
        self.imageSelectWidget.updateNotifyPos()

    def getMD5(self, path):
        ## 获取文件的MD5值
        file = open(path, 'rb')
        md5 = hashlib.md5(file.read()).hexdigest()
        file.close()
        return md5

    def getIds(self, imgList):
        """获取图片名称列表

        :param imgList:
        :return:
        """
        ids = []
        for i in range(len(imgList)):
            ids.append(imgList[i].id)
        return ids

    def getImgIndex(self, id):
        '''从列表中获取对应ID的索引'''
        index = -1
        for i in range(len(self.imgList)):
            if self.imgList[i].id == id:
                index = i
                break
        return index

    def processOver(self, id, debug):
        imgIndex = self.getImgIndex(id)
        if debug:
            self.imgList[imgIndex].isRecognitionWithDebug = True
        else:
            self.imgList[imgIndex].isRecognition = True

    def modelLoadOver(self, str):
        """模型加载完成"""
        self.print(str)
        self.showDefaultTip()

    def showDefaultTip(self):
        self.timer = QTimer(self, timeout=self.print)
        self.timer.setSingleShot(True)  # 只触发一次
        self.timer.start(3000)

    def print(self, str="北海沐光队"):
        self.hintButton.setText(str)


class Img():
    def __init__(self, md5, name, path):
        self.id = md5  # 以md5值作为ID
        self.name = name  # 图片名称,默认区别不同图片的标志
        self.isRecognition = False  # 是否完成卡号识别
        self.isRecognitionWithDebug = False  # 能显示中间详细过程处理结果
        self.path = path  # 路径


class ImgProcessThread(QThread):
    processOver = pyqtSignal(str, bool)  # 处理完成之后的信号
    modelLoadSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.processQueue = []
        self.model = None

    def run(self):
        while (True):
            self.msleep(2)
            if self.model != None:
                ##模型加载完成
                if self.processQueue != []:
                    pendImg = deepcopy(self.processQueue[0])
                    del self.processQueue[0]
                    try:
                        self.processOne(pendImg.path, debug=False)
                        # 告诉主线程处理完成
                        self.processOver.emit(pendImg.id, False)
                        self.processOne(pendImg.path, debug=True)
                        # 告诉主线程处理完成
                        self.processOver.emit(pendImg.id, True)
                    except:
                        print("识别错误")
            else:
                self.loadModel()
                self.modelLoadSignal.emit("模型加载完成")

    def processOne(self, imgPath, debug):
        '''处理一张图片'''
        stratTime = time.time()
        processOne(imgPath, self.model, debug)
        endTime = time.time()
        #print("recognition_time:{}".format(endTime - stratTime))

    def loadModel(self):
        '''加载模型'''
        self.model = loadModel()

    def adjustPriority(self, id):
        '''调整处理的优先级'''
        for i in range(len(self.processQueue)):
            if self.processQueue[i].id == id:
                if i != 0:  # 如果不是第一个
                    processQueueT = deepcopy(self.processQueue)
                    process = deepcopy(processQueueT[i])
                    processQueueT.remove(processQueueT[i])
                    processQueueT.insert(0, process)
                    if len(processQueueT) == len(self.processQueue):
                        self.processQueue = processQueueT
                        break
                    else:
                        pass


class JudgeThread(QThread):
    '''判断是否处理完成的线程'''
    showSignal = pyqtSignal()

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.isJudge = False  # 是否正在做判断

    def run(self):
        while (True):
            self.msleep(2)
            if self.isJudge == True:
                self.msleep(50)
                self.isJudge = False
                self.showSignal.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BaseWidget()
    main_window.show()
    sys.exit(app.exec_())
