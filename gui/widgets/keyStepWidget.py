from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import os
from PyQt5.QtCore import Qt
import math
from os.path import join


class KeyStepWidget(QWidget):
    '''定义关键步骤界面'''
    def __init__(self, parent=None, objectName="KeyResultWidget"):
        super(KeyStepWidget, self).__init__(parent)
        # 默认文件夹名为空
        self.name = ''
        self.setObjectName(objectName)
        self.initUi()
        # 默认测试图片路径
        self.testPath = '../demo/test_images/'
        # 默认生成测试结果的路径
        self.resultPath = '../demo/test_result/'

    def setName(self, imgName):
        """设置处理文件夹的名字

        :param imgName: 文件夹名
        """
        self.name = imgName

    def initUi(self):
        pass

    def paintEvent(self, event):
        """创建绘图对象
        """
        qp = QPainter()
        qp.begin(self)
        self.drawBackground(qp)
        if self.name != '':
            # 绘制主要步骤图
            self.drawImg(qp)
            # 绘制箭头
            self.drawArrow(qp)
            # 显示预测结果
            self.drawPredict(event, qp)
        qp.end()

    # 画箭头
    def drawArrow(self, qp):
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.02, 0.06, 0.25, 0.3, 2)
        qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(':/image/arrow_right.png')))

        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.51, 0.56, 0.12, 0.17, 2)
        qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(':/image/arrow_right.png')))

        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.7, 0.75, 0.25, 0.35, 0.5)
        qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(':/image/arrow_down.png')))

    # 画背景
    def drawBackground(self, qp):
        col = QColor(0, 0, 0)
        col.setNamedColor("white")
        # 设置边框颜色
        qp.setPen(col)
        # 背景颜色
        qp.setBrush(QColor(255, 255, 255))
        qp.drawRect(0, 0, self.width(), self.height())

    # 画显示的各个图片
    def drawImg(self, qp):

        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.06, 0.5, 0.1, 0.6, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(QImage(join(self.resultPath, self.name, '1.切割和矫正.jpg'))))

        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.6, 0.88, 0.08, 0.22, 15)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(QImage(join(self.resultPath, self.name, '2.银行卡卡号区域.jpg'))))

        # 要统计的文件夹
        DIR = join(self.resultPath, self.name, '3.卡号分割')

        # 获取数字文件夹的文件个数，即卡号数目
        totalNums = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        poi = 1

        for row in range(math.ceil(totalNums / 10)):
            for col in range(10):
                # 水平方向每个银行卡卡号的间隔比例
                padingRatio = 0.03
                x, y, w, h = self.drawArea1(self.width(), self.height(),
                                            0.55 + padingRatio * col, 0.65 + padingRatio * (col + 1),
                                            0.4 + 0.05 * row, 0.4 + 0.05 * (row + 1),
                                            0.8)
                qp.drawPixmap(x, y, w, h,
                              QPixmap.fromImage(QImage(join(self.resultPath, self.name, '3.卡号分割', str(poi) + '.jpg'))))
                poi += 1

    def drawPredict(self, event, qp):
        """用于绘制预测结果"""

        # 打开预测结果的txt文件
        f = open(join(self.resultPath, self.name, '4.预测结果.txt'), 'r')
        with f:
            data = f.read()
        f.close()

        # 设置画笔及颜色
        qp.setPen(QColor(100, 0, 0))
        # 计算预测结果矩形框的位置及大小
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.85, 0.7, 0.9, 12)
        qp.drawRect(x, y, w, h)
        # 画笔颜色
        qp.setPen(QColor(0, 0, 0))
        # 字体设置
        qp.setFont(QFont('黑体', 35))
        qp.drawText(x - 220, y, 240, h, Qt.AlignCenter, '识别结果：')
        # 画笔颜色
        qp.setPen(QColor(255, 0, 0))
        # 字体设置
        qp.setFont(QFont('黑体', 35))
        qp.drawText(x, y, w, h, Qt.AlignCenter, data)

    def drawArea1(self, width, hight, beginWidthRatio, endWidthRatio, beginHightRatio, endHightRatio, ratio):
        """设置矩形框宽度与高度范围，根据宽高比计算绘制内容的位置，使其在满足宽高比的情况下在此矩形框中居中

        :param width: 矩形框总宽度
        :param hight: 矩形框总高度
        :param beginWidthRatio: 开始绘制的位置与总宽度的比例
        :param endWidthRatio: 结束绘制的位置与总宽度的比例
        :param beginHightRatio: 开始绘制的位置与总高度的比例
        :param endHightRatio: 结束绘制的位置与总高度的比例
        :param ratio: 宽高比
        :return: 绘制起点x坐标，y坐标，绘制图像宽度，绘制图像高度
        """

        # 限制框宽度
        aeraWidth = width * endWidthRatio - width * beginWidthRatio
        # 限制框高度
        aeraHight = hight * endHightRatio - hight * beginHightRatio

        if aeraWidth / aeraHight <= ratio:
            # 图片的高度
            imgHight = aeraWidth / ratio
            imgWidth = aeraWidth
            # 图片起始x坐标
            beginX = width * beginWidthRatio + (aeraWidth - imgWidth) / 2
            # 图片起始y坐标
            beginY = hight * beginHightRatio + (aeraHight - imgHight) / 2
        else:
            # 图片的宽度
            imgWidth = aeraHight * ratio
            imgHight = aeraHight
            # 图片起始x坐标
            beginX = width * beginWidthRatio + (aeraWidth - imgWidth) / 2
            # 图片起始y坐标
            beginY = hight * beginHightRatio + (aeraHight - imgHight) / 2

        return beginX, beginY, imgWidth, imgHight
