from PyQt5.QtWidgets import QWidget, QScrollArea, QStyleOption, QStyle
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from os.path import join, dirname, isfile, basename
import os
from PyQt5.QtCore import Qt, QDir
import math


class detailProcessWidget(QScrollArea):
    '''创建详细步骤滑动界面类'''
    def __init__(self, parent=None, objectName="DetailProcessWidget"):
        super(QScrollArea, self).__init__(parent)
        # 设置样式
        self.setObjectName(objectName)
        # 默认文件夹名为空
        self.name = ''
        self.initUi()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setPath(self, imgPath):
        """设置读取的文件路径

        :param imgPath: 读取的文件路径
        """
        # 正在处理的文件名
        self.name = basename(imgPath)
        # 传递文件夹名
        self.topFiller.setName(self.name)
        # 传递路径
        self.topFiller.setPath(imgPath)

    def setWidth(self, screenWidth):
        """# 传递屏幕显示宽度"""
        self.screenWidth = screenWidth

    def initUi(self):
        # 在本滑动界面中添加详细步骤界面
        self.topFiller = detailWidget()
        self.setWidget(self.topFiller)

    def resizeEvent(self, e):
        # 设置页面的宽度和高度
        self.topFiller.setFixedSize(self.parentWidget().width(),8000)

class detailWidget(QWidget):
    '''新建详细界面类'''
    def __init__(self):
        super(detailWidget, self).__init__()
        # 默认文件名为空
        self.name = ''
        self.initUi()
        # 默认测试文件路径为空
        self.imgPath = ""
        # 设置读取的文件路径
        self.resultPath = join(dirname(QDir(QDir().currentPath()).currentPath()),"demo","test_result")

    def initUi(self):
        pass

    def setName(self, imgName):
        """设置文件名

        :param imgName: 传入的文件名
        """
        self.name = imgName

    def setPath(self, imgPath):
        """设置文件路径

        :param imgPath: 传入的文件路径
        """
        self.imgPath = imgPath

    def paintEvent(self, event):
        """新建一个绘制类"""
        qp = QPainter()
        qp.begin(self)
        # 文件名不为空的时候进行绘制
        if self.name != '':
            # 绘制详细过程图以及介绍
            self.drawImg(qp)
        qp.end()

        # 以下几行代码的功能是避免在多重传值后的功能失效
        opt=QStyleOption()
        opt.initFrom(self)
        p=QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget,opt,p,self)

    # 画显示的各个图片
    def drawImg(self, qp):

        # 预测结果文件读取
        f = open(join(self.resultPath, self.name, '4.预测结果.txt'), 'r')
        with f:
            data = f.read()
        f.close()

        # 画原图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.005, 0.055, 1.57)
        qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(self.imgPath)))

        # 开始，原图
        qp.setPen(QColor(255, 0, 0))
        qp.setFont(QFont('黑体', 15))  # 字体设置
        qp.drawText(x + 0.4 * w, y - 40, 60, 40, Qt.AlignCenter, '原图')

        # 画第一步：进行LSD_line处理
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.055, 0.105, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(QImage(join(self.resultPath,self.name,'详细识别过程','1.LSD_Line.jpg'))))

        # 第一步介绍
        qp.drawText(x - 0.45 * w, y - h * 0.6, w, h, Qt.AlignCenter, '1：')
        qp.drawText(x - 0.05 * w, y - h * 0.6, w, h, Qt.AlignCenter, '进行LSD线段检测')

        # 画第二步：线段合并，生成所有水平直线
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.12, 0.42, 0.105, 0.155, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '2.allHLine.jpg'))))
        # 第二步介绍
        qp.drawText(x - 0.45 * w, y - h * 0.6, w, h, Qt.AlignCenter, '2.1：')
        qp.drawText(x - 0.15 * w, y - h * 0.6, w + 100, h, Qt.AlignCenter, '得到所有可能的水平分割线')

        # 画第二步：线段合并，生成所有垂直直线
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.47, 0.77, 0.105, 0.155, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '2.allVLine.jpg'))))
        # 第二步介绍
        qp.drawText(x - 0.45 * w, y - h * 0.6, w, h, Qt.AlignCenter, '2.2：')
        qp.drawText(x - 0.15 * w, y - h * 0.6, w + 100, h, Qt.AlignCenter, '得到所有可能的垂直分割线')

        # 画第三步：线段筛选，四条最长水平直线
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.12, 0.42, 0.155, 0.205, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '3.FourHLine.jpg'))))
        # 第三步介绍
        qp.drawText(x - 0.15 * w, y - h * 0.75, w*1.5, h, Qt.AlignCenter, '3.1：根据线段检测结果，选择')
        qp.drawText(x - 0.08 * w, y - h * 0.6, w*1.5, h, Qt.AlignCenter, '最有可能的四条水平分割线')

        # 画第三步：线段筛选，四条最长垂直直线
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.47, 0.77, 0.155, 0.205, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '3.FourVLine.jpg'))))

        # 第三步介绍
        qp.drawText(x - 0.15 * w, y - h * 0.75, w*1.5, h, Qt.AlignCenter, '3.2：根据线段检测结果，选择')
        qp.drawText(x - 0.08 * w, y - h * 0.6, w*1.5, h, Qt.AlignCenter, '最有可能的四条水平分割线')

        # 画第四步：选出银行卡水平边缘
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.12, 0.42, 0.205, 0.255, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '4.TwoHLine.jpg'))))
        # 第四步介绍
        qp.drawText(x - 0.15 * w, y - h * 0.75, w*1.5, h, Qt.AlignCenter, '4.1：根据四条水平分割线的斜率以')
        qp.drawText(x - 0.08 * w, y - h * 0.6, w*1.5, h, Qt.AlignCenter, '及距离比例确定银行卡水平边缘')

        # 画第四步：选出银行卡垂直边缘
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.47, 0.77, 0.205, 0.255, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '4.TwoVLine.jpg'))))
        # 第四步介绍
        qp.drawText(x - 0.15 * w, y - h * 0.75, w*1.5, h, Qt.AlignCenter, '4.2：根据四条线段距离与其中最大')
        qp.drawText(x - 0.08 * w, y - h * 0.6, w*1.5, h, Qt.AlignCenter, '距离的比例确定银行卡垂直边缘')

        # 画第五步：选出银行卡区域
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.255, 0.305, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '5.cut_and_rectify.jpg'))))
        # 第五步介绍
        qp.drawText(x, y - h * 0.6, w*1.1, h, Qt.AlignCenter, '5：得到银行卡的切割矫正图')

        # 画第六步：银行卡Sobel_1算子图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.305, 0.355, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '6.Sobel_1.jpg'))))
        # 第六步介绍
        qp.drawText(x-0.2*w, y - h * 0.6, w*1.5, h, Qt.AlignCenter, '6：使用Sobel_1算子对银行卡图进行处理')

        # 画第七步：银行卡Sobel_3算子图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.355, 0.405, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '7.Sobel_3.jpg'))))
        # 第七步介绍
        qp.drawText(x-0.3*w, y - h * 0.6, w*1.8, h, Qt.AlignCenter, '7：对上一步处理结果使用Sobel_3算子进行处理')

        # 画第八步：银行卡Sobel_5算子图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.405, 0.455, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '8.Sobel_5.jpg'))))
        # 第八步介绍
        qp.drawText(x-0.3*w, y - h * 0.6, w*1.8, h, Qt.AlignCenter, '8：对上一步处理结果使用Sobel_5算子进行处理')

        # 画第九步：银行卡均值滤波处理图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.455, 0.505, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '9.medianBlur.jpg'))))
        # 第九步介绍
        qp.drawText(x-0.3*w, y - h * 0.6, w*2, h, Qt.AlignCenter, '9：对Sobel算子处理后的边缘特征图进行均值滤波操作')

        # 画第十步：银行卡二值化处理图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.505, 0.555, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '10.adaptive_binary.jpg'))))
        # 第十步介绍
        qp.drawText(x, y - h * 0.6, w, h, Qt.AlignCenter, '10：进行二值化处理')

        # 画第十一步：去除连通区域处理图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.555, 0.605, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '11.del_connected_region.jpg'))))
        # 第十一步介绍
        qp.drawText(x-0.1*w, y - h * 0.6, w*1.2, h, Qt.AlignCenter, '11：去除二值化图像中的连通区域')

        # 画第十二步：卡号区域图
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.605, 0.655, 1.57)
        qp.drawPixmap(x, y, w, h,
                      QPixmap.fromImage(
                          QImage(join(self.resultPath, self.name, '详细识别过程', '12.num_aera.jpg'))))
        # 第十二步介绍
        qp.drawText(x-0.1*w, y - h * 0.6, w*1.2, h, Qt.AlignCenter, '12：根据像素密度框选出卡号区域')

        # 要统计的文件夹
        cardCategoryDir= join(self.resultPath,str(self.name),'详细识别过程')
        ducNums = len([name for name in os.listdir(cardCategoryDir) if isfile(os.path.join(cardCategoryDir, name))])
        # 根据文件数目判断凹凸字体与印刷字体
        if ducNums == 22:

            # 画第十三步：卡号区域图进行Sobel_3算子处理
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.655, 0.675, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '13.num_aeraSobel_3.jpg'))))
            # 第十三步介绍
            qp.drawText(x - 0.45 * w, y - h*1.3, w*2, h*1.1, Qt.AlignCenter, '13：使用Sobel_3算子对卡号区域进行处理')

            # 画第十四步：对Sobel_3算子处理结果进行二值化
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.675, 0.695, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '14.num_aeraBinary.jpg'))))
            # 第十四步介绍
            qp.drawText(x - 0.45*w, y - h*1.3, w*2, h*1.1, Qt.AlignCenter, '14：对Sobel_3算子处理结果进行二值化')

            # 画第十五步：对二值化结果进行膨胀操作
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.695, 0.715, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '15.num_aeraDilate.jpg'))))

            # 第十五步介绍
            qp.drawText(x-0.2*w, y - h*1.3, w*1.3, h*1.1, Qt.AlignCenter, '15：对二值化结果进行膨胀操作')

            # 画第十六步：去除膨胀结果图的连通区域
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.715, 0.735, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '16.num_aeraDelConnected.jpg'))))

            # 第十六步介绍
            qp.drawText(x, y - h*1.3, w, h*1.1, Qt.AlignCenter, '16：去除膨胀结果图的连通区域')

            # 画第十七步：银行卡号区域处理后的直方图
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.735, 0.755, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '17.num_aeraHistogram.jpg'))))

            # 第十七步介绍
            qp.drawText(x-0.2*w, y - h*1.3, w*1.5, h*1.1, Qt.AlignCenter, '17：得到银行卡号区域处理后的直方图')

            # 画第十八步：根据直方图像密度素分布规律得到号码分割位置
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.755, 0.775, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '18.num_aeraSplit.jpg'))))

            # 第十八步介绍
            qp.drawText(x - 0.5*w, y-h*1.3, w*2, h*1.1, Qt.AlignCenter, '18：根据直方图像密度素分布规律得到号码分割位置')

            # 要统计的文件夹
            DIR = join(self.resultPath, self.name, '3.卡号分割')
            totalNums = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            # 用于确定文件位置（第几个图）
            poi = 0
            # 获取数字文件夹的文件个数，即卡号数目
            for row in range(math.ceil(totalNums / 10)):
                for col in range(10):
                    # 水平方向每个银行卡卡号的间隔比例
                    padingRatio = 0.05
                    x, y, w, h = self.drawArea1(self.width(), self.height(),
                                                0.15 + padingRatio * col, 0.25 + padingRatio * (col + 1),
                                                0.77 + 0.003 * row, 0.77 + 0.003 * (row + 1),
                                                0.8)
                    qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(join(DIR, (str(poi + 1) + '.jpg')))))
                    poi += 1

            # 绘制预测结果
            qp.setPen(QColor(100, 0, 0))
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.85, 0.69, 0.89, 12)
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

        else:

            # 画第十三步：对卡号区域图进行二值化处理
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.655, 0.675, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name,'详细识别过程', '13.num_aeraBinary.jpg'))))

            # 第十三步介绍
            qp.drawText(x-0.15*w, y - h*1.3, w*1.3, h*1.1, Qt.AlignCenter, '13：对卡号区域图进行二值化处理')

            # 画第十四步：对二值化结果进行膨胀操作
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.675, 0.695, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '14.num_aeraDilate.jpg'))))

            # 第十四步介绍
            qp.drawText(x-0.2*w, y - h*1.3, w*1.3, h*1.1, Qt.AlignCenter, '14：对二值化结果进行膨胀操作')

            # 画第十五步：银行卡号区域处理后的直方图
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.695, 0.715, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '15.num_aeraHistogram.jpg'))))

            # 第十五步介绍
            qp.drawText(x - 0.1 * w, y - h*1.3, w*1.2, h*1.1, Qt.AlignCenter, '15：银行卡号区域处理后的直方图')

            # 画第十六步：根据直方图像密度素分布规律得到号码分割位置
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.6, 0.715, 0.735, 15)
            qp.drawPixmap(x, y, w, h,
                          QPixmap.fromImage(
                              QImage(join(self.resultPath, self.name, '详细识别过程', '16.num_aeraSplit.jpg'))))

            # 第十六步介绍
            qp.drawText(x - 0.4 * w, y - h*1.3, w*2, h*1.1, Qt.AlignCenter, '16：根据直方图像密度素分布规律得到号码分割位置')

            # 要统计的文件夹
            DIR = join(self.resultPath,self.name,'3.卡号分割')
            totalNums = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            poi = 0
            # 获取数字文件夹的文件个数，即卡号数目
            for row in range(math.ceil(totalNums / 10)):
                for col in range(10):
                    # 水平方向每个银行卡卡号的间隔比例
                    padingRatio = 0.05
                    x, y, w, h = self.drawArea1(self.width(), self.height(),
                                                0.15 + padingRatio * col, 0.25 + padingRatio * (col + 1),
                                                0.735 + 0.003 * row, 0.735 + 0.003 * (row + 1),
                                                0.8)
                    qp.drawPixmap(x, y, w, h, QPixmap.fromImage(QImage(join(DIR, (str(poi+1) + '.jpg')))))
                    poi += 1

            qp.setPen(QColor(100, 0, 0))
            x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.85, 0.65, 0.85, 12)
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

    def drawTexts(self, event, qp):
        f = open(join(self.resultPath,self.name,'4.预测结果.txt'), 'r')
        with f:
            data = f.read()
        f.close()

        qp.setPen(QColor(100, 0, 0))
        x, y, w, h = self.drawArea1(self.width(), self.height(), 0.3, 0.85, 0.8, 0.95, 12)
        qp.drawRect(x, y, w, h)
        # 画笔颜色
        qp.setPen(QColor(0, 0, 0))
        # 字体设置
        qp.setFont(QFont('黑体', 35))
        qp.drawText(x-220, y, 240, h, Qt.AlignCenter, '识别结果：')
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
            imgWidth = aeraHight * ratio  # 图片的宽度
            imgHight = aeraHight
            # 图片起始x坐标
            beginX = width * beginWidthRatio + (aeraWidth - imgWidth) / 2
            # 图片起始y坐标
            beginY = hight * beginHightRatio + (aeraHight - imgHight) / 2

        imgWidth = imgWidth*0.8
        imgHight = imgHight*0.8

        return beginX, beginY, imgWidth, imgHight

