'''公共函数文件'''

import cv2
import os
import numpy as np
from os.path import join, basename, abspath, dirname
from PyQt5.QtCore import QDir
from copy import deepcopy as dp


# 设置全局变量
def init():
    global global_dict
    global_dict = {}


def check_folder(dir):
    '''初始化目录'''
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        print("check_folder_error")


init()
check_folder(join(dirname(QDir(QDir().currentPath()).currentPath()), 'demo', 'test_result'))


def set_value(name, value):
    global_dict[name] = value


def get_value(name, defValue=None):
    try:
        return global_dict[name]
    except KeyError:
        return defValue


set_value('save_root_path', join(dirname(QDir(QDir().currentPath()).currentPath()), 'demo', 'test_result'))


def fixed_thresh_binary(img, min_thresh=127, max_thresh=255):
    """固定阈值分割

    :param img: 灰度图像
    :param min_thresh: 分割阈值下界
    :param max_thresh: 阈值分割上届
    :return: 二值化图像
    """

    _, th = cv2.threshold(img, min_thresh, max_thresh, cv2.THRESH_BINARY)
    return th


def adaptive_binary(img, method='OTSU'):
    """自适应二值化

    :param img: 灰度图像
    :param method: 自适应二值化方法选择method=['MEAN_C','GAUSSIAN_C','OTSU']，默认'OTSU'.
    :return: 二值化图像
    """

    if method == 'MEAN_C':
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    elif method == 'GAUSSIAN_C':
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    elif method == 'OTSU':

        # 大津阈值法
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th


def mkdir(path):
    """创建十一个分类的文件夹,保存数据集处理结果

    :param path: 创建文件夹的路径
    :return: bool
    """

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


def sobel_x(img, ksize=3):
    """水平方向sobel边缘检测算子

    :param img: 灰度图像
    :param ksize: 卷积核大小
    :return: sobel特征边缘图像
    """

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=ksize)
    # convert 转换  scale 缩放
    return cv2.convertScaleAbs(x)


def sobel_y(img, ksize=3):
    """垂直方向sobel边缘检测算子

    :param img: 灰度图像
    :param ksize: 卷积核大小
    :return: sobel特征边缘图像
    """

    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=ksize)
    return cv2.convertScaleAbs(y)


def sobel(img, ksize_x=3, ksize_y=3):
    """sobel边缘检测算子

    :param img: 灰度图像
    :param ksize_x: 水平方向，卷积核大小
    :param ksize_y: 垂直方向，卷积核大小
    :return: sobel特征边缘图像
    """

    return cv2.addWeighted(sobel_x(img, ksize_x), 0.5, sobel_y(img, ksize_y), 0.5, 0)


def scharr(img):
    """scharr边缘检测算子

    :param img: 灰度图像
    :return: scharr特征边缘图像
    """

    x = sobel_x(img, ksize=-1)
    y = sobel_y(img, ksize=-1)
    # convert 转换  scale 缩放
    Scharr_absX = cv2.convertScaleAbs(x)
    Scharr_absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(Scharr_absX, 0.5, Scharr_absY, 0.5, 0)


def laplacian(img, ksize=1):
    """laplacian边缘检测算子

    :param img: 灰度图像
    :param ksize: 卷积核大小
    :return: laplacian特征边缘图像
    """

    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)


def canny(img, min_threshold=50, max_threshold=150):
    """

    :param img: 灰度图像
    :param min_threshold: 阈值上届
    :param max_threshold: 阈值上界
    :return: canny特征边缘图像
    """

    return cv2.Canny(img, min_threshold, max_threshold)


def vertical_histogram(img):
    """统计图像垂直方向上的直方图

    :param img: 二值图像
    :return: 二值化的直方图
    """

    h = img.shape[0]
    w = img.shape[1]
    img_histogram = np.zeros(img.shape)
    # 统计每一列的白色像素点个数
    for i in range(w):
        count = 0
        for j in range(h):
            if img[j][i] == 255:
                count += 1
        for m in range(h):
            if m > h - count:
                img_histogram[m][i] = 255
    return img_histogram


def horizontal_histogram(img):
    """统计图像水平方向上的直方图

    :param img: 二值图像
    :return: 二值化的直方图
    """

    h = img.shape[0]
    w = img.shape[1]
    img_histogram = np.zeros([w, h])
    # 统计每一行的白色像素点个数
    for i in range(h):
        count = 0
        for j in range(w):
            if img[i][j] == 255:
                count += 1
        for m in range(w):
            if m > w - count:
                img_histogram[m][i] = 255
    return img_histogram


def cv2_imread(file_path):
    """重写cv2.imread()函数,通过cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式

    :param file_path: 图像文件路径
    :return: 返回uint8格式图像
    """

    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def cv2_imwrite(file_path, file):
    """重写cv2.imwrite()函数

    :param file_path: 图像文件路径
    :param file: uint8格式图像写入文件
    :return: 返回uint8格式图像
    """
    """重写cv2.imwrite()函数."""
    try:
        suffix = '.' + str(basename(file_path.rstrip()).split('.')[-1])
        cv2.imencode(suffix, file)[1].tofile(file_path)
    except:
        print(file_path)
        print("cv2_imwrite_error")


def del_connected_region(img, method="less", area_range=400):
    """删除小于给定面积的连通区域

    :param img: 二值图像
    :param method: 方法名
    :param area_range: 面积范围上界
    :return: 删除小于给定面积的连通区域的二值化图像
    """

    # 查找连通区域
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if method == 'less':
            # 判断连通区域面积
            if area < area_range:
                # 画出轮廓
                cv2.drawContours(img, [contours[i]], 0, 0, -1)
        else:
            # 判断连通区域面积
            if area > area_range:
                # 画出轮廓
                cv2.drawContours(img, [contours[i]], 0, 0, -1)
    return img


def saveProcessStep(img, debug, save_path, dir, file_name):
    # debug模式下复制一张图像用于保存中间结果,生成文件
    img_ = dp(img) if debug == True else None
    if debug == True:
        img_t = dp(img_)
        cv2_imwrite(join(save_path, dir, file_name), img_t)
        del img_t
