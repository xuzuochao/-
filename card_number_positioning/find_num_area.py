'''寻找银行卡号区域'''

import cv2
from copy import deepcopy as dp
import numpy as np
from os.path import join
from card_number_positioning.util import sobel_x, adaptive_binary, del_connected_region, get_value, cv2_imwrite, \
    saveProcessStep

# 全局变量用于切换运行模式
debug = get_value("debug")
name = get_value("name")
save_path = get_value("save_path")


def seg_num_area(img):
    """分割银行卡号区域,通过Sobel算子进行边缘检测后，进行均值滤波、二值化等操作，
        最后分割银行卡卡号区域.

    :param img: 银行卡灰度图片
    :return: 银行卡号区域框位置,上边界和下边界
    """

    set_global_variable()
    # 复制输入图像
    dst = dp(img)
    # 图像灰度化
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # 水平方向Sobel算子(卷积核大小=1)
    dst = sobel_x(dst, ksize=1)

    # debug模式下复制一张图像用于保存中间结果,生成Sobel_1处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '6.Sobel_1.jpg')

    # 水平方向Sobel算子(卷积核大小=3)
    dst = sobel_x(dst, ksize=3)

    # debug模式下复制一张图像用于保存中间结果,生成Sobel_3处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '7.Sobel_3.jpg')

    # 水平方向Sobel算子(卷积核大小=5)
    dst = sobel_x(dst, ksize=5)

    # debug模式下复制一张图像用于保存中间结果,生成Sobel_5处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '8.Sobel_5.jpg')

    # 均值滤波
    dst = cv2.medianBlur(dst, 5)

    # debug模式下复制一张图像用于保存中间结果,生成均值滤波处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '9.medianBlur.jpg')

    # 大津阈值法二值化
    dst = adaptive_binary(dst, 'OTSU')

    # debug模式下复制一张图像用于保存中间结果,生成大津阈值法二值化处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '10.adaptive_binary.jpg')

    dst = del_connected_region(dst)  # 去除小连通区域

    # debug模式下复制一张图像用于保存中间结果,生成去除连通区域后的处理图
    saveProcessStep(dst, debug, save_path, '详细识别过程', '11.del_connected_region.jpg')

    # 从上往下扫描，计算得分最高的区域
    box_height=31
    upper_border = calculate_roi_area(dst, step_size=1, box_height=box_height)

    num_rectAera = dp(dst)
    dst = cv2.cvtColor(num_rectAera, cv2.COLOR_GRAY2BGR)
    num_rectAera = cv2.rectangle(dst, (0, upper_border), (480, upper_border + box_height), (0, 0, 255), 4)

    # debug模式下复制一张图像用于保存中间结果,生成去除连通区域后的处理图
    saveProcessStep(num_rectAera, debug, save_path, '详细识别过程', '12.num_aera.jpg')

    return upper_border, upper_border + box_height


def set_global_variable():
    """在当前文件夹设置全局变量，用于保存中间结果

    :return:
    """

    global debug, name, save_path
    debug = get_value("debug")
    name = get_value("name")
    save_path = get_value("save_path")


def calculate_roi_area(img, step_size=4, box_height=35, padding=30):
    """寻找银行卡卡号区域，按固定画框尺寸从上到下对银行卡区域进行判断并计算白色像素占比情况（密度），密度最大
    即认为是银行卡号区域（银行卡上1/4区域不包括卡号）

    :param img: 传入的银行卡二值图像
    :param step_size: 滑动步长
    :param box_height: 滑框高度
    :param padding: 左右Padding范围内不计算分数
    :return: 银行卡卡号区域的边界
    """

    # 滑框分数
    score = []
    # 滑动总次数
    for i in range(int(img.shape[1] / step_size)):
        if i * step_size + box_height < img.shape[1]:
            # 当前滑框位置
            box = img[i * step_size:i * step_size + box_height, padding:img.shape[0] - padding]
            # 计算此画框内白色像素总数
            score.append(np.sum(box == 255))

    # 复制总数
    score_rank = dp(score)
    # 总数按照递减排序
    score_rank.sort(reverse=True)

    for i in range(len(score)):
        # 判断坐标范围是否在银行卡上1/4区域
        if (score.index(score_rank[i])) * step_size > 0.4 * img.shape[0] and \
                (score.index(score_rank[i])) * step_size+box_height < 0.74 * img.shape[0]:
            # 找到分数最高的垂直方向坐标
            x = (score.index(score_rank[i])) * step_size
            break
    return x
