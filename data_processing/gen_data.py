'''运行本文件生成训练数据'''

import cv2
from card_number_positioning.util import cv2_imread
from os.path import join
import os
import numpy as np
import copy
from card_number_positioning.util import laplacian
from card_number_positioning.util import adaptive_binary
from card_number_positioning.util import del_connected_region
from card_number_positioning.util import vertical_histogram
from data_processing.num_split import single_num, data_class, write_num

# 项目根目录，例"D:\ExperimentalPapers\MyPaperCode\version1"
root_dir = os.path.abspath(os.path.dirname(__file__))

# 测试图片目录
datasets_dir = join(root_dir, 'dataset')


def process_data(image_gray, image_rgb, file_name, cur_count, ratio):
    """通过一系列图像处理操作，调用num_split与write_num，完成数据的生成

    :param image_gray: 数据集的灰度图像
    :param image_rgb: 数据集图片
    :param file_name: 正在切分的数据的文件路径名
    :param cur_count: 当前数据序号
    ：:param ratio: 数据集中生成训练集的比例
    :return:
    """

    # 边缘检测
    img_dege = laplacian(image_gray, 3)
    # 二值化
    dst = adaptive_binary(img_dege, 'OTSU')
    # 膨胀
    kernel2 = np.ones((2, 2), np.uint8)
    dst = cv2.dilate(dst, kernel2, iterations=1)
    # 去除小区域
    dst = del_connected_region(dst, 9)
    # 生成直方图
    dst_histogram = vertical_histogram(dst)
    # 进行分割
    seg_area = single_num(dst_histogram, 28, 2)
    # 显示分割后的图像
    img_show3 = copy.deepcopy(image_rgb)
    for i in range(len(seg_area) - 1):
        cv2.rectangle(img_show3, (seg_area[i], 0), (seg_area[i + 1], 120), (255, 0, 0), 1)
        cv2.line(img_show3, (seg_area[i], 0), (seg_area[i], 120), (255, 0, 0), 1)

    # 数字分割
    # 仅仅进行切割的文件数目
    num = int(1084 * (1 - ratio))
    only_split = np.random.randint(1084, size=num)
    data, test_data, cur_count, file_name = data_class(image_rgb, seg_area, file_name, cur_count, only_split)
    # 将数字图写入文件
    # 训练集
    write_num(data, cur_count, file_name, 'train')
    # 测试集
    write_num(test_data, cur_count, file_name, 'test')


def gen_data(ratio):
    """运行此函数完成训练数据以及测试数据的生成

    :param:ratio 数据集中生成训练集的比例
    :return:
    """

    # 当前数据序号
    cur_count = 0
    # 报错数据量
    error_count = 0
    # 测试图像路径
    test_img_path = datasets_dir
    for file_name in os.listdir(test_img_path):
        # 路径
        path = join(test_img_path, file_name)
        # 读取图像(黑白)
        img = cv2_imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 读取图像(彩色)
        img_color = cv2_imread(join(test_img_path, file_name))

        # 循环处理一张图像，下标异常处理
        try:
            process_data(img, img_color, file_name, cur_count, ratio)
        except IndexError:
            error_count += 1
            print(error_count)
            print("--------------------")
            print("index Error" + file_name)

        print("第%d个数据处理完成" % cur_count)
        # 计算总数
        cur_count += 1

    print(cur_count)


if __name__ == '__main__':
    ratio = 0.8
    gen_data(ratio)
