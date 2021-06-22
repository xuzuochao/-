'''找到数据集的卡号图片的单个数字分割位置，完成数字切割并保存文件。'''

import cv2
import numpy as np
from data_processing.augmentation_func import augmentation
from os.path import join
import random
from card_number_positioning.util import mkdir
from torchvision import transforms


def single_num(dst_histogram, step_size, min_poi):
    """通过直方图判断银行卡号开始和结束的位置

    :param dst_histogram:数据集图片二值化后的直方图
    :param step_size:切割位置移动的步长
    :param min_poi:第一个数字的起始位置
    :return:银行卡号每个数字位置的开始位置和结束位置
    """

    # 每个数字水平方向的切割位置
    seg_area = []
    # 将第一个切割位置加入列表
    seg_area.append(min_poi)
    # 当前切割数字的水平方向位置
    cur_poi = min_poi
    # 当前切割位置加上步长后小于数据集图片长度，则继续寻找下一切割位置
    while cur_poi + step_size <= 120:
        for poi_col in range(cur_poi, cur_poi + step_size):
            # 在当前切割位置到加上步长后的区域内，
            # 若直方图白色像素占比大于0则认为该区域是数字

            total_pixel = 0
            for poi_row in range(dst_histogram.shape[0]):
                if dst_histogram[poi_row][poi_col] == 255:
                    total_pixel += 1
            mean = total_pixel / (dst_histogram.shape[0] * step_size)
            if mean >= 0:
                if cur_poi not in seg_area:
                    seg_area.append(cur_poi)
                cur_poi += step_size
        # 添加最后切割位置，默认倒数第二个像素处（120-2）
        seg_area.append(118)

        # 判断是否划了五条线,如果少于5条线则会进行添加
        if len(seg_area) < 5:
            for index in range(len(seg_area)):
                dis = seg_area[index + 1] - seg_area[index]
                # 若两条线的距离大于步长1.5倍，则认为缺少一个切割位置
                if dis >= 1.5 * step_size:
                    seg_area.append(118)
                    for temp in range(len(seg_area) - 1, 1, -1):
                        seg_area[temp] = seg_area[temp - 1]
                        seg_area[index + 1] = seg_area[index + 2] - step_size

    return seg_area

def num_split(image_rgb, seg_area, num_resize=46):
    """ 通过四个数字各自的起始于结束位置，将四个数字复制

       :param image_rgb: 数据集图片
       :param seg_area: 银行卡号每个数字位置的开始位置和结束位置
       :param num_resize: 重新设置每个数字图片的尺寸
       :return: 四个银行卡号数字
       """
    # 第一个数字
    num_one = np.zeros([46, 27, 3], np.uint8)
    # 第二个数字
    num_two = np.zeros([46, 27, 3], np.uint8)
    # 第三个数字
    num_three = np.zeros([46, 27, 3], np.uint8)
    # 第四个数字
    num_four = np.zeros([46, 29, 3], np.uint8)

    # 复制前三个数字
    for row in range(46):
        for col in range(27):
            for channel in range(3):
                num_one[row][col][channel] = image_rgb[row][col + seg_area[0]][channel]
                num_two[row][col][channel] = image_rgb[row][col + seg_area[1]][channel]
                num_three[row][col][channel] = image_rgb[row][col + seg_area[2]][channel]

    # 复制第四个数字,因为尺寸不同的原因，所以分开复制
    for row in range(46):
        for col in range(29):
            for channel in range(3):
                num_four[row][col][channel] = image_rgb[row][col + seg_area[3]][channel]

    # 设置第一个图片尺寸
    num_one = cv2.resize(num_one, (num_resize, num_resize))
    # 设置第二个图片尺寸
    num_two = cv2.resize(num_two, (num_resize, num_resize))
    # 设置第三个图片尺寸
    num_three = cv2.resize(num_three, (num_resize, num_resize))
    # 设置第四个图片尺寸
    num_four = cv2.resize(num_four, (num_resize, num_resize))

    return num_one, num_two, num_three, num_four

def data_class(image_rgb, seg_area, file_name, cur_count, only_split):
    """ 根据only_split列表确定仅仅进行分割的数据，将其作为测试集，剩余的数据作为训练集

    :param image_rgb: 数据集图片
    :param seg_area: 银行卡号每个数字位置的开始位置和结束位置
    :param file_name: 正在切分的数据的文件路径名
    :param cur_count: 当前数据序号
    :param only_split:仅仅进行分割的数据序号
    :return: 训练集数据，测试集数据，当前数据序号，正在切分的数据的文件路径名
    """

    # 用于保存测试集
    test_data = []
    # 用于保存测试集，增强后的数据
    data = [[], [], [], []]
    # 随机生成仅仅进行分割的数据序号,如果当前文件序号在随机选择的只进行分割的序号中则执行
    if cur_count in only_split:
        num_one, num_two, num_three, num_four = num_split(image_rgb, seg_area)
        test_data.extend([num_one, num_two, num_three, num_four])

    else:
        num_one, num_two, num_three, num_four = num_split(image_rgb, seg_area)

        data[0] = augmentation(num_one)
        data[1] = augmentation(num_two)
        data[2] = augmentation(num_three)
        data[3] = augmentation(num_four)

        # 随机将四个数字中的40个resize为15*15，20*20，25*25，30*30
        for i in range(4):
            # 随机选10个resize为15
            for size_15 in random.sample(range(0, 80), 10):
                data[i][size_15] = cv2.resize(data[i][size_15], (15, 15))

            # 随机选10个resize为20
            for size_20 in random.sample(range(0, 80), 10):
                data[i][size_20] = cv2.resize(data[i][size_20], (20, 20))

            # 随机选10个resize为25
            for size_25 in random.sample(range(0, 80), 10):
                data[i][size_25] = cv2.resize(data[i][size_25], (25, 25))

            # 随机选10个resize为30
            for size_30 in random.sample(range(0, 80), 10):
                data[i][size_30] = cv2.resize(data[i][size_30], (30, 30))

    return data, test_data, cur_count, file_name

def write_num(data, cur_count, file_name, file_type):
    """通过获得内存中的复制数字将数字写入文件中

    :param data: 存放四个数字的八十张数据增强图片，共320张
    :param cur_count: 当前数据集图片的序号
    :param file_name: 当前数据集图片的文件名
    :return:

            文件命名格式：当前数字标签_当前原文件数据集中位置_当前文件个数
    """

    # 写入数据的编号
    data_count = 0
    # 创建文件夹
    # 判断是训练集还是测试集
    if file_type == 'train':
        path_0 = "data/train/0"
        path_1 = "data/train/1"
        path_2 = "data/train/2"
        path_3 = "data/train/3"
        path_4 = "data/train/4"
        path_5 = "data/train/5"
        path_6 = "data/train/6"
        path_7 = "data/train/7"
        path_8 = "data/train/8"
        path_9 = "data/train/9"
        path_none = "data/train/10"

    else:
        path_0 = "data/test/0"
        path_1 = "data/test/1"
        path_2 = "data/test/2"
        path_3 = "data/test/3"
        path_4 = "data/test/4"
        path_5 = "data/test/5"
        path_6 = "data/test/6"
        path_7 = "data/test/7"
        path_8 = "data/test/8"
        path_9 = "data/test/9"
        path_none = "data/test/10"

    mkdir(path_0)
    mkdir(path_1)
    mkdir(path_2)
    mkdir(path_3)
    mkdir(path_4)
    mkdir(path_5)
    mkdir(path_6)
    mkdir(path_7)
    mkdir(path_8)
    mkdir(path_9)
    mkdir(path_none)

    # 当前数字是该数据集的第几个数字
    cur_num = 0
    # 根据数据集的标签将切割的数字图写入相应文件夹中
    for file_name_num in file_name[0:4]:
        # 写入训练集数据
        if file_type == 'train' and len(data[0]) > 0:
            # 每个数字写入80张
            for poi in range(80):
                # 写入0文件夹
                if file_name_num == '0':
                    cv2.imwrite(join(path_0, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入1文件夹
                elif file_name_num == '1':
                    cv2.imwrite(join(path_1, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入2文件夹
                elif file_name_num == '2':
                    cv2.imwrite(join(path_2, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入3文件夹
                elif file_name_num == '3':
                    cv2.imwrite(join(path_3, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入4文件夹
                elif file_name_num == '4':
                    cv2.imwrite(join(path_4, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入5文件夹
                elif file_name_num == '5':
                    cv2.imwrite(join(path_5, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入6文件夹
                elif file_name_num == '6':
                    cv2.imwrite(join(path_6, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入7文件夹
                elif file_name_num == '7':
                    cv2.imwrite(join(path_7, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入8文件夹
                elif file_name_num == '8':
                    cv2.imwrite(join(path_8, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入9文件夹
                elif file_name_num == '9':
                    cv2.imwrite(join(path_9, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入none文件夹
                elif file_name_num == '_':
                    cv2.imwrite(join(path_none, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num][poi], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1
        else:
            # 写入测试集数据
            if file_type == 'test' and len(data) > 0:
                # 写入0文件夹
                if file_name_num == '0':
                    cv2.imwrite(join(path_0, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入1文件夹
                elif file_name_num == '1':
                    cv2.imwrite(join(path_1, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入2文件夹
                elif file_name_num == '2':
                    cv2.imwrite(join(path_2, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入3文件夹
                elif file_name_num == '3':
                    cv2.imwrite(join(path_3, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入4文件夹
                elif file_name_num == '4':
                    cv2.imwrite(join(path_4, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入5文件夹
                elif file_name_num == '5':
                    cv2.imwrite(join(path_5, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入6文件夹
                elif file_name_num == '6':
                    cv2.imwrite(join(path_6, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入7文件夹
                elif file_name_num == '7':
                    cv2.imwrite(join(path_7, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入8文件夹
                elif file_name_num == '8':
                    cv2.imwrite(join(path_8, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入9文件夹
                elif file_name_num == '9':
                    cv2.imwrite(join(path_9, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1

                # 写入none文件夹
                elif file_name_num == '_':
                    cv2.imwrite(join(path_none, file_name_num + "_" + str(cur_count) + "_" + str(data_count) + ".png"),
                                cv2.normalize(data[cur_num], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    data_count += 1


        cur_num += 1

