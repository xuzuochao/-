'''分割银行卡号数字'''

import cv2
import numpy as np
from copy import deepcopy as dp
from os.path import join, dirname
from card_number_positioning.util import sobel_y, adaptive_binary, fixed_thresh_binary, del_connected_region, \
    vertical_histogram, horizontal_histogram, get_value, cv2_imwrite, saveProcessStep, canny


def num_segment(scale_img, original_img, upper_lower_border, scale):
    """对银行卡号区域进行切割，得到每个数字区域，具体为先判断卡号种类（凹凸字体或者印刷体），然后经过边缘检测后，
    进行均值滤波、二值化等操作，初步找到数字分割线，然后去掉不可信分割线，最后分割银行卡每个卡号区域

    :param scale_img: 缩放过的原图
    :param original_img: 原图
    :param upper_lower_border: 银行卡卡号区域上下边界(相对于缩放的图像的边界)
    :param scale: 图像缩放比例
    :return: 分割的卡号列表
    """

    set_global_variable()
    # 分割除银行卡数字区域
    dst = scale_img[upper_lower_border[0]:upper_lower_border[1], :, :]
    if is_print_font(dst) == False:
        # 图像灰度化
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        dst = sobel_y(dst, ksize=3)

        # debug模式下复制一张图像用于保存中间结果,生成sobel_y,卷积核为3的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '13.num_aeraSobel_3.jpg')

        # 二值化
        dst = adaptive_binary(dst, 'OTSU')

        # debug模式下复制一张图像用于保存中间结果,生成二值化的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '14.num_aeraBinary.jpg')

        kernel2 = np.ones((2, 2), np.uint8)
        # 膨胀
        dst = cv2.dilate(dst, kernel2, iterations=1)

        # debug模式下复制一张图像用于保存中间结果,生成膨胀后的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '15.num_aeraDilate.jpg')

        # 去除小连通区域
        dst = del_connected_region(dst, method="less", area_range=10)

        # debug模式下复制一张图像用于保存中间结果,生成去除小连通区域后的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '16.num_aeraDelConnected.jpg')

        # 生成直方图
        dst_histogram = vertical_histogram(dst)

        # debug模式下复制一张图像用于保存中间结果,生成直方图
        saveProcessStep(dst_histogram, debug, save_path, '详细识别过程', '17.num_aeraHistogram.jpg')

        # 获取水平方向卡号开始和结束的位置
        seg_pos_start, seg_pos_end = seg_pos(dst_histogram)

        # 获取直方图波谷
        trough_list = get_histogram_trough(dst_histogram, scope=10)

        # 根据直方图获取卡号可能的分割
        possible_seg = calculate_possible_seg(trough_list, seg_pos_start, seg_pos_end)

        imshow_1 = dp(scale_img[upper_lower_border[0]:upper_lower_border[1], :, :])
        for i in range(len(possible_seg)):
            cv2.line(imshow_1, (possible_seg[i], 0), (possible_seg[i], 304), (255, 0, 0), 1)

            # debug模式下复制一张图像用于保存中间结果,生成初步分割图
            saveProcessStep(imshow_1, debug, save_path, '详细识别过程', '18.num_aeraOriSplit.jpg')

        # 移除卡号错误的分割
        imshow_2 = dp(scale_img[upper_lower_border[0]:upper_lower_border[1], :, :])
        seg_area = del_seg_error(possible_seg)

        for i in range(len(seg_area)):
            cv2.line(imshow_2, (seg_area[i][0], 0), (seg_area[i][0], 304), (255, 0, 0), 2)
            # debug模式下复制一张图像用于保存中间结果,生成分割图
            saveProcessStep(imshow_2, debug, save_path, '详细识别过程', '19.num_aeraSplit.jpg')
            img_ = dp(imshow_2) if debug == False else None
            if debug == False:
                img_t = dp(img_)
                cv2_imwrite(join(dirname(save_path), 'card_' + get_value("name")), img_t)
                del img_t

        # 获取卡号的图片列表
        seg_num_img_list = seg_img(original_img, upper_lower_border, scale, seg_area)

    else:
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        (b, g, r) = cv2.split(dst)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        dst = cv2.merge((bH, gH, rH))

        dst = contrast_brightness(dst, 0.9, 10)
        # cv2.imshow("test", dst)
        # cv2.waitKey(0)
        # 图像灰度化
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # 二值化
        dst = 255 - fixed_thresh_binary(dst, min_thresh=60, max_thresh=255)

        # debug模式下复制一张图像用于保存中间结果,生成二值化的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '13.num_aeraBinary.jpg')

        kernel2 = np.ones((7, 7), np.uint8)
        # 膨胀
        dst = cv2.dilate(dst, kernel2, iterations=1)

        # debug模式下复制一张图像用于保存中间结果,生成膨胀的处理图
        saveProcessStep(dst, debug, save_path, '详细识别过程', '14.num_aeraDilate.jpg')

        # 生成直方图
        dst_histogram = vertical_histogram(dst)

        # debug模式下复制一张图像用于保存中间结果,生成直方图
        saveProcessStep(dst_histogram, debug, save_path, '详细识别过程', '15.num_aeraHistogram.jpg')

        # 获取水平方向卡号开始和结束的位置
        seg_pos_start, seg_pos_end = print_seg_pos(dst_histogram)

        # 获取直方图波谷
        trough_list = get_histogram_trough(dst_histogram, scope=10)

        # 根据直方图获取卡号可能的分割
        # 计算数字间距平均值
        mean = print_calculate_ave(trough_list, seg_pos_start, seg_pos_end)
        # 找到分割线位置
        possible_seg = print_calculate_possible_seg(trough_list, seg_pos_start, seg_pos_end, mean)
        imshow_2 = dp(scale_img[upper_lower_border[0]:upper_lower_border[1], :, :])
        for i in range(len(possible_seg)):
            cv2.line(imshow_2, (possible_seg[i], 0), (possible_seg[i], 304), (255, 0, 0), 1)

        # debug模式下复制一张图像用于保存中间结果,生成直方图
        saveProcessStep(imshow_2, debug, save_path, '详细识别过程', '16.num_aeraSplit.jpg')

        # 移除卡号错误的分割
        seg_area = print_del_seg_error(possible_seg)

        img_ = dp(imshow_2) if debug == False else None
        if debug == False:
            img_t = dp(img_)
            cv2_imwrite(join(dirname(save_path), 'card_' + get_value("name")), img_t)
            del img_t

        # 获取卡号的图片列表
        seg_num_img_list = print_seg_img(original_img, upper_lower_border, scale, seg_area)

    return seg_num_img_list


def contrast_brightness(image, c, b):
    """改变输入图的对比度与亮度, 调节亮度(b)和对比度(c)

    :param image: 输入RGB图
    :param c: 输出图所占权重
    :param b: 输出图所占权重
    :return: 改变对比度与亮度后的RGB图
    """

    h, w, channel = image.shape
    blank = np.zeros([h, w, channel], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, b)

    return dst


def set_global_variable():
    """在当前文件夹设置全局变量，用于保存中间结果

    :return:
    """

    global debug, name, save_path
    debug = get_value("debug")
    name = get_value("name")
    save_path = get_value("save_path")


def seg_pos(dst_histogram, padding=10):
    """通过直方图计算凹凸字体水平方向卡号区域开始和结束的位置

    :param dst_histogram: 直方图
    :param padding: 不包括计算范围的边界
    :return: 银行卡号位置的开始位置和结束位置
    """

    # 直方图的高
    h = dst_histogram.shape[0]
    # 直方图的宽
    w = dst_histogram.shape[1]
    seg_pos_start = padding
    seg_pos_end = w - padding

    # 寻找开始位置
    for i in range(padding, w - padding):
        if int(dst_histogram[h - 1][i]) == 255:
            seg_pos_start = i
            break

    # 寻找结束位置
    for i in range(padding, w - padding):
        if int(dst_histogram[h - 1][w - padding - i]) == 255:
            seg_pos_end = w - padding - i
            break

    return seg_pos_start, seg_pos_end


def print_seg_pos(dst_histogram, padding=1):
    """通过直方图计算印刷字体水平方向卡号区域开始和结束的位置

    :param dst_histogram: 直方图
    :param padding: 不包括计算范围的边界
    :return: 银行卡号位置的开始位置和结束位置
    """

    # 直方图的高
    h = dst_histogram.shape[0]
    # 直方图的宽
    w = dst_histogram.shape[1]
    seg_pos_start = padding
    seg_pos_end = w - padding

    # 寻找开始位置
    for i in range(padding, w - padding):
        if int(dst_histogram[h - 1][i]) == 255:
            seg_pos_start = i
            break

    # 寻找结束位置
    for i in range(padding, w - padding):
        if int(dst_histogram[h - 1][w - padding - i]) == 255:
            seg_pos_end = w - padding - i
            break

    return seg_pos_start, seg_pos_end


def horizontal_seg_pos(dst_histogram):
    """获取水平方向单个数字开始和结束的位置，对切割数字的优化，避免垂直方向切割过长

    :param dst_histogram: 单个数字的直方图
    :return: 单个银行卡号位置的开始位置和结束位置
    """

    # 直方图的高
    h = dst_histogram.shape[0]
    # 直方图的宽
    w = dst_histogram.shape[1]
    seg_pos_start = 0
    seg_pos_end = w

    try:
        # 寻找开始位置
        for i in range(0, w):
            seg_pos_start = 0
            # h - int((0.1*h))是为了避免少量的直方图显示干扰（花纹）
            if int(dst_histogram[h - int((0.1 * h))][i]) == 255:
                if i > 5:
                    i -= 4
                    seg_pos_start = i
                else:
                    seg_pos_start = 0
                break

        # 寻找结束位置
        for i in range(0, w):
            if int(dst_histogram[h - 1][w - i - 1]) == 255:
                seg_pos_end = w - i
                break
    except:
        print("horizontal_seg_pos_error")

    return seg_pos_start, seg_pos_end


def get_histogram_trough(img, scope=6):
    """计算直方图可能的波谷，通过判断该列像素点总数是否比scope范围内的每一列白色像素点总数小，如果都小的话，
    即为波谷。是波谷记录值为1，不是记为0

    :param img: 直方图
    :param scope: 判断范围
    :return: 可能的波谷位置
    """

    h = img.shape[0]
    w = img.shape[1]

    # 记录图像每一列白色像素点的值
    column_pixels_statistic = np.zeros(w, dtype=int)
    # 记录是否是极值点
    min_point_histogram = np.zeros(w, dtype=int)

    # 统计每一列的白色像素点总数
    for i in range(w):
        cout = 0
        for j in range(h):
            if int(img[h - j - 1][i]) == 255:
                cout += 1
            else:
                break
        column_pixels_statistic[i] = cout

    # 在scope范围内比较
    for i in range(scope, w - scope - 1):
        # 如果全黑，肯定为波谷
        if column_pixels_statistic[i] == 0:
            min_point_histogram[i] = 1
        else:
            cur_pixel = column_pixels_statistic[i]
            scope_t = scope
            for j in range(scope):
                if cur_pixel > column_pixels_statistic[i - j - 1]:
                    break
                elif cur_pixel > column_pixels_statistic[i + j + 1]:
                    break
                scope_t -= 1
            if scope_t == 0:
                min_point_histogram[i] = 1

    return min_point_histogram


def calculate_possible_seg(trough_list, seg_pos_start, seg_pos_end):
    """根据是否是波谷判断可能的分割,以该列像素点总数是否比scope范围内的每一列白色像素点总数小作为判断依据

    :param trough_list: 所有的波谷位置
    :param seg_pos_start: 分割起始位置
    :param seg_pos_end: 分割结束位置
    :return: 可能的波谷位置，作为分割位置
    """

    possible_seg = []
    # 添加开始的分割线
    possible_seg.append(seg_pos_start)
    cur_seg = seg_pos_start
    while cur_seg + 1 <= seg_pos_end:
        # 依次判断下一列是否是分割线
        next_seg = cur_seg + 1
        # 分割线不能超过图像宽度
        if next_seg >= len(trough_list) - 1:
            break
        # 如果为波谷就判断为分割点
        if trough_list[next_seg] == 1:
            possible_seg.append(next_seg)
        else:
            pass
        cur_seg = next_seg

    possible_seg.append(seg_pos_end)

    return possible_seg


def print_calculate_ave(trough_list, seg_pos_start, seg_pos_end):
    """根据是否是波谷判断分割线的平均距离, 以该列像素点总数是否比scope范围内的每一列白色像素点总数小作为依据

    :param trough_list: 所有的波谷位置
    :param seg_pos_start: 分割起始位置
    :param seg_pos_end: 分割结束位置
    :return: 分割线距离平均值
    """

    cur_seg = seg_pos_start
    # 可信间距，保证两天线之间只有一个数字
    trust_dis = 0
    # 保存所有间隔用于求平均间隔
    means = []
    while cur_seg + 1 <= seg_pos_end:
        # 依次判断下一列是否是分割线
        next_seg = cur_seg + 1
        # 分割线不能超过图像宽度
        if next_seg >= len(trough_list) - 1:
            break

        # 如果为波谷就判断为分割点，当有可信线段时距离置0，从新开始计算数字间距
        if trough_list[next_seg] == 1:
            if trust_dis != 0:
                means.append(trust_dis)
                trust_dis = 0

        # 若没有分割线则代表为数字区域，计算单个数字区域宽度
        else:
            trust_dis += 1

        cur_seg = next_seg

    # 计算数字平均宽度
    mean = int(sum(means) / len(means))

    return mean


def print_calculate_possible_seg(trough_list, seg_pos_start, seg_pos_end, mean):
    """根据是否是波谷判断可能的分割,以判断该列像素点总数是否比scope范围内的每一列白色像素点总数小作为依据

    :param trough_list: 所有的波谷位置
    :param seg_pos_start: 分割起始位置
    :param seg_pos_end: 分割结束位置
    :param mean: 数字间隔平均值
    :return: 可能的波谷位置，即切割位置
    """

    possible_seg = []
    # 添加开始的分割线
    possible_seg.append(seg_pos_start)
    cur_seg = seg_pos_start
    # 可信间距，保证两条线之间只有一个数字
    trust_dis = 0
    while cur_seg + 1 <= seg_pos_end:
        # 依次判断下一列是否是分割线
        next_seg = cur_seg + 1
        # 分割线不能超过图像宽度
        if next_seg >= len(trough_list) - 1:
            break

        # 如果为波谷就判断为分割点
        if trough_list[next_seg] == 1:
            possible_seg.append(next_seg)
            trust_dis = 0

        else:
            trust_dis += 1

            # 如果信任距离大于平均距离一定值，则认为该区域有两个数，应该增加一条分割线
            if (trust_dis - mean) > 10:
                possible_seg.append(next_seg - (mean - 5))
                trust_dis = 0

        cur_seg = next_seg

    possible_seg.append(seg_pos_end)

    return possible_seg


def del_seg_error(possible_seg, seg_area_range=0, min_width=19, max_width=21, trusted_area_width=10):
    """删除错误的分割,首先通过所有可能分割计算每个卡号的宽度，然后根据卡号宽度确定可信的线段，然后以可信线段为
    基准线，以固定宽度分割卡号

    :param possible_seg: 可能的分割线
    :param seg_area_range: 分割起始位置
    :param min_width: 最小卡号宽度
    :param max_width: 最大卡号宽度
    :param trusted_area_width: 可信的分割线区域宽度
    :return: 分割线的区域
    """

    # 分割线(有可能分割线为一块区域)
    seg_area = [[]]

    # 生成分割线区域
    # 连续几条分割线即为一个分割区域
    for i in range(len(possible_seg)):
        seg_area[-1].append(possible_seg[i])
        if i != len(possible_seg) - 1:
            if possible_seg[i] + 1 != possible_seg[i + 1]:
                seg_area.append([])

    # 去掉分割区域小于seg_area_range的分割
    for i in range(len(seg_area)):
        if len(seg_area[i]) < seg_area_range:
            t = []
            t.append(seg_area[i][int(len(seg_area[i]) / 2)])
            seg_area[i] = t

    # 计算卡号的宽度
    distance_list = []
    for i in range(len(seg_area) - 1):
        distance_list.append(int(seg_area[i + 1][0] - seg_area[i][-1]))
    distance_list.sort(reverse=True)
    distance = []

    for i in distance_list:
        if i >= max_width or i <= min_width:
            pass
        else:
            distance.append(i)

    if len(distance) == 0:
        distance_number = 20

    else:
        distance_number = int(sum(distance) / len(distance))

    # 找到可信的线段
    trust_flag = []
    for i in range(len(seg_area)):
        trust_flag.append(0)

    for i in range(len(seg_area)):
        if len(seg_area[i]) > trusted_area_width:
            trust_flag[i] = 1

    # 可信线段范围范围
    for i in range(len(seg_area) - 1):
        if abs(seg_area[i + 1][0] - seg_area[i][-1] - distance_number) <= 3:
            trust_flag[i] = 1

    if abs(seg_area[len(seg_area) - 1][0] - seg_area[len(seg_area) - 2][-1] - distance_number) <= 2:
        trust_flag[len(seg_area) - 1] = 1

    # 按固定距离生成线段
    seg_area_insert = []
    trusted_flag_t = dp(trust_flag)

    # 计算头部
    if trusted_flag_t[0] == 0:
        for i in range(len(seg_area)):
            if trusted_flag_t[i] == 1:
                pos_right = i
                break
            else:
                trusted_flag_t[i] = 1
        n = round((seg_area[pos_right][0] - seg_area[0][-1]) / distance_number)

        for j in range(n):
            seg_area_insert.append(seg_area[pos_right][0] - distance_number * (j + 1))

    # 计算尾部
    if trusted_flag_t[-1] == 0:
        for i in range(len(trusted_flag_t)):
            pos = len(trusted_flag_t) - i - 1
            if trusted_flag_t[pos] == 1:
                pos_left = len(trusted_flag_t) - i - 1
                break
            else:
                trusted_flag_t[pos] = 1
        n = round((seg_area[-1][0] - seg_area[pos_left][-1]) / distance_number)

        for j in range(n):
            seg_area_insert.append(seg_area[pos_left][-1] + distance_number * (j + 1))

    # 计算中间
    pos_left = [0]
    pos_right = [0]
    seek_flag = False

    for i in range(len(seg_area)):
        if trusted_flag_t[i] == 0 and seek_flag == False:
            pos_left[0] = i
            seek_flag = True
        if seek_flag == True:
            if trusted_flag_t[i] == 1:
                pos_right[0] = i
                n = round((seg_area[pos_right[0]][0] - seg_area[pos_left[0] - 1][-1]) / distance_number)
                if n > 1:
                    dis = round((seg_area[pos_right[0]][0] - seg_area[pos_left[0] - 1][-1]) / n)
                    for j in range(n - 1):
                        seg_area_insert.append(seg_area[pos_left[0] - 1][-1] + dis * (j + 1))
                seek_flag = False

    # 生成新的分割区域
    seg_area_new = []
    for i in range(len(trust_flag)):
        if trust_flag[i] == 1:
            seg_area_new.append(seg_area[i])
    for i in range(len(seg_area_insert)):
        t = []
        t.append(seg_area_insert[i])
        seg_area_new.append(t)

    seg_area_new = sort_asc(seg_area_new)

    # 去掉开头明显不是的分割线
    # if seg_area_new[0][0] < 33:
    #     seg_area_new.remove(seg_area_new[0])

    return seg_area_new


def print_del_seg_error(possible_seg, seg_area_range=12, min_width=13, max_width=25, trusted_area_width=2):
    """删除印刷体错误的分割，首先通过所有可能分割计算每个卡号的宽度，然后根据卡号宽度确定可信的线段，然后以可信
    线段为基准线，以固定宽度分割卡号

    :param possible_seg: 可能的分割线
    :param seg_area_range: 分割起始位置
    :param min_width: 最小卡号宽度
    :param max_width: 最大卡号宽度
    :param trusted_area_width: 可信的分割线区域宽度
    :return: 分割线的区域
    """

    # 分割线(有可能分割线为一块区域)
    seg_area = [[]]

    # 生成分割线区域
    # 连续几条分割线即为一个分割区域
    for i in range(len(possible_seg)):
        seg_area[-1].append(possible_seg[i])
        if i != len(possible_seg) - 1:
            if possible_seg[i] + 1 != possible_seg[i + 1]:
                seg_area.append([])

    # 去掉分割区域小于seg_area_range的分割
    # for i in range(len(seg_area)):
    #     if len(seg_area[i]) < seg_area_range:
    #         t = []
    #         t.append(seg_area[i][int(len(seg_area[i]) / 2)])
    #         seg_area[i] = t

    # 计算卡号的宽度
    distance_list = []
    for i in range(len(seg_area) - 1):
        distance_list.append(int(seg_area[i + 1][0] - seg_area[i][-1]))
    distance_list.sort(reverse=True)
    distance = []

    for i in distance_list:
        if i >= max_width or i <= min_width:
            pass
        else:
            distance.append(i)
    if len(distance) == 0:
        distance_number = 15
    else:
        distance_number = int(sum(distance) / len(distance))

    # 找到可信的线段
    trust_flag = []
    for i in range(len(seg_area)):
        trust_flag.append(0)

    for i in range(len(seg_area)):
        if len(seg_area[i]) > trusted_area_width:
            trust_flag[i] = 1

    for i in range(len(seg_area) - 1):
        if abs(seg_area[i + 1][0] - seg_area[i][-1] - distance_number) <= 3:  # 可信线段范围范围
            trust_flag[i] = 1

    if abs(seg_area[len(seg_area) - 1][0] - seg_area[len(seg_area) - 2][-1] - distance_number) <= 2:
        trust_flag[len(seg_area) - 1] = 1

    # 按固定距离生成线段
    seg_area_insert = []
    trusted_flag_t = dp(trust_flag)

    # 计算头部
    if trusted_flag_t[0] == 0:
        for i in range(len(seg_area)):
            if trusted_flag_t[i] == 1:
                pos_right = i
                break
            else:
                trusted_flag_t[i] = 1
        n = round((seg_area[pos_right][0] - seg_area[0][-1]) / distance_number)

        for j in range(n):
            seg_area_insert.append(seg_area[pos_right][0] - distance_number * (j + 1))

    # 计算尾部
    if trusted_flag_t[-1] == 0:
        for i in range(len(trusted_flag_t)):
            pos = len(trusted_flag_t) - i - 1
            if trusted_flag_t[pos] == 1:
                pos_left = len(trusted_flag_t) - i - 1
                break
            else:
                trusted_flag_t[pos] = 1
        n = round((seg_area[-1][0] - seg_area[pos_left][-1]) / distance_number)

        for j in range(n):
            seg_area_insert.append(seg_area[pos_left][-1] + distance_number * (j + 1))

    # 计算中间
    pos_left = [0]
    pos_right = [0]
    seek_flag = False

    for i in range(len(seg_area)):
        if trusted_flag_t[i] == 0 and seek_flag == False:
            pos_left[0] = i
            seek_flag = True
        if seek_flag == True:
            if trusted_flag_t[i] == 1:
                pos_right[0] = i
                n = round((seg_area[pos_right[0]][0] - seg_area[pos_left[0] - 1][-1]) / distance_number)
                if n > 1:
                    dis = round((seg_area[pos_right[0]][0] - seg_area[pos_left[0] - 1][-1]) / n)

                    for j in range(n - 1):
                        seg_area_insert.append(seg_area[pos_left[0] - 1][-1] + dis * (j + 1))
                seek_flag = False

    # 生成新的分割区域
    seg_area_new = []

    for i in range(len(trust_flag)):
        if trust_flag[i] == 1:
            seg_area_new.append(seg_area[i])

    for i in range(len(seg_area_insert)):
        t = []
        t.append(seg_area_insert[i])
        seg_area_new.append(t)

    seg_area_new = sort_asc(seg_area_new)

    # 去掉开头明显不是的分割线
    # if seg_area_new[0][0] < 33:
    #     seg_area_new.remove(seg_area_new[0])

    return seg_area_new


def sort_asc(seg_area):
    """对分割区域按照起始位置递增排序

    :param seg_area: 分割区域
    :return: 递增排序的分割区域
    """

    length = len(seg_area)
    seg_area_return = dp(seg_area)
    seg_list = []

    # 分割区域起始位置
    for i in range(length):
        seg_list.append(seg_area[i][0])

    # 冒泡排序算法
    for i in range(0, length - 1):
        for j in range(0, length - 1 - i):
            if seg_list[j] > seg_list[j + 1]:
                tmp = seg_list[j]
                seg_list[j] = seg_list[j + 1]
                seg_list[j + 1] = tmp

                t = seg_area_return[j]
                seg_area_return[j] = seg_area_return[j + 1]
                seg_area_return[j + 1] = t

    return seg_area_return


def seg_img(original_img, upper_lower_border, scale, seg_area):
    """按照分割区域分割凹凸字体图像

    :param original_img: 原图
    :param upper_lower_border: 上下边界(相对于缩放图)
    :param scale: 缩放图
    :param seg_area: 分割线的区域
    :return: 递增排序的分割区域
    """

    seg_num_img_list = []
    # 两个区域之间为数字区域
    for i in range(len(seg_area) - 1):
        left_border = int(seg_area[i][-1] * scale[0])
        right_border = int(seg_area[i + 1][0] * scale[0])
        upper_border = int(upper_lower_border[0] * scale[1] - 3)
        lower_border = int(upper_lower_border[1] * scale[1] + 4)
        seg_num_img_list.append(original_img[upper_border:lower_border, left_border:right_border])

    return seg_num_img_list


def print_seg_img(original_img, upper_lower_border, scale, seg_area):
    """按照分割区域分割印刷体图像

    :param original_img: 原图
    :param upper_lower_border: 上下边界(相对于缩放图)
    :param scale: 缩放图
    :param seg_area: 分割线的区域
    :return: 递增排序的分割区域
    """

    seg_num_img_list = []
    # 两个区域之间为数字区域
    for i in range(len(seg_area) - 1):
        left_border = int(seg_area[i][-1] * scale[0])
        right_border = int(seg_area[i + 1][0] * scale[0])
        upper_border = int(upper_lower_border[0] * scale[1] - 3)
        lower_border = int(upper_lower_border[1] * scale[1] + 4)
        seg_num_img_list.append(original_img[upper_border:lower_border, left_border:right_border])

    # 防止垂直方向切割过度
    # for dst_num in range(len(seg_num_img_list)):
    #     # 图像灰度化
    #     dst = cv2.cvtColor(seg_num_img_list[dst_num], cv2.COLOR_BGR2GRAY)
    #     # 二值化
    #     dst = 255 - fixed_thresh_binary(dst, min_thresh=45, max_thresh=255)
    #     # 生成直方图
    #     dst_histogram = horizontal_histogram(dst)
    #     seg_pos_start, seg_pos_end = horizontal_seg_pos(dst_histogram)
    #
    #     # seg_pos_start-5是为了避免图片上方被截掉过多
    #     if (seg_pos_start - 2) > 0:
    #         seg_pos_start = seg_pos_start - 2
    #     else:
    #         pass

    # seg_num_img_list[dst_num] = seg_num_img_list[dst_num][seg_pos_start:seg_pos_end,
    #                             0:seg_num_img_list[dst_num].shape[1]]

    return seg_num_img_list


def is_print_font(img):
    """判断是否是印刷体

    :param img: 切割好的银行卡号区域图片
    :return: bool
    """

    img = dp(img)
    # 图像灰度化
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = 255 - fixed_thresh_binary(dst, min_thresh=45, max_thresh=255)
    kernel2 = np.ones((4, 3), np.uint8)
    # 膨胀
    dst = cv2.dilate(dst, kernel2, iterations=1)
    # 腐蚀
    kernel = np.ones((4, 2), np.uint8)
    dst = cv2.erode(dst, kernel, iterations=1)
    # 去除小连通区域
    dst = del_connected_region(dst, method="more", area_range=250)
    # 去除小连通区域
    dst = del_connected_region(dst, method="less", area_range=90)

    white = getWhitePixel(dst)
    if white > 1000:
        return True
    else:
        return False


def getWhitePixel(image):
    """计算白色像素点的面积

    :param image: 输入的二值化图像
    :return: 返回面积值
    """
    area = 0
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = th3.shape
    for i in range(height):
        for j in range(width):
            if th3[i, j] == 255:
                area += 1
    return area
