'''本文件用于对银行卡原图进行切割与矫正'''

import cv2
import numpy as np
from copy import deepcopy as dp
# from card_number_positioning.pylsd.lsd import lsd
#from pylsd.lsd import lsd
from math import atan, pi, sqrt
from card_number_positioning.util import cv2_imwrite, get_value
from os.path import join
from card_number_positioning.util import saveProcessStep, adaptive_binary
from card_number_positioning.pylsd.lsd import lsd


class line(object):
    '''
    自定义线段类
    '''

    def __init__(self, pt1, pt2, extend_pt1=None, extend_pt2=None, width=None, angle=None, ):
        """初始化

        :param pt1: 线段左侧点
        :param pt2: 线段右侧点
        :param extend_pt1: 左侧点向左延长交于左边界的点
        :param extend_pt2: 右侧点向右延长交于右侧边界的点
        :param width: 线段宽度
        :param angle: 线段与水平风向的夹角
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.width = width
        self.angle = angle
        self.extend_pt1 = extend_pt1
        self.extend_pt2 = extend_pt2
        self.cal_length()
        if abs(pt1[0] - pt2[0]) == 0:
            self.k = None
            self.b = pt1[0]
        else:
            self.k = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
            self.b = pt2[1] - self.k * pt2[0]

    def cal_length(self):
        """计算线段长度

        :return: 线段长度
        """
        self.length = sqrt((self.pt1[0] - self.pt2[0]) * (self.pt1[0] - self.pt2[0]) + (self.pt1[1] - self.pt2[1]) * (
                self.pt1[1] - self.pt2[1]))


def cut_and_rectify(org_img):
    """从原图将银行卡进行切割和矫正

    :param org_img: 测试原图
    :return: 切割出银行卡区域并矫正银行卡
    """
    new_method(org_img)
    set_global_variable()
    # 缩放原图用于切割
    img = dp(org_img)
    img = cv2.resize(img, (480, 305))

    # debug模式下复制一张图像用于保存中间结果
    img_ = dp(img) if debug == True else None

    # resize后的图的尺寸
    img_h, img_w = img.shape[0:2]
    # 原图尺寸
    original_h, original_w = org_img.shape[0:2]
    # x坐标比例
    ratio_x = original_w / img_w
    # y坐标比例
    ratio_y = original_h / img_h
    # 图像灰度化
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lsd线段检测
    lines = lsd(dst)
    # debug模式执行，用于生成过程图
    if debug == True:
        img_t = dp(img_)
        for i in range(len(lines)):
            pt1 = (int(lines[i][0]), int(lines[i][1]))
            pt2 = (int(lines[i][2]), int(lines[i][3]))
            cv2.line(img_t, pt1, pt2, (0, 0, 255), int(2))
        cv2_imwrite(join(save_path, "详细识别过程", "1.LSD_Line.jpg"), img_t)
        del img_t

    k_up, b_up, k_lower, b_lower = up_lower_border(lines, img)
    k_left, b_left, k_right, b_right = left_right_border(lines, img)
    p1, p2, p3, p4 = cal_point_intersection(k_up, b_up, k_lower, b_lower, k_left, b_left, k_right, b_right)

    # 由缩小图坐标根据坐标比例变换到原图坐标
    ori_p1 = (int(p1[0] * ratio_x), int(p1[1] * ratio_y))
    ori_p2 = (int(p2[0] * ratio_x), int(p2[1] * ratio_y))
    ori_p3 = (int(p3[0] * ratio_x), int(p3[1] * ratio_y))
    ori_p4 = (int(p4[0] * ratio_x), int(p4[1] * ratio_y))

    pts1 = np.float32([ori_p1, ori_p2, ori_p3, ori_p4])
    pts2 = np.float32([[0, 0], [ori_p2[0] - ori_p1[0], 0], [0, ori_p3[1] - ori_p1[1]],
                       [ori_p2[0] - ori_p1[0], ori_p3[1] - ori_p1[1]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(org_img, M, (ori_p2[0] - ori_p1[0], ori_p3[1] - ori_p1[1]))

    # debug模式下复制一张图像用于保存中间结果,生成水平所有线条
    saveProcessStep(dst, debug, save_path, '详细识别过程', '5.cut_and_rectify.jpg')

    return dst


def set_global_variable():
    """在当前文件夹设置全局变量，用于保存中间结果

    :return:
    """

    global debug, name, save_path
    debug = get_value("debug")
    name = get_value("name")
    save_path = get_value("save_path")


def cal_extension_y(pt1, pt2, img):
    """垂直方向计算延长线

    :param pt1: 线段第一个点
    :param pt2: 线段第二个点
    :param img: 银行卡原图
    :return: 垂直方向的一条分割线
    """

    h = img.shape[0]
    w = img.shape[1]
    # 计算方程斜率和截距
    if abs(pt1[0] - pt2[0]) == 0:
        x1 = pt1[0]
        y1 = 0
        x2 = pt1[0]
        y2 = img.shape[0] - 1
    else:
        k = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        b = pt2[1] - k * pt2[0]

        # 假设y=0
        y = 0
        x = -1 * b / k

        if x < 0:
            x1 = 0
            y1 = int(b)
        elif 0 <= x and x < w:
            y1 = 0
            x1 = int(-1 * b / k)
        else:
            x1 = w - 1
            y1 = int(k * x1 + b)

        # 假设y=h-1
        y = h - 1
        x = (y - b) / k
        if x < 0:
            x2 = 0
            y2 = int(b)
        elif 0 <= x and x < w:
            y2 = h - 1
            x2 = int((y2 - b) / k)
        else:
            x2 = w - 1
            y2 = int(k * x2 + b)

    return (x1, y1), (x2, y2)


def cal_extension_x(pt1, pt2, img):
    """水平方向计算延长线

    :param pt1: 线段第一个点
    :param pt2: 线段第二个点
    :param img: 银行卡原图
    :return: 水平方向的一条分割线
    """

    h = img.shape[0]
    w = img.shape[1]
    # 计算方程斜率和截距
    if abs(pt1[0] - pt2[0]) == 0:
        x1 = pt1[0]
        y1 = 0
        x2 = pt1[0]
        y2 = img.shape[0] - 1
    else:
        k = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
        b = pt2[1] - k * pt2[0]

        # 假设x=0
        x = 0
        y = k * x + b

        if y < 0:
            y1 = 0
            x1 = int(-1 * b / k)
        elif 0 <= y and y < h:
            y1 = int(y)
            x1 = 0
        else:
            y1 = h - 1
            x1 = int((y1 - 1 * b) / k)

        # 假设x=w-1
        x = w - 1
        y = k * x + b
        if y < 0:
            y2 = 0
            x2 = int((y - b) / k)
        elif 0 <= y and y < h:
            x2 = w - 1
            y2 = int(y)
        else:
            y2 = h - 1
            x2 = int((y1 - 1 * b) / k)

    return (x1, y1), (x2, y2)


def judge_same_element_num(array1, array2):
    """判断两个相同大小的数组中相同元素的个数

    :param array1: 数组1
    :param array2: 数组2
    :return: 相同元素的个数
    """

    w = array1.shape[0]
    h = array1.shape[1]
    count = 0
    for i in range(w):
        for j in range(h):
            if int(array1[i][j]) == 255 and int(array1[i][j]) == int(array2[i][j]):
                count += 1
    return count


def remove_element(remove_pos, list):
    """删除list中索引为列表remove_pos中的元素的位置的元素

    :param remove_pos:列表，用于list的元素索引
    :param list:待处理的列表
    :return:删除元素后的列表
    """
    list_ = []
    for i in range(len(list)):
        if i not in remove_pos:
            list_.append(list[i])
    return list_


def averagenum(num):
    """求列表元素的平均数

    :param num: 带计算平均数的列表
    :return: 列表平均数
    """
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def up_lower_border(lines, img):
    """计算银行卡上下边界的位置

    :param lines: 所有可能的水平分割线
    :param img: 银行卡原图
    :return: 银行卡的上下边界直线的斜率与截距
    """
    dst = dp(img)
    img_show = dp(img)
    # 将检测出来的线段用自自定义的数据结构保存
    # 水平方向的线段
    lines_x = []
    for i in range(len(lines)):
        t = []
        # 保证pt1是靠近图像左边的点
        if int(lines[i][0]) < int(lines[i][2]):
            pt1 = (int(lines[i][0]), int(lines[i][1]))
            pt2 = (int(lines[i][2]), int(lines[i][3]))
        else:
            pt1 = (int(lines[i][2]), int(lines[i][3]))
            pt2 = (int(lines[i][0]), int(lines[i][1]))

        width = lines[i][4]
        angle = int(
            atan(abs(pt1[1] - pt2[1]) / abs(pt1[0] - pt2[0] + 0.0000001)) * 180 / pi)
        length = sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        # 如果倾斜角小于10度则判断为水平线段
        # 过滤掉长度小于30的线段
        if angle < 10 and length > 25:
            pt3, pt4 = cal_extension_x(pt1, pt2, img)
            t.append(line(pt1, pt2, pt3, pt4, width, angle))
            cv2.line(img_show, pt1, pt3, (0, 255, 255), int(1))
            cv2.line(img_show, pt1, pt2, (0, 0, 255), int(3))
            cv2.line(img_show, pt4, pt2, (255, 0, 255), int(1))
            # 将所有的水平线段添加到列表
            lines_x.append(t)

    # 删除银行卡银联区域的线段
    # lines_x = remove_unionpay_x(lines_x)

    # 合并是同一直线的线段
    lines_x_merge = []
    while len(lines_x) > 0:
        # 待判断的线段
        cur_line = dp(lines_x[0][0])
        t = []
        t.append(cur_line)
        lines_x.remove(lines_x[0])
        remove_pos = []
        for j in range(len(lines_x)):
            compare_line = dp(lines_x[j][0])
            ditanse1 = sqrt((cur_line.extend_pt1[1] - compare_line.extend_pt1[1]) * (
                    cur_line.extend_pt1[1] - compare_line.extend_pt1[1]))
            ditanse2 = sqrt((cur_line.extend_pt2[1] - compare_line.extend_pt2[1]) * (
                    cur_line.extend_pt2[1] - compare_line.extend_pt2[1]))
            # 若两个线段左边界交点与右边界交点的距离都小于5则判断为共线
            if ditanse1 <= 10 and ditanse2 <= 10:
                t.append(compare_line)
                remove_pos.append(j)
        # 将最长的线段作为最终线段
        max_len_index = -1
        max_len = -1
        for i in range(len(t)):
            if t[i].length > max_len:
                max_len_index = i
                max_len = t[i].length
        t[0], t[max_len_index] = t[max_len_index], t[0]
        lines_x_merge.append(t)
        lines_x = remove_element(remove_pos, lines_x)

    # 确定一个上边界线段和一个下边界线段
    img_line_merge = dp(dst)
    # 每条线与其共线所有线段的长度
    line_length = []
    for i in range(len(lines_x_merge)):
        length = 0
        for j in range(len(lines_x_merge[i])):
            length += lines_x_merge[i][j].length
        line_length.append(length)

    # 将可能的线段按长度进行递减排序
    line_length_rank = dp(line_length)
    line_length_rank, org_order = line_sort(line_length_rank)

    # debug模式下复制一张图像用于保存中间结果,生成水平所有线条
    img_ = dp(img) if debug == True else None
    if debug == True:
        img_t = dp(img_)
        for i in range(len(lines_x_merge)):
            cur_line = lines_x_merge[i][0]
            cv2.line(img_t, cur_line.extend_pt1, cur_line.extend_pt2, 255, int(2))
        cv2_imwrite(join(save_path, "详细识别过程", "2.allHLine.jpg"), img_t)
        del img_t

    # 保存选择的四条可能分割线
    possible_line = []
    all_x_line = []
    for i in range(len(line_length_rank)):
        if i < 4:
            # 选择长度最长的四个
            cur_line = lines_x_merge[org_order[str(i)]][0]
            possible_line.append(cur_line)
            cv2.line(img_line_merge, cur_line.extend_pt1, cur_line.extend_pt2, 255, int(2))
        all_x_line.append(lines_x_merge[org_order[str(i)]][0])
    all_x_line = sort_x_line(all_x_line)

    # debug模式下复制一张图像用于保存中间结果，生成水平四条线条
    saveProcessStep(img_line_merge, debug, save_path, '详细识别过程', '3.FourHLine.jpg')

    # 保存延长后的水平分割线
    y = []
    for i in range(len(possible_line)):
        y.append(possible_line[i].extend_pt1[1])
    y_order = dp(y)
    y_order.sort(reverse=False)

    # 保存上边界分割线
    up_line = []
    # 保存下边界分割线
    lower_line = []

    # 获取任意两条分割线的斜率差值
    dis_up1 = abs(possible_line[y.index(y_order[0])].k - possible_line[y.index(y_order[2])].k)
    dis_up2 = abs(possible_line[y.index(y_order[0])].k - possible_line[y.index(y_order[3])].k)
    dis_up3 = abs(possible_line[y.index(y_order[0])].k - possible_line[y.index(y_order[1])].k)
    dis_low1 = abs(possible_line[y.index(y_order[1])].k - possible_line[y.index(y_order[2])].k)
    dis_low2 = abs(possible_line[y.index(y_order[0])].k - possible_line[y.index(y_order[3])].k)
    dis_low3 = abs(possible_line[y.index(y_order[2])].k - possible_line[y.index(y_order[3])].k)

    # 存放每两条线的间距
    dis_max = [abs(y_order[0] - y_order[1]), abs(y_order[0] - y_order[2]), abs(y_order[0] - y_order[3]),
               abs(y_order[1] - y_order[2]), abs(y_order[1] - y_order[3]), abs(y_order[2] - y_order[3])]
    max_dis = max(dis_max)

    # 最大间距，若两条线的斜率接近且与最大间距比例接近阈值0.75，则认为是上下边界
    if dis_up2 < 0.05 and abs(y_order[0] - y_order[3]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[0])])
        lower_line.append(possible_line[y.index(y_order[3])])

    if dis_up1 < 0.05 and abs(y_order[0] - y_order[2]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[0])])
        lower_line.append(possible_line[y.index(y_order[2])])

    if dis_up3 < 0.05 and abs(y_order[0] - y_order[1]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[0])])
        lower_line.append(possible_line[y.index(y_order[1])])

    if dis_low1 < 0.05 and abs(y_order[1] - y_order[2]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[1])])
        lower_line.append(possible_line[y.index(y_order[2])])

    if dis_low2 < 0.05 and abs(y_order[0] - y_order[3]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[0])])
        lower_line.append(possible_line[y.index(y_order[3])])

    if dis_low3 < 0.05 and abs(y_order[2] - y_order[3]) / max_dis > 0.75:
        up_line.append(possible_line[y.index(y_order[2])])
        lower_line.append(possible_line[y.index(y_order[3])])

    # 如果上边界为空
    if up_line != []:
        if up_line[0].extend_pt1[1] > 0.4 * 305:
            up_line = []
        for i in range(len(all_x_line)):
            if all_x_line[i].extend_pt1[1] != 0 and all_x_line[i].angle < 10:  # 选择尽量垂直的那条线
                up_line.append(all_x_line[i])
                break
        if lower_line[0].extend_pt1[1] < 0.7 * 305 or abs(
                lower_line[0].extend_pt1[0] - lower_line[0].extend_pt2[0]) < 100:
            lower_line = []
        for i in range(len(all_x_line)):
            cur_line = all_x_line[len(all_x_line) - i - 1]
            if cur_line.extend_pt1[1] != 0 and cur_line.angle < 10 and abs(
                    cur_line.extend_pt1[0] - cur_line.extend_pt2[0]) > 100 :
                lower_line.append(cur_line)
                break
    else:
        up_line.append(possible_line[y.index(y_order[0])])
        lower_line.append(possible_line[y.index(y_order[3])])

    img_seg = dp(dst)
    cv2.line(img_seg, up_line[0].extend_pt1, up_line[0].extend_pt2, 255, int(3))
    cv2.line(img_seg, lower_line[0].extend_pt1, lower_line[0].extend_pt2, 255, int(3))

    # debug模式下复制一张图像用于保存中间结果,生成水平两条线条
    saveProcessStep(img_seg, debug, save_path, '详细识别过程', '4.TwoHLine.jpg')

    k_up = up_line[0].k
    b_up = up_line[0].b
    k_lower = lower_line[0].k
    b_lower = lower_line[0].b
    return k_up, b_up, k_lower, b_lower


def left_right_border(lines, img):
    """计算银行卡的左右边界

    :param lines:所有可能的左右边界线段
    :param img:银行卡原图
    :return:左右银行卡分割线的斜率与截距
    """
    dst = dp(img)
    img_show = dp(img)

    # 将检测出来的线段用自自定义的数据结构保存
    # 水平方向的线段
    lines_y = []
    for i in range(len(lines)):
        t = []
        # 保证pt1是靠近图像左边的点
        if int(lines[i][1]) < int(lines[i][3]):
            pt1 = (int(lines[i][0]), int(lines[i][1]))
            pt2 = (int(lines[i][2]), int(lines[i][3]))
        else:
            pt1 = (int(lines[i][2]), int(lines[i][3]))
            pt2 = (int(lines[i][0]), int(lines[i][1]))

        width = lines[i][4]
        angle = int(
            atan(abs(pt1[1] - pt2[1]) / abs(pt1[0] - pt2[0] + 0.0000001)) * 180 / pi)  # 此处误
        length = sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

        # 如果倾斜角大于70度则判断为垂直线段
        # 过滤掉长度小于30的线段
        if angle > 70:
            pt3, pt4 = cal_extension_y(pt1, pt2, img)
            t.append(line(pt1, pt2, pt3, pt4, width, angle))
            cv2.line(img_show, pt1, pt2, (0, 0, 255), int(1))
            cv2.line(img_show, pt4, pt2, (255, 0, 255), int(1))
            possible_line_img = np.zeros((img.shape[0], img.shape[1]))
            cv2.line(possible_line_img, pt3, pt4, 255, int(1))
            # 将所有的水平线段添加到列表
            lines_y.append(t)

    # 删除银行卡银联区域的线段
    lines_y = remove_unionpay_y(lines_y)

    # 合并是同一直线的线段
    lines_y_merge = []
    while len(lines_y) > 0:
        # 当前待判断的线段
        cur_line = dp(lines_y[0][0])
        t = []
        t.append(cur_line)
        lines_y.remove(lines_y[0])
        remove_pos = []
        for j in range(len(lines_y)):
            compare_line = dp(lines_y[j][0])
            ditanse1 = sqrt((cur_line.extend_pt1[0] - compare_line.extend_pt1[0]) * (
                    cur_line.extend_pt1[0] - compare_line.extend_pt1[0]))
            ditanse2 = sqrt((cur_line.extend_pt2[0] - compare_line.extend_pt2[0]) * (
                    cur_line.extend_pt2[0] - compare_line.extend_pt2[0]))
            # 若两个线段上边界交点与下边界交点的距离都小于10则判断为共线
            if ditanse1 <= 10 and ditanse2 <= 10:
                t.append(compare_line)
                remove_pos.append(j)
        # 将最长的线段作为最终线段
        max_len_index = -1
        max_len = -1
        for i in range(len(t)):
            if t[i].length > max_len:
                max_len_index = i
                max_len = t[i].length
        t[0], t[max_len_index] = t[max_len_index], t[0]
        lines_y_merge.append(t)  # 加入到合并的线段
        lines_y = remove_element(remove_pos, lines_y)

    # 确定一个左边界线段和一个右边界线段
    img_line_merge = dp(dst)

    # 每条线与其共线所有线段的长度
    line_length = []
    for i in range(len(lines_y_merge)):
        length = 0
        for j in range(len(lines_y_merge[i])):
            length += lines_y_merge[i][j].length
        line_length.append(length)

    # 将可能的线段按长度进行递减排序
    line_length_rank = dp(line_length)
    line_length_rank, org_order = line_sort(line_length_rank)

    # debug模式下复制一张图像用于保存中间结果,生成垂直所有线条
    img_ = dp(img) if debug == True else None
    if debug == True:
        img_t = dp(img_)
        for i in range(len(lines_y_merge)):
            cur_line = lines_y_merge[i][0]
            cv2.line(img_t, cur_line.extend_pt1, cur_line.extend_pt2, 255, int(2))
        cv2_imwrite(join(save_path, "详细识别过程", "2.allVLine.jpg"), img_t)
        del img_t

    # 保存四条最可能的分割线
    possible_line = []
    all_y_line = []
    for i in range(len(line_length_rank)):
        if i < 4:
            # 选择长度最长的四个
            cur_line = lines_y_merge[org_order[str(i)]][0]
            possible_line.append(cur_line)
            cv2.line(img_line_merge, cur_line.extend_pt1, cur_line.extend_pt2, 255, int(2))
        all_y_line.append(lines_y_merge[org_order[str(i)]][0])
    all_y_line = sort_y_line(all_y_line)
    # debug模式下复制一张图像用于保存中间结果,生成垂直四条线条
    saveProcessStep(img_line_merge, debug, save_path, '详细识别过程', '3.FourVLine.jpg')

    # 从这四根线选择两条
    x = []
    for i in range(len(possible_line)):
        x.append(possible_line[i].pt1[0])
    y_order = dp(x)
    y_order.sort(reverse=False)

    # 通过判断密度再选取2根线
    left_line = []
    right_line = []
    if x.index(y_order[0]) < x.index(y_order[1]):
        left_line.append(possible_line[x.index(y_order[0])])
    else:
        left_line.append(possible_line[x.index(y_order[1])])
    if x.index(y_order[-1]) < x.index(y_order[-2]):
        right_line.append(possible_line[x.index(y_order[-1])])
    else:
        right_line.append(possible_line[x.index(y_order[-2])])

    # 如果得到的两根线太近，则选取最边界线
    if abs(left_line[0].extend_pt1[0] - right_line[0].extend_pt1[0]) < 240:
        left_line = []
        right_line = []
        left_line.append(possible_line[x.index(y_order[0])])
        right_line.append(possible_line[x.index(y_order[-1])])

    # 如果左边的线太靠右，选择最左边的线
    if left_line[0].extend_pt1[0] > 480 * 0.3:
        left_line = []
    for i in range(len(all_y_line)):
        if all_y_line[i].extend_pt1[0] != 0 and all_y_line[i].angle > 82:  # 选择尽量垂直的那条线
            left_line.append(all_y_line[i])
            break

    img_seg = dp(dst)
    cv2.line(img_seg, left_line[0].extend_pt1, left_line[0].extend_pt2, 255, int(3))
    cv2.line(img_seg, right_line[0].extend_pt1, right_line[0].extend_pt2, 255, int(3))

    # debug模式下复制一张图像用于保存中间结果,生成垂直两条线条
    saveProcessStep(img_seg, debug, save_path, '详细识别过程', '4.TwoVLine.jpg')

    k_left = left_line[0].k
    b_left = left_line[0].b
    k_right = right_line[0].k
    b_right = right_line[0].b
    return k_left, b_left, k_right, b_right


def cal_point_intersection(k_up, b_up, k_lower, b_lower, k_left, b_left, k_right, b_right):
    """计算水平分割线与垂直分割线的交点

    :param k_up: 上边界斜率
    :param b_up: 上边界截距
    :param k_lower: 下边界斜率
    :param b_lower: 下边界截距
    :param k_left: 左边界斜率
    :param b_left: 左边界截距
    :param k_right: 右边界斜率
    :param b_right: 右边界截距
    :return: 四个交点的左边值
    """

    # 计算上与左线的交点
    if k_left != None:
        upper_left_x = (b_left - b_up) / (k_up - k_left)
        upper_left_y = k_up * upper_left_x + b_up
    else:
        upper_left_x = b_left
        upper_left_y = k_up * upper_left_x + b_up

    # 计算上与右线的交点
    if k_right != None:
        upper_right_x = (b_right - b_up) / (k_up - k_right)
        upper_right_y = k_up * upper_right_x + b_up
    else:
        upper_right_x = b_right
        upper_right_y = k_up * upper_right_x + b_up

    # 计算下与左线的交点
    if k_left != None:
        lower_left_x = (b_left - b_lower) / (k_lower - k_left)
        lower_left_y = k_lower * lower_left_x + b_lower
    else:
        lower_left_x = b_left
        lower_left_y = k_lower * lower_left_x + b_lower

    # 计算下与右线的交点
    if k_right != None:
        lower_right_x = (b_right - b_lower) / (k_lower - k_right)
        lower_right_y = k_lower * lower_right_x + b_lower
    else:
        lower_right_x = b_right
        lower_right_y = k_lower * lower_right_x + b_lower

    # 返回左上，右上，左下，右下的坐标点
    return (int(upper_left_x), int(upper_left_y)), (int(upper_right_x), int(upper_right_y)), \
           (int(lower_left_x), int(lower_left_y)), (int(lower_right_x), int(lower_right_y))


def line_sort(nums, sort='dsc'):
    """对线段进行降序排序，并返回调换顺序的索引值"""
    org_order = {}
    nums_t = dp(nums)
    nums = []
    for i in range(len(nums_t)):  # 遍历 len(nums)-1 次
        if sort == 'dsc':
            maxIndex = nums_t.index(max(nums_t))
            nums.append(nums_t[maxIndex])
            nums_t[maxIndex] = -1
            org_order[str(i)] = maxIndex  # 添加索引
        elif sort == 'asc':
            minIndex = nums_t.index(min(nums_t))
            nums.append(nums_t[minIndex])
            nums_t[minIndex] = 10000
            org_order[str(i)] = minIndex  # 添加索引

    return nums, org_order


def new_method(org_img):
    org_img_t = dp(org_img)
    org_img_t = cv2.resize(org_img_t, (480, 300), cv2.INTER_CUBIC)
    # cv2.imshow("org_img", org_img_t)
    dst = cv2.cvtColor(org_img_t, cv2.COLOR_BGR2GRAY)
    # 大津阈值法二值化
    dst = adaptive_binary(dst, 'OTSU')
    # cv2.imshow("OTSU", dst)


def remove_unionpay_x(lines_x):
    lines_x_new = []
    while len(lines_x) > 0:
        cur_line = lines_x[0][0]
        remove_pos = []
        is_unionpay_line = False
        for j in range(len(lines_x)):
            if j != 0:
                compare_line = lines_x[j][0]
                if abs(cur_line.angle - compare_line.angle) <= 3 and abs(
                        cur_line.pt1[0] - compare_line.pt1[0]) < 15 and abs(cur_line.pt2[0] - compare_line.pt2[0]) < 15 \
                        and abs(
                    cur_line.pt1[1] - compare_line.pt1[1]) < 30 and abs(
                    cur_line.pt2[1] - compare_line.pt2[1]) < 30 and cur_line.length < 110:
                    remove_pos.append(j)
                    is_unionpay_line = True
        if is_unionpay_line == False:
            lines_x_new.append(lines_x[0])
        remove_pos.append(0)
        lines_x = remove_element(remove_pos, lines_x)
    return lines_x_new


def remove_unionpay_y(lines_y):
    lines_y_new = []
    while len(lines_y) > 0:
        cur_line = lines_y[0][0]
        remove_pos = []
        is_unionpay_line = False
        for j in range(len(lines_y)):
            if j != 0:
                compare_line = lines_y[j][0]
                if abs(cur_line.angle - compare_line.angle) <= 3 and abs(
                        cur_line.pt1[0] - compare_line.pt1[0]) < 15 and abs(cur_line.pt2[0] - compare_line.pt2[0]) < 15 \
                        and abs(
                    cur_line.pt1[1] - compare_line.pt1[1]) < 30 and abs(
                    cur_line.pt2[1] - compare_line.pt2[1]) < 30 and cur_line.length < 70:
                    remove_pos.append(j)
                    is_unionpay_line = True
        if is_unionpay_line == False:
            lines_y_new.append(lines_y[0])
        remove_pos.append(0)
        lines_y = remove_element(remove_pos, lines_y)
    return lines_y_new


def sort_y_line(y_line):
    """对y方向线段安装坐标的前后顺序进行排序"""
    x = []
    for i in range(len(y_line)):
        x.append(y_line[i].extend_pt1[0])
    x, order = line_sort(x, sort='asc')
    y_line_new = []
    for i in range(len(y_line)):
        y_line_new.append(y_line[order[str(i)]])
    return y_line_new


def sort_x_line(x_line):
    """对y方向线段安装坐标的前后顺序进行排序"""
    x = []
    for i in range(len(x_line)):
        x.append(x_line[i].extend_pt1[1])
    x, order = line_sort(x, sort='asc')
    x_line_new = []
    for i in range(len(x_line)):
        x_line_new.append(x_line[order[str(i)]])
    return x_line_new
