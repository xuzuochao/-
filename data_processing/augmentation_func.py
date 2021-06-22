'''用于数据增强的方法，输入一个银行卡号数字，得到八十三个数据，只选取前80个，使用时直接调用augmentation函数'''

import cv2
import numpy as np
from torchvision import transforms
import random
# from imgaug import augmenters as iaa


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


def saturation(img, s):
    """通过将RGB图转换到HLS，改变原图饱和度

    :param img: 输入RGB图
    :param s: 越大饱和度越高，负数饱和度相反（低）
    :return: 改变饱和度后的RGB图
    """

    # 图像归一化，且转换为浮点型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    # 复制
    hlsCopy = np.copy(hlsImg)
    # 饱和度
    hlsCopy[:, :, 2] = (1.0 + s / float(10)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)

    return lsImg


def clamp(pv):
    """避免像素值溢出（大于255）

    :param pv: 像素值
    :return: 像素值
    """

    if pv > 255:
        return 255

    if pv < 0:
        return 0

    else:
        return pv


def gaussian_noise(image):
    """在像素值中根据高斯分布随机添加像素值，产生高斯噪声

    :param image: 输入RGB图
    :return: 添加高斯噪声后的RGB图
    """

    h, w, c = image.shape
    for row in range(h):
        for column in range(w):
            # 从0到20生成3个随机数
            s = np.random.normal(0, 20, 3)
            b = image[row, column, 0]
            g = image[row, column, 1]
            r = image[row, column, 2]
            image[row, column, 0] = clamp(b + s[0])
            image[row, column, 1] = clamp(g + s[1])
            image[row, column, 2] = clamp(g + s[2])

    return image


def contrast_brightness_pics(imgs, init_c, init_b, nums):
    """将原图增强为多张改变了对比度与亮度的图像

    :param imgs: 输入RGB图
    :param init_c: 初始对比度参数
    :param init_b: 初始亮度参数
    :param nums: 得到的数据量
    :return: 经过对比度与亮度数据增强后的列表
    """

    contrast_nums = []
    step = (1.3 - init_c)/nums
    for contrast in imgs:
        # 对比度参数
        c = init_c
        # 亮度参数
        b = init_b
        while c < 1.3:
            dst = contrast_brightness(contrast, c, b)
            contrast_nums.append(dst)
            # 亮度度增加比例
            c += step
            # 对比度增加比例
            b += 1

    return contrast_nums

def saturation_pics(imgs, contrast_nums):
    """将原图增强为多张改变了对比度与亮度的图像

    :param imgs: 输入RGB图
    :param contrast_nums: 经过对比度与亮度数据增强后的列表
    :return: 经过饱和度数据增强后的列表
    """

    change_stura = []
    for stura1 in imgs:
        # 1倍饱和度
        dst1_stura1 = saturation(stura1, 10)
        # -1倍饱和度
        dst2_stura1 = saturation(stura1, -10)
        change_stura.append(dst1_stura1)
        change_stura.append(dst2_stura1)

    for stura2 in contrast_nums:
        # 1倍饱和度
        dst1_stura2 = saturation(stura2, 10)
        # -1倍饱和度
        dst2_stura2 = saturation(stura2, -10)
        change_stura.append(dst1_stura2)
        change_stura.append(dst2_stura2)

    return change_stura


def gaussian_noise_pics(imgs, contrast_nums):
    """将原图增强为多张改变了对比度与亮度的图像

    :param imgs: 输入RGB图
    :param contrast_nums: 经过对比度与亮度数据增强后的列表
    :return: 经过添加高斯噪声数据增强后的列表
    """

    # 存放增加高斯噪声的数字图片
    gauss_num = []
    for gauss1 in imgs:
        dst1_gauss1 = gaussian_noise(gauss1)
        gauss_num.append(dst1_gauss1)

    for gauss2 in contrast_nums:
        dst1_gauss2 = gaussian_noise(gauss2)
        gauss_num.append(dst1_gauss2)

    return gauss_num

def GaussianBlur_pics(imgs, contrast_nums):
    """将原图增强为多张改变了高斯模糊的图像

    :param imgs: 输入RGB图
    :param contrast_nums: 经过对比度与亮度数据增强后的列表
    :return: 经过高斯模糊数据增强后的列表
    """

    # 存放高斯模糊操作后的数字图
    gaussblur_num = []
    for gauss1_blur in imgs:
        dst1_gaussblur = cv2.GaussianBlur(gauss1_blur, (7, 7), 0)
        gaussblur_num.append(dst1_gaussblur)

    for gauss2_blur in contrast_nums:
        dst2_gaussblur = cv2.GaussianBlur(gauss2_blur, (7, 7), 0)
        gaussblur_num.append(dst2_gaussblur)

    return gaussblur_num

def augmentation(img):
    """数据增强,通过改变对比度、改变饱和度、添加噪声、模糊处理、随机裁剪等处理得到更多数据

    :param img: 输入RGB图
    :return: 返回80个数据增强后的图片
    """

    # 默认不进行旋转
    rotation_num = [img]
    # 存放对比度和亮度调节后的数字图
    contrast_nums = contrast_brightness_pics(rotation_num, 0.6, 1.5, 10)
    # 存放改变饱和度的图片
    change_stura = saturation_pics(rotation_num, contrast_nums)
    # 存放增加高斯噪声的数字图片
    gauss_num = gaussian_noise_pics(rotation_num,contrast_nums)
    # 存放高斯模糊操作后的数字图
    gaussblur_num = GaussianBlur_pics(rotation_num, contrast_nums)

    # 第一个随机裁剪函数
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop((35, 35)),])

    # 第二个随机裁剪函数
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop((42, 42)),])

    # 存放需裁剪的图像
    all_pic = []
    pics = rotation_num + contrast_nums + change_stura + gauss_num + gaussblur_num

    # 进行第一部分裁剪
    for crop_num in range(0, len(pics), 4):
        croped_img = transform1(pics[crop_num])
        imgs = np.asarray(croped_img)
        all_pic.append(imgs)

    # 进行第二部分裁剪
    for crop_num in range(1, len(pics)-1, 4):
        croped_img = transform2(pics[crop_num])
        imgs = np.asarray(croped_img)
        all_pic.append(imgs)

    all_pic += rotation_num + contrast_nums + change_stura + gauss_num + gaussblur_num

    # 进行透视变换
    is_trans = 0
    for img in all_pic:
        h, w, ch = img.shape
        img = cv2.resize(img, (int(w * 1.2), int(h * 1.2)), cv2.INTER_LINEAR)  # 放大

        # 扩展图像，保证内容不超出可视范围
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        fov = 10

        w, h = img.shape[0:2]  # 必须要注意，此处不是h,w = img.shape[0:2]，要不然图像会h,w会相反

        # 能够整除3，4，5随机的一个向左透视
        if is_trans % random.randint(3, 5) == 0:
            anglex = 10
            angley = 10
            anglez = 5
            result = coordinate_transform(img, anglex, angley, anglez, fov, w, h)
            all_pic[is_trans] = result
            # cv2.imshow("result1", result)

        # 能够整除11，12，13随机的一个向右透视
        elif is_trans % random.randint(11, 13) == 0:
            anglex = 10
            angley = 10
            anglez = -5
            result = coordinate_transform(img, anglex, angley, anglez, fov, w, h)
            all_pic[is_trans] = result
            # cv2.imshow("result1", result)
        is_trans += 1

    return all_pic



def rad(x):
    return x * np.pi / 180


def coordinate_transform(img, anglex, angley, anglez, fov, w, h):
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    result = cv2.warpPerspective(img, warpR, (h, w))

    return result



