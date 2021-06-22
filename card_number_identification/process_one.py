import cv2
from os.path import join, basename, dirname
from card_number_identification.predict_num import predict_number, load_model
from card_number_positioning.util import cv2_imread, check_folder, set_value, get_value, cv2_imwrite
from card_number_positioning.cut_rectify import cut_and_rectify
from card_number_positioning.find_num_area import seg_num_area
from card_number_positioning.num_segment import num_segment
from PyQt5.QtCore import QDir
import os


def processOne(img_path, model, debug=False):
    '''处理一张图像'''
    # 在全局变量中设置debug模式
    set_value('debug', debug)  # if debug == True else util.set_value('debug', False)
    # 设置图像名称
    name = basename(img_path)
    set_value('name', name)
    # 初始化目录
    # 以图像名称为文件夹名称
    save_path = get_value("save_root_path")
    set_value('save_path', join(save_path, name))
    check_folder(join(save_path, name, "详细识别过程"))

    # 读取图片
    org_img = cv2_imread(img_path)

    # 银行卡切割
    # 在缩放图上计算切割
    img = cut_and_rectify(org_img)
    cv2_imwrite(join(save_path, name, "1.切割和矫正.jpg"), img)

    # 获取银行卡号区域的上下边界
    scale_img = cv2.resize(img, (480, 305), cv2.INTER_CUBIC)

    upper_border, lower_border = seg_num_area(scale_img)
    # 保存银行卡卡号区域
    num_area_img = scale_img[int(upper_border):int(lower_border), :, :]
    cv2_imwrite(join(save_path, name, "2.银行卡卡号区域.jpg"), num_area_img)

    # 卡号分割
    # 银行卡缩放比例
    w_scale = img.shape[1] / 480
    h_scale = img.shape[0] / 305
    seg_num_img_list = num_segment(scale_img, img,
                                   (upper_border, lower_border),
                                   (w_scale, h_scale))

    # 卡号识别
    str_predict = ''
    check_folder(join(save_path, name, "3.卡号分割"))
    for i in range(len(seg_num_img_list)):
        cv2_imwrite(join(save_path, name, "3.卡号分割", str(i + 1) + ".jpg"), seg_num_img_list[i])
        predict = predict_number(seg_num_img_list[i], model)
        if predict != 10:
            str_predict += str(predict)
        else:
            str_predict += '\t'
    with open(join(save_path, name, "4.预测结果.txt"), 'w') as f:
        f.write("{}".format(str_predict))
    print("预测结果：", str_predict)
    # 保存预测结果
    if debug == False:
        with open(join(save_path, "result.txt"), 'a+') as f:
            f.write("{}:{}\n".format('card_' + name, str_predict))
    print("name:" + name)
    return str_predict


def loadModel():
    #model_path = join(dirname(QDir(QDir().currentPath()).currentPath()), "card_number_identification/result/exp1", "29_dict.pth")
    model_path = join(dirname(QDir(QDir().currentPath()).currentPath()), "card_number_identification","model_dict.pth")
    model = load_model(model_path)
    return model


def main():
    root_dir = dirname(os.path.abspath(os.path.dirname(__file__)))
    test_img_path = join(root_dir,'demo', 'test_images')
    model = loadModel()
    for file_name in os.listdir(test_img_path):
        img_path = join(test_img_path, file_name)
        try:
            processOne(img_path, model, debug=True)
        except:
            print("识别错误")
        # 按下任意键后进入下一张图像处理
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model = loadModel()
    img_path = "../demo/test_images/image(31).jpg"
    processOne(img_path, model, debug=True)
    #main()
