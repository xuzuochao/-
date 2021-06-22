import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from os.path import join
import cv2
from card_number_identification.googlenet import GoogLeNet

def predict_number(seg_num_img, model):
    seg_num_img = cv2.resize(seg_num_img, (30, 30))
    seg_num_img = seg_num_img.transpose((2, 0, 1))
    seg_num_img=torch.from_numpy(seg_num_img)
    seg_num_img = seg_num_img.type(torch.FloatTensor)
    seg_num_img = seg_num_img.view(1, 3, 30, 30)

    inputs = Variable(seg_num_img)

    CUDA = torch.cuda.is_available()
    if CUDA:
        inputs = inputs.float().cuda()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    return predicted.item()


def load_model(model_path):
    model = GoogLeNet()

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    return model
