import torch.utils.data as data
import os
from os.path import join
import cv2


class Dataset(data.Dataset):
    def __init__(self, root='data', split='train'):
        self.image_path = []
        self.image_label = []
        if split == 'train':
            for i in range(0, 11):
                path = join(root, split, str(i))
                for file in os.listdir(path):
                    self.image_path.append(join(path, file))
                    self.image_label.append(i)
        elif split == 'test':
            for i in range(0, 11):
                path = join(root, split, str(i))
                for file in os.listdir(path):
                    self.image_path.append(join(path, file))
                    self.image_label.append(i)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        label = self.image_label[index]
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (30, 30))
            image = image.transpose((2, 0, 1))
            return image, label, image_path
        except:
            print(image_path)
            return None, None, None

    def __len__(self):
        return len(self.image_label)

