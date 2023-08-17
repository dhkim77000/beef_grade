import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, RandomCrop, ColorJitter, CenterCrop
import albumentations
import albumentations.pytorch
import albumentations.augmentations
import cv2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, h, w, mean, std, **args):

        self.transform = albumentations.Compose([ 
            albumentations.RandomCrop(h, w),
            albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.Normalize(mean=mean, std=std),
            albumentations.OneOf([
                              albumentations.HorizontalFlip(p=1),
                              albumentations.RandomRotate90(p=1),
                              albumentations.VerticalFlip(p=1),
                              albumentations.ShiftScaleRotate(shift_limit=0.0625, 
                                                scale_limit=0.1, 
                                                rotate_limit=45, 
                                                p=1),            
            ], p=1),
            albumentations.OneOf([
                              albumentations.OpticalDistortion(p=1),
                              albumentations.GaussNoise(p=1)                 
            ], p=1),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        
    def __call__(self, image):
        return self.transform(image=image)



class BeefDataset(Dataset):
    num_classes = 5

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.label = pd.read_csv(os.path.join(data_dir, 'grade_labels.csv'))
        self.label.set_index('imname',inplace = True)
        self.label_encoder = {'1++':0, '1+':1, '1':2, '2':3, '3':4}
        self.label_decoder = {0:'1++', 1:'1+', 2:'1', 3:'2', 4:'3'}
        self.labels = []
        self.setup()
        self.calc_statistics()

        

    def setup(self):

        img_folder = os.path.join(self.data_dir, 'images')
        for file_name in os.listdir(img_folder):
            
            if is_image_file(file_name):

                img_path = os.path.join(img_folder, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)

                grade_label = self.label_encoder[self.label.loc[file_name,'grade']]

                self.image_paths.append(img_path)
                self.labels.append(grade_label)


    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):

        image = self.read_image(index)
        image_transform = self.transform(image)['image'].float()
        
        return image_transform, self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        # 이미지의 Red 채널 선택
        red_channel = image[:, :, 0]
        # Red 채널 강화 (증폭)
        equalized_red_channel = cv2.equalizeHist(red_channel)
       
        equalized_image = np.copy(image)
        equalized_image[:, :, 0] = equalized_red_channel 
        
        return equalized_image

    def decode_labels(self, label):
        return self.label_decoder[label]
    
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        self.label_to_index = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_index:
                self.label_to_index[label] = []
            self.label_to_index[label].append(idx)

        train_indices = []
        val_indices = []
        for label, indices in self.label_to_index.items():
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

        self.train_dataset = Subset(self, train_indices)
        self.val_dataset = Subset(self, val_indices)

        return self.train_dataset, self.val_dataset

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            CenterCrop((128, 128)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
