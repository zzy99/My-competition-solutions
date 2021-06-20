import glob
import os
import random
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Normalize, ToTensor, Compose
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = cv2.imread('img/' + file)[:, :, ::-1]
        image = cv2.resize(image, (128, 128))
        image = (image / 255.0).astype(np.float32)
        image = np.moveaxis(image, 2, 0)
        return file, image


if __name__ == '__main__' :

    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]

    transforms = Compose([
        ToTensor(),
        Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])
    # model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True).cuda()
    model = r2plus1d_18(pretrained=True).cuda()
    model.eval()
    model.fc = torch.nn.Identity()

    # files = os.listdir('data_C')
    files = os.listdir('spectrogram')
    with torch.no_grad():
        for file in tqdm(files):
            feat_stack = []
            start = file.split('.')[0].split('_')[-1] == 'start'
            if start:
                length = 200*25
            else:
                length = 180*25
            input = []
            for i in range(1, length+1, 25):
                images = []
                for k in range(25):
                    path = 'frames/' + file.split('.')[0] + '_' + str(i+k).rjust(4, '0') + '.jpg'
                    image = cv2.imread(path)[:, :, ::-1].copy()
                    image = transforms(image)
                    images.append(image.unsqueeze(1)) #(3, 1, 128, 128)
                images = torch.cat(images, 1) #(3, fps, 128, 128)
                images = images.unsqueeze(0) #(1, 3, fps, 128, 128)
                input.append(images)

            input = torch.cat(input, 0) #(200/180, 3, fps, 128, 128)
            y = torch.zeros((input.size(0)))
            dataset = TensorDataset(input, y)
            loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

            outputs = []
            for i, (x, y) in enumerate(loader):
                x = x.cuda() #(32, 3, fps, 128, 128)
                output = model(x).cpu().numpy()  #(32, 512)
                outputs.extend(output)
            outputs = np.array(outputs) #(200/180, 512)

            np.save('frames_npy/' + file.split('.')[0] + '.npy', outputs)


