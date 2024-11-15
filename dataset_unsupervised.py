import os
import numpy as np
from os.path import join
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image
import matplotlib.pyplot as plt

class Real_Dataset(Dataset):
    def __init__(self, path, transform = False, train = True, ep = 1e-6, clip = 10):
        super(Real_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()
        
        self.train = train
        self.ep = ep
        self.clip = clip
            
        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if train:
            self.dir_t = 'train'
            self.label = self.label[:7]
        else:
            self.dir_t = 'test'

        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)

    def __getitem__(self, index):
        
        # Label
        label = self.data_label[index]

        # File Name
        file_name = self.file_name[index]

        # Data Load
        # img = self.transform(Image.open(self.data_path[index]).convert('L'))
        # Logarithm
        img = loadmat(self.data_path[index])['complex_img']
        mag = abs(img)
        phase = np.angle(img)
        mag = np.log10(mag + self.ep)
        mag[mag > np.log10(self.clip)] = np.log10(self.clip)
        mag = (mag - np.log10(self.ep)) / (np.log10(self.clip) - np.log10(self.ep))

        img_r, img_i = mag * np.cos(phase), mag * np.sin(phase)

        img_r = self.transform(img_r)
        img_i = self.transform(img_i)

        img = torch.complex(img_r, img_i)

        return img.type(torch.complex64), label, file_name
    
    def __len__(self):
        return len(self.data_path)
    

class Synth_Dataset(Dataset):
    def __init__(self, path, transform = False, train = True, ep = 1e-6, clip = 10):
        super(Synth_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()
        
        self.train = train
        self.ep = ep
        self.clip = clip

        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if train:
            self.dir_t = 'train'
            self.label = self.label[:7]
        else:
            self.dir_t = 'test'


        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)

    def __getitem__(self, index):
        
        # Label
        label = self.data_label[index]

        # File Name
        file_name = self.file_name[index]
        file_name = file_name.replace('synth', 'refine')

        # Logarithm
        img = loadmat(self.data_path[index])['complex_img']
        mag = abs(img)
        phase = np.angle(img)
        mag = np.log10(mag + self.ep)
        mag[mag > np.log10(self.clip)] = np.log10(self.clip)
        mag = (mag - np.log10(self.ep)) / (np.log10(self.clip) - np.log10(self.ep))

        img_r, img_i = mag * np.cos(phase), mag * np.sin(phase)

        img_r = self.transform(img_r)
        img_i = self.transform(img_i)

        img = torch.complex(img_r, img_i)

        return img.type(torch.complex64), label, file_name
    
    def __len__(self):
        return len(self.data_path)
    
if __name__ == '__main__':
    
    path = './Data/SAR Data/SAMPLE/results/CycleGAN_230817_ep1e-6_clip_10/refine/2s1/2s1_refine_A_elevDeg_015_azCenter_016_22_serial_b01.png'
    img = Image.open(path).convert('L')
    # img = plt.imread(path)
    plt.imshow(img, cmap = 'gray')
    plt.savefig('k.png')
    print(np.array(img).shape)
    print('a')