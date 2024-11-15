import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import tifffile as tif
import numpy as np

from network import Generator
from dataset_unsupervised import Synth_Dataset, Real_Dataset
from dataset_supervised import SAR_Dataset
from utils import test_Crop

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mu = 1
    gamma = 0
    batch_size = 8

    ep = 1e-3
    clip = 2

    path2synth = './Data/SAR Data/SAMPLE/mat_files_pix2pix_60%/synth'
    path2real = './Data/SAR Data/SAMPLE/mat_files_pix2pix_60%/real'
    path2model = './Weight/ComplexCycleGAN/231025~/weights_gen_complex_cycle_consistency_tanh.pt'
    path2save = './Data/SAR Data/SAMPLE/results/231025~/ComplexCycleGAN_complex_cycle_consistency_tanh'

    transform = transforms.Compose([transforms.ToTensor(), test_Crop()])
    
    test_ds = Synth_Dataset(path2synth, train = False, transform = transform, ep = ep, clip = clip)
    real_ds = Real_Dataset(path2real, train = False, transform = transform, ep = ep, clip = clip)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    real_dl = DataLoader(real_ds, batch_size = batch_size, shuffle = False)

    Gen_A2B = Generator(bias = False).to(device)

    weight = torch.load(path2model)
    Gen_A2B.load_state_dict(weight)
    Gen_A2B.eval()
    to_pil = ToPILImage()
    with torch.no_grad():
        for synth, label, name in test_dl:
            fake_real = abs(Gen_A2B(synth.to(device)).detach().cpu())

            for ii, img in enumerate(fake_real):
                path = os.path.join(path2save, 'refine', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                # tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))

            for ii, img in enumerate(synth):
                img = abs(img)
                path = os.path.join(path2save, 'synth', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                # tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))

        for real, label, name in real_dl:
            for ii, img in enumerate(real):
                img = abs(img)
                path = os.path.join(path2save, 'real', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
                # tif.imsave(os.path.join(path, name[ii][:-3]+'tif'), np.array(img.squeeze()))

    
    # a
    # test_ds = SAR_Dataset('QPM', transform = transform, train = False)
    # test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    # with torch.no_grad():
    #     for synth, real, label, name in test_dl:
    #         fake_real = Gen_A2B(synth.to(device)).detach().cpu()

    #         for ii, img in enumerate(fake_real):
    #             path = os.path.join(path2save, 'refine', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))
            
    #         for ii, img in enumerate(synth):
    #             path = os.path.join(path2save, 'synth', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))

    #         for ii, img in enumerate(real):
    #             path = os.path.join(path2save, 'real', label[ii])
    #             os.makedirs(path, exist_ok = True)
    #             img = to_pil(img * mu + gamma)
    #             img.save(os.path.join(path, name[ii][:-3]+'png'))