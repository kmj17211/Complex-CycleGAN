import torch
from torchvision import transforms
from torchvision.transforms.functional import rotate
import random

from scipy.signal import wiener, medfilt
from scipy.stats import truncexpon

class ReplayBuffer():
    def __init__(self, max_size = 50):
        assert (max_size > 0), '0보다 크게 해라'
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for element in data:
            # element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1) > 0.5:
                    i = torch.randint(0, self.max_size, size = [1])
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.stack(to_return, dim = 0)
    
class Crop(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            trans = transforms.RandomCrop((100, 100))
        else:
            trans = test_Crop()
        return trans(img)
    
class test_Crop(object):
    def __init__(self):
        self.x1 = 5
        self.x2 = 105
        self.y1 = int(128 / 2 - 50)
        self.y2 = int(128 / 2 + 50)
    def __call__(self, img):
        cropped_img = img[:, self.y1:self.y2, self.x1:self.x2]
        return cropped_img
    
class Rotation(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:

            trans = transforms.Compose([transforms.Pad(30, padding_mode = 'reflect'),
                                transforms.RandomRotation((-20, 20)),
                                transforms.CenterCrop(img.size(1))])
            
            return trans(img)
        else:
            return img
        
class Angle_Rotation(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            img = rotate(img, random.randint(1, 3) * 90)
            return img
        else:
            return img
        
class Add_Noise(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            img = wiener(img)
            img = medfilt(img, kernel_size = (1, 3, 3))
            noise = truncexpon.rvs(1, size = img.shape)
            noise_img = torch.tensor(img * noise)
            noise_img[noise_img > 1] = 1
            return noise_img #torch.tensor(img)
        else:
            return img