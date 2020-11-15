import torchvision.datasets as dset
import torchvision
from PIL import Image
import numpy as np

class STL10_Relabel(dset.STL10):
    def __init__(self, split = 'unlabeled'):
        super().__init__(root = './', split = split, download = True)
        self.Timg = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),torchvision.transforms.ToTensor()])
        self.Ttar = torchvision.transforms.ToTensor()
    def __getitem__(self, index: int):
        #reprogramme torchvision's dataset so that it return orginal img as label
        
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        target = self.Ttar(img)
        
        img = self.Timg(img)

        return img, target
