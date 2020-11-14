import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in, out, pool):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in, (in+out)//2, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(pool))
        self.layer2 = nn.Sequential(nn.Conv2d((in+out)//2, out, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(pool))
        self.pool = pool
    def forward(self, x):
        X = x
        x = self.layer1(x)
        x = self.layer2(x)
        X = F.max_pool2d(self.pool**2, X)
        return X + x
    
class Colorizer(nn.Module):
    def __init__(self,layers = 20, input_size = 96):
        self.layers = nn.ModuleList()
