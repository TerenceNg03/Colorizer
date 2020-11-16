import torch.nn as nn
import torch.nn.functional as F
import torch

class Down(nn.Module):
    def __init__(self, inc, out, pool = 2, relu = True):
        super().__init__()
        self.relu = relu
        self.layer1 = nn.Sequential(nn.Conv2d(inc, (inc+out)//2, 3, padding = 1))
        self.layer2 = nn.Sequential(nn.Conv2d((inc+out)//2, out, 3, padding = 1))
        self.pool = pool
    def forward(self, x):
        x = self.layer1(x)
        if self.relu:
            x = F.relu(x)
        x = self.layer2(x)
        if self.relu:
            x = F.relu(x)
        x = F.max_pool2d(x, self.pool)
        return x
    
class Up(nn.Module):
    def __init__(self, inc, out, relu = True):
        super().__init__()
        self.relu = relu
        self.layer1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer2 = nn.Conv2d(inc, out, 3, padding = 1)
    def forward(self, x):
        x = self.layer1(x)
        if self.relu:
            x = F.relu(x)
        x = self.layer2(x)
        if self.relu:
            x = F.relu(x)
        return x

class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = Down(1,8)
        self.d2 = Down(8,64)
        self.d3 = Down(64,128)
        self.d4 = Down(128,256)
        self.u1 = Up(256,128)
        self.u2 = Up(256,64)
        self.u3 = Up(128,32)
        self.u4 = Up(40,32)
        self.out = Down(33,3,pool = 1, relu = False)
        
    def forward(self, x):
        orig = x
        x = self.d1(x)
        out1 = x
        x = self.d2(x)
        out2 = x
        x = self.d3(x)
        out3 = x
        x = self.d4(x)
        
        x = self.u1(x)
        x = torch.cat([x,out3], dim = 1)
        x = self.u2(x)
        x = torch.cat([x,out2], dim = 1)
        x = self.u3(x)
        x = torch.cat([x,out1], dim = 1)
        x = self.u4(x)
        x = torch.cat([x,orig], dim = 1)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x
        
class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.d1 = Down(3,24)
        self.d2 = Down(24,192)
        self.d3 = Down(192,384)
        self.d4 = Down(384,384)
        self.affine = nn.Sequential(nn.Linear(384*36, 1000), nn.ReLU(), nn.Linear(1000, 2))
        
    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x =torch.flatten(x, start_dim=1)
        x = self.affine(x)
        x = F.log_softmax(x, dim = 1)
        
        return x
        
