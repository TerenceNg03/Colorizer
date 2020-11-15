import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import time
import datetime


class Trainer:
    def __init__(self, model_G, model_D, dataset, batchsize = 100):
        self.model_G = model_G
        self.model_D = model_D
        self.dataset = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
        self.optimizer_D = optim.Adam(model_D.parameters())
        self.optimizer_G = optim.Adam(model_G.parameters())
        self.cuda = torch.cuda.is_available()
        self.T = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Grayscale(),torchvision.transforms.ToTensor()])

        if self.cuda:
            self.model_G = self.model_G.cuda()
            self.model_D = self.model_D.cuda()
            
    def pretrain(self, epochs, timeout = -1):
        for epoch in range(epochs):
            time_start = time.time()
            total = self.dataset.__len__()
            i = 0
            timedout = False
            for _, data in enumerate(self.dataset): 
                gray, orig = data
                if self.cuda:
                    gray = gray.cuda()
                    orig = orig.cuda()
                self.optimizer_G.zero_grad()
                output = self.model_G(gray)
                loss = F.mse_loss(output, orig)
                loss.backward()
                self.optimizer_G.step()
                i += 1
                time_end = time.time()
                print('(pretrain Epcho %d / %d) Training process : %.3f%% '%(epoch+1, epochs, 100*i/total)+'Time used : '+str(datetime.timedelta(seconds=int(time_end-time_start))),end='\r')
                if timeout>0 and (time_end-time_start>=timeout):
                    print('(pretrain Epcho %d / %d) Training process : %.3f%% Timed out'%(epoch+1, epochs, 100*i/total)+' '*40)
                    timedout = True
                    break
            if not timedout:
                print('(pretrain Epcho %d / %d) Done in time %.3f s'%(epoch+1, epochs, time_end-time_start)+' '*40)
            
    def train(self, epochs, timeout = -1):
        #timeout : control train time
        
        self.model_D.train()
        self.model_G.train()
        for epoch in range(epochs):
            time_start = time.time()
            total = len(trainset)
            i = 0
            for data in enumerate(self.dataset):  
                X, _ = data 
                self._train(X)
                i += 1
                print('(Epcho %d / %d) Training process : %.3f%%'%(epoch+1, epochs, 100*i/total),end='\r')
                if timeout>0 and (time.time()-time_start>=timeout):
                    break
            time_end = time.time()
            print('(Epcho %d / %d) Done in time %.3f s'%(epoch+1, epochs, time_end-time_start)+' '*40, end = '\r')

        
        