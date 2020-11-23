from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

content_path = "none"
style_path = "none"

class filedialogdemo(QWidget):

    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QVBoxLayout()

        vlayout = QHBoxLayout()
        vwg = QWidget()
        vwg.setLayout(vlayout)
        layout.addWidget(vwg)

        vlayout1 = QVBoxLayout()
        vwg1 = QWidget()
        vwg1.setLayout(vlayout1)
        vlayout.addWidget(vwg1)

        vlayout2 = QVBoxLayout()
        vwg2 = QWidget()
        vwg2.setLayout(vlayout2)
        vlayout.addWidget(vwg2)

        self.btn1 = QPushButton()
        self.btn1.clicked.connect(self.loadFile_1)
        self.btn1.setText("Choose Content Image")
        vlayout1.addWidget(self.btn1)

        self.label1 = QLabel()
#        self.label1.setPixmap(QPixmap('content_tmp.jpg'))
        vlayout1.addWidget(self.label1)

        self.btn2 = QPushButton()
        self.btn2.clicked.connect(self.loadFile_2)
        self.btn2.setText("Choose Style Image")
        vlayout2.addWidget(self.btn2)

        self.label2 = QLabel()
#        self.label2.setPixmap(QPixmap('style_tmp.jpg'))
        vlayout2.addWidget(self.label2)

        self.btn3 = QPushButton("START and Run Now (need about 10 minutes)")
        self.btn3.clicked.connect(self.start_transfer)
        layout.addWidget(self.btn3)

        self.btn4 = QPushButton("START and See the finished result")
        self.btn4.clicked.connect(self.directly_print)
        layout.addWidget(self.btn4)

        self.splider = QSlider(Qt.Horizontal)
        self.splider.setMinimum(0)
        self.splider.setMaximum(100)
        self.splider.setSingleStep(5)
        self.splider.setValue(100)
        self.splider.valueChanged.connect(self.modify)
        layout.addWidget(self.splider)

        self.label3 = QLabel()
        layout.addWidget(self.label3)
#        self.label3.setPixmap(QPixmap('output_tmp.jpg'))
        self.label3.setAlignment(Qt.AlignCenter)

        self.setWindowTitle("Image Style Transfer")
        self.setWindowIcon(QIcon('logo.jpg'))

        palette1 = QPalette()
#        palette1.setColor(self.backgroundRole(), QColor(192,253,123))
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)

        self.setLayout(layout)
        self.move(700, 20)

    def loadFile_1(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'd:\\picture_content\\', 'Image files(*.jpg *.gif *.png)')
        self.label1.setPixmap(QPixmap(fname))
        global content_path
        if fname.startswith('D'):
            content_path = fname
        print(fname)
        im = Image.open(fname)
        scale = 256 / max(im.size)
        size = np.array(im.size) * scale
        im = im.resize(size.astype(int), Image.ANTIALIAS)
        plt.imshow(im)
        plt.savefig('input_tmp.png')

    def loadFile_2(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'd:\\picture_style\\', 'Image files(*.jpg *.gif *.png)')
        self.label2.setPixmap(QPixmap(fname))
        global style_path
        if fname.startswith('D'):
            style_path = fname
        print(fname)

    output_path = 'D:/picture_output/001-002.png'
    def modify(self):
        value = self.splider.value()
        im_out = np.array(Image.open(output_path))
        im_in = np.array(Image.open('input_tmp.png'))
 #       im_out = Image.fromarray(cv2.cvtColor(im_out,cv2.COLOR_BGR2RGB))
 #       im_in = Image.fromarray(cv2.cvtColor(im_in,cv2.COLOR_BGR2RGB))
        im_tmp = im_out
        rows,cols,dims=im_in.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(dims):
                    im_tmp[i,j,k] = (float)(im_out[i,j,k])/100*value+(float)(im_in[i,j,k])/100*(100-value)
        im_tmp = Image.fromarray(im_tmp)  
        im_tmp.save('result.png')
        self.label3.setPixmap(QPixmap('result.png'))

    def start_transfer(self):
        if content_path == 'none':
            self.label1.setText('You should choose an image')
            return
        if style_path == 'none':
            self.label2.setText('You should choose an image')
            return
        self.label3.setText('Please Wait For A Moment')
        run()
        self.label3.setPixmap(QPixmap('result.png'))
 #       self.label3.setPixmap(QPixmap('output.png'))

    def directly_print(self):
        if content_path == 'none':
            self.label1.setText('You should choose an image')
            return
        if style_path == 'none':
            self.label2.setText('You should choose an image')
            return
        tmp1 = content_path.lstrip('D:/picture_content/').rstrip('.jpg')
        tmp2 = style_path.lstrip('D:/picture_style/').rstrip('.jpg')
        global output_path
        output_path = 'D:/picture_output/' + tmp1 + '-' + tmp2 + '.png'
        self.label3.setPixmap(QPixmap(output_path))

 #       self.label3.setPixmap(QPixmap(output_path))





def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 256  # use small size if no gpu
 
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # 将图片转化为张量
 
    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0) #添加一个0维度 batch 适应网络输入
        return image.to(device, torch.float)
 
    style_img = image_loader(style_path)
    content_img = image_loader(content_path)
 
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    unloader = transforms.ToPILImage()  # reconvert into PIL image
 
    plt.ion()
 
    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension 去掉0维度
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated
 
    plt.figure()
    imshow(style_img, title='Style Image')
 
    plt.figure()
    imshow(content_img, title='Content Image')

    class ContentLoss(nn.Module):
 
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            '''
            we 'detach' the target content from the tree used
            to dynamically compute the gradient: this is a stated value,
            not a variable. Otherwise the forward method of the criterion
            will throw an error.
            '''
            self.target = target.detach()
 
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    # 计算风格损失
    def gram_matrix(input):
        a, b, c, d = input.size()
        # a = batch size(=1)
        # b = number of feature maps 
        # (c,d) = dimensions of a f. map (N = c*d)
 
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
 
        G = torch.mm(features, features.t())  # compute the gram product
 
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):
 
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
 
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    # 导入模型
    cnn = models.vgg16(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
 
    # create a module to normalize input image so we can easily put it in a nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)  #（3，1，1）
            self.std = torch.tensor(std).view(-1, 1, 1)
 
        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4'] #在第四个卷积层后进行计算 内容损失
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] #在每个卷积层后计算风格损失
 
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
 
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)
 
        # just in order to have an iterable access to or list of content/syle losses
        content_losses = []
        style_losses = []
 
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization) #先在model 中加上标准化层
 
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
 
            model.add_module(name, layer) #每遇到一个层 就加到model中
 
            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
 
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
 
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
 
        model = model[:(i + 1)]
 
        return model, style_losses, content_losses

    input_img = content_img.clone()
    plt.figure()
    imshow(input_img, title='Input Image')

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)
 
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
 
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)
 
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
 
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
 
                style_score *= style_weight
                content_score *= content_weight
 
                loss = style_score + content_score
                loss.backward()
 
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
 
                return style_score + content_score
 
            optimizer.step(closure)
 
        # a last correction...
        input_img.data.clamp_(0, 1)
 
        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
 
    plt.figure()
    imshow(output, title='Output Image')
 
    # sphinx_gallery_thumbnail_number = 4
    plt.savefig('output.png', format = 'png')
#    plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    fileload =  filedialogdemo()
    fileload.show()
    sys.exit(app.exec_())