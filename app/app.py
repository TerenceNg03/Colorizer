from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
import torch
import torchvision.transforms as transforms
import cv2
import os
from pretrain.model import Colorizer


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Colorizer")


        layout = QVBoxLayout()

        vlayout = QHBoxLayout()
        vwg = QWidget()
        vwg.setLayout(vlayout)
        layout.addWidget(vwg)

        ### Load panel ###
        vlayout1 = QVBoxLayout()
        vwg1 = QWidget()
        vwg1.setLayout(vlayout1)
        vlayout.addWidget(vwg1)

        ### Output panel ###
        vlayout2 = QVBoxLayout()
        vwg2 = QWidget()
        vwg2.setLayout(vlayout2)
        vlayout.addWidget(vwg2)

        self.btn1 = QPushButton()

        self.btn1.setText("Choose a image")
        self.btn1.clicked.connect(self.loadimage)
        vlayout1.addWidget(self.btn1)

        self.graph1 = QLabel()
        vlayout1.addWidget(self.graph1)
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        vlayout1.addWidget(self.label1)

        self.btn2 = QPushButton()
        self.btn2.setText("Save output as")
        self.btn2.clicked.connect(self.saveimage)
        vlayout2.addWidget(self.btn2)

        self.graph2 = QLabel()
        vlayout2.addWidget(self.graph2)
        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        vlayout2.addWidget(self.label2)

        self.btn3 = QPushButton("Convert(This may take a few seconds)")
        self.btn3.clicked.connect(self.start_thread)
        layout.addWidget(self.btn3)


        widget = QWidget()
        widget.setLayout(layout)

        
        self.setCentralWidget(widget)
        
        ###init ###
        self.image = None
        self.output = None
        self.thread = None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Colorizer()
        model.load_state_dict(torch.load('./pretrain/colorizer.pt', map_location=device))
        self.model = model.eval()
        
    def loadimage(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', 
   './example/',"Image files (*.jpg *.gif *.png *.bmp)")
        if not filename:
            return
        self.filename = filename
        image = Image.open(filename)
        image = image.convert('RGB')
        image = ImageOps.grayscale(image)
        ### display image ###
        qim = ImageQt(image)
        qim = qim.scaledToWidth(256)
        pix = QPixmap.fromImage(qim)
        self.graph1.setPixmap(pix)
        self.label1.setText(os.path.split(filename)[-1])
        
        self.graph2.setPixmap(pix)
        self.label2.setText('Click convert to convert')
        
        ### save as ndarray ###
        self.w, self.h = image.size
        w = self.w if self.w%32 == 0 else self.w - self.w%32 + 32
        h = self.h if self.h%32 == 0 else self.h - self.h%32 + 32
        image = image.resize((w,h))
        T = transforms.Compose([transforms.ToTensor()])
        image = T(image).unsqueeze(0)
        self.image = image
        
    def start_thread(self):
        if self.thread != None:
            QMessageBox.warning(self, 'Error : Convert is not finished', 'Please wait until calculation is finished')
            return
        self.thread = Thread(self.convert, self.end_thread)
        self.thread.start()
        
    def end_thread(self):
        self.thread = None

    def convert(self):
        if self.image == None:
            QMessageBox.warning(self, 'Error : No iupt', 'Choose an input before convert!')
            return
        self.label2.setText('Converting')
        output = self.model(self.image)
        output = torch.squeeze(output, 0)
        T = transforms.Compose([transforms.ToPILImage()])
        output = T(output)
        output = output.resize((self.w, self.h))
        output = output.convert('RGBA')
        self.output = output.convert('RGB')
        qim = ImageQt(output)
        qim = qim.scaledToWidth(256)
        pix = QPixmap.fromImage(qim)
        self.graph2.setPixmap(pix)
        self.label2.setText('Convert done!')
        
    def saveimage(self):
        if self.output == None:
            QMessageBox.warning(self, 'Error : Output not avaliable', 'Must convert before save output')
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Save file', 
   './example/'+self.filename,"Image files (*.jpg *.gif *.png *.bmp)")
        if not filename:
            return
        self.output.save(filename)
        
class Thread(QThread):
    def __init__(self, command, exit_cmd):
        self.command = command
        self.exit_cmd = exit_cmd
        super().__init__()
        
    def run(self):
        self.command()
        self.exit_cmd()
        self.exit()
        
class App():
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.window.show() 
    
    def exec_(self):
        self.app.exec_()
        
app = App()
app.exec_()
