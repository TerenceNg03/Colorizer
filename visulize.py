import matplotlib.pyplot as plt
import torch

def Datavisulize(n, dataset, transform = None, cmap = 'viridis', datapos = 0, title = None):
    f, imgs = plt.subplots(n,n)
    if title:
        f.suptitle(title)
    f.set_figheight(8)
    f.set_figwidth(8)
    for i in range(n):
        for j in range(n):
            img = dataset.__getitem__(i*n+j)[datapos]
            if transform:
                img = transform(img)
            img = img.permute((1,2,0)).detach()
            if img.shape[2] == 1:
                img = img.reshape((img.shape[0],img.shape[1]))
            imgs[i,j].imshow(img, aspect='auto', cmap = cmap)
            imgs[i,j].axis('off')
            
def compare(n, dataset, model):
    f, imgs = plt.subplots(n,3)
    f.set_figheight(8)
    f.set_figwidth(8)
    imgs[0,0].set_title('Grayscale')
    imgs[0,1].set_title('Original')
    imgs[0,2].set_title('Output')

    for i in range(n):
        gray, orig = dataset.__getitem__(i)
        img = model(gray.unsqueeze(0)).squeeze(0).permute((1,2,0)).detach()
        imgs[i,0].imshow(gray.squeeze(0), aspect='auto', cmap = 'gray')
        imgs[i,1].imshow(orig.permute((1,2,0)), aspect='auto')
        imgs[i,2].imshow(img, aspect='auto')

        imgs[i,0].axis('off')
        imgs[i,1].axis('off')
        imgs[i,2].axis('off')