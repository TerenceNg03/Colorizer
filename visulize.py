import matplotlib.pyplot as plt
import torch

def visulize(n, dataset, transform = None, cmap = 'viridis', datapos = 0, title = None):
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