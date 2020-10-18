import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True)

classes = ("t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle-boot")

train = Dataset(trainset)
dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

imgs, labels = next(iter(dataloader))
# (32,784) es el shape. 
# 784 ya que las imagenes son de 28 x 28.

r, c = 3, 5
plt.figure(figsize=(c*3, r*3))
for row in range(r):
    for col in range(c):
        index = c*row + col
        plt.subplot(r, c, index + 1)
        ix = random.randint(0, len(train)-1)
        img, label = train[ix]
        plt.imshow(img.reshape(28,28).cpu())
        plt.axis('off')
        plt.title(classes[label.item()])
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()