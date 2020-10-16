import torch
import torchvision
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True)

classes = ("t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle-boot")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, trainset):
        self.imgs = torch.tensor([np.array(i[0]).flatten() / 255. for i in trainset], dtype = torch.float, device = device)
        self.labels = torch.tensor([i[1] for i in trainset], device = device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        return self.imgs[ix], self.labels[ix]

train = Dataset(trainset)
dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

imgs, labels = next(iter(dataloader))
# (32,784) es el shape. 
# 784 ya que las imagenes son de 28 x 28.

