import torch
import torchvision
import numpy as np
from rnn import *
import matplotlib.pyplot as plt

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

n_in, n_out = 30, 28 * 28
# Creamos el generador
generator = MLP(n_in, n_out)
# Recibe 30 hyperparametros y genera una imagen 28x28

# Le pasamos 64 "imagenes"o hyperparametros.
output = generator(torch.randn(64, 30))
# Agarramos la primera imagen que genero
plt.imshow(output[0].reshape(28, 28).detach().numpy())
plt.show()
# Nos genero ruido

# Creamos el discriminador
discriminator = MLP(28*28, 1)
# Recibe una imagen 28x28 y devuelve un valor (Falso o True)