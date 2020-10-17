import torch
import torchvision
import numpy as np
from rnn import *
import matplotlib.pyplot as plt
from tqdm import tqdm

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
generador = MLP(n_in, n_out)
# Recibe 30 hyperparametros y genera una imagen 28x28

# Le pasamos 64 "imagenes"o hyperparametros.
output = generador(torch.randn(64, 30))
# Agarramos la primera imagen que genero
if False:
    plt.imshow(output[0].reshape(28, 28).detach().numpy())
    plt.show()
# Nos genero ruido

# Creamos el discriminador
discriminador = MLP(28*28, 1)
# Recibe una imagen 28x28 y devuelve un valor (Falso o True)

# Funcion de entrenamiento
def fit(g, d, dataloader, epochs = 30, crit = None):
    g.to(device)
    d.to(device)
    g_optimizer = torch.optim.Adam(g.parameters(), lr = 3e-4)
    d_optimizer = torch.optim.Adam(d.parameters(), lr = 3e-4)

    crit = torch.nn.BCEWithLogitsLoss() if crit == None else crit
    g_loss, d_loss = [], []
    
    #mb = master_bar(range(1, epochs + 1))
    hist = {'g_loss': [], 'd_loss': []}

    for epoch in range(1, epochs + 1):
        bar = tqdm(dataloader)
        for X, y in bar:
            # Entrenamos al discriminador
            g.eval()
            d.train()

            # Generamos imagenes falsas
            noise = torch.randn((X.size(0), g.input_size)).to(device)
            generated_images = g(noise)
            
            # Input del discriminador (Imagenes generadas, Imagenes reales)
            d_input = torch.cat([generated_images, X.view(X.size(0), -1)])
            # Objetivo del discriminador
            d_gt = torch.cat([torch.zeros(X.size(0)), torch.ones(X.size(0))]).view(-1, 1).to(device)
            # Las generadas tienen que resultar en 0 la etiqueta y las reales 1

            # Optimizacion
            d_optimizer.zero_grad()
            d_output = d(d_input)
            d_l = crit(d_output, d_gt)
            d_l.backward()
            d_optimizer.step()
            d_loss.append(d_l.item())

            # Entrenamos al generador
            g.train()
            d.eval()
            
            # Generamos un batch de imagenes falsas
            noise = torch.randn((X.size(0), g.input_size)).to(device)
            generated_images = g(noise)
            # Salidas del discriminador
            d_output = d(generated_images)

            # Gorund truth.
            g_gt = torch.ones(X.size(0)).view(-1, 1).to(device)

            #Optimizacion
            g_optimizer.zero_grad()
            g_l = crit(d_output, g_gt)
            g_l.backward()
            g_optimizer.step()
            g_loss.append(g_l.item())

            bar.set_description(f'g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}')

        print(f'Epoch {epoch}/{epochs} g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}')
        hist['g_loss'].append(np.mean(g_loss))
        hist['d_loss'].append(np.mean(d_loss))
    return hist

hist = fit(generador, discriminador, dataloader)

torch.save(discriminador.state_dict(), "./models/MLP/discriminador_state_dict.pt")
torch.save(generador.state_dict(), "./models/MLP/generador_state_dict.pt")