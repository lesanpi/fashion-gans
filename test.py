import torch
import torchvision
import numpy as np
from rnn import *
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
n_in, n_out = 30, 28 * 28

# Generador
#generador = MLP(n_in, n_out)
generador = Generator()
generador.load_state_dict(torch.load("./models/ConvNets/generador_state_dict.pt", map_location=torch.device(device)))

#Discriminador
#discriminador = MLP(28*28, 1)
discriminador = Discriminator()
discriminador.load_state_dict(torch.load("./models/ConvNets/discriminador_state_dict.pt", map_location=torch.device(device)))

#discriminador.eval()
generador.eval()
with torch.no_grad():
    noise = torch.randn((10, generador.input_size)).to(device)
    generated_images = generador(noise)
    fig, axs = plt.subplots(2, 5, figsize = (15, 5))
    i = 0
    for ax in axs:
        for _ax in ax:
            img = generated_images[i].view(28,28).cpu()
            _ax.imshow(img)
            i += 1

    plt.show()