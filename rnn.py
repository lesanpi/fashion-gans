import torch

def block(n_in, n_out):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, n_out),
        torch.nn.ReLU(inplace= True)
    )

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = block(input_size, 150)
        self.fc2 = block(150, 100)
        self.fc3 = torch.nn.Linear(100, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 100
        self.inp = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 7*7*128),
            torch.nn.BatchNorm1d(7*7*128)
        )
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, stride = 2, padding= 1, bias = False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 1, 4, stride = 2, padding= 1, bias= False),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        x = self.inp(x)
        x = x.view(-1, 128, 7, 7)
        x = self.main(x)
        x = x.view(x.size(0), 28*28)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 4, stride = 2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, 4, stride = 2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(128 * 7* 7, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # esperamos vectores a la entrada 28*28
        x = x.view(x.size(0), 1, 28, 28)
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x