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