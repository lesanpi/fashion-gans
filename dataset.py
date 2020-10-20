import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, trainset, normalize = False):
        self.imgs = torch.tensor([np.array(i[0]).flatten() / 255. for i in trainset], dtype = torch.float, device = device)
        if normalize:
            self.imgs = self.imgs * 2. - 1.
        self.labels = torch.tensor([i[1] for i in trainset], device = device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        return self.imgs[ix], self.labels[ix]