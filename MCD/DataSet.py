from torch import tensor
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
import torch
class DataSet64x64(Dataset):
    def __init__(self,data, target) -> None:
        self.data  = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) :
        return torch.tensor(self.data[index].reshape((1,64,64)),dtype=torch.float32) , self.target[index]

class DataSet16x16(Dataset):
    def __init__(self,data, target) -> None:
        self.data  = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) :
        return torch.tensor(self.data[index].reshape((1,16,16)),dtype=torch.float32) , self.target[index]

class DataSet1d64x64(Dataset):
    def __init__(self,data, target) -> None:
        self.data  = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) :
        return torch.tensor(self.data[index].reshape((1,-1)),dtype=torch.float32) , self.target[index]
