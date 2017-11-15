import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size, data_type):
        super(Dataset, self).__init__()
        self.root = root
        self.paths = glob(os.path.join(self.root, f"{data_type}/*"))
        self.transform = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.paths)


def imageLoader(data_path, batch_size, image_size, workers=2, shuffle=True):
    a_dataset = Dataset(data_path, image_size, "A")
    b_dataset = Dataset(data_path, image_size, "B")
    a_dataloader = DataLoader(dataset=a_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    b_dataloader = DataLoader(dataset=b_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    return a_dataloader, b_dataloader
