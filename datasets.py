import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def infiniteloop(dataloader):
    while True:
        for *x, y in iter(dataloader):
            yield *x, y

class LF5x5_Dataset(Dataset):
    def __init__(self, root, size=None):

        transforms_list = []
        if size is not None:
            transforms_list.append(transforms.Resize(size))
        transforms_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transforms_list)

        self.imgs, self.masks = [], []

        training_names = []
        for name in sorted(glob.glob(f"{root}/*.png")):
            r = int(name.split('/')[-1].split('_')[1])
            c = int(name.split('/')[-1].split('_')[2])
            if r % 4 > 0 or c % 4 > 0:
                continue
            training_names.append(name.split("/")[-1])
            img = np.asarray(Image.open(name).convert('RGB')) / 255.
            self.imgs.append(self.transform(Image.fromarray(np.uint8(255 * img))))

        self.hw = self.imgs[0].shape[1:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return self.imgs[item], item

class LLFF_Dataset(Dataset):
    def __init__(self, root, size=None):

        transforms_list = []
        if size is not None:
            transforms_list.append(transforms.Resize(size))
        transforms_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transforms_list)

        self.imgs, self.masks = [], []
        training_names = []

        for i, name in enumerate(sorted(glob.glob(f"{root}/*.png"))):
            if i >= 30:
                break
            img = np.asarray(Image.open(name).convert('RGB')) / 255.
            self.imgs.append(self.transform(Image.fromarray(np.uint8(255 * img))))
            training_names.append(name.split("/")[-1])
        self.hw = self.imgs[0].shape[1:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return self.imgs[item], item
