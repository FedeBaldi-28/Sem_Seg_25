import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class Cityscapes(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.image_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'gtFine', split)
        self.transform = transform
        self.target_transform = target_transform

        self.images = []
        self.masks = []

        for city in os.listdir(self.image_dir):
            img_folder = os.path.join(self.image_dir, city)
            mask_folder = os.path.join(self.mask_dir, city)
            for file_name in os.listdir(img_folder):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_folder, file_name)
                    mask_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    mask_path = os.path.join(mask_folder, mask_name)
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])  # già in scala di grigi

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Converti la maschera in tensor long [H, W] (con valori di classe interi)
        mask = T.PILToTensor()(mask).long().squeeze(0)

        return img, mask
