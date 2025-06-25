import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GTA5(Dataset):
    def __init__(self, root, transform=None, target_transform=None, joint_transform=None):
        """
        Args:
            root (str): cartella principale con sottocartelle 'images' e 'converted'
            transform (callable): trasformazione da applicare all'immagine
            target_transform (callable): trasformazione da applicare alla maschera
            joint_transform (callable): trasformazione congiunta immagine+maschera
        """
        self.images_dir = os.path.join(root, 'images')
        self.converted_masks_dir = os.path.join(root, 'converted')
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        # Lista dei nomi file immagine .png (es. 00001.png)
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])

        # Percorsi completi
        self.image_paths = [os.path.join(self.images_dir, fname) for fname in self.images]
        self.mask_paths = [
            os.path.join(self.converted_masks_dir, fname.replace('.png', '_converted.png'))
            for fname in self.images
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica immagine
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # Carica maschera già convertita (come immagine a singolo canale)
        label_mask = Image.open(self.mask_paths[idx])

        # 👇 Prima applica joint_transform (se definita)
        if self.joint_transform is not None:
            img, label_mask = self.joint_transform(img, label_mask)
        else:
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                label_mask = self.target_transform(label_mask)

        # Maschera → tensor long, rimuovo canale (C=1)
        label_mask = T.PILToTensor()(label_mask).long().squeeze(0)

        return img, label_mask

class GTA5Wrapper(Dataset):
    def __init__(self, subset, joint_transform=None):
        self.subset = subset
        self.joint_transform = joint_transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]     # prendi immagine e label dal subset
        if self.joint_transform is not None:
            img, label = self.joint_transform(img, label)  # applica trasformazione
        return img, label

    def __len__(self):
        return len(self.subset)
