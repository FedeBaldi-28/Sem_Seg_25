import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from .labels import GTA5Labels_TaskCV2017


class GTA5(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        """
        Args:
            root (str): cartella principale con sottocartelle 'images' e 'labels'
            transform (callable): trasformazione da applicare all'immagine
            target_transform (callable): trasformazione da applicare alla maschera
        """
        self.images_dir = os.path.join(root, 'images')
        self.masks_dir = os.path.join(root, 'labels')
        self.converted_masks_dir = os.path.join(root, 'converted')
        self.transform = transform
        self.target_transform = target_transform

        # Crea cartella 'converted' se non esiste
        os.makedirs(self.converted_masks_dir, exist_ok=True)

        # Lista file .png
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        self.labels = GTA5Labels_TaskCV2017()
        self.rgb_to_id = {label.color: label.ID for label in self.labels.list_}

        # Percorsi completi immagini e maschere
        self.image_paths = [os.path.join(self.images_dir, fname) for fname in self.images]
        self.mask_paths = [os.path.join(self.masks_dir, fname) for fname in self.images]

    def __len__(self):
        return len(self.image_paths)

    def rgb_to_label_id(self, rgb_mask):
        """Converte maschera RGB in maschera con ID (0–18), 255 per pixel sconosciuti"""
        h, w, _ = rgb_mask.shape
        mask_id = np.full((h, w), 255, dtype=np.uint8)  # 255 = ignoto

        for color, label_id in self.rgb_to_id.items():
            matches = np.all(rgb_mask == color, axis=-1)
            mask_id[matches] = label_id

        return mask_id

    def __getitem__(self, idx):
        # Carica immagine
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # Carica e converte maschera
        rgb_mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
        label_mask = self.rgb_to_label_id(rgb_mask)

        # Salva maschera convertita (solo una volta)
        mask_name = os.path.basename(self.image_paths[idx]).replace('.png', '_converted.png')
        save_path = os.path.join(self.converted_masks_dir, mask_name)
        if not os.path.exists(save_path):
            Image.fromarray(label_mask).save(save_path)

        # Converto maschera in PIL Image per le trasformazioni
        label_mask = Image.fromarray(label_mask)

        # Applica trasformazioni
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label_mask = self.target_transform(label_mask)

        # Maschera → tensor long, rimuovo canale (C=1)
        label_mask = T.PILToTensor()(label_mask).long().squeeze(0)

        return img, label_mask
