import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GTA5(Dataset):
    def __init__(self, root, transform=None, target_transform=None, joint_transform=None):
        self.images_dir = os.path.join(root, 'images')
        self.converted_masks_dir = os.path.join(root, 'converted')
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])

        self.image_paths = [os.path.join(self.images_dir, fname) for fname in self.images]
        self.mask_paths = [
            os.path.join(self.converted_masks_dir, fname.replace('.png', '_converted.png'))
            for fname in self.images
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        label_mask = Image.open(self.mask_paths[idx])

        if self.joint_transform is not None:
            img, label_mask = self.joint_transform(img, label_mask)
        else:
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                label_mask = self.target_transform(label_mask)

        label_mask = T.PILToTensor()(label_mask).long().squeeze(0)

        return img, label_mask
