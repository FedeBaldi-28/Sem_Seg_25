import random
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image

class JointTransform:
    def __init__(
        self,
        base_transform_img,
        base_transform_mask,
        augment=True,
        strategy='none'  # 'flip', 'jitter', 'flip-jitter', or 'none'
    ):
        self.base_transform_img = base_transform_img
        self.base_transform_mask = base_transform_mask
        self.augment = augment
        self.strategy = strategy.lower()
        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def random_resized_crop(self, img, mask):
        scale = (0.8, 1.0)
        ratio = (1.3, 1.8)
        
        seed = random.randint(0, 99999)
    
        random.seed(seed)
        cropper_img = T.RandomResizedCrop(size=(720, 1280), scale=scale, ratio=ratio, interpolation=Image.BILINEAR)
        img = cropper_img(img)
    
        random.seed(seed)
        cropper_mask = T.RandomResizedCrop(size=(720, 1280), scale=scale, ratio=ratio, interpolation=Image.NEAREST)
        mask = cropper_mask(mask)
    
        return img, mask

    def __call__(self, img, mask):
        if self.augment and self.strategy != 'none':
            if random.random() < 0.5:
                if self.strategy == 'flip':
                    img = F.hflip(img)
                    mask = F.hflip(mask)
                elif self.strategy == 'jitter':
                    img = self.color_jitter(img)
                elif self.strategy == 'flip-jitter':
                    img = F.hflip(img)
                    mask = F.hflip(mask)
                    img = self.color_jitter(img)
                elif self.strategy == 'blur':
                    img = F.gaussian_blur(img, kernel_size=7, sigma=(1.0, 2.5))
                elif self.strategy == 'flip-jitter-crop':
                    img = F.hflip(img)
                    mask = F.hflip(mask)
                    img = self.color_jitter(img)
                    img, mask = self.random_resized_crop(img, mask)
                elif self.strategy == 'crop':
                    img, mask = self.random_resized_crop(img, mask)
                else:
                    raise ValueError(f"Unknown augmentation strategy: {self.strategy}")

        img = self.base_transform_img(img)
        mask = self.base_transform_mask(mask)
        return img, mask
