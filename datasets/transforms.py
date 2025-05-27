import random
import torchvision.transforms.functional as F
import torchvision.transforms as T

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
            brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1
        )

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
                    img = F.gaussian_blur(img, kernel_size=5, sigma=(0.1, 2.0))
                elif self.strategy == 'flip-jitter-blur':
                    img = F.hflip(img)
                    mask = F.hflip(mask)
                    img = self.color_jitter(img)
                    img = F.gaussian_blur(img, kernel_size=5, sigma=(0.1, 2.0))
                elif self.strategy == 'crop':
                    i, j, h, w = T.RandomCrop.get_params(img, output_size=(720, 960))
                    img = F.crop(img, i, j, h, w)
                    mask = F.crop(mask, i, j, h, w)
                elif self.strategy == 'crop+jitter':
                    i, j, h, w = T.RandomCrop.get_params(img, output_size=(720, 960))
                    img = F.crop(img, i, j, h, w)
                    mask = F.crop(mask, i, j, h, w)
                    img = self.color_jitter(img)
                else:
                    raise ValueError(f"Unknown augmentation strategy: {self.strategy}")

        img = self.base_transform_img(img)
        mask = self.base_transform_mask(mask)
        return img, mask