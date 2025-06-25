import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        if class_weights is not None:
            self.class_weights = class_weights.view(1, -1)
        else:
            self.class_weights = None

    def forward(self, input, target):
        # input: [N, C, H, W]
        # target: [N, H, W]
        input, target = self.flatten(input, target, self.ignore_index)
        log_probs = torch.log_softmax(input, dim=1)
        probs = torch.exp(log_probs)                 

        focal_factor = (1 - probs) ** self.gamma      

        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)     
        target_focal_factor = focal_factor.gather(1, target.unsqueeze(1)).squeeze(1) 

        ce_loss = -target_log_probs

        if self.class_weights is not None:
            weights = self.class_weights[0].gather(0, target)  # [num_pixels]
            loss = weights * target_focal_factor * ce_loss
        else:
            loss = target_focal_factor * ce_loss

        return loss.mean()

    def flatten(self, input, target, ignore_index):
        num_classes = input.size(1)
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)
        mask = target != ignore_index
        return input[mask], target[mask]

class DiceLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        if class_weights is not None:
            self.class_weights = class_weights.view(1, -1)
        else:
            self.class_weights = None

    def forward(self, input, target):
        input, target = self.flatten(input, target, self.ignore_index)
        input = F.softmax(input, dim=1) 

        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).float() 
        
        intersection = (input * target_one_hot).sum(dim=0)
        union = input.sum(dim=0) + target_one_hot.sum(dim=0)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss_per_class = 1 - dice  

        dice_loss_per_pixel = dice_loss_per_class.gather(0, target)

        if self.class_weights is not None:
            weights = self.class_weights[0].gather(0, target)  
            weighted_loss = dice_loss_per_pixel * weights
            return weighted_loss.mean()
        else:
            return dice_loss_per_pixel.mean()

    def flatten(self, input, target, ignore_index):
        num_classes = input.size(1)
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1) 
        mask = target != ignore_index
        return input[mask], target[mask]


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, class_weights=None, ignore_index=255, gamma=1.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.focal = FocalLoss(class_weights=None, ignore_index=ignore_index, gamma=gamma)
        self.dice = DiceLoss(class_weights=class_weights, ignore_index=ignore_index, smooth=smooth)

    def forward(self, input, target):
        loss_focal = self.focal(input, target)
        loss_dice = self.dice(input, target)
        return self.alpha * loss_focal + (1 - self.alpha) * loss_dice
