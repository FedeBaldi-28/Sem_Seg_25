import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from datasets.gta5 import GTA5
from datasets.cityscapes import CityscapesTarget
from models.bisenet.build_bisenet import BiSeNet
from utils import fast_hist, per_class_iou, poly_lr_scheduler, compute_pixel_frequency, median_frequency_balancing
import math
import random
from datasets.transforms import JointTransform
from torch.utils.data import Subset
from torch.cuda import amp
from models.discriminator import FCDiscriminator
from itertools import cycle
import torch.nn.functional as F
from losses import FocalLoss, DiceLoss, HybridLoss


#################### CONFIGURATION ####################
CONTEXT_PATH = 'resnet18'
ALPHA = 1
NUM_CLASSES = 19
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 2.5e-2
IMG_SIZE = (720, 1280)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]


#################### TRANSFORM ####################
input_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
])

transform_cityscapes = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#################### DATASET ####################
dataset_root = "/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/GTA5/GTA5"

train_joint_transform = JointTransform(input_transform, target_transform, augment=True, strategy='jitter')
val_joint_transform = JointTransform(input_transform, target_transform, augment=False, strategy='none')

train_full = GTA5(root=dataset_root, joint_transform=train_joint_transform)
val_full = GTA5(root=dataset_root, joint_transform=val_joint_transform)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
indices = list(range(len(train_full)))

random.seed(42)
random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

target_dataset = CityscapesTarget(
    root='/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='train',
    transform=transform_cityscapes
)

train_loader_gta = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader_gta = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
cityscapes_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


#################### COMPUTE WEIGHTS ####################
class_pixel_count, image_class_pixels = compute_pixel_frequency(train_loader_gta, NUM_CLASSES)
weights = median_frequency_balancing(class_pixel_count, image_class_pixels, NUM_CLASSES)
print("Class Weights (Median Frequency Balancing):")
print(weights)

normalized_weights = weights / weights.sum() * len(weights)
print(normalized_weights)
weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(DEVICE)


#################### MODEL & DISCRIMINATOR ####################
model = BiSeNet(num_classes=NUM_CLASSES, context_path='resnet18').to(DEVICE)
D = FCDiscriminator(num_classes=NUM_CLASSES).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)
    D = nn.DataParallel(D)

model = model.to(DEVICE)
D = D.to(DEVICE)


#################### LOSS & OPTIMIZER ####################
criterion = HybridLoss(alpha=0.7, class_weights=weights_tensor)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))
scaler = GradScaler()

power = 0.9
steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
max_iter = steps_per_epoch * EPOCHS

bce_loss = nn.BCEWithLogitsLoss()
lambda_adv = 0.001


#################### PAPER LOSS ####################
def SegLoss(output, target, criterion, cx1=None, cx2=None, alpha=1.0):
    output = output.float()
    main_loss = criterion(output, target)
    auxiliary_loss = 0
    if cx1 is not None and cx2 is not None:
        cx1 = cx1.float()
        cx2 = cx2.float()
        auxiliary_loss += criterion(cx1, target)
        auxiliary_loss += criterion(cx2, target)
    return main_loss + alpha * auxiliary_loss


#################### TRAINING ####################
def train(model, train_loader, optimizer, criterion, device, num_classes, epoch):
    model.train()
    D.train()
    running_total_loss = running_seg_loss = running_adv_loss = 0.0
    total_correct = total_pixels = 0
    total_batches = len(train_loader)
    start_time = time.time()
    iter_t = cycle(cityscapes_loader)

    for batch_idx, (imgs_s, labels_s) in enumerate(tqdm(train_loader)):
        imgs_t = next(iter_t)
        source_inputs, source_targets = imgs_s.to(DEVICE), labels_s.to(DEVICE).long()
        target_inputs = imgs_t.to(DEVICE)

        current_iter = (epoch - 1) * steps_per_epoch + batch_idx
        poly_lr_scheduler(optimizer, LEARNING_RATE, current_iter, max_iter=max_iter, power=0.9)
        poly_lr_scheduler(optimizer_D, 1e-4, current_iter, max_iter=max_iter, power=0.9)

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        for param in D.parameters(): param.requires_grad = False

        with autocast():
            source_outputs = model(source_inputs)
            if isinstance(source_outputs, tuple):
                src_cx1, src_cx2 = source_outputs[1], source_outputs[2]
                source_outputs = source_outputs[0]


            seg_loss = SegLoss(source_outputs, source_targets, criterion, src_cx1, src_cx2, alpha=ALPHA)
            target_outputs = model(target_inputs)
            if isinstance(target_outputs, tuple): target_outputs = target_outputs[0]

            D_out = D(F.softmax(target_outputs.float(), dim=1))
            adv_loss = lambda_adv * bce_loss(D_out, torch.full(D_out.size(), 0.0, device=device))

            total_loss = seg_loss + adv_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Attention: total_loss contains NaN or Inf! Stopping the training.")
                exit(1)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for param in D.parameters(): param.requires_grad = True

        optimizer_D.zero_grad()

        D_out_source = D(F.softmax(source_outputs.detach().float(), dim=1))
        D_loss_source = bce_loss(D_out_source, torch.full(D_out_source.size(), 1.0, device=device))

        D_out_target = D(F.softmax(target_outputs.detach().float(), dim=1))
        D_loss_target = bce_loss(D_out_target, torch.full(D_out_target.size(), 0.0, device=device))

        D_loss = 0.5 * (D_loss_source + D_loss_target)
        scaler.scale(D_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        running_total_loss += total_loss.item()
        running_seg_loss += seg_loss.item()
        running_adv_loss += adv_loss.item()

        with torch.no_grad():
            preds = torch.argmax(source_outputs, dim=1)
            mask = (source_targets != 255)
            total_correct += ((preds == source_targets) & mask).sum().item()
            total_pixels += mask.sum().item()

    avg_total_loss = running_total_loss / total_batches
    avg_seg_loss = running_seg_loss / total_batches
    avg_adv_loss = running_adv_loss / total_batches
    pixel_acc = total_correct / total_pixels
    epoch_time = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']

    print(f"[Epoch {epoch}] | Total Loss: {avg_total_loss:.4f} | Seg Loss: {avg_seg_loss:.4f} | Adv Loss: {avg_adv_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
    return avg_total_loss, pixel_acc


#################### VALIDATION ####################
def validate(model, val_loader, criterion, device, num_classes, epoch):
    model.eval()
    val_loss = total_correct = total_pixels = 0
    total_batches = len(val_loader)
    start_time = time.time()
    hist = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1).long()

            with autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = SegLoss(outputs, targets, criterion, None, None, alpha=ALPHA)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            mask = (targets != 255)

            total_correct += ((preds == targets) & mask).sum().item()
            total_pixels += mask.sum().item()

            for lp, pp in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                hist += fast_hist(lp.flatten(), pp.flatten(), num_classes)

    avg_loss = val_loss / total_batches
    pixel_acc = total_correct / total_pixels
    ious = per_class_iou(hist)
    mIoU = np.nanmean(ious) * 100
    ious_per_class = ious * 100
    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch}] | [Val] Loss: {avg_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | mIoU: {mIoU:.2f}% | Time: {epoch_time:.1f}s")
    for idx, iou_cls in enumerate(ious_per_class):
        print(f"  {CLASS_NAMES[idx]}: {iou_cls:.2f}%")

    return pixel_acc, mIoU
    

#################### MAIN ####################
if __name__ == '__main__':
    print("start of training")
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        _, _ = train(model, train_loader_gta, optimizer, criterion, DEVICE, NUM_CLASSES, epoch)
        pixel_acc, mean_iou = validate(model, val_loader_gta, criterion, DEVICE, NUM_CLASSES, epoch)

        torch.cuda.empty_cache()

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), 'best_model_5_combo_07_focal_03_dice.pth')
            print(f"New best mIoU: {best_miou:.2f}% → model saved!")

        print(f"Epoch {epoch} completed! Best mIoU so far: {best_miou:.2f}%\n\n")

    torch.save(model.state_dict(), f'final_model_epoch_5_combo_07_focal_03_dice{EPOCHS}.pth')
    print(f"Training completed: final model saved as final_model_epoch_5_combo_07_focal_03_dice{EPOCHS}.pth")
