%%writefile /kaggle/working/dataset1/Sem_Seg_25/train_2a_no_weights.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.cityscapes import Cityscapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import time
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from utils import fast_hist, per_class_iou
import math
from torch.cuda import amp


#################### CONFIGURATION ####################
NUM_CLASSES = 19
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001
IMG_SIZE = (512, 1024)
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


#################### DATASET ####################
train_dataset = Cityscapes(
    root='/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='train',
    transform=input_transform,
    target_transform=target_transform
)

val_dataset = Cityscapes(
    root='/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='val',
    transform=input_transform,
    target_transform=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


#################### MODEL ####################
model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path='/kaggle/input/punto-3/Seg_sem_25/Seg_sem_25/deeplab_resnet_pretrained_imagenet.pth')

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)


#################### LOSS & OPTIMIZER ####################
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
scaler = amp.GradScaler()


#################### TRAINING ####################
def train(model, train_loader, optimizer, criterion, device, num_classes, epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_batches = len(train_loader)
    start_time = time.time()

    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(1).long()

        optimizer.zero_grad()

        with amp.autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            mask = (targets != 255)
            total_correct += ((preds == targets) & mask).sum().item()
            total_pixels += mask.sum().item()

    avg_loss = running_loss / total_batches
    pixel_acc = total_correct / total_pixels

    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch}] | [Train] Loss: {avg_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | Time: {epoch_time:.1f}s")

    return avg_loss, pixel_acc


#################### VALIDATION ####################
def validate(model, val_loader, criterion, device, num_classes, epoch):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_batches = len(val_loader)
    start_time = time.time()
    
    hist = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1).long()
            
            with amp.autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            loss = criterion(outputs, targets)
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

    print("IoU per classe (val):")
    for idx, iou_cls in enumerate(ious_per_class):
        print(f"  {CLASS_NAMES[idx]}: {iou_cls:.2f}%")

    return pixel_acc, mIoU


#################### MAIN ####################
if __name__ == '__main__':
    print("start of training")
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"epoch {epoch}/{EPOCHS}")
        train(model, train_loader, optimizer, criterion, DEVICE, NUM_CLASSES, epoch)
        pixel_acc, mean_iou = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES, epoch)

        torch.cuda.empty_cache()

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), 'best_model_2a_no_weight.pth')
            print(f"New best accuracy: {best_miou:.2f}% → model saved!")

        print(f"Epoch {epoch} completed! Best accuracy so far: {best_miou:.2f}%\n\n")

    torch.save(model.state_dict(), f'final_model_epoch_2a_no_weight{EPOCHS}.pth')
    print(f"Training finished: final model saved as final_model_epoch_2a_no_weight{EPOCHS}.pth")
