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
from utils import compute_mIoU
from utils import fast_hist, per_class_iou


print("\u2705 Mixed Precision Training attivo con torch.cuda.amp")

# CONFIGURAZIONE
NUM_CLASSES = 19
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.0001
IMG_SIZE = (512, 1024)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NOMI DELLE CLASSI CITYSCAPES
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]

# TRANSFORM
input_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
])

# DATASET
train_dataset = Cityscapes(
    root='/content/drive/MyDrive/MLDL2024_project/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='train',
    transform=input_transform,
    target_transform=target_transform
)

val_dataset = Cityscapes(
    root='/content/drive/MyDrive/MLDL2024_project/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='val',
    transform=input_transform,
    target_transform=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# FUNZIONE PER CALCOLARE I PESI

def compute_class_weights(dataloader, num_classes):
    class_counts = np.zeros(num_classes)
    print("\n→ Calcolo dei pesi per classi...")
    for _, masks in tqdm(dataloader):
        for mask in masks:
            mask_np = mask.numpy()
            for c in range(num_classes):
                class_counts[c] += np.sum(mask_np == c)

    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print("Pesi per classe:", weights)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# MODELLO
model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=True, pretrain_model_path='/content/drive/MyDrive/MLDL2024_project/deeplab_resnet_pretrained_imagenet.pth')

if torch.cuda.device_count() > 1:
    print(f"🚀 Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

# LOSS & OPTIMIZER
class_weights = compute_class_weights(train_loader, NUM_CLASSES)
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
optimizer = optim.SGD(model.optim_parameters(lr=LEARNING_RATE), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
scaler = GradScaler()  # Mixed Precision

# METRICHE
def compute_metrics(preds, targets, num_classes):
    mask = targets != 255
    preds = preds[mask]
    targets = targets[mask]

    ious = [float('nan')] * num_classes
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious[cls] = float('nan')
        else:
            ious[cls] = intersection / union

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    correct = (preds == targets).sum().item()
    total = targets.numel()
    pixel_acc = correct / total

    return pixel_acc, mean_iou, ious

# TRAINING
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

        with autocast():
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

    # Calcolo mIoU su tutto il train set
    mIoU, _ = compute_mIoU(model, train_loader, num_classes, device)
    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch}] | [Train] Loss: {avg_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | mIoU: {mIoU:.2f}% | Time: {epoch_time:.1f}s")

    return avg_loss, pixel_acc, mIoU


# VALIDAZIONE
def validate(model, val_loader, criterion, device, num_classes, epoch):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_batches = len(val_loader)
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1).long()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            mask = (targets != 255)
            total_correct += ((preds == targets) & mask).sum().item()
            total_pixels += mask.sum().item()

    avg_loss = val_loss / total_batches
    pixel_acc = total_correct / total_pixels

    mIoU, ious_per_class = compute_mIoU(model, val_loader, num_classes, device)
    epoch_time = time.time() - start_time

    print(f"[Epoch {epoch}] | [Val] Loss: {avg_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | mIoU: {mIoU:.2f}% | Time: {epoch_time:.1f}s")

    print("IoU per classe (val):")
    for idx, iou_cls in enumerate(ious_per_class):
        print(f"  {CLASS_NAMES[idx]}: {iou_cls:.2f}%")

    return pixel_acc, mIoU


# MAIN LOOP
if __name__ == '__main__':
    print("avvio training")
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"epoch {epoch}/{EPOCHS}")
        train(model, train_loader, optimizer, criterion, DEVICE, NUM_CLASSES, epoch)
        pixel_acc, mean_iou = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES, epoch)

        torch.cuda.empty_cache()

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Nuova best accuracy: {best_miou:.2f}% → modello salvato!")

        print(f"Epoch {epoch} completato! Best accuracy finora: {best_miou:.2f}%\n\n")

    torch.save(model.state_dict(), f'final_model_epoch_{EPOCHS}.pth')
    print(f"📦 Training finito: modello finale salvato come final_model_epoch_{EPOCHS}.pth")
