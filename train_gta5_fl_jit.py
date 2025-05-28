import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from datasets.gta5 import GTA5
from models.bisenet.build_bisenet import BiSeNet
from utils import compute_mIoU, fast_hist, per_class_iou, poly_lr_scheduler
import math
import random
from datasets.transforms import JointTransform
from torch.utils.data import Subset


print("\u2705 Mixed Precision Training attivo con torch.cuda.amp")

# CONFIGURAZIONE
NUM_CLASSES = 19
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 2.5e-2
IMG_SIZE = (720, 1280)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# alpha per funzione loss
ALPHA = 1

# NOMI DELLE CLASSI
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

# PERCORSI
dataset_root = "/kaggle/working/punto-3/Seg_sem_25/datasets/GTA5/GTA5"

# --- JOINT TRANSFORM INSTANCES ---
train_joint_transform = JointTransform(
    base_transform_img=input_transform,
    base_transform_mask=target_transform,
    augment=True,
    strategy='flip-jitter'  # oppure 'flip', 'jitter', o 'none'
)

val_joint_transform = JointTransform(
    base_transform_img=input_transform,
    base_transform_mask=target_transform,
    augment=False,
    strategy='none'
)

# --- DATASET CONFIGURATI CON JOINT TRANSFORM ---
train_full = GTA5(root=dataset_root, joint_transform=train_joint_transform)
val_full = GTA5(root=dataset_root, joint_transform=val_joint_transform)

# --- SPLIT INDICI ---
train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
indices = list(range(len(train_full)))

SEED = 42  # Qualsiasi intero fisso
random.seed(SEED)
random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# --- CREAZIONE DEI SOTTOSET ---
train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

# DATALOADER
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# MODELLO
model = BiSeNet(num_classes=NUM_CLASSES, context_path='resnet18').to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"🚀 Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

def compute_pixel_frequency(dataloader, num_classes):
    class_pixel_count = np.zeros(num_classes, dtype=np.int64)
    num_images = len(dataloader.dataset)
    image_class_pixels = np.zeros((num_images, num_classes), dtype=np.int64)
    img_counter = 0

    for _, targets in tqdm(dataloader):
        for j in range(targets.size(0)):
            label = np.array(targets[j])
            for class_id in range(num_classes):
                pixel_count = np.sum(label == class_id)
                class_pixel_count[class_id] += pixel_count
                image_class_pixels[img_counter, class_id] = pixel_count
            img_counter += 1

    return class_pixel_count, image_class_pixels

def median_frequency_balancing(class_pixel_count, image_class_pixels):
    frequencies = np.zeros(NUM_CLASSES)
    for class_id in range(NUM_CLASSES):
        image_pixels = image_class_pixels[:, class_id]
        image_pixels = image_pixels[image_pixels > 0]
        if len(image_pixels) > 0:
            frequencies[class_id] = np.mean(image_pixels)
        else:
            frequencies[class_id] = 0.0

    median_freq = np.median(frequencies[frequencies > 0])

    weights = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if frequencies[i] > 0:
            weights[i] = median_freq / frequencies[i]
        else:
            weights[i] = 0.0

    return weights

print("🔍 Calcolo class weights con MFB su GTA5...")
class_pixel_count, image_class_pixels = compute_pixel_frequency(train_loader, NUM_CLASSES)
weights = median_frequency_balancing(class_pixel_count, image_class_pixels)
print("✅ Class Weights (Median Frequency Balancing):")
print(weights)

weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# LOSS, OPTIMIZER, SCALER
criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=255)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
scaler = GradScaler()

# PARAMETRI POLY LR
power = 0.9
N = len(train_dataset)
steps_per_epoch = math.ceil(N / BATCH_SIZE)
max_iter = steps_per_epoch * EPOCHS

# FUNZIONE DI PERDITA DA PAPER
def Loss(output, target, criterion, cx1=None, cx2=None, alpha=1.0):
    main_loss = criterion(output, target)
    auxiliary_loss = 0
    if cx1 is not None and cx2 is not None:
        auxiliary_loss += criterion(cx1, target)
        auxiliary_loss += criterion(cx2, target)
    joint_loss = main_loss + alpha * auxiliary_loss
    return joint_loss

def train(model, train_loader, optimizer, criterion, device, num_classes, epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(1).long()

        # Aggiornamento manuale del learning rate
        current_iter = epoch * steps_per_epoch + batch_idx
        poly_lr_scheduler(optimizer, LEARNING_RATE, current_iter, max_iter=max_iter, power=power)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                cx1 = outputs[1]
                cx2 = outputs[2]
                outputs = outputs[0]
            else:
                cx1 = cx2 = None
            loss = Loss(outputs, targets, criterion, cx1, cx2, alpha=ALPHA)

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



# VALIDAZIONE
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
            cx1 = cx2 = None    # Per stare sicuri
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1).long()

            with autocast():  # ✅ Aggiunto qui
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            loss = Loss(outputs, targets, criterion, None, None, alpha=ALPHA)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            mask = (targets != 255)
            total_correct += ((preds == targets) & mask).sum().item()
            total_pixels += mask.sum().item()

            # Aggiorna confusion matrix per IoU
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

# MAIN LOOP
if __name__ == '__main__':
    print("Avvio training")
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train(model, train_loader, optimizer, criterion, DEVICE, NUM_CLASSES, epoch)
        pixel_acc, mean_iou = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES, epoch)

        torch.cuda.empty_cache()

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), 'best_model_3b_fl_jit.pth')
            print(f"Nuova best accuracy: {best_miou:.2f}% → modello salvato!")

        print(f"Epoch {epoch} completato! Best accuracy finora: {best_miou:.2f}%\n\n")

    torch.save(model.state_dict(), f'final_model_epoch_3b_fl_jit{EPOCHS}.pth')
    print(f"📦 Training finito: modello finale salvato come final_model_epoch_3b_fl_jit{EPOCHS}.pth")