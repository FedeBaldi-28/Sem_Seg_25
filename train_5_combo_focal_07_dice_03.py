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
from datasets.cityscapes import CityscapesTarget
from models.bisenet.build_bisenet import BiSeNet
from utils import compute_mIoU, fast_hist, per_class_iou, poly_lr_scheduler
import math
import random
from datasets.transforms import JointTransform
from torch.utils.data import Subset
from torch.cuda import amp
from train_gta5 import FCDiscriminator
from itertools import cycle
import torch.nn.functional as F

print("\u2705 Mixed Precision Training attivo con torch.cuda.amp")

# ==============================
# ✅ FocalLoss personalizzata
# ==============================

class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        if class_weights is not None:
            self.class_weights = class_weights.view(1, -1)  # [1, num_classes]
        else:
            self.class_weights = None

    def forward(self, input, target):
        # input: [N, C, H, W]
        # target: [N, H, W]
        input, target = self.flatten(input, target, self.ignore_index)
        log_probs = torch.log_softmax(input, dim=1)  # [num_pixels, C]
        probs = torch.exp(log_probs)                  # [num_pixels, C]

        focal_factor = (1 - probs) ** self.gamma      # [num_pixels, C]

        # Gather the log_probs and focal_factor for the target class of each pixel
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)      # [num_pixels]
        target_focal_factor = focal_factor.gather(1, target.unsqueeze(1)).squeeze(1)  # [num_pixels]

        ce_loss = -target_log_probs  # cross entropy per pixel

        if self.class_weights is not None:
            # Apply class weights per pixel
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
            self.class_weights = class_weights.view(1, -1)  # [1, num_classes]
        else:
            self.class_weights = None

    def forward(self, input, target):
        input, target = self.flatten(input, target, self.ignore_index)
        input = F.softmax(input, dim=1)  # [num_pixels, C]

        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).float()  # [num_pixels, C]

        intersection = (input * target_one_hot).sum(dim=0)
        union = input.sum(dim=0) + target_one_hot.sum(dim=0)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # [num_classes]
        dice_loss_per_class = 1 - dice  # [num_classes]

        # Calcola la loss per pixel (loss della classe target per ciascun pixel)
        dice_loss_per_pixel = dice_loss_per_class.gather(0, target)  # [num_pixels]

        if self.class_weights is not None:
            weights = self.class_weights[0].gather(0, target)  # [num_pixels]
            weighted_loss = dice_loss_per_pixel * weights
            return weighted_loss.mean()
        else:
            return dice_loss_per_pixel.mean()

    def flatten(self, input, target, ignore_index):
        num_classes = input.size(1)
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # [num_pixels, C]
        target = target.view(-1)  # [num_pixels]
        mask = target != ignore_index
        return input[mask], target[mask]


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, class_weights=None, ignore_index=255, gamma=1.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.focal = FocalLoss(class_weights=None, ignore_index=ignore_index, gamma=gamma)
        # Per DiceLoss qui PASSO class_weights
        self.dice = DiceLoss(class_weights=class_weights, ignore_index=ignore_index, smooth=smooth)

    def forward(self, input, target):
        loss_focal = self.focal(input, target)
        loss_dice = self.dice(input, target)
        return self.alpha * loss_focal + (1 - self.alpha) * loss_dice

# ==============================
# Configurazione
# ==============================

NUM_CLASSES = 19
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 2.5e-2
IMG_SIZE = (720, 1280)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHA = 1
CLASS_NAMES = [ 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle' ]

input_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
target_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
])

dataset_root = "/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/GTA5/GTA5"
train_joint_transform = JointTransform(input_transform, target_transform, augment=True, strategy='flip-jitter')
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

transform_cityscapes = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
target_dataset = CityscapesTarget(
    root='/kaggle/working/punto-3/Seg_sem_25/Seg_sem_25/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='train',
    transform=transform_cityscapes
)

train_loader_gta = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader_gta = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
cityscapes_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
class_pixel_count, image_class_pixels = compute_pixel_frequency(train_loader_gta, NUM_CLASSES)
weights = median_frequency_balancing(class_pixel_count, image_class_pixels)
print("✅ Class Weights (Median Frequency Balancing):")
print(weights)

normalized_weights = weights / weights.sum() * len(weights)
print(normalized_weights)
weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(DEVICE)


# ==============================
# Modello & Discriminatore
# ==============================

model = BiSeNet(num_classes=NUM_CLASSES, context_path='resnet18').to(DEVICE)
D = FCDiscriminator(num_classes=NUM_CLASSES).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"🚀 Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)
    D = nn.DataParallel(D)

model = model.to(DEVICE)
D = D.to(DEVICE)

# ==============================
# Ottimizzatori e perdita
# ==============================

criterion = HybridLoss(alpha=0.7, class_weights=weights_tensor)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))
scaler = GradScaler()

power = 0.9
steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
max_iter = steps_per_epoch * EPOCHS

bce_loss = nn.BCEWithLogitsLoss()
lambda_adv = 0.001

# ==============================
# Funzioni di perdita ausiliaria
# ==============================

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

# ==============================
# Training
# ==============================

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
                print("❌ Attenzione: total_loss contiene NaN o Inf! Interrompo il training.")
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

# ==============================
# Validazione
# ==============================

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
    
# ==============================
# MAIN
# ==============================

if __name__ == '__main__':
    start_epoch = 1
    checkpoint_path = "/kaggle/working/checkpoint__.pth"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Checkpoint trovato! Riprendo da epoca {start_epoch}")
    except FileNotFoundError:
        print("⚠️ Nessun checkpoint trovato, partenza da zero.")
    
    print("Avvio training")
    best_miou = 0.0

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        _, _ = train(model, train_loader_gta, optimizer, criterion, DEVICE, NUM_CLASSES, epoch)
        pixel_acc, mean_iou = validate(model, val_loader_gta, criterion, DEVICE, NUM_CLASSES, epoch)
        torch.cuda.empty_cache()

        # Salvataggio del checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, checkpoint_path)

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), 'best_model_5_combo_7_3.pth')
            print(f"Nuova best mIoU: {best_miou:.2f}% → modello salvato!")

        print(f"Epoch {epoch} completato! Best mIoU finora: {best_miou:.2f}%\n\n")

    torch.save(model.state_dict(), f'final_model_epoch_5_combo_7_3{EPOCHS}.pth')
    print(f"📦 Training finito: modello finale salvato come final_model_epoch_5_combo_7_3{EPOCHS}.pth")
