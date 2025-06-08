import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from build_bisenet import BiSeNet
from datasets.cityscapes import Cityscapes
import time
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import math
from torch.optim.lr_scheduler import LambdaLR
from utils import fast_hist, per_class_iou


# CONFIGURAZIONE
CONTEXT_PATH = 'resnet18'
ALPHA = 1  # Same as the paper
NUM_CLASSES = 19
BATCH_SIZE = 16
EPOCHS = 50
INIT_LR = 2.5e-2
IMG_SIZE = (512, 1024)  # (H, W)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    root='/kaggle/working/prova1/MLDL2024_project/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='train',
    transform=input_transform,
    target_transform=target_transform
)

val_dataset = Cityscapes(
    root='/kaggle/working/prova1/MLDL2024_project/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='val',
    transform=input_transform,
    target_transform=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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

# MODELLO
model = BiSeNet(num_classes=NUM_CLASSES, context_path=CONTEXT_PATH)

if torch.cuda.device_count() > 1:
    print(f"🚀 Usando {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=255)
optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=1e-4)
scaler = GradScaler() 

# PARAMTETRI POLY LR
power = 0.9
N = len(train_dataset)                         
steps_per_epoch = math.ceil(N / BATCH_SIZE)    
max_iter = steps_per_epoch * EPOCHS            

# POLY LR
def poly_scheduler(iteration: int) -> float:
  return (1.0 - iteration / max_iter) ** power

scheduler = LambdaLR(optimizer, lr_lambda=poly_scheduler)

# FUNZIONE DI PERDITA
def Loss(output, target, criterion, cx1=None, cx2=None, alpha=1.0):
  
  main_loss = criterion(output,target)
  auxiliary_loss = 0 
  if cx1 is not None and cx2 is not None:
    auxiliary_loss += criterion(cx1,target)
    auxiliary_loss += criterion(cx2,target)

  joint_loss = main_loss + alpha * auxiliary_loss

  return joint_loss


# FUNZIONE DI TRAINING
def train(model, train_loader, optimizer, criterion, device, num_classes, epoch, scheduler):
  model.train()
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
        cx1 = outputs[1]
        cx2 = outputs[2]
        outputs = outputs[0]
      loss = Loss(outputs,targets,criterion,cx1,cx2,alpha=ALPHA)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

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


# FUNZIONE DI VALIDAZIONE
def validate(model, val_loader, criterion, device, num_classes, epoch):
  model.eval()
  val_loss = 0.0
  total_correct = 0
  total_pixels = 0
  total_batches = len(val_loader)
  start_time = time.time()

  hist = np.zeros((num_classes, num_classes))

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
      cx1 = cx2 = None    # Per stare sicuri
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
      targets = targets.squeeze(1).long()  # [B, 1, H, W] → [B, H, W]

      with autocast():  # ✅ Aggiunto qui
          outputs = model(inputs)
          if isinstance(outputs, tuple):
              outputs = outputs[0]

      loss = Loss(outputs,targets,criterion,None,None,alpha=ALPHA)
      val_loss += loss.item()

      preds = torch.argmax(outputs, dim=1)

      mask = (targets != 255)
      total_correct += ((preds == targets) & mask).sum().item()
      total_pixels += mask.sum().item()

      # Aggiorna confusion matrix per IoU
      for lp, pp in zip(targets.cpu().numpy(), preds.cpu().numpy()):
          hist += fast_hist(lp.flatten(), pp.flatten(), num_classes)

  avg_loss = val_loss / total_batches
  avg_pixel_acc = total_pixel_acc / total_batches

  ious = per_class_iou(hist)
  mIoU = np.nanmean(ious) * 100
  ious_per_class = ious * 100

  epoch_time = time.time() - start_time

  print(f"[Epoch {epoch}] | [Val] Loss: {avg_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | mIoU: {mIoU:.2f}% | Time: {epoch_time:.1f}s")

  print("IoU per classe (val):")
  for idx, iou_cls in enumerate(ious_per_class):
      print(f"  {CLASS_NAMES[idx]}: {iou_cls:.2f}%")

  return pixel_acc, mIoU
  
# CICLO DI TRAINING
if __name__ == '__main__':
  print("avvio training")
  best_miou = 0.0

  for epoch in range(1, EPOCHS + 1):
    print(f"epoch{epoch}/{EPOCHS}")
    train(model, train_loader, optimizer, criterion, DEVICE, NUM_CLASSES, epoch, scheduler)
    pixel_acc, mean_iou = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES, epoch)

    torch.cuda.empty_cache()

    if mean_iou > best_miou:
        best_miou = mean_iou
        torch.save(model.state_dict(), 'best_model_2b_weight.pth')
        print(f"Nuova best accuracy: {best_miou:.2f}% → modello salvato!")
    
    print(f"Epoch {epoch} completato! Best accuracy finora: {best_miou:.2f}%\n\n")

  torch.save(model.state_dict(), f'final_model_epoch_2b_weight{EPOCHS}.pth')
  print(f"📦 Training finito: modello finale salvato come final_model_epoch_2b_weight{EPOCHS}.pth")

