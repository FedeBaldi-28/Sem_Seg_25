import torch
import time
import argparse
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.cityscapes import Cityscapes
from utils import fast_hist, per_class_iou
from utils import compute_mIoU
from PIL import Image


# -------------------
# Configurazione
# -------------------
HEIGHT, WIDTH = 512, 1024
NUM_CLASSES = 19
ITERATIONS = 1000
IMG_SIZE = (512, 1024)

# -------------------
# Caricamento modello
# -------------------
def load_model(model_name, model_path):
    if model_name == "deeplabv2":
        from models.deeplabv2.deeplabv2 import get_deeplab_v2
        model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=False)
    elif model_name == "bisenet":
        from model.bisenet import BiSeNet
        model = BiSeNet(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

# -------------------
# Calcolo FLOPs
# -------------------
def compute_flops(model):
    device = next(model.parameters()).device
    dummy_image_flops = torch.zeros((1, 3, HEIGHT, WIDTH)).to(device)
    flops = FlopCountAnalysis(model, dummy_image_flops)
    print("FLOPs and Parameters:")
    print(flop_count_table(flops, max_depth=2))

# -------------------
# Calcolo Latency & FPS
# -------------------
def compute_latency_fps(model):
    device = next(model.parameters()).device
    latency = []
    fps = []

    print("Calcolo latency e FPS...")

    model.eval()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            dummy_input = torch.zeros((1, 3, HEIGHT, WIDTH)).to(device)
            start = time.time()
            _ = model(dummy_input)
            end = time.time()

            elapsed = end - start
            latency.append(elapsed)
            fps.append(1.0 / elapsed)

    latency = np.array(latency)
    fps = np.array(fps)

    print(f"\nMean Latency: {latency.mean() * 1000:.2f} ms")
    print(f"Std Latency: {latency.std() * 1000:.2f} ms")
    print(f"Mean FPS: {fps.mean():.2f}")
    print(f"Std FPS: {fps.std():.2f}")

# -------------------
# Loader Cityscapes val
# -------------------
def get_cityscapes_val_loader(root, height, width, batch_size=4):
    input_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
    ])
    
    val_set = Cityscapes(
    root='/content/drive/MyDrive/MLDL2024_project/datasets/Cityscapes/Cityscapes/Cityspaces',
    split='val',
    transform=input_transform,
    target_transform=target_transform
)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione FLOPs, Latency, FPS e mIoU")
    parser.add_argument("--model", type=str, required=True, help="Nome modello: deeplabv2 | bisenet")
    parser.add_argument("--weights", type=str, required=True, help="Path file .pth del modello")
    parser.add_argument("--cityscapes", type=str, required=True, help="Path alla directory root di Cityscapes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model.lower(), args.weights).to(device)

    compute_flops(model)
    compute_latency_fps(model)

    print("\nCalcolo mIoU su Cityscapes...")
    val_loader = get_cityscapes_val_loader(args.cityscapes, HEIGHT, WIDTH)
    miou, _ = compute_mIoU(model, val_loader, NUM_CLASSES)
    print(f"Mean IoU: {miou:.2f}%")

