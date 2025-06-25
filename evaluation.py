import torch
import time
import argparse
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.cityscapes import Cityscapes
from utils import fast_hist, per_class_iou
from datasets.labels import GTA5Labels_TaskCV2017
from PIL import Image
from tqdm import tqdm
from torch import amp
import os
import csv
import matplotlib.pyplot as plt


#################### CONFIGURAZIONE ####################
HEIGHT, WIDTH = 512, 1024
NUM_CLASSES = 19
ITERATIONS = 1000
IMG_SIZE = (512, 1024)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]


#################### COLORS CITYSCAPES ####################
def get_cityscapes_palette():
    palette = [0] * (256 * 3)
    for label in GTA5Labels_TaskCV2017.list_:
        palette[label.ID * 3:label.ID * 3 + 3] = list(label.color)
    return palette

def label_to_color(label_array):
    color_img = Image.fromarray(label_array.astype(np.uint8), mode='P')
    color_img.putpalette(get_cityscapes_palette())
    return color_img


#################### LOAD MODEL ####################
def load_model(model_name, model_path):
    if model_name == "deeplabv2":
        from models.deeplabv2.deeplabv2 import get_deeplab_v2
        model = get_deeplab_v2(num_classes=NUM_CLASSES, pretrain=False)
    elif model_name == "bisenet":
        from models.bisenet.build_bisenet import BiSeNet
        model = BiSeNet(num_classes=NUM_CLASSES, context_path='resnet18')
    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")

    state_dict = torch.load(model_path, map_location="cpu")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    return model


#################### FLOPS ####################
def compute_flops(model):
    device = next(model.parameters()).device
    dummy = torch.zeros((1, 3, HEIGHT, WIDTH)).to(device)
    flops = FlopCountAnalysis(model, dummy)
    print("FLOPs and Parameters:")
    print(flop_count_table(flops, max_depth=2))


#################### LATENCY & FPS ####################
def compute_latency_fps(model):
    device = next(model.parameters()).device
    latency = []
    fps = []
    print("Calcolo latency e FPS...")

    with torch.no_grad():
        for _ in range(ITERATIONS):
            dummy_input = torch.zeros((1, 3, HEIGHT, WIDTH)).to(device)
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            elapsed = end - start
            latency.append(elapsed)
            fps.append(1.0 / elapsed)

    print(f"\nMean Latency: {np.mean(latency)*1000:.2f} ms")
    print(f"Mean FPS: {np.mean(fps):.2f}")


#################### LOAD VALIDATION ####################
def get_cityscapes_val_loader(root, height, width, batch_size=1):
    input_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
    ])

    val_set = Cityscapes(
        root=root,
        split='val',
        transform=input_transform,
        target_transform=target_transform
    )
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)


#################### IMAGE VIEWING ####################
def visualize_image_gt_pred(model, val_loader, model_name, device):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze(1).cpu().numpy()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1).squeeze(1).cpu().numpy()

            input_img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            input_img = (input_img * np.array([0.229, 0.224, 0.225]) +
                         np.array([0.485, 0.456, 0.406])).clip(0, 1)

            gt_color = label_to_color(targets[0])
            pred_color = label_to_color(preds[0])

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(input_img)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_color)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_color)
            plt.title(f"Prediction ({model_name})")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"/kaggle/working/predizione_{model_name}.png")
            plt.show()

            break


#################### MAIN ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="deeplabv2 | bisenet")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--cityscapes", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model.lower(), args.weights).to(device)
    model.eval()

    compute_flops(model)
    compute_latency_fps(model)

    print("\nCalcolo mIoU su Cityscapes...")
    val_loader = get_cityscapes_val_loader(args.cityscapes, HEIGHT, WIDTH)

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1).long()
            with amp.autocast('cuda'):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)
            for lp, pp in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                hist += fast_hist(lp.flatten(), pp.flatten(), NUM_CLASSES)

    ious = per_class_iou(hist)
    mIoU = np.nanmean(ious) * 100
    print(f"Mean IoU: {mIoU:.2f}%")
    for idx, iou_cls in enumerate(ious * 100):
        print(f"  {CLASS_NAMES[idx]}: {iou_cls:.2f}%")

    results_path = "/kaggle/working/miou_results_ultimi.csv"
    file_exists = os.path.exists(results_path)
    with open(results_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['model_name', 'weights_file', 'mIoU'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model_name': args.model.lower(),
            'weights_file': os.path.basename(args.weights),
            'mIoU': f"{mIoU:.2f}"
        })

    weights_filename = os.path.basename(args.weights)
    visualize_image_gt_pred(model, val_loader, weights_filename, device)
