import numpy as np


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

import numpy as np
import torch


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def compute_mIoU(model, dataloader, num_classes, device='cuda'):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze(1).cpu().numpy()

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for lp, pp in zip(labels, preds):
                hist += fast_hist(lp.flatten(), pp.flatten(), num_classes)

    ious = per_class_iou(hist)  # array di iou per classe
    mIoU = np.nanmean(ious) * 100  # media
    return mIoU, ious * 100  # restituisco anche IoU per classe in % (comodo per stampa)

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets.labels import GTA5Labels_TaskCV2017

def convert_masks_once(root):
    images_dir = os.path.join(root, 'images')
    labels_dir = os.path.join(root, 'labels')
    converted_dir = os.path.join(root, 'converted')
    os.makedirs(converted_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    label_def = GTA5Labels_TaskCV2017()
    rgb_to_id = {label.color: label.ID for label in label_def.list_}

    def rgb_to_label_id(rgb_mask):
        h, w, _ = rgb_mask.shape
        mask_id = np.full((h, w), 255, dtype=np.uint8)
        for color, label_id in rgb_to_id.items():
            matches = np.all(rgb_mask == color, axis=-1)
            mask_id[matches] = label_id
        return mask_id

    print(f"🔁 Conversione maschere RGB in ID (salvate in: {converted_dir})")

    for fname in tqdm(images):
        converted_name = fname.replace(".png", "_converted.png")
        save_path = os.path.join(converted_dir, converted_name)

        if os.path.exists(save_path):
            continue

        label_path = os.path.join(labels_dir, fname)
        rgb_mask = np.array(Image.open(label_path).convert('RGB'))
        label_mask = rgb_to_label_id(rgb_mask)
        Image.fromarray(label_mask).save(save_path)

    print("✅ Conversione completata.")
