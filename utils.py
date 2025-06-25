import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
from datasets.labels import GTA5Labels_TaskCV2017


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

    print(f"Conversione maschere RGB in ID (salvate in: {converted_dir})")

    for fname in tqdm(images):
        converted_name = fname.replace(".png", "_converted.png")
        save_path = os.path.join(converted_dir, converted_name)

        if os.path.exists(save_path):
            continue

        label_path = os.path.join(labels_dir, fname)
        rgb_mask = np.array(Image.open(label_path).convert('RGB'))
        label_mask = rgb_to_label_id(rgb_mask)
        Image.fromarray(label_mask).save(save_path)

    print("Conversione completata.")


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

def median_frequency_balancing(class_pixel_count, image_class_pixels, NUM_CLASSES):
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

