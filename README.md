## Real-time Domain Adaptation in Semantic Segmentation
In this report, we address the challenges of real-time domain adaptation in semantic segmentation, with a focus on the domain shift between synthetic and real world images, evaluating how different strategies, such as architectural choices, data augmentation, loss functions and domain adaptation methods can reduce the domain gap while maintaining real-time inference capabilities.

**DeepLab petrained weights**: https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing

## Datasets

**Cityscapes**: https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing

**GTA5**: https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing

## utils

utils.py contains helper functions to compute: learning rate with polynomial decay, calculation of IoU by class, conversion of the dataset masks from GTA5 from RGB to numeric IDs, calculation of balancing weights using the Median Frequency Balancing technique.

## Step 2 – Supervised Training on Cityscapes

**Train DeepLabV2:**

- Without class weights: train_2a_no_weight.py

- With class weights: train_2a_weight.py

**Train BiSeNet:**

- Without class weights: train_2b_no_weight.py

- With class weights: train_2b_weight.py

**Evaluate models:**

Use evaluation.py on Cityscapes validation set.

## Step 3 – Domain Shift Evaluation

Before training convert_gta5_mask.py converts RGB segmentation masks to class IDs using labels.py.

**Train BiSeNet on GTA5 and test on Cityscapes:**

No augmentation: 

- train_3a.py

With augmentations:

- Flip only: train_3b_flip.py

- Color jitter: train_3b_jitter.py

- Jitter + flip: train_3b_jitter_flip.py

- Jitter + flip + crop: train_3b_jitter_flip_crop.py

Use evaluation.py on Cityscapes validation set.

## Step 4 – Domain Adaptation

Adversarial domain adaptation: train_4.py

The discriminator was implemented in discriminator.py

Use evaluation.py on Cityscapes validation set.

## Step 5 – Loss Extensions

Experiments with advanced loss functions for rare class emphasis.

losses.py contains the definition of Focal loss, Dice loss and their combination.

**Focal loss:**

- Weighted: train_5_focal_weighted.py

- Not weighted: train_5_focal_no_weighted.py

**Dice loss:**

- Weighted: train_5_dice_weighted.py

- Not weighted: train_5_dice_no_weighted.py

**Combo loss (Focal + Dice):**

- α = 0.7: train_5_combo_focal_03_dice_07.py

- α = 0.5: train_5_combo_focal_05_dice_05.py

- α = 0.3: train_5_combo_focal_07_dice_03.py





