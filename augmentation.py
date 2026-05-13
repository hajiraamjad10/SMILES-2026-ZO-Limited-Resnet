"""
augmentation.py — Enriched training augmentation pipeline for CIFAR100.

The key additions over the skeleton:
- RandomCrop(224, padding=28): translation invariance without information loss.
- ColorJitter: robustness to brightness, contrast, saturation shifts common
  across CIFAR100 superclasses (animals, vehicles, household objects, etc.)
- RandomRotation(15): moderate rotational invariance.
- RandomErasing(p=0.2): simulates occlusion, reduces overfitting to texture.

Validation pipeline is unchanged (deterministic).
"""

import torchvision.transforms as T

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the CIFAR100 image transform pipeline.

    Args:
        train: If True, return the augmented training pipeline.
               If False, return the fixed validation pipeline.
    """
    if train:
        return T.Compose([
            T.Resize(232),                                    # slight over-size for crop
            T.RandomCrop(224),                                # translation invariance
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),                     # moderate rotation
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2)),       # occlusion robustness
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
