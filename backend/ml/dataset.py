"""
This is the dataset loader for trail detection training.
It processes JSON masks and creates training data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import cv2
from PIL import Image, ImageDraw
import logging

logger = logging.getLogger(__name__)

class TrailDataset:
    """ This is the dataset class for loading labeled trail images and masks """

    def __init__(self, data_dir: Path, image_size: Tuple[int, int] = (512, 512), augment: bool = True):
        """
        data_dir: directory containing PNG images and JSON labelme files,
        image_size: target size for images (height, width)
        augment: whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Here we find all JSON files (these indicate labeled images)
        self.json_files = list(self.data_dir.glob("*.json"))
        logger.info(f"Found {len(self.json_files)} labeled images")

        # Now we validate that corresponding images exist
        self.valid_samples = []
        for json_file in self.json_files:
            image_file = json_file.with_suffix('.png')
            if image_file.exists():
                self.valid_samples.append((image_file, json_file))
            else:
                logger.warning(f"Image not found for {json_file}")

        logger.info(f"Valid samples: {len(self.valid_samples)}")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Here we get a training sample

        It returns an image (normalized image array (H, W, C)) and a binary mask array (H, W, 1)
        """
        image_path, json_path = self.valid_samples[idx]

        # Load image
        image = self._load_image(image_path)

        # Load and create mask from labelme JSON
        mask = self._create_mask_from_json(json_path, image.shape[:2])

        # Resize to target size
        image = cv2.resize(image, self.image_size[::-1])  # cv2 wants (W, H)
        mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Apply augmentations if enabled
        if self.augment:
            image, mask = self._augment(image, mask)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add channel dimension to mask if needed
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

        mask = mask.astype(np.float32) / 255.0

        return image, mask

    def _load_image(self, image_path: Path) -> np.ndarray:
        """ Here we load image as numpy array """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _create_mask_from_json(
            self,
            json_path: Path,
            image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """ Here we create a binary mask from the labelme JSON annotation """

        with open(json_path, 'r') as f:
            label_data = json.load(f)

        # We create a blank mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # We draw each shape onto the mask
        for shape in label_data.get('shapes', []):
            if shape['label'].lower() in ['trail', 'satellite', 'streak']:
                points = shape['points']

                if shape['shape_type'] == 'polygon':
                    # If it's a polygon we convert points to a numpy array
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)

                elif shape['shape_type'] == 'line' or shape['shape_type'] == 'linestrip':
                    # For lines, draw with thickness
                    pts = np.array(points, dtype=np.int32)
                    cv2.polylines(mask, [pts], False, 255, thickness=5)

        return mask

    def _augment(
            self,
            image: np.ndarray,
            mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
         Here we apply data augmentation
         For example, we do:
            - Random flips (horizontal and vertical)
            - Random rotations (90, 180, 270 degrees)
            - Random brightness/contrast adjustments
            - Gaussian noise
            """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # Random 90-degree rotation
        k = np.random.randint(0,4) # Each 1 = 90 degrees (2 = 180 deg for example)
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        # Random brightness adjusment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image-mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image, mask

    def split_train_val(
            self,
            val_split: float = 0.2,
            seed: int = 42
    ) -> Tuple['TrailDataset', 'TrailDataset']:
        """ Here we split the dataset into training and validation sets """
        np.random.seed(seed)
        indices = np.random.permutation(len(self.valid_samples))

        split_idx = int(len(indices) * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Create train dataset
        train_dataset = TrailDataset.__new__(TrailDataset)
        train_dataset.data_dir = self.data_dir
        train_dataset.image_size = self.image_size
        train_dataset.augment = self.augment
        train_dataset.valid_samples = [self.valid_samples[i] for i in train_indices]

        # Create validation dataset with no augmentation
        val_dataset = TrailDataset.__new__(TrailDataset)
        val_dataset.data_dir = self.data_dir
        val_dataset.image_size = self.image_size
        val_dataset.augment = False  # No augmentation for validation
        val_dataset.valid_samples = [self.valid_samples[i] for i in val_indices]

        logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

        return train_dataset, val_dataset

def visualize_sample(dataset: TrailDataset, idx: int = 0, save_path: str = None):
    """ Visualize a dataset sample for debugging """

    import matplotlib.pyplot as plt

    image, mask = dataset[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask[:, :, 0], cmap='gray')
    axes[1].set_title('Trail Mask')
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(mask[:, :, 0], cmap='Reds', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test the dataset loader
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data/raw")
    dataset = TrailDataset(data_dir, image_size=(512,512))

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Mask unique values: {np.unique(mask)}")

        # Visualize a few samples
        visualize_sample(dataset, 0, "sample_0.png")
