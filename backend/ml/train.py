"""
Training script for the trail detection U-Net model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

from .model import UNet, CombinedLoss
from .dataset import TrailDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """
    This is the training manager

    It handles the training loop, validation, checkpointing, metrics tracking
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cuda',
            learning_rate: float = 1e-4,
            checkpoint_dir: Path = Path('data/models')
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss and optimizer
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.epochs_trained = 0

    def train_epoch(self) -> dict:
        """
        Train for one epoch.

        An epoch = one complete pass through all training data.
        """
        self.model.train()

        total_loss = 0
        total_iou = 0
        num_batches = 0

        # tqdm creates the progress bar seen during training
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Change (B, H, W, C) to (B, C, H, W) if needed
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)
            if masks.shape[1] != 1:
                masks = masks.permute(0, 3, 1, 2)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)

            # Backwards pass
            loss.backward()
            self.optimizer.step()

            # Calculate IoU
            iou = self._calculate_iou(predictions, masks)

            # Update metrics
            total_loss += loss.item()
            total_iou += iou
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        return {
            'loss': avg_loss,
            'iou': avg_iou
        }

    def validate(self) -> dict:
        """
        Validate the model on the validation set
        """

        self.model.eval()

        total_loss = 0
        total_iou = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if images.shape[1] != 3:
                    images = images.permute(0, 3, 1, 2)
                if masks.shape[1] != 1:
                    masks = masks.permute(0, 3, 1, 2)

                # Forward pass only
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)

                # Calculate IoU
                iou = self._calculate_iou(predictions, masks)

                # Update metrics
                total_loss += loss.item()
                total_iou += iou
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}'
                })

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        return {
            'loss': avg_loss,
            'iou': avg_iou
        }

    def _calculate_iou(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            threshold: float = 0.5
    ) -> float:

        # Binarize predictions
        pred_binary = (predictions > threshold).float()

        # Calculate intersection and union
        intersection = (pred_binary * targets).sum()
        union = pred_binary.sum() + targets.sum() - intersection

        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / union
        return iou.item()

    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model for multiple epochs.

        This is the main training loop that:
        1. Trains for one epoch
        2. Validates
        3. Updates learning rate
        4. Saves checkpoints
        5. Repeats

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"IoU: {train_metrics['iou']:.4f}"
            )

            # Validate
            val_metrics = self.validate()
            logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"IoU: {val_metrics['iou']:.4f}"
            )

            # Update learning rate based on validation loss
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(current_lr)

            self.epochs_trained += 1

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint('best_model.pth')
                logger.info(f"Saved new best model (val_loss: {val_metrics['loss']:.4f})")

        # Save final model
        self._save_checkpoint('final_model.pth')
        self._save_training_history()

        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        torch.save({
            'epoch': self.epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _save_training_history(self):
        """Save training history as JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved training history: {history_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.epochs_trained = checkpoint['epoch']

        logger.info(f"Loaded checkpoint from epoch {self.epochs_trained}")


def collate_fn(batch):
    """
    Custom collate function to handle batching.

    This converts lists of numpy arrays into PyTorch tensors.
    """
    images = []
    masks = []

    for image, mask in batch:
        images.append(torch.from_numpy(image))
        masks.append(torch.from_numpy(mask))

    images = torch.stack(images)
    masks = torch.stack(masks)

    return images, masks


def main():
    """Main training function."""

    # Configuration
    DATA_DIR = Path("data/raw")
    MODEL_DIR = Path("data/models")
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = TrailDataset(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        augment=True
    )

    # Split into train and validation
    train_dataset, val_dataset = full_dataset.split_train_val(
        val_split=VAL_SPLIT,
        seed=42
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device == 'cuda')
    )

    # Create model
    logger.info("Creating model...")
    model = UNet(in_channels=3, out_channels=1)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=MODEL_DIR
    )

    # Train
    trainer.train(num_epochs=NUM_EPOCHS, save_every=5)


if __name__ == "__main__":
    main()