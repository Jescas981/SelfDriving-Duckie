"""Train with Lightning; view live loss with: tensorboard --logdir logs"""
import argparse
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from dataset.duckietown_dataset import DuckietownDataset
from dataset.episode_sampler import EpisodeShuffleSampler
from model.model import SelfDrivingModel
import torch.nn.functional as F


class DrivingTrainer(pl.LightningModule):
    def __init__(self, lr=1e-3, lambda_att=10.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = SelfDrivingModel()

    def forward(self, images):
        return self.model(images)

    def _map_actions_to_invradius(self, actions):
        # use inverse radius for numerical stability
        v = actions[:, :, 0:1]      # (B, T, 1)
        omega = actions[:, :, 1:2]  # (B, T, 1)
        u_gt = omega / (torch.abs(v) + 1e-6)
        u_gt = torch.clamp(u_gt, min=-10, max=10)
        return u_gt

    def _compute_loss(self, batch):
        images = batch['image']  # (B, T, 3, 80, 160)
        # Lightning has already transferred the batch to the training device.
        # Keep images compact in worker caches, prefetch queues and transfers.
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        actions = batch['action']  # (B, T, 2)
        # use inverse radius for numerical stability
        u_target = self._map_actions_to_invradius(actions)  # (B, T, 1)
        predictions, attentions = self(images)
        B, T, L = attentions.shape
        att_sum = attentions.sum(dim=1)
        target = T / L

        l1_loss = F.l1_loss(predictions, u_target, reduction='mean')
        att_loss = torch.mean((target - att_sum) ** 2)
        total_loss = l1_loss + self.hparams.lambda_att * att_loss
        
        return total_loss, l1_loss, att_loss

    def training_step(self, batch, batch_idx):
        loss, l1_loss, att_loss = self._compute_loss(batch)
        self.log_dict(
            {'train_loss': loss, 'train_l1_loss': l1_loss,
             'train_att_loss': att_loss}, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch['image'].size(0),
        )
        # Lightning performs backward(), zero_grad(), and optimizer.step().
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l1_loss, att_loss = self._compute_loss(batch)
        self.log_dict(
            {'val_loss': loss, 'val_l1_loss': l1_loss,
             'val_att_loss': att_loss}, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch['image'].size(0),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train(model, loader, val_loader, epochs=10, log_dir='logs'):
    # Flush frequently so TensorBoard updates while training is running.
    logger = TensorBoardLogger(
        save_dir=str(log_dir), name='self_driving', flush_secs=5,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    print(f'TensorBoard logs: {logger.log_dir}')
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)
    return trainer


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def worker_count(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be at least 0')
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be a finite number greater than 0')
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be a finite number greater than or equal to 0')
    return value


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--num-workers', type=worker_count, default=3,
                        help='Training data loader workers; use 0 for debugging.')
    parser.add_argument('--val-num-workers', type=worker_count, default=None,
                        help='Validation workers (default: same as --num-workers).')
    parser.add_argument('--prefetch-factor', type=positive_int, default=1,
                        help='Batches prepared per worker; ignored with 0 workers.')
    parser.add_argument('--window-size', '--T', dest='window_size',
                        type=positive_int, default=30, help='Frames per sequence.')
    parser.add_argument('--epochs', type=positive_int, default=10)
    parser.add_argument('--lr', type=positive_float, default=1e-4)
    parser.add_argument('--lambda-att', '--lambda_att', type=nonnegative_float,
                        default=10.0, help='Attention loss weight; 0 disables its contribution.')
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--log-dir', type=Path, default=root / 'logs')
    args = parser.parse_args(argv)
    if args.val_num_workers is None:
        args.val_num_workers = args.num_workers
    return args


def main():
    args = parse_args()
    print(f'Training settings: {vars(args)}', flush=True)
    print('Loading training dataset...', flush=True)
    dataset = DuckietownDataset(
        root=args.data_dir, split='train', normalize=False,
        channels_first=True, T=args.window_size, training_only=True,
    )
    # Shuffle episodes each epoch while keeping their windows consecutive.
    sampler = EpisodeShuffleSampler(dataset._ends)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        **({'prefetch_factor': args.prefetch_factor}
                           if args.num_workers > 0 else {}))
    print(f'Training windows: {len(dataset)}. Loading validation dataset...', flush=True)
    val_dataset = DuckietownDataset(
        root=args.data_dir, split='validation', normalize=False,
        channels_first=True, T=args.window_size, training_only=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_num_workers, pin_memory=True,
        **({'prefetch_factor': args.prefetch_factor}
           if args.val_num_workers > 0 else {}),
    )
    print(f'Validation windows: {len(val_dataset)}. Starting training...', flush=True)
    model = DrivingTrainer(lr=args.lr, lambda_att=args.lambda_att)
    train(model, loader, val_loader, epochs=args.epochs, log_dir=args.log_dir)


if __name__ == '__main__':
    main()
