import pytorch_lightning as pl
from torch.utils.data import DataLoader

from surface_predictor.data.base_dataset import SurfaceDataset


class SurfaceDatamodule(pl.LightningDataModule):

    def __init__(self, train_sequences, val_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SurfaceDataset(self.train_sequences)
        self.val_dataset = SurfaceDataset(self.val_sequences)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=2)
