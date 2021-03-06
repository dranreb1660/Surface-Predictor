import torch
from torch.utils.data import Dataset


class SurfaceDataset(Dataset):

    def __init__(self, sequences) -> None:
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label).long()
        )
