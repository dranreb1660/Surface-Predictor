import torch.nn as nn


class SurfaceModel(nn.Module):

    def __init__(self, n_features, n_classes, n_hidden, dropout, n_layers=3):
        super().__init__()
        print(dropout)
        self.n_fearues = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.n_fearues,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.cassifier = nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # print(type(x))
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.cassifier(out)
