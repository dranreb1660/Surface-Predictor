import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import wandb

from surface_predictor.models.model import SurfaceModel


class SurfacePredictor(pl.LightningModule):
    def __init__(self, n_features, n_classes, n_hidden, seq_length, dropout, n_layers=3, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.seq_length = seq_length
        self.n_features = n_features

        self.model = SurfaceModel(
            n_features, n_classes, n_hidden, n_layers=n_layers, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.train_ac = torchmetrics.Accuracy()
        self.val_ac = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x, labels=None):
        output = self.model(x)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
            return loss, output

        return output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_ac = self.train_ac(predictions, labels)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_ac, prog_bar=True, logger=True)

        return dict(loss=loss, accuracy=step_ac)

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_ac = self.train_ac(predictions, labels)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_ac, prog_bar=True, logger=True)

        return dict(loss=loss, accuracy=step_ac, loggits=outputs)

    def validation_epoch_end(self, validation_step_outputs):
        dummyImput = torch.zeros(
            (1, self.seq_length, self.n_features), device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummyImput, model_filename, opset_version=11)
        # wandb.save(model_filename,)
        outputs = [out['loggits'] for out in validation_step_outputs]
        flattened_outputs = torch.flatten(
            torch.cat(outputs))
        self.logger.experiment.log(
            {'valid/logits': wandb.Histogram(flattened_outputs.to('cpu')),
             'epoch': self.current_epoch}
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
