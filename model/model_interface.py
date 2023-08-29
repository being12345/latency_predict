import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, model, loss='mse', lr=1.0 * 1e-3, **kargs):
        """
        model: str or model object
        loss: str or configure yourselves
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.configure_loss(loss)

    def forward(self, X, edge_index, edge_weight):
        if edge_weight:
            return self.model(X, edge_index, edge_weight)
        else:
            return self.model(X, edge_index)

    def training_step(self, batch, batch_idx):
        print(batch)
        snapshot = batch
        out = self(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = self.loss_function(out, snapshot.y)

        self.log('MSE loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        snapshot = batch
        out = self(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = self.loss_function(out, snapshot.y)

        MS = torch.sum((out - snapshot.y) ** 2)
        L1 = torch.mean(out - snapshot.y)
        self.log('L2', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('MS', MS, on_step=True, on_epoch=True, prog_bar=True)
        self.log('L1', L1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        snapshot = batch
        out = self(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = self.loss_function(out, snapshot.y)

        MS = torch.sum((out - snapshot.y) ** 2)
        L1 = torch.mean(out - snapshot.y)
        self.log('L2', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('MS', MS, on_step=True, on_epoch=True, prog_bar=True)
        self.log('L1', L1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if not hasattr(self.hparams, 'weight_decay'):
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self, loss):
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
