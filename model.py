import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
import pickle


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            nn.Conv1d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        self.convs_sigm = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.convs_tanh = _conv_stack(dilations, 1, num_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        out = x
        skips = []

        for conv_sigm, conv_tanh, residual in zip(
            self.convs_sigm, self.convs_tanh, self.residuals
        ):
            x = out
            out_sigm, out_tanh = conv_sigm(x), conv_tanh(x)
            # gated activation
            out = torch.tanh(out_tanh) * torch.sigmoid(out_sigm)
            skips.append(out)
            out = residual(out)
            out = out + x[:, :, -out.size(2) :]  # fit input with layer output

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class PedalNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PedalNet, self).__init__()
        self.wavenet = WaveNet(
            num_channels=hparams.num_channels,
            dilation_depth=hparams.dilation_depth,
            num_repeat=hparams.num_repeat,
            kernel_size=hparams.kernel_size,
        )
        self.hparams = hparams

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(self.hparams.data, "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.wavenet.parameters(), lr=self.hparams.learning_rate
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4
        )

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}
