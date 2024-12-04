from typing import Any, Mapping
from lightning.pytorch import LightningModule
from model.layers.basic_linear_layer import BasicLinearModel
from torch.nn import functional as F
import torch
import numpy as np
from .eval.eval_vs_result import get_fscore_from_predscore


class VideoSumModule(LightningModule):
    def __init__(self, model, T_max, eta_min):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.val_fscore_step = []
        self.val_fscore_epoch = []

    def training_step(self, batch):
        pred, attn = self.model(batch)
        loss = self.loss_fn(pred, batch["gtscore"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, attn = self.model(batch)
        pred_scores = pred.squeeze(0).detach().cpu().numpy()
        for key in batch.keys():
            if not isinstance(batch[key][0], str):
                if key == "n_frames":
                    batch[key] = int(batch[key].detach().cpu().numpy())
                else:
                    batch[key] = batch[key].squeeze(0).detach().cpu().numpy()

        shot_bound = batch["change_points"].astype(int)
        n_frames = batch["n_frames"]
        positions = batch["positions"].astype(int)
        user_summary = batch["user_summary"]

        fscore = get_fscore_from_predscore(
            pred_scores=pred_scores,
            shot_bound=shot_bound,
            n_frames=n_frames,
            positions=positions,
            user_summary=user_summary,
        )
        
        self.val_fscore_step.append(fscore)
        self.log("val/fscore/step", fscore, batch_size=1)

    def on_validation_epoch_end(self):
        mean_fscore = np.mean(self.val_fscore_step)
        self.val_fscore_epoch.append(mean_fscore)
        
        self.log("val/fscore/epoch", mean_fscore, prog_bar=True, batch_size=1)
        self.log("val/fscore/max", max(self.val_fscore_epoch), prog_bar=True, batch_size=1)
        
        self.val_fscore_step.clear()

    def configure_optimizers(self):
        # Choose an optimizer or implement your own.
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.eta_min
        )
        optim_dict = {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optim_dict
