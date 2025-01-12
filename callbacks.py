import os
from pathlib import Path

import imageio
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from data_module import GeoSRDataset
from models.edsr import EDSR


class LogResults(Callback):
    def __init__(
        self,
        lr_path,
        hr_path,
        lr_names,
        hr_names,
        log_every_n_epochs=1,
        device="cuda:0",
        log_to_wandb=True,
        log_to_disk=True,
        save_dir="logs/",
    ):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.log_every_n_epochs = log_every_n_epochs
        self.device = device
        self.log_to_wandb = log_to_wandb
        self.log_to_disk = log_to_disk
        self.save_dir = save_dir
        self.lr_files = sorted(lr_names)
        self.hr_files = sorted(hr_names)
        self.lr_files = [self.lr_path + file for file in self.lr_files]
        self.hr_files = [self.hr_path + file for file in self.hr_files]

    def on_fit_start(self, trainer, pl_module):
        self.ds_pred = GeoSRDataset(
            lr_path=self.lr_path,
            hr_path=self.hr_path,
            scale=trainer.datamodule.scale,
            extension=trainer.datamodule.extension,
            patch_size=None,
        )

        self.ds_pred.files_lr = [
            file for file in self.ds_pred.files_lr if file in self.lr_files
        ]
        self.ds_pred.files_hr = [
            file for file in self.ds_pred.files_hr if file in self.hr_files
        ]
        self.save_dir = self.save_dir + pl_module.logger.experiment.name + "/"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            with torch.no_grad():
                for i in range(len(self.ds_pred)):
                    # Apply model
                    lr, hr = self.ds_pred[i]
                    hr_pred = pl_module(lr.to(self.device))

                    # Correct ranges if you like
                    lr = self.visual_corrections(lr)
                    hr_pred = self.visual_corrections(hr_pred)  # numpy
                    hr = self.visual_corrections(hr)

                    imgs_rgb = [lr[0:3, :, :], hr_pred[0:3, :, :], hr[0:3, :, :]]
                    imgs_nir = [lr[3, :, :], hr_pred[3, :, :], hr[3, :, :]]
                    # imgs_rgb = [lr[0:3,:,:],hr_pred[:,:,0:3],hr[0:3,:,:]]
                    # imgs_nir = [lr[3,:,:],hr_pred[:,:,3],hr[3,:,:]]

                    if self.log_to_wandb:
                        pl_module.logger.log_image(
                            f"epoch_{trainer.current_epoch}_img_{i}_rgb", imgs_rgb
                        )
                        pl_module.logger.log_image(
                            f"epoch_{trainer.current_epoch}_img_{i}_nir", imgs_nir
                        )
                    if self.log_to_disk:
                        hr_pred = hr_pred.cpu().detach().numpy()
                        hr_pred = hr_pred.swapaxes(0, 1).swapaxes(1, 2)
                        hr_pred = np.uint16(hr_pred)
                        imageio.v3.imwrite(
                            self.save_dir
                            + f"epoch_{trainer.current_epoch}_img_{i}.tif",
                            hr_pred,
                            extension=".tif",
                        )

    def visual_corrections(sele, img):
        """Here is a place for processing before wandb"""
        # Example:
        # img[:3] = img[:3] / img[:3].mean()
        # img[3] = img[3] / img[3].mean()
        # img = img / 2
        # img = torch.clip(img, 0,1)
        return img
