import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as torch_data
import lightning
import os
from torchvision.transforms.v2.functional import crop
import imageio
from torch.utils.data import DataLoader


class GeoSRData(lightning.LightningDataModule):
    def __init__(
        self,
        scale=4,
        extension=".tif",
        patch_size=(300, 300),
        batch_size=32,
        n_data_jobs=4,
    ):
        super().__init__()
        self.scale = scale
        self.extension = extension
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_data_jobs = n_data_jobs
        self.prefetch_factor = 1 if self.n_data_jobs > 0 else None

        self.main_data_path = os.environ["MAIN_DATA_PATH"]
        self.save_hyperparameters()

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.ds_train = GeoSRDataset(
                lr_path=self.main_data_path + os.environ["TRAIN_DATA_SUB_PATH_LR"],
                hr_path=self.main_data_path + os.environ["TRAIN_DATA_SUB_PATH_HR"],
                scale=self.scale,
                extension=self.extension,
                patch_size=self.patch_size,
            )
            self.ds_val = GeoSRDataset(
                lr_path=self.main_data_path + os.environ["EVAL_DATA_SUB_PATH_LR"],
                hr_path=self.main_data_path + os.environ["EVAL_DATA_SUB_PATH_HR"],
                scale=self.scale,
                extension=self.extension,
                patch_size=None,
            )
        if stage == "test":
            self.ds_test = GeoSRDataset(
                lr_path=self.main_data_path + os.environ["TEST_DATA_SUB_PATH_LR"],
                hr_path=self.main_data_path + os.environ["TEST_DATA_SUB_PATH_HR"],
                scale=self.scale,
                extension=self.extension,
                patch_size=None,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_data_jobs,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min((2, self.n_data_jobs)),
            pin_memory=True,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_data_jobs,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
        )

    def pred_dataloader(self):
        pass


class GeoSRDataset(torch_data.Dataset):
    def __init__(
        self,
        lr_path,
        hr_path,
        scale=4,
        extension=".tif",
        patch_size=(300, 300),
    ):
        super().__init__()

        self.scale = scale
        self.extension = extension
        self.patch_size = patch_size
        self.crop = type(self.patch_size) is tuple

        self.files_lr = [
            lr_path + file
            for file in os.listdir(lr_path)
            if file.endswith(self.extension)
        ]
        self.files_hr = [
            hr_path + file
            for file in os.listdir(hr_path)
            if file.endswith(self.extension)
        ]
        if len(self.files_lr) != len(self.files_hr):
            raise FileExistsError("Length on lr and hr pictures is not the same")

        self.files_lr.sort()
        self.files_hr.sort()

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, idx):
        f_lr = self.files_lr[idx]
        f_hr = self.files_hr[idx]

        lr = imageio.v3.imread(f_lr)
        hr = imageio.v3.imread(f_hr)
        # img shape (H, W, C)
        if self.crop:
            iy = torch.randint(0, lr.shape[0] - self.patch_size[0], (1,)).item()
            ix = torch.randint(0, lr.shape[1] - self.patch_size[1], (1,)).item()
            lr = lr[iy : iy + self.patch_size[0], ix : ix + self.patch_size[1], :]
            hr = hr[
                iy * self.scale : iy * self.scale + self.patch_size[0] * self.scale,
                ix * self.scale : ix * self.scale + self.patch_size[1] * self.scale,
                :,
            ]
        lr = np.float32(lr.swapaxes(0, 1).swapaxes(2, 0))
        hr = np.float32(hr.swapaxes(0, 1).swapaxes(2, 0))
        lr = torch.tensor(lr)
        hr = torch.tensor(hr)
        # img shape (C, H, W)
        return lr, hr


class GeoSRDatasetInference(torch_data.Dataset):
    def __init__(
        self,
        lr_path,
        extension=".tif",
    ):
        super().__init__()

        self.extension = extension
        self.files_lr = [
            lr_path + file
            for file in os.listdir(lr_path)
            if file.endswith(self.extension)
        ]
        self.files_lr.sort()

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, idx):
        f_lr = self.files_lr[idx]

        lr = imageio.v3.imread(f_lr)
        # img shape (H, W, C)

        lr = np.float32(lr.swapaxes(0, 1).swapaxes(2, 0))
        lr = torch.tensor(lr)
        # img shape (C, H, W)
        f_lr = f_lr.split("/")[-1].split(".")[0]
        return lr, f_lr