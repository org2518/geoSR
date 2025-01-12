import os
import sys
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import torch
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data_module import GeoSRData, GeoSRDatasetInference
from models.edsr import EDSR
from models.srcnn import SRCNN
from training_module import GeoSR

load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)
output_save_dir = "output/"
device = "cuda"

checkpoint_dir = "ckpt/"
# checkpoint_name = "SRCNN_S2_PS_x4_16_v2.ckpt"
checkpoint_name = "EDSR_S2_PS_x4_16_v2.ckpt"
checkpoint_path = Path(checkpoint_dir) / checkpoint_name

# Download checkpoint from wandb:
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "<user>/GeoSR/model-3j0ng5qq:latest"
# from utils import download_checkpoint
# checkpoint_path = download_checkpoint(checkpoint_reference)

MAIN_DATA_PATH = os.getenv("MAIN_DATA_PATH")
EVAL_DATA_SUB_PATH_LR = os.getenv("EVAL_DATA_SUB_PATH_LR")
if MAIN_DATA_PATH is None or EVAL_DATA_SUB_PATH_LR is None:
    raise EnvironmentError("Setup .env file")

dataset = GeoSRDatasetInference(
    lr_path=Path(MAIN_DATA_PATH) / EVAL_DATA_SUB_PATH_LR,
    extension=".tif",
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=1,
)

##########################
# Loading model
geoSR_model = GeoSR.load_from_checkpoint(checkpoint_path)

# Preparing output
save_dir = Path(output_save_dir)
save_dir.mkdir(exist_ok=True)
##########################

# Main loop
geoSR_model.eval()
geoSR_model = geoSR_model.to(device)
with torch.no_grad():
    for lr, f_lr in tqdm(dataloader):  # iterate over batches (img, img_file_name)
        lr = lr.to(device)
        hr_pred = geoSR_model(lr)
        for hr_predi, f_lri in zip(
            hr_pred, f_lr
        ):  # iterate over pictures and file names
            hr_predi = hr_predi.cpu().detach().numpy()
            hr_predi = hr_predi.swapaxes(0, 1).swapaxes(1, 2)
            hr_predi = np.uint16(hr_predi)
            imageio.v3.imwrite(
                save_dir / (f_lri + "_123.tif"),
                hr_predi,
                extension=".tif",
            )
