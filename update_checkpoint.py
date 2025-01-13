import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from callbacks import LogResults
from data_module import GeoSRData
from loss import DoubleLoss, oneminusx
from models.edsr import EDSR
from models.srcnn import SRCNN
from training_module import GeoSR

load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

checkpoint_dir = "ckpt/"
# checkpoint_name = "SRCNN_S2_PS_x4_16.ckpt"
# checkpoint_new_name = "SRCNN_S2_PS_x4_16_v3"
checkpoint_name = "EDSR_S2_PS_x4_16.ckpt"
checkpoint_new_name = "EDSR_S2_PS_x4_16_v3"
checkpoint_path = Path(checkpoint_dir) / checkpoint_name

# Download checkpoint from wandb:
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "<user>/GeoSR/model-3j0ng5qq:latest"
# from utils import download_checkpoint
# checkpoint_path = download_checkpoint(checkpoint_reference)


# Choose model
# model_class = SRCNN
# model_params = {
#     "num_channels" : 4,
#     "scale_factor" : 4,
#     "n_feats" : 128,
# }

model_class = EDSR
model_params = {
    "n_resblocks": 64,
    "n_feats": 128,
    "scale": 4,
    "kernel_size": 3,
}

# model_class = SwinIR
# model_params = {
#     "upscale":4,
#     "img_size":(128, 128),
#     "window_size":8,
#     "img_range":1.,
#     "in_chans":4,
#     "depths":[6, 6, 6, 6],
#     "embed_dim":60,
#     "num_heads":[6, 6, 6, 6],
#     "mlp_ratio":2,
#     "upsampler":'pixelshuffledirect',
#     }


trainer = Trainer(
    # accelerator="gpu",
    accelerator="cpu",
    deterministic=True,  # turn off/on random seed
    precision=32,  # "16-mixed"
    log_every_n_steps=1,
    max_epochs=1,
    accumulate_grad_batches=4,
    limit_train_batches=2,
    limit_val_batches=1,
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(
            every_n_train_steps=1,
            dirpath=checkpoint_dir,
            filename=checkpoint_new_name,
        ),
    ],
)

lightning_module = GeoSR.load_from_checkpoint(
    checkpoint_path,
    model_class=model_class,
    model_params=model_params,
    learning_rate=1e-10,
    scc_on_cpu=False,
    # loss_function=nn.MSELoss(),  # or nn.L1Loss()
    loss_function=DoubleLoss(
        loss1=nn.L1Loss(), loss2="SCC", f1=None, f2=oneminusx, merge_function=torch.mul
    ),
)

ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
trainer.fit(lightning_module, GeoSRData(**ckpt["datamodule_hyper_parameters"]))
