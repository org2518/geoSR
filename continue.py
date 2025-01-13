import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from callbacks import LogResults
from data_module import GeoSRData
from models.srcnn import SRCNN
from training_module import GeoSR

load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

checkpoint_dir = "ckpt/"
checkpoint_name = "SRCNN_S2_PS_x4_16.ckpt"
checkpoint_path = Path(checkpoint_dir) / checkpoint_name

# Download checkpoint from wandb:
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "<user>/GeoSR/model-3j0ng5qq:latest"
# from utils import download_checkpoint
# checkpoint_path = download_checkpoint(checkpoint_reference)

trainer = Trainer(
    accelerator="gpu",
    # accelerator = "cpu",
    deterministic=True,  # turn off/on random seed
    # gradient_clip_val=0.5,
    # gradient_clip_algorithm="norm",
    precision=32,  # "16-mixed"
    accumulate_grad_batches=4,
    log_every_n_steps=50,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    max_epochs=1,
    # limit_train_batches = 11,
    limit_val_batches=4,
    logger=WandbLogger(
        project="GeoSR",
        log_model="all",  # Must be "all" for checkpointing
        offline=False,
        save_dir="logs/",
    ),
    callbacks=[
        LogResults(
            lr_path=os.environ["MAIN_DATA_PATH"] + os.environ["EVAL_DATA_SUB_PATH_LR"],
            hr_path=os.environ["MAIN_DATA_PATH"] + os.environ["EVAL_DATA_SUB_PATH_HR"],
            lr_names=[
                "T43QEC_20220404_RGBN_10m_14_1.tif",
                "T35TMJ_20210823_RGBN_10m_13_2.tif",
            ],
            hr_names=[
                "T43QEC_20220404_RGBN_PS_2_5m_14_1.tif",
                "T35TMJ_20210823_RGBN_PS_2_5m_13_2.tif",
            ],
            log_to_wandb=False,
            log_to_disk=True,
            save_dir="logs/tif/",
            log_every_n_epochs=1,
        ),
        ModelCheckpoint(monitor="val_scc", mode="max"),
        # ModelCheckpoint(
        #     dirpath=None,
        #     filename=None,
        #     monitor="val_scc",
        #     mode='max',
        #     save_last=True,
        #     save_top_k=1,
        #     every_n_train_steps=None,
        #     train_time_interval=None,
        #     every_n_epochs=None,
        #     ),
        # EarlyStopping(monitor="val_scc", min_delta=0.0001, patience=3, mode="max"),
        # LearningRateFinder(min_lr=1e-7, max_lr=1e-4, num_training_steps=100),
        LearningRateMonitor(),
        # RichModelSummary(),
        # LogModelHyperParams(),
    ],
    # profiler="simple",  # use to check what is working slow
)

##########################
ckpt = torch.load(checkpoint_path)
lightning_module = GeoSR.load_from_checkpoint(checkpoint_path)
trainer.fit(
    lightning_module,
    GeoSRData(**ckpt["datamodule_hyper_parameters"]),
    ckpt_path=checkpoint_path,
)