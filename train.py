from data_module import GeoSRData
from train_module import GeoSR
from callbacks import LogResults
from dotenv import load_dotenv
import os
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    RichModelSummary,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch import Trainer, seed_everything
import torch.nn as nn
import torch

from models.srcnn import SRCNN
# from models.edsr import EDSR, LogModelHyperParams
from models.swinir import SwinIR, LogModelHyperParams

load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

model = SRCNN(
    num_channels = 4,
    scale_factor = 4,
    )

# model = EDSR(
#     n_resblocks=64,
#     n_feats=128,
#     scale=4,
#     kernel_size=3,
# )

    # window_size = 8
    # height = (1024 // upscale // window_size + 1) * window_size
    # width = (720 // upscale // window_size + 1) * window_size

# model = SwinIR(
#     upscale=4, 
#     img_size=(128, 128),
#     window_size=8, 
#     img_range=1.,
#     in_chans=4, 
#     depths=[6, 6, 6, 6],
#     embed_dim=60, 
#     num_heads=[6, 6, 6, 6], 
#     mlp_ratio=2, 
#     upsampler='pixelshuffledirect',
#     )

lightning_module = GeoSR(
    model=model,
    loss_function=nn.MSELoss(),  # or nn.L1Loss()
    spectrum_end=12000,  # for pixels values scaling
    # torch.tensor([[[1]],[[1]], [[1]], [[1]]])
    # Adam settings
    learning_rate=1e-5,  # Initial learning rate
    weight_decay=0,  # Make optimizer forgot old steps, 0 = turn off
    # Metrics
    scc_window=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    scc_ws=8,
    data_range=1,  # PSNR and SSIM
    border_size=8,
    # You can multiply the learning rate by 0.1 after 100 and 150 epochs
    # scheduler_MultiStepLR_milestones = (100,150),
    scheduler_MultiStepLR_milestones=None,  # None = turn off
    scheduler_MultiStepLR_multiplier=0.1,
    watch = True,
)
data_module = GeoSRData(
    scale=4,
    extension=".tif",
    patch_size=(128, 128),
    batch_size=4,
    n_data_jobs=2,
    # spectrum_end = 12000, # for pixels values scaling
)

trainer = Trainer(
    accelerator="gpu",
    # accelerator = "cpu",
    deterministic=True,  # turn off/on random seed
    # gradient_clip_val=0.5,
    # gradient_clip_algorithm="norm",
    precision=32, # "16-mixed"
    accumulate_grad_batches=4,
    log_every_n_steps=50,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    max_epochs=10,
    # limit_train_batches = 11,
    limit_val_batches=4,
    logger=WandbLogger(
        project="GeoSR",
        log_model=False,  # "all", True or False
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
            log_to_wandb=True,
            log_to_disk=True,
            save_dir="logs/tif/",
            log_every_n_epochs=1,
        ),
        # ModelCheckpoint(monitor="val_scc", mode="max"),
        # EarlyStopping(monitor="val_scc", min_delta=0.0001, patience=3, mode="max"),
        # LearningRateFinder(min_lr=1e-7, max_lr=1e-4, num_training_steps=100),
        LearningRateMonitor(),
        # RichModelSummary(),
        # LogModelHyperParams(),
    ],
    # profiler="simple",  # use to check what is working slow
    # default_root_dir=os.environ["WORK_PATH"],
)

trainer.fit(lightning_module, data_module)
# trainer.test(lightning_module, data_module)
