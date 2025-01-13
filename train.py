import os
from torchmetrics.image import SpatialCorrelationCoefficient as SCC

import torch
import torch.nn as nn
from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger

from callbacks import LogResults
from data_module import GeoSRData
from models.srcnn import SRCNN
from loss import DoubleLoss

# from models.edsr import EDSR, LogModelHyperParams
from models.swinir import LogModelHyperParams, SwinIR
from training_module import GeoSR

load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

# Choose model
model_class = SRCNN
model_params = {
    "num_channels": 4,
    "scale_factor": 4,
    "n_feats" : 128,
}

# model_class = EDSR
# model_params = {
#     "n_resblocks":64,
#     "n_feats":128,
#     "scale":4,
#     "kernel_size":3,
# }

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

lightning_module = GeoSR(
    model_class = model_class,
    model_params = model_params,
    # loss_function=nn.MSELoss(),  # or nn.L1Loss() 
    
    loss_function=DoubleLoss(
        loss1=nn.L1Loss(),
        loss2=SCC(high_pass_filter=torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device = "cuda"), window_size=8),
        f1 = None,
        f2 = lambda x: 1 - x, 
        merge_function= torch.mul
    ),


    spectrum_end=12000,  # for pixels values scaling
    # torch.tensor([[[1]],[[1]], [[1]], [[1]]])
    # Adam settings
    learning_rate=1e-5,  # Initial learning rate
    weight_decay=0,  # Make optimizer forgot old steps, 0 = turn off
    # Metrics
    scc_window=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    scc_ws=8,
    data_range=65535,  # max pixel values (bit range: 8bit-255, 16bit-65535)
    border_size=8,
    # You can multiply the learning rate by 0.1 after 100 and 150 epochs
    # scheduler_MultiStepLR_milestones = (100,150),
    scheduler_MultiStepLR_milestones=None,  # None = turn off
    scheduler_MultiStepLR_multiplier=0.1,
    watch=True,
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
    precision=32,  # "16-mixed"
    accumulate_grad_batches=1,
    log_every_n_steps=50,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    max_epochs=1,
    # limit_train_batches = None,
    # limit_val_batches=None,
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
            log_to_wandb=True,
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

trainer.fit(lightning_module, data_module)
# trainer.test(lightning_module, data_module)
