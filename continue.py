
import torch
from data_module import GeoSRData
from train_module import GeoSR
from lightning.pytorch import Trainer, seed_everything
from models.srcnn import SRCNN
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
import wandb
from pathlib import Path
from collections import OrderedDict
import sys
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
checkpoint_reference = "<user>/GeoSR/model-3j0ng5qq:latest"

# Define model with the same parameters as in the initial training
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
    # default_root_dir=os.environ["WORK_PATH"],
)

##########################
# download checkpoint locally (if not already cached)
run = wandb.init(project="GeoSR")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

# loading checkpoint
ckpt = torch.load(Path(artifact_dir) / "model.ckpt")
model_ckpt = OrderedDict({k.removeprefix("model."):v for k,v in ckpt["state_dict"].items() if k.startswith("model.")})
try:
    model.load_state_dict(model_ckpt)
except KeyError as ex:
    print("KeyError: Model parameters do not match the source model.")
    sys.exit(1)
lightning_module = GeoSR.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", model=model)


trainer.fit(lightning_module, GeoSRData(**ckpt["datamodule_hyper_parameters"]), ckpt_path=Path(artifact_dir) / "model.ckpt")