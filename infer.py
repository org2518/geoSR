import torch
from torch.utils.data import DataLoader
from data_module import GeoSRData, GeoSRDatasetInference
from train_module import GeoSR
from lightning.pytorch import Trainer, seed_everything
from models.srcnn import SRCNN
from models.edsr import EDSR
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
import wandb
from tqdm import tqdm
from pathlib import Path
import imageio
from collections import OrderedDict
import sys
import numpy as np
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
load_dotenv()

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)
save_dir = "/path/to/output"

# loading checkpoint
# checkpoint from wandb
# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
checkpoint_reference = "<user>/GeoSR_PhD/model-r5p0tmvu:latest"
api = wandb.Api()
artifact = api.artifact(checkpoint_reference)
artifact_dir = artifact.download()
ckpt_path = Path(artifact_dir) / "model.ckpt"

# checkpoint from disk
# ckpt_path  = torch.load(".ckpt")

# Define model with the same parameters as in the initial training
# model = SRCNN(
#     num_channels = 4,
#     scale_factor = 2,
#     )

model = EDSR(
    n_resblocks=32,
    n_feats=256,
    scale=2,
    kernel_size=3,
    n_colors=4,
)

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

dataset = GeoSRDatasetInference(
    lr_path = "/path/to/lr_data/",
    extension=".tif",
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=1,
)

trainer = Trainer(
    accelerator="gpu",
)

##########################
# download checkpoint locally (if not already cached)

ckpt = torch.load(ckpt_path)
model_ckpt = OrderedDict({k.removeprefix("model."):v for k,v in ckpt["state_dict"].items() if k.startswith("model.")})
try:
    model.load_state_dict(model_ckpt)
except KeyError as ex:
    print("KeyError: Model parameters do not match the source model.")
    sys.exit(1)
lightning_module = GeoSR.load_from_checkpoint(ckpt_path, model=model)

# Preparing output
save_dir = Path(save_dir)
save_dir.mkdir(exist_ok=True)
##########################

# Main loop
lightning_module.eval()
with torch.no_grad():
    for lr,f_lr in tqdm(dataloader): # iterate over batches (img, img_file_name)
        lr = lr.to("cuda")
        hr_pred = lightning_module(lr)
        for hr_predi,f_lri in zip(hr_pred, f_lr): # iterate over pictures and file names
            hr_predi = hr_predi.cpu().detach().numpy()
            hr_predi = hr_predi.swapaxes(0, 1).swapaxes(1, 2)
            hr_predi = np.uint16(hr_predi)
            imageio.v3.imwrite(
                save_dir / (f_lri + ".tif"),
                hr_predi,
                extension=".tif",
            )
