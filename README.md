# geoSR
A basic setup for training and testing super-resolution models on geospatial images. 

## Workflow
This project follows [lightning](https://lightning.ai/docs/pytorch/stable/) code organization principles. Main files:
- `train.py` initialize all parts nad make them spin. Here you need to setup all parameters.
- `lightning_module` control training/validation/test/prediction step, as wall as metrics calculation and logging.
- `data_module`:
    - `GeoSRDataset` extract pairs of pictures from two folders low resolution (lr) and high resolution (hr). 
    - `GeoSRData` decide how to load the data to the training loop.
- `models` - the torch `nn.Module` with a forward function that takes a batch of low resolution pictures and returns their higher resolution version. The pictures are torch 4-dim tensors with a shape (B,C,H,W).
- `callbacks` - additional stuff that is not a core part of a training or data loading but it is nice to have. 
- `logs`
## Data preparation
Suggested data structure:
```
your_data_folder_with_great_backups
├── eval
│   ├── low_res
│   │   ├── img1_23212.tif
│   │   └── img2_54351.tif
│   └── high_res
│       ├── img1_56234.tif
│       └── img2_34121.tif
├── test
│   ├── low_res
│   │   ├── img1_53454.tif
│   │   └── img2_12312.tif
│   └── high_res
│       ├── img1_09034.tif
│       └── img2_21341.tif
└── train
    ├── low_res
    │   ├── img1_90293.tif
    │   └── img2_12340.tif
    └── high_res
        ├── img1_09213.tif
        └── img2_23191.tif
```
The folders should have only matching pictures. The number of pictures has to be the same in both low- and high-res folder. The names of folders and files are irrelevant, but the order matters. Pictures are matches by the alphabetical order. Only pictures with a selected extension are used.

## Run

### Setup
Prepare a virtual environment. 
```python
python -m venv venv
pip install -r requirements.txt
```
Add `.env` file with paths:
```bash
MAIN_DATA_PATH="/your/path/data/"
TRAIN_DATA_SUB_PATH_HR = "train/hr/"
TRAIN_DATA_SUB_PATH_LR = "train/lr/"
EVAL_DATA_SUB_PATH_HR  = "eval/hr/"
EVAL_DATA_SUB_PATH_LR = "eval/lr/"
TEST_DATA_SUB_PATH_HR  = "test/hr/"
TEST_DATA_SUB_PATH_LR = "test/lr/"
```
With this setup, the training data will be taken from `/your/path/data/train/hr` and `/your/path/data/train/lr`.

Before a first training, run `wandb login`, if you are using the [WandB](https://wandb.ai) callback.

### Train

```python
python train.py
```