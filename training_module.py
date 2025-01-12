import lightning
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sewar.full_ref import scc as SCC_sewar
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from torchmetrics.image import SpatialCorrelationCoefficient as SCC
from torchvision.transforms.v2.functional import center_crop

# PSNR from: https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html
# Here is how the psnr is calculated:
# psnr_base_e = 2 * torch.log(data_range) - torch.log(mean_squared_error)
# psnr_vals = psnr_base_e * (10 / torch.log(tensor(base)))
# return mean(psnr_vals)


class GeoSR(lightning.LightningModule):
    def __init__(
        self,
        model_class,
        model_params,
        loss_function=nn.MSELoss(),  # or nn.L1Loss()
        spectrum_end=65535,
        # Adam settings
        learning_rate=1e-5,  # Initial learning rate
        betas=(0.9, 0.999),  # I have no idea what it is
        epsilon=1e-8,  # I have no idea what it is
        weight_decay=0,  # Make optimizer forgot old steps, 0 = turn off
        # SCC
        scc_window=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        scc_ws=8,
        scc_on_cpu=False,
        # PSNR
        data_range=65535,  # max pixel values (bit range: 8bit-255, 16bit-65535)
        border_size=0,  # how many pixels to crop befor calculating loss and metrics
        scheduler=None,  # MultiStepLR object or None
        scheduler_MultiStepLR_milestones=None,
        scheduler_MultiStepLR_multiplier=0.1,
        watch=True,
    ):
        super().__init__()
        self.model_class = model_class
        self.model_params = model_params
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.scc_window = scc_window
        self.scc_ws = scc_ws
        self.scc_on_cpu = scc_on_cpu
        self.data_range = data_range
        self.spectrum_end = spectrum_end
        self.border_size = border_size
        self.scheduler_MultiStepLR_milestones = scheduler_MultiStepLR_milestones
        self.scheduler_MultiStepLR_multiplier = scheduler_MultiStepLR_multiplier
        self.watch = watch
        self.save_hyperparameters(ignore=['loss_function'])

        self.model = model_class(**model_params)
    
    def setup(self, stage):
        if self.scc_on_cpu:
            scc_device = "cpu"
        else:
            scc_device = self.device

        self.scc = SCC(
            high_pass_filter=torch.tensor(self.scc_window, device=scc_device),
            window_size=self.scc_ws,
        )

        if isinstance(self.spectrum_end, torch.Tensor):
            self.spectrum_end = self.spectrum_end.to(self.device)
        

    def forward(self, imgs):
        singleImg = False
        if len(imgs.shape) == 3:  # is it a single picture, or a batch
            singleImg = True
            imgs = imgs.view(1, *imgs.shape)

        imgs = self.process(imgs)
        imgs = self.model(imgs)
        imgs = self.inv_process(imgs)
        if singleImg:
            imgs = imgs[0, ...]
        return imgs

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here
        # for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = optim.AdamW(
            self.parameters(),  # Give all weights to optimizer
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            maximize=False,
        )

        if self.scheduler_MultiStepLR_milestones is not None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.scheduler_MultiStepLR_milestones,
                gamma=self.scheduler_MultiStepLR_multiplier,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        lr, hr = batch
        lr = self.process(lr)
        pred_hr = self.model(lr)

        hr = self.remove_border(hr)
        pred_hr = self.remove_border(pred_hr)
        pred_hr = self.inv_process(pred_hr)  # clamp could be removed

        loss = self.loss_function(pred_hr, hr)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        lr = self.process(lr)
        pred_hr = self.model(lr)

        hr = self.remove_border(hr)
        pred_hr = self.remove_border(pred_hr)
        pred_hr = self.inv_process(pred_hr)  # clamp could be removed

        loss = self.loss_function(pred_hr, hr)

        # SCC is quite slow.
        scc = self.mean_scc_over_batch(hr, pred_hr)
        # SSIM takes all the ram. Use with small batch and patch sizes
        # ssim = SSIM(pred_hr.cpu(), hr.cpu(), data_range=self.data_range)
        psnr = PSNR(pred_hr, hr, data_range=self.data_range)

        self.log("val_loss", loss)
        self.log("val_scc", scc)
        self.log("val_psnr", psnr)
        # self.log("val_ssim", ssim)

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        lr = self.process(lr)
        pred_hr = self.model(lr)

        hr = self.remove_border(hr)
        pred_hr = self.remove_border(pred_hr)
        pred_hr = self.inv_process(pred_hr)  # clamp could be removed

        # SCC is quite slow.
        scc = self.mean_scc_over_batch(hr, pred_hr)
        # SSIM takes all the ram. Use with small batch and patch size
        # ssim = SSIM(pred_hr.cpu(), hr.cpu(), data_range=self.data_range)
        psnr = PSNR(pred_hr, hr, data_range=self.data_range)

        self.log("test_scc", scc)
        self.log("test_psnr", psnr)
        # self.log("val_ssim", ssim)

    def predict_step(self, batch, batch_idx):
        lr, hr = batch
        lr = self.process(lr)
        hr_pred = self.model(lr)
        return self.inv_process(hr_pred)

    def on_fit_start(self):
        self.save_hyperparameters()
        if self.watch:
            self.logger.watch(self.model, log="all", log_freq=2, log_graph=True)

    def process(self, imgs):
        imgs = imgs.div(self.spectrum_end)
        return imgs

    def inv_process(self, imgs):
        imgs = imgs.mul(self.spectrum_end)
        imgs = torch.clamp(imgs, 0, self.data_range)
        return imgs

    def mean_scc_over_batch_sewar(self, hr, hr_pred):
        sccs = []
        for hri, hr_predi in zip(hr, hr_pred):
            hri = hri.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
            hr_predi = hr_predi.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
            scc_val = SCC_sewar(hri, hr_predi, win=self.scc_window, ws=self.scc_ws)
            sccs.append(scc_val)
        return np.mean(sccs)

    def mean_scc_over_batch(self, hr, hr_pred):
        sccs = []
        for hri, hr_predi in zip(hr, hr_pred):
            if self.scc_on_cpu:
                hr_predi = hr_predi.cpu()
                hri = hri.cpu()
            scc_val = self.scc(hr_predi, hri)
            sccs.append(scc_val.item())
        return np.mean(sccs)

    def remove_border(self, img):
        if self.border_size > 0:
            after_crop_size = np.array(img.shape[2:]) - self.border_size * 2
            img = center_crop(img, after_crop_size)
        return img
