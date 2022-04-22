from pytorch_lightning.callbacks import ModelCheckpoint
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl


class VideoDiffusion(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        model = Unet3D(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            model,
            image_size=args.resolution,
            num_frames=args.sequence_length,
            timesteps=args.timesteps,  # number of steps
            loss_type=args.loss  # L1 or L2
        )

    def forward(self, x):
        loss = self.diffusion(x)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch['video']

        loss = self(x)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.lr)
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

