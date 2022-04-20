import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import torch.optim.lr_scheduler as lr_scheduler
from data import VideoData

import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

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


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--loss', type=str, default='l1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    model = VideoDiffusion(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1))

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps)

    if args.ckpt_path:
        print("Loading from checkpoint!!!")
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, data)


if __name__ == '__main__':
    main()
