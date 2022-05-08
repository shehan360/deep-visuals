from pytorch_lightning.callbacks import ModelCheckpoint
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from SoundNet import SoundNet
import torch
from transformers import TransfoXLConfig, TransfoXLModel

class VideoDiffusion(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Audio
        self.soundNet = SoundNet()
        self.soundNet.load_state_dict(torch.load('soundnet.pth'))
        self.audio_lin = torch.nn.Linear(1024, 512)

        #Transformer
        configuration = TransfoXLConfig(d_model=512, d_embed=512, n_head=8, d_head=64, d_inner=2048, n_layer=12,
                                        mem_len=512)
        self.transformer = TransfoXLModel(configuration)


        model = Unet3D(
            dim=64,
            cond_dim=512,
            dim_mults=(1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            model,
            image_size=args.resolution,
            num_frames=args.sequence_length,
            timesteps=args.timesteps,  # number of steps
            loss_type=args.loss  # L1 or L2
        )

    def get_audio_embedding(self, audio):
        audio_embedding = self.soundNet(audio)
        audio_embedding = torch.nn.functional.max_pool2d(audio_embedding.squeeze(), 2)
        audio_embedding = torch.flatten(audio_embedding, start_dim=1)
        audio_embedding = self.audio_lin(audio_embedding)
        return audio_embedding

    def forward(self, video, audio):
        audio_embedding = self.soundNet(audio)
        audio_embedding = torch.nn.functional.max_pool2d(audio_embedding.squeeze(), 2)
        audio_embedding = torch.flatten(audio_embedding, start_dim=1)
        audio_embedding = self.audio_lin(audio_embedding)

        audio_tansfomed_embedding = self.transformer(inputs_embeds=audio_embedding.unsqueeze(0))

        loss = self.diffusion(video, cond=audio_tansfomed_embedding['last_hidden_state'].squeeze())
        return loss

    def training_step(self, batch, batch_idx):
        video = batch['video']
        audio = batch['audio']

        loss = self(video, audio)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.args.lr)
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

