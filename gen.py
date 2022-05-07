import torch
import argparse
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from matplotlib import animation
from VideoDiffusion import VideoDiffusion
from pydub import AudioSegment
import os
import numpy as np
from librosa.util import fix_length

pl.seed_everything(1234)

parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('--loss', type=str, default='l1')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--cond_data', type=str, default="visuals-dataset-validate24/validate")
args = parser.parse_args()

kwargs = {'args': args}
model = VideoDiffusion.load_from_checkpoint(checkpoint_path=args.ckpt_path, **kwargs)

videos = os.listdir(args.cond_data)
count = 0
audio_arr = []
for video_name in sorted(videos):
    if video_name.endswith('mp4'):
        audio_segment = AudioSegment.from_file(args.cond_data + "/" + video_name, "mp4")
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(22050)

        channel_sounds = audio_segment.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]

        audio = np.array(samples).T.astype(np.float32)
        audio /= np.iinfo(samples[0].typecode).max
        audio = audio.reshape(-1)
        audio = fix_length(audio, size=5 * 22050)
        audio *= 256.0
        audio = np.reshape(audio, (1, -1, 1))
        audio_arr.append(torch.Tensor(audio))

audio_cond = torch.cat(audio_arr).unsqueeze(dim=1)
audio_embedding = model.get_audio_embedding(audio_cond)

videos = model.diffusion.sample(batch_size=4, cond=audio_embedding)

videos = ((videos + 0.5) * 255).cpu().numpy().astype('uint8')

fig = plt.figure()
plt.axis('off')
im = plt.imshow(videos[0, :, :, :])

def init():
    im.set_data(videos[0, :, :, :])

def animate(i):
    im.set_data(videos[i, :, :, :])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=videos.shape[0], interval=40)  # 200ms = 5 fps
plt.show()
