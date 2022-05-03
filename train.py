import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data import VideoData
from VideoDiffusion import VideoDiffusion


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
