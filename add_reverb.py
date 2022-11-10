import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import AddReverb

# python add_reverb.py ./rir_list.csv /home/fujimura/nas03/home/linux/project01/00ASJ2022autumn/dataset/LibriSpeech_train_clean_100_10000 /home/fujimura/nas03/home/linux/project01/00ASJ2022autumn/dataset/LibriSpeech_train_clean_100_10000_reverb


def main(args):
    # """
    pl.seed_everything(1234)
    reverb = AddReverb(args.csv_file)
    dir_path = Path(args.dir_path)
    save_dir = Path(args.save_dir_path)
    # save_dir.mkdir(parents=True, exist_ok=True)
    for aud_path in dir_path.glob("*.wav"):
        speech, fs = torchaudio.load(aud_path)
        # [ch, time]. but ch is 1, so it can be viewed as [batch, time] where batch=1.
        meta_data = torchaudio.info(aud_path)
        breakpoint()
        # reverbed = reverb(speech, torch.ones(1))
        # print(reverbed)
    # """
    """
        torchaudio.save(
            save_dir_path / aud_path.name,
            reverbed,
            fs,
            encoding=meta_data.encoding,
            bits_per_sample=meta_data.bits_per_sample,
        )
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file of the rir")
    parser.add_argument("dir_path", type=str, help="target speech directory")
    parser.add_argument(
        "save_dir_path",
        type=str,
        help="directory where the reverbed signal is saved on",
    )
    args = parser.parse_args()
    main(args)
