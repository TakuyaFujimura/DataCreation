import argparse
import os
import random
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torchaudio
from speechbrain.processing.speech_augmentation import AddReverb
from util import check_format

# python add_reverb.py ./rir_list.csv ../../00ASJ2022autumn/dataset/LibriSpeech_train_clean_100_10000 ../../00ASJ2022autumn/dataset/LibriSpeech_train_clean_100_10000_reverb


class MyAddReverb:
    def __init__(self, rir_scale_factors, csv_file, rir_folder, reverb_sr, clean_sr):
        self.reverb_funcs = torch.nn.ModuleList([])
        self.rir_scale_factors = rir_scale_factors
        for scale in rir_scale_factors:
            self.reverb_funcs.append(
                AddReverb(
                    csv_file,
                    rir_scale_factor=scale,
                    replacements={"rir_folder": rir_folder},
                    reverb_sample_rate=reverb_sr,
                    clean_sample_rate=clean_sr,
                )
            )

    def reverb(self, speech):
        """Returns a reverbed speech. A more efficient implementation is desirable.
        cfg:
            speech (tensor): speech_batch. (n_batch, n_time)
        Returns:
            reverbed (tensor): reverbed speech. (n_batch, n_time)
        """
        reverbed = speech.new_zeros(speech.shape)
        for i in range(len(speech)):
            func_id = random.randint(0, len(self.rir_scale_factors) - 1)
            reverbed[i] = self.reverb_funcs[func_id](speech[i][None], torch.ones(1))[0]
        return reverbed


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)
    reverber = MyAddReverb(
        cfg.rir_scale_factors,
        cfg.csv_file,
        cfg.rir_folder,
        cfg.reverb_sr,
        cfg.clean_sr,
    )

    dir_path = Path(cfg.dir_path)
    save_dir = Path(cfg.save_dir_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    with open(f"{str(save_dir)}/readme.txt", "a") as f:
        f.write("This dataset is created by DataCreation/add_reverb.py\n")
        f.write(f"csv_file: {cfg.csv_file}\n")
        f.write(f"dir_path: {cfg.dir_path}\n")
        f.write(f"save_dir_path: {cfg.save_dir_path}\n")
        f.write(f"rir_folder: {cfg.rir_folder}\n")
        f.write(f"rir_scale_factors: {cfg.rir_scale_factors}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"For the detail, please refer the DataCreation/hydra_data/{cfg.name}")

    for aud_path in dir_path.glob("*.wav"):
        speech, sr = torchaudio.load(aud_path)
        # [ch, time]. but ch is 1, so it can be viewed as [batch, time] where batch=1.
        check_format(cfg.old_params, speech, sr)
        meta_data = torchaudio.info(aud_path)
        reverbed = reverber.reverb(speech)
        torchaudio.save(
            save_dir / aud_path.name,
            reverbed,
            cfg.new_params.sr,
            encoding=meta_data.encoding,
            bits_per_sample=meta_data.bits_per_sample,
        )


if __name__ == "__main__":
    main()
