import argparse
import os
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from speechbrain.processing.speech_augmentation import AddReverb

# python add_reverb_test.py ./rircsv/tmp.csv ../../00ASJ2022autumn/dataset/LibriSpeech_train_clean_100_10000/19-198-0032.wav ./test --rir_scale_factors 1
# python add_reverb.py ./rircsv/rir_list.csv ../../00ASJ2022autumn/dataset/LibriSpeech_train_noisy_100_10000_CHiME1 ../../00ASJ2022autumn/dataset/LibriSpeech_train_noisy_100_10000_CHiME1_reverb --rir_scale_factors 1
#0,1.0,RIRS1ch/RVB2014_type1_rir_largeroom1_far_angla.wav,wav,
#18,1.0,RIRS1ch/RVB2014_type1_rir_smallroom1_near_angla.wav,wav,

class MyAddReverb:
    def __init__(self, rir_scale_factors, csv_file):
        self.reverb_funcs = torch.nn.ModuleList([])
        self.rir_scale_factors = rir_scale_factors
        for scale in rir_scale_factors:
            self.reverb_funcs.append(
                AddReverb(
                    csv_file,
                    rir_scale_factor=scale,
                    replacements={
                        "rir_folder": "/nas01/homes/fujimura22-1000060/linux/project01/gitrepo/DataCreation/RIRS1ch"
                    },
                )
            )

    def reverb(self, speech):
        """Returns a reverbed speech. A more efficient implementation is desirable.
        Args:
            speech (tensor): speech_batch. (n_batch, n_time)
        Returns:
            reverbed (tensor): reverbed speech. (n_batch, n_time)
        """
        reverbed = speech.new_zeros(speech.shape)
        for i in range(len(speech)):
            func_id = random.randint(0, len(self.rir_scale_factors) - 1)
            reverbed[i] = self.reverb_funcs[func_id](speech[i][None], torch.ones(1))[0]
        return reverbed


def main(args):
    seed = 1234
    pl.seed_everything(seed)
    reverber = MyAddReverb(args.rir_scale_factors, args.csv_file)

    nas03_dir = os.getcwd().replace("nas01", "nas03")
    aud_path = Path(nas03_dir) / Path(args.aud_path)
    save_dir = Path(args.save_dir_path)

    speech, fs = torchaudio.load(aud_path)
    # [ch, time]. but ch is 1, so it can be viewed as [batch, time] where batch=1.
    meta_data = torchaudio.info(aud_path)
    reverbed = reverber.reverb(speech)
    torchaudio.save(
        save_dir / aud_path.name,
        reverbed,
        fs,
        encoding=meta_data.encoding,
        bits_per_sample=meta_data.bits_per_sample,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file of the rir")
    parser.add_argument("aud_path", type=str, help="target speech path")
    parser.add_argument(
        "save_dir_path",
        type=str,
        help="directory where the reverbed signal is saved on",
    )
    parser.add_argument(
        "--rir_scale_factors",
        required=True,
        nargs="+",
        type=float,
        help="list of the rir scale factors",
    )
    args = parser.parse_args()
    main(args)
