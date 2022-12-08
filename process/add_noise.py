import argparse
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import soundfile as sf


def wavread(filepath):
    data, sr = sf.read(filepath)
    f = sf.info(filepath)
    return data, sr, f.subtype

def wavwrite(filepath, data, sr, subtype):
    sf.write(filepath, data, sr, subtype)

def cal_adjusted_rms(s, n, snr):
    s_amp = np.abs(s)
    S = np.mean(s_amp**2)
    n_amp = np.abs(n)
    N = np.mean(n_amp**2)
    c = np.sqrt(S/(N*10**(snr/10)))
    return c

def randomcrop_repeat(data, crop_len):
    dif = len(data) - crop_len
    if dif < 0:
        data = np.tile(data, int(np.ceil(crop_len / len(data))))
        dif = len(data) - crop_len
    start = random.randint(0, dif)
    return data[start : start + crop_len]


def main(args):
    seed = 1234
    pl.seed_everything(seed)
    nas03_dir = os.getcwd().replace("nas01", "nas03")
    clean_dir = Path(nas03_dir) / Path(args.clean_dir)
    noise_dir = Path(nas03_dir) / Path(args.noise_dir)
    clean_path_list = sorted(list(clean_dir.glob("*.wav")))
    noise_path_list = sorted(list(noise_dir.glob("*.wav")))
    save_dir = Path(nas03_dir) / Path(args.save_dir_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    snr_list = args.snr_list
    with open(f"{str(save_dir)}/readme.txt", "a") as f:
        f.write("This dataset is created by DataCreation/add_noise.py\n")
        f.write(f"nas03_dir: {nas03_dir}\n")
        f.write(f"clean_dir: {args.clean_dir}\n")
        f.write(f"noise_dir: {args.noise_dir}\n")
        f.write(f"save_dir_path: {args.save_dir_path}\n")
        f.write(f"snr_list: {args.snr_list}\n")
        f.write(f"seed: {seed}\n")
        # f.write("Only this time, snr=random.uniform(-5,5)\n")
    for clean_path in clean_path_list:
        #print(f"\r{i}/{len(clean_path_list)}", end="")
        s, sr, subtype = wavread(clean_path)
        snr = np.random.choice(snr_list)
        # snr = random.uniform(-5, 5)
        n_idx = np.random.randint(0, len(noise_path_list))
        n, sr, subtype = wavread(noise_path_list[n_idx])
        n = randomcrop_repeat(n, len(s))
        assert len(s)==len(n)
        x = s + cal_adjusted_rms(s, n, snr) * n
        wavwrite(
            save_dir / clean_path.name,
            x,
            sr,
            subtype,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_dir", type=str)
    parser.add_argument("noise_dir", type=str)
    parser.add_argument("save_dir_path",type=str)
    parser.add_argument(
        "--snr_list",
        required=True,
        nargs="+",
        type=float,
        help="list of the snr",
    )
    args = parser.parse_args()
    main(args)
