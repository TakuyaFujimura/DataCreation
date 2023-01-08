import random
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import soundfile as sf
import torch
from omegaconf import DictConfig


def make_readme(savedir, exec_filename, hydra_name):
    with open(f"{str(savedir)}/readme.txt", "a") as f:
        f.write(f"This dataset is created by {exec_filename}\n")
        f.write(
            f"For the detail, please refer the DataCreation/hydra_data/{hydra_name}"
        )


def myround(x, point="0.01"):
    x_rounded = Decimal(str(x)).quantize(Decimal(point), rounding=ROUND_HALF_UP)
    return x_rounded


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
    c = np.sqrt(S / (N * 10 ** (snr / 10)))
    return c


def randomcrop_repeat(data, crop_len):
    dif = len(data) - crop_len
    if dif < 0:
        data = np.tile(data, int(np.ceil(crop_len / len(data))))
        dif = len(data) - crop_len
    start = random.randint(0, dif)
    return data[start : start + crop_len]


def check_format(params: DictConfig, data, sr=None, subtype=None):
    if getattr(params, "ch", None) is not None:
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 1:
                assert 1 == params.ch
            else:
                assert data.shape[0] == params.ch
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                assert 1 == params.ch
            else:
                assert data.shape[1] == params.ch
        else:
            raise ValueError(f"{type(data)} is not supported.")
    if (getattr(params, "sr", None) is not None) and (sr is not None):
        assert sr == params.sr
    if (getattr(params, "subtype", None) is not None) and (subtype is not None):
        assert subtype == params.subtype
