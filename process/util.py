import random

import numpy as np
import soundfile as sf
import torch


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


def check_format(old_params, data, sr_old=None, subtype_old=None):
    if getattr(old_params, "ch", None) is not None:
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 1:
                assert 1 == old_params.ch
            else:
                assert data.shape[0] == old_params.ch
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                assert 1 == old_params.ch
            else:
                assert data.shape[1] == old_params.ch
        else:
            raise ValueError(f"{type(data)} is not supported.")
    if (getattr(old_params, "sr", None) is not None) and (sr_old is not None):
        assert sr_old == old_params.sr
    if (getattr(old_params, "subtype", None) is not None) and (subtype_old is not None):
        assert subtype_old == old_params.subtype
