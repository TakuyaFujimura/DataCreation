import random

import numpy as np
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
    c = np.sqrt(S / (N * 10 ** (snr / 10)))
    return c


def randomcrop_repeat(data, crop_len):
    dif = len(data) - crop_len
    if dif < 0:
        data = np.tile(data, int(np.ceil(crop_len / len(data))))
        dif = len(data) - crop_len
    start = random.randint(0, dif)
    return data[start : start + crop_len]
