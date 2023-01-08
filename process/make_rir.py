import random
from pathlib import Path

import hydra
import numpy as np
import pyroomacoustics as pra
import pytorch_lightning as pl
from tqdm import tqdm
from util import make_readme, wavwrite


def make_room(rt60tgt: float, roomdim: list, fs: float) -> pra.room.ShoeBox:
    """Returns a room which satisfies a rt60 and roomdim constraint using Sabine's Formula.
    cfg:
        rt60tgt (float): Target RT60
        roomdim (list): Room size
        fs (float): sampling frequency
    Returns:
        src_loc (numpy.ndarray): Location of sound source
        mic_loc (numpy.ndarray): Location of microphone
    """
    e_absorption, max_order = pra.inverse_sabine(rt60tgt, roomdim)
    room = pra.ShoeBox(
        roomdim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        use_rand_ism=True,
    )
    return room


def get_loc(roomdim: list, r: float = 1.0):
    """Returns a randomly selected locations of source and microphone with fixed distance.
    cfg:
        roomdim (list): Room size
        r (float): Fixed distance between source and microphone.
    Returns:
        room (pyroomacoustics.room.ShoeBox): A room created so that satisfies rt60tgt and roomdim.
    """
    # depth * width * height
    roomdim = np.array(roomdim)
    border = np.zeros(3)

    src_loc = np.array(
        [
            random.uniform(0, roomdim[0]),
            random.uniform(0, roomdim[1]),
            random.uniform(0, roomdim[2]),
        ]
    )
    mic_loc = np.array(roomdim) + 1  # initialization
    while np.any(mic_loc > roomdim) or np.any(mic_loc < border):
        elevation = random.uniform(0, np.pi)
        azimuth = random.uniform(0, 2 * np.pi)
        delta_loc = np.array(
            [
                r * np.sin(elevation) * np.cos(azimuth),
                r * np.sin(elevation) * np.sin(azimuth),
                r * np.cos(elevation),
            ]
        )
        mic_loc = src_loc + delta_loc

    return src_loc, mic_loc


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)
    fs = cfg.sr
    rt60tgt_list = cfg.rt60tgt_list
    roomdim_dict = cfg.roomdim_dict.items()
    n_samples = cfg.n_samples
    save_dir_parent = Path(cfg.save_dir_path)

    save_dir_parent.mkdir(parents=False, exist_ok=getattr(cfg, "exist_ok", False))
    make_readme(save_dir_parent, "DataCreation/process/make_rir.py", cfg.name)
    if cfg.tqdm:
        roomdim_dict = tqdm(roomdim_dict)
    for roomname, roomdim in roomdim_dict:
        save_dir = save_dir_parent / roomname
        save_dir.mkdir(parents=False, exist_ok=False)
        for rt60tgt in rt60tgt_list:
            for i in range(n_samples):
                room = make_room(rt60tgt, np.array(roomdim), fs)
                src_loc, mic_loc = get_loc(roomdim)
                room.add_source(src_loc)
                room.add_microphone(mic_loc)
                room.compute_rir()
                rir = room.rir[0][0]
                rt60tgt_str = "".join(str(rt60tgt).split("."))
                wavwrite(
                    save_dir / f"{rt60tgt_str}_{i:03}.wav",
                    rir,
                    fs,
                    cfg.subtype,
                )


if __name__ == "__main__":
    main()
