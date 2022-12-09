from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
from util import cal_adjusted_rms, check_format, randomcrop_repeat, wavread, wavwrite


class PickupFilepath:
    def __init__(self, noise_dir: Path):
        self.noise_dir = noise_dir
        self.noise_path_list = sorted(list(noise_dir.glob("*.wav")))

    def randomly_get(self):
        n_idx = np.random.randint(0, len(self.noise_path_list))
        return self.noise_path_list[n_idx]

    def get_paired(self, filepath: Path):
        return self.noise_dir / filepath.name


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)
    clean_dir = Path(cfg.clean_dir)
    noise_dir = Path(cfg.noise_dir)
    clean_path_list = sorted(list(clean_dir.glob("*.wav")))
    pickup = PickupFilepath(noise_dir)
    save_dir = Path(cfg.save_dir_path)
    save_dir.mkdir(parents=True, exist_ok=False)
    snr_list = cfg.snr_list
    with open(f"{str(save_dir)}/readme.txt", "a") as f:
        f.write("This dataset is created by DataCreation/add_noise.py\n")
        f.write(f"clean_dir: {cfg.clean_dir}\n")
        f.write(f"noise_dir: {cfg.noise_dir}\n")
        f.write(f"save_dir_path: {cfg.save_dir_path}\n")
        f.write(f"snr_list: {cfg.snr_list}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"For the detail, please refer the DataCreation/hydra_data/{cfg.name}")
        # f.write("Only this time, snr=random.uniform(-5,5)\n")
    for clean_path in clean_path_list:
        s, sr, subtype = wavread(clean_path)
        check_format(cfg.old_params, s, sr, subtype)
        snr = np.random.choice(snr_list)
        # snr = random.uniform(-5, 5)
        if cfg.pickup_mode == "random":
            noise_path = pickup.randomly_get()
        elif cfg.pickup_mode == "pair":
            noise_path = pickup.get_paired(clean_path)
        else:
            raise ValueError(f"{cfg.pickup_mode} is not supported.")
        n, sr, subtype = wavread(noise_path)
        check_format(cfg.old_params, n, sr, subtype)
        n = randomcrop_repeat(n, len(s))
        assert len(s) == len(n)
        x = s + cal_adjusted_rms(s, n, snr) * n
        wavwrite(
            save_dir / clean_path.name,
            x,
            cfg.new_params.sr,
            cfg.new_params.subtype,
        )


if __name__ == "__main__":
    main()
