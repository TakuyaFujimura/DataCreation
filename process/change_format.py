from pathlib import Path

import hydra
import numpy as np
from pytorch_lightning import seed_everything
from scipy import signal
from util import wavread, wavwrite


class ChangeFormat:
    def __init__(self, new_params: dict, old_params: dict):
        self.new_params = new_params
        self.old_params = old_params

    def resample(self, data):
        data = signal.resample(
            data, int(len(data) * self.new_params.sr / self.old_params.sr)
        )
        return data

    def compress_ch(self, data):
        data = np.mean(data, axis=1)
        return data


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    seed_everything(cfg.seed)
    old_params = cfg.old_params
    new_params = cfg.new_params
    changer = ChangeFormat(new_params, old_params)
    target_dir = Path(cfg.target_dir)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)
    with open(f"{str(save_dir)}/readme.txt", "a") as f:
        f.write("This dataset is created by DataCreation/process/change_format.py\n")
        f.write(f"target_dir: {cfg.target_dir}\n")
        f.write(f"save_dir: {cfg.save_dir}\n")
        f.write(f"seed: {cfg.seed}\n")

    for aud_path in target_dir.glob("*.wav"):
        data, sr_old, subtype_old = wavread(aud_path)
        # confirm whether the data format is correct
        if old_params.ch is not None:
            if len(data.shape) == 1:
                assert 1 == old_params.ch
            else:
                assert data.shape[1] == old_params.ch
        if old_params.sr is not None:
            assert sr_old == old_params.sr
        if old_params.subtype is not None:
            assert subtype_old == old_params.subtype

        # change the format
        if cfg.proc_func_names is not None:
            for func_name in cfg.proc_func_names:
                if func_name == "resample":
                    data = changer.resample(data)
                elif func_name == "compress_ch":
                    data = changer.compress_ch(data)
                else:
                    raise ValueError(f"{func_name} is not supported")

        wavwrite(
            save_dir / aud_path.name,
            data,
            new_params.sr,
            new_params.subtype,
        )


if __name__ == "__main__":
    main()
