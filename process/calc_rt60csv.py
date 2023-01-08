import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from pyroomacoustics.experimental.rt60 import measure_rt60
from util import check_format, myround, wavread


def make_rt60csv(rirdir: Path, csv_file: Path, params: dict):
    if csv_file.exists():
        print(f"{csv_file} exists.")
        return
    csv_dict_list = []
    for rir_path in rirdir.glob("*.wav"):
        h, fs, subtype = wavread(rir_path)
        check_format(params, h, fs, subtype)
        rt60 = measure_rt60(h, fs=fs, decay_db=60)
        csv_dict_list.append({"filename": rir_path, "RT60": rt60})
    with open(csv_file, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "RT60"])
        writer.writeheader()
        writer.writerows(csv_dict_list)


def analyze_rt60csv(
    csv_path: Path, save_path_stem: Path, rt60_step: float = 0.1, plot: bool = True
):
    import matplotlib.pyplot as plt

    rt60_bins = np.arange(0, 2.5, rt60_step)
    rt60_cnt = np.zeros(len(rt60_bins) + 1)

    rt60_list = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            measured_rt60 = float(row[1])
            rt60_cnt[np.digitize(measured_rt60, rt60_bins)] += 1
            rt60_list.append(measured_rt60)

    with open(f"{save_path_stem}.txt", mode="w") as f:
        for i in range(len(rt60_cnt)):
            if i == 0:
                bin_text = f"rt60<{rt60_bins[i]:.2f}"
            elif i == len(rt60_cnt) - 1:
                bin_text = f"{rt60_bins[i-1]:.2f}<rt60"
            else:
                bin_text = f"{rt60_bins[i-1]:.2f}<rt60<{rt60_bins[i]:.2f}"
            f.write(f"{bin_text}: {int(rt60_cnt[i])}\n")
        f.write(f"total: {int(np.sum(rt60_cnt))}\n")

    if plot:
        plt.hist(rt60_list, bins=40)
        plt.savefig(f"{save_path_stem}.png")
        plt.clf()
        plt.close()


def main_make_rt60csv_myRIR():
    params = OmegaConf.create({"sr": 16000, "ch": 1, "subtype": "PCM_16"})
    rirdir_parent = Path(
        "/nas03/homes/fujimura22-1000060/linux/project01/mydataset/myRIR"
    )
    csvdir = Path(
        "/nas01/homes/fujimura22-1000060/linux/project01/gitrepo/DataCreation/Data/RT60/myRIR"
    )
    for rirdir in rirdir_parent.glob("*"):
        if rirdir.is_dir():
            csvfile = Path(f"{str(csvdir)}/{rirdir.stem}.csv")
            make_rt60csv(rirdir, csvfile, params)


def main_analyze_rt60csv_myRIR():

    csvdir = Path(
        "/nas01/homes/fujimura22-1000060/linux/project01/gitrepo/DataCreation/Data/RT60/myRIR"
    )
    for csv_path in csvdir.glob("*.csv"):
        save_path_stem = Path(f"{str(csvdir)}/{csv_path.stem}")
        analyze_rt60csv(csv_path, save_path_stem)


def main_make_rt60csv_myRIR2():
    params = OmegaConf.create({"sr": 16000, "ch": 1, "subtype": "PCM_16"})
    rirdir_parent = Path(
        "/nas03/homes/fujimura22-1000060/linux/project01/mydataset/myRIR"
    )
    csvdir = Path(
        "/nas01/homes/fujimura22-1000060/linux/project01/gitrepo/DataCreation/Data/RT60/myRIR"
    )
    for rirdir in rirdir_parent.glob("*"):
        if rirdir.is_dir():
            if str(rirdir.stem)[4:] in ["12", "13", "14", "15", "16", "17"]:
                csvfile = Path(f"{str(csvdir)}/{rirdir.stem}.csv")
                make_rt60csv(rirdir, csvfile, params)


def main_analyze_rt60csv_myRIR2():

    csvdir = Path(
        "/nas01/homes/fujimura22-1000060/linux/project01/gitrepo/DataCreation/Data/RT60/myRIR"
    )
    for csv_path in csvdir.glob("*.csv"):
        if str(csv_path.stem)[4:] in ["12", "13", "14", "15", "16", "17"]:
            save_path_stem = Path(f"{str(csvdir)}/{csv_path.stem}")
            analyze_rt60csv(csv_path, save_path_stem)


if __name__ == "__main__":
    # main_make_rt60csv_myRIR()
    main_make_rt60csv_myRIR2()
    main_analyze_rt60csv_myRIR2()
