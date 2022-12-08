import argparse
import csv
import random
from pathlib import Path

import torchaudio

# python make_csv_sep.py


def make_csv(rir_path_list, csv_file, replaces=None):
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    for rir_path in rir_path_list:
        meta_data = torchaudio.info(rir_path)
        duration = meta_data.num_frames / meta_data.sample_rate
        with open(csv_file, "a") as f:
            writer = csv.writer(f)
            if replaces is None:
                writer.writerow([i, duration, f"{rir_path}", "wav", ""])
            else:
                rir_path = str(rir_path).replace(replaces[0], replaces[1])
                writer.writerow([i, duration, f"{rir_path}", "wav", ""])
        i += 1


def typeA():
    dst_dir = Path("RIRS1ch")
    rir_path_list = list(dst_dir.glob("*.wav"))
    assert len(rir_path_list) == 325
    all_index = list(range(325))
    random.shuffle(all_index)
    rir1_index = all_index[:155]
    rir2_index = all_index[155:310]
    rirtest_index = all_index[310:]
    rir1_path_list = [rir_path_list[i] for i in rir1_index]
    rir2_path_list = [rir_path_list[i] for i in rir2_index]
    rirtest_path_list = [rir_path_list[i] for i in rirtest_index]

    make_csv(dst_dir, rir1_path_list, "rir1_list.csv")
    make_csv(dst_dir, rir2_path_list, "rir2_list.csv")
    make_csv(dst_dir, rirtest_path_list, "rirtest_list.csv")


def typeB():
    dst_dir = Path("RIRS_NOISES")
    room_sizes = ["large", "medium", "small"]
    rir_path_dict = {"rir_add": [], "rir_obs": [], "rir_test": []}

    # rir_add
    for size in room_sizes:
        for i in range(1, 101):
            rir_path = random.choice(
                list(dst_dir.glob(f"simulated_rirs/{size}room/Room{i:03}/*.wav"))
            )
            rir_path_dict["rir_add"].append(rir_path)
    # rir_obs
    for size in room_sizes:
        for i in range(101, 131):
            rir_path = random.choice(
                list(dst_dir.glob(f"simulated_rirs/{size}room/Room{i:03}/*.wav"))
            )
            rir_path_dict["rir_obs"].append(rir_path)
    # rir_test
    for size in room_sizes:
        for i in range(196, 201):
            rir_path = random.choice(
                list(dst_dir.glob(f"simulated_rirs/{size}room/Room{i:03}/*.wav"))
            )
            rir_path_dict["rir_test"].append(rir_path)
    make_csv(
        rir_path_dict["rir_add"],
        "rircsv/rir_add_sim.csv",
        replaces=["RIRS_NOISES", "$rir_folder"],
    )
    make_csv(
        rir_path_dict["rir_obs"],
        "rircsv/rir_obs_sim.csv",
        replaces=["RIRS_NOISES", "$rir_folder"],
    )
    make_csv(
        rir_path_dict["rir_test"],
        "rircsv/rir_test_sim.csv",
        replaces=["RIRS_NOISES", "$rir_folder"],
    )


def main():
    random.seed(1234)
    typeB()


if __name__ == "__main__":
    main()
