import argparse
import csv
import random
from pathlib import Path

import torchaudio

# python make_csv_sep.py


def make_csv(dst_dir, rir_path_list, csv_file):
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    for rir_path in rir_path_list:
        meta_data = torchaudio.info(rir_path)
        duration = meta_data.num_frames / meta_data.sample_rate
        with open(csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [i, duration, f"{dst_dir / Path(rir_path).name}", "wav", ""]
            )
        i += 1


def main():
    random.seed(1234)
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


if __name__ == "__main__":
    main()
