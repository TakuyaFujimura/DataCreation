import argparse
import csv
from pathlib import Path

import torchaudio


def main(args):
    with open(args.csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    with open("rir_list.txt", "r") as f:
        rir_path_list = f.read().split()[4::5]
    i = 0
    for rir_path in rir_path_list:
        meta_data = torchaudio.info(rir_path)
        duration = meta_data.num_frames / meta_data.sample_rate
        writer.writerow([i, duration, rir_path, "wav", ""])
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="save name of the csv file")
    args = parser.parse_args()
    main(args)
