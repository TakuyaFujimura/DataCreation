import argparse
import csv
from pathlib import Path

import torchaudio

# python make_csv.py ./rir_list.csv


def make_csv(csv_file, filename_list):
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    for filename in filename_list:
        rir_path = filename
        meta_data = torchaudio.info(rir_path)
        if meta_data.num_channels != 1:
            print(rir_path)
        duration = meta_data.num_frames / meta_data.sample_rate
        with open(csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([i, duration, rir_path, "wav", ""])
        i += 1


def A(args):
    with open(args.csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    with open("rir_list.txt", "r") as f:
        rir_path_list = f.read().split()[4::5]
    i = 0
    dst_dir = Path("RIRS1ch")
    for rir_path in rir_path_list:
        meta_data = torchaudio.info(rir_path)
        rir, fs = torchaudio.load(rir_path)
        if len(rir) != 1:
            rir = rir.mean(dim=0, keepdim=True)
        torchaudio.save(
            dst_dir / Path(rir_path).name,
            rir,
            fs,
            encoding=meta_data.encoding,
            bits_per_sample=meta_data.bits_per_sample,
        )
        duration = meta_data.num_frames / meta_data.sample_rate
        with open(args.csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [i, duration, f"{dst_dir / Path(rir_path).name}", "wav", ""]
            )
        i += 1


def B(args):
    with open(args.csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    src_dir = Path("RIRS_NOISES/simulated_rirs/largeroom")
    for rir_path in src_dir.glob("*/*.wav"):
        meta_data = torchaudio.info(rir_path)
        if meta_data.num_channels != 1:
            print(rir_path)
        duration = meta_data.num_frames / meta_data.sample_rate
        with open(args.csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([i, duration, rir_path, "wav", ""])
        i += 1


def C(args):
    with open(args.csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    with open("RIRS1ch_real_RT60_1_25_files.csv") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            rir_path = f"RIRS1ch/{str(row[0])}"
            meta_data = torchaudio.info(rir_path)
            if meta_data.num_channels != 1:
                print(rir_path)
            duration = meta_data.num_frames / meta_data.sample_rate
            with open(args.csv_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([i, duration, rir_path, "wav", ""])
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="save name of the csv file")
    # parser.add_argument("dir_path", type=str, help="path to the RIR_NOISES")
    args = parser.parse_args()
    C(args)
