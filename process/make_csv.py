import csv
from pathlib import Path

import numpy as np
import torchaudio


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


def merge_csv(save_name, csv_list):
    with open(save_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    i = 0
    for csvfile in csv_list:
        with open(csvfile) as f:
            reader = csv.reader(f)
            _ = next(reader)
            with open(save_name, "a") as f:
                writer = csv.writer(f)
                for row in reader:
                    writer.writerow([i, row[1], row[2], row[3], row[4]])
                    i += 1


# ====================main functions====================#


def main_myRIR_room():
    """Make csvfile for addReverb function.
    csvfile will be created for each room.
    """
    csvdir_path = Path(
        "/home/fujimura22/nas01home/linux/project01/gitrepo/DataCreation/Data/myRIR"
    )
    for csv_path in csvdir_path.glob("*.csv"):
        # make csv file of each room for addReverb.
        filename_list = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                filename_list.append(row[0])  # rir_path
        addReverb_csvfile_dir = Path(
            "/home/fujimura22/nas01home/linux/project01/gitrepo/DataCreation/Data/myRIR/addReverb_csv"
        )
        addReverb_csvfile = addReverb_csvfile_dir / csv_path.name
        make_csv(addReverb_csvfile, filename_list)


def main_myRIR_obs_RT60():
    """Make csvfile for addReverb function.
    csvfile will be created on room3-4(rir_obs) and
    it'll be created for each RT60 interval.
    """
    csvdir_path = Path(
        "/home/fujimura22/nas01home/linux/project01/gitrepo/DataCreation/Data/myRIR"
    )
    filename_rt60_list = [[] for _ in range(13)]  # "0.2-0.3" to "1.4-1.5"
    for room_n in [3, 4]:
        csv_path = csvdir_path / f"room{room_n}.csv"
        # make csv file of each room for addReverb.
        rt60_borders = np.arange(0.2, 1.6, 0.1)
        with open(csv_path) as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                rir_path, rt60 = row
                rt60 = float(rt60)
                if 0.2 <= rt60 and rt60 < 1.5:
                    idx = np.digitize(rt60, rt60_borders) - 1
                    filename_rt60_list[idx].append(rir_path)

        addReverb_csvfile_dir = Path(
            "/home/fujimura22/nas01home/linux/project01/gitrepo/DataCreation/Data/myRIR/addReverb_csv"
        )
    for i in range(13):
        lower = "".join((f"{0.2 + 0.1 * i :.1f}").split("."))
        upper = "".join((f"{0.3 + 0.1 * i :.1f}").split("."))
        name = f"rirobs_{lower}-{upper}.csv"  # lower<=rt60<lower+0.1
        addReverb_csvfile = addReverb_csvfile_dir / name
        make_csv(addReverb_csvfile, filename_rt60_list[i])


def main_merge_csv():
    addReverb_csvfile_dir = Path(
        "/home/fujimura22/nas01home/linux/project01/gitrepo/DataCreation/Data/myRIR/addReverb_csv"
    )
    save_csv_name = addReverb_csvfile_dir / "rirtest.csv"
    csv_list = [
        addReverb_csvfile_dir / "room5.csv",
        addReverb_csvfile_dir / "room6.csv",
    ]
    merge_csv(save_csv_name, csv_list)


if __name__ == "__main__":
    # main_myRIR_room()
    # main_myRIR_obs_RT60()
    main_merge_csv()
