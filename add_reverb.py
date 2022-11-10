import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.speech_augmentation import AddReverb


def main(args):
    # signal = read_audio("/Users/fujimuratakuya/Desktop/p232_001.wav")
    # clean = signal.unsqueeze(0)  # [batch, time, channels]
    # clean, fs = torchaudio.load("/Users/fujimuratakuya/Desktop/p232_002.wav")
    # [ch, time]. but ch is 1, so it can be viewed as [batch, time] where batch=1.
    reverb = AddReverb(args.csv_file)
    # reverbed = reverb(clean, torch.ones(1))
    # torchaudio.save("./reverbed.wav", reverbed, fs, encoding="PCM_S", bits_per_sample=16)
    dir_path = Path(args.dir_path)
    for aud_path in dir_path.glob("*.wav"):
        speech, fs = torchaudio.load(aud_path)
        meta_data = torchaudio.info(aud_path)
        reverbed = reverb(speech, torch.ones(1))
        torchaudio.save(
            f"{args.save_dir_path}/{aud_path.name}",
            reverbed,
            fs,
            encoding=meta_data.encoding,
            bits_per_sample=meta_data.bits_per_sample,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="csv file of the rir")
    parser.add_argument("dir_path", type=str, help="target speech directory")
    parser.add_argument(
        "save_dir_path",
        type=str,
        help="directory where the reverbed signal is saved on",
    )
    args = parser.parse_args()
    main(args)
