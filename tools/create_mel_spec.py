"""Create a mel-spectogram dataset from MP3/WAV file audio.

1.) Convert sampling rate of audio to 16kHz
2.) Pad short audio to 4s long
3.) Extract the spectrogram with the FFT size of 1024, hop size of 256 and
    crop it to a mel-spectrogram of size 80 x 250 (4 seconds)
"""

import argparse
import librosa
import numpy as np
import soundata
from tqdm import tqdm
import torch

from xdiffusion.layers.audio import wav_to_mel

# 62.5 is the mel length for 1 second
MEL_LENGTH_PER_SECOND = 62.5
AUDIO_LENGTH = 4  # Seconds


def create_mel_spectograms(
    output_path: str,
    sample_rate: int = 16000,
    target_mel_length: int = 256,
    num_mel_bins: int = 128,
):
    dataset = soundata.initialize("urbansound8k")
    dataset.download()
    dataset.validate()

    all_clips = dataset.load_clips()

    all_mel = []
    all_labels = []
    for key, audio_clip in tqdm(all_clips.items()):
        audio_path = audio_clip.audio_path
        audio_class_id = audio_clip.class_id

        wav, original_sample_rate = librosa.load(audio_path, sr=sample_rate, mono=True)

        # Pad to 4 seconds long
        wav = librosa.util.fix_length(wav, size=sample_rate * AUDIO_LENGTH)

        # Convert to mel spectrogram
        mel_spec = wav_to_mel(wav, sample_rate=sample_rate, num_mel_bins=num_mel_bins)

        if mel_spec.shape[1] > target_mel_length:
            mel_spec = mel_spec[:, :target_mel_length]
        elif mel_spec.shape[1] < target_mel_length:
            mel_spec = np.pad(
                mel_spec, ((0, 0), (0, target_mel_length - mel_spec.shape[1]))
            )
        assert mel_spec.shape[1] == target_mel_length
        all_mel.append(mel_spec)
        all_labels.append(audio_class_id)

    # Save the output
    np.savez(
        "urban8k.npz", mel_spectrograms=np.array(all_mel), labels=np.array(all_labels)
    )


def main(override=None):
    parser = argparse.ArgumentParser(description="Command line options")
    parser.add_argument("--output_path", type=str, default="mel_spectogram")
    args = parser.parse_args()

    create_mel_spectograms(output_path=args.output_path)


if __name__ == "__main__":
    main()
