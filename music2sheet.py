import argparse
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


A4_FREQ = 440.0
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class NoteEvent:
    start_time: float
    end_time: float
    note: str
    frequency: float


def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(num_frames)

    if sample_width == 1:
        dtype = np.uint8
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        dtype = np.int16
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data /= 32768.0
    elif sample_width == 4:
        dtype = np.int32
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    return data, sample_rate


def read_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".wav":
        return read_wav_mono(path)

    if suffix == ".mp3":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to read MP3 files but was not found.")

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "converted.wav"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(file_path), str(wav_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return read_wav_mono(str(wav_path))

    raise ValueError(f"Unsupported audio format: {suffix}")


def freq_to_note_name(freq: float) -> str:
    if freq <= 0:
        return "Rest"
    midi = int(round(69 + 12 * math.log2(freq / A4_FREQ)))
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


def extract_pitch_events(
    samples: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    min_freq: float = 60.0,
    max_freq: float = 2000.0,
) -> List[NoteEvent]:
    window = np.hanning(frame_size)
    events: List[NoteEvent] = []

    last_note = None
    last_start = 0.0
    last_freq = 0.0

    for frame_start in range(0, len(samples) - frame_size, hop_size):
        frame = samples[frame_start : frame_start + frame_size]
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

        valid = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(valid):
            continue

        peak_idx = np.argmax(magnitudes[valid])
        peak_freq = freqs[valid][peak_idx]
        note_name = freq_to_note_name(peak_freq)

        timestamp = frame_start / sample_rate
        if last_note is None:
            last_note = note_name
            last_start = timestamp
            last_freq = peak_freq
        elif note_name != last_note:
            events.append(
                NoteEvent(
                    start_time=last_start,
                    end_time=timestamp,
                    note=last_note,
                    frequency=last_freq,
                )
            )
            last_note = note_name
            last_start = timestamp
            last_freq = peak_freq

    end_time = len(samples) / sample_rate
    if last_note is not None:
        events.append(
            NoteEvent(
                start_time=last_start,
                end_time=end_time,
                note=last_note,
                frequency=last_freq,
            )
        )

    return events


def write_events(path: str, events: Iterable[NoteEvent]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("start_time,end_time,note,frequency\n")
        for event in events:
            file.write(
                f"{event.start_time:.3f},{event.end_time:.3f},{event.note},{event.frequency:.2f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a WAV audio file into a simple music sheet representation by "
            "detecting dominant frequencies and mapping them to notes."
        )
    )
    parser.add_argument("input", help="Path to a WAV audio file")
    parser.add_argument("output", help="Output CSV file for detected notes")
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--min-freq", type=float, default=60.0)
    parser.add_argument("--max-freq", type=float, default=2000.0)
    args = parser.parse_args()

    samples, sample_rate = read_audio_mono(args.input)
    events = extract_pitch_events(
        samples,
        sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )
    write_events(args.output, events)


if __name__ == "__main__":
    main()