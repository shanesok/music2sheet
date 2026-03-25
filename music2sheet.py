#!/usr/bin/env python3
"""
melody_sheet_pipeline.py

Full-mix audio (song) -> MELODY ONLY transcription:
1) Demucs separates stems -> we take vocals.wav
2) Basic Pitch transcribes vocals.wav -> outputs MIDI (melody)
3) (Optional) Convert MIDI -> MusicXML (if music21 is installed)

This does NOT create a full piano arrangement. It creates a single-line melody
you can open as sheet music (MuseScore: File -> Open the .mid or .musicxml).

Requirements (you install):
- demucs  (source separation)
- basic-pitch (melody transcription)
Optional:
- music21 (MIDI -> MusicXML)

Example:
python melody_sheet_pipeline.py "input.wav" --outdir "out" --make-musicxml

Confidence: 0.88 that this pipeline will produce a recognizable melody on typical pop songs,
assuming Demucs + Basic Pitch are installed and the vocal is reasonably clear.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def which_or_fail(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise RuntimeError(
            f"Missing command '{cmd}'.\n"
            f"Install it first, then re-run.\n"
            f"  - demucs:      pip install demucs\n"
            f"  - basic-pitch: pip install basic-pitch\n"
        )
    return path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def find_vocals_stem(separated_dir: Path) -> Path:
    """
    Demucs output pattern:
      separated/<model>/<track_name>/vocals.wav
    We search for a vocals.wav below separated_dir.
    """
    matches = list(separated_dir.rglob("vocals.wav"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find vocals.wav under: {separated_dir}\n"
            "Demucs may have failed or used a different output layout."
        )
    # pick the most recently modified
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def find_basic_pitch_midi(outdir: Path) -> Path:
    """
    basic-pitch CLI typically writes a .mid into the output directory,
    but filenames can vary. We search for newest .mid.
    """
    mids = list(outdir.rglob("*.mid")) + list(outdir.rglob("*.midi"))
    if not mids:
        raise FileNotFoundError(
            f"Could not find any MIDI file written under: {outdir}\n"
            "Basic Pitch may have failed or wrote elsewhere."
        )
    mids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return mids[0]


def midi_to_musicxml(midi_path: Path, musicxml_path: Path) -> None:
    """
    Optional: Convert MIDI -> MusicXML using music21.
    """
    try:
        from music21 import converter  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "music21 is not installed (or failed to import).\n"
            "Install it with:\n"
            "  pip install music21\n"
            f"Original error: {e}"
        )

    score = converter.parse(str(midi_path))
    # Write MusicXML
    score.write("musicxml", fp=str(musicxml_path))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a full-mix audio file into MELODY-ONLY sheet music via Demucs + Basic Pitch."
    )
    ap.add_argument("input", type=str, help="Path to input audio (WAV/MP3/etc.)")
    ap.add_argument("--outdir", type=str, default="melody_out", help="Output directory")
    ap.add_argument("--demucs-model", type=str, default="htdemucs", help="Demucs model name (default: htdemucs)")
    ap.add_argument("--device", type=str, default="", help="Demucs device override (e.g. 'cuda' if you have GPU)")
    ap.add_argument("--no-demucs", action="store_true", help="Skip Demucs (if you already have vocals.wav)")
    ap.add_argument("--vocals", type=str, default="", help="Provide a vocals stem directly (skips Demucs)")
    ap.add_argument("--make-musicxml", action="store_true", help="Also export MusicXML (requires music21)")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure tools exist (unless skipping)
    if not args.no_demucs and not args.vocals:
        which_or_fail("demucs")
    which_or_fail("basic-pitch")

    vocals_path: Optional[Path] = None

    # Step 1: get vocals stem
    if args.vocals:
        vocals_path = Path(args.vocals).expanduser().resolve()
        if not vocals_path.exists():
            raise FileNotFoundError(f"--vocals not found: {vocals_path}")
        print(f"Using provided vocals stem: {vocals_path}")

    elif args.no_demucs:
        raise ValueError("If you use --no-demucs, you must also pass --vocals PATH_TO_VOCALS.wav")

    else:
        # Run Demucs
        separated_root = outdir / "separated"
        demucs_cmd = [
            "demucs",
            "-n",
            args.demucs_model,
            "-o",
            str(separated_root),
        ]
        if args.device.strip():
            demucs_cmd += ["-d", args.device.strip()]

        demucs_cmd += [str(input_path)]
        run(demucs_cmd)

        vocals_path = find_vocals_stem(separated_root)
        print(f"Found vocals stem: {vocals_path}")

    assert vocals_path is not None

    # Step 2: Basic Pitch -> MIDI
    bp_out = outdir / "basic_pitch"
    bp_out.mkdir(parents=True, exist_ok=True)

    # basic-pitch CLI supports output directory via -o / --output-dir in many installs,
    # but versions vary. We use a safe approach:
    #   run inside bp_out so outputs land there.
    # If your basic-pitch version supports --output-dir, you can add it.
    cmd = ["basic-pitch", str(vocals_path)]
    print("\nRunning Basic Pitch (this can take a bit on long audio)...")
    subprocess.run(cmd, cwd=str(bp_out), check=True)

    midi_path = find_basic_pitch_midi(bp_out)
    final_midi = outdir / "melody.mid"
    shutil.copy2(midi_path, final_midi)
    print(f"\n✅ Melody MIDI saved to: {final_midi}")

    # Step 3 (optional): MIDI -> MusicXML
    if args.make_musicxml:
        musicxml_path = outdir / "melody.musicxml"
        midi_to_musicxml(final_midi, musicxml_path)
        print(f"✅ MusicXML saved to: {musicxml_path}")

    print("\nNext step (sheet music):")
    print("- Open MuseScore -> File -> Open -> melody.mid (or melody.musicxml)")
    print("- Then Export as PDF to print.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)