"""Song -> sheet music (+ guitar tab), end to end.

    audio ──Demucs──▶ instrument stem ──basic-pitch──▶ notes ─┐
       └──beat_this──▶ beats + downbeats ─────────────────────┴─▶ score ─▶ MusicXML / SVG / MIDI
                                      guitar: stringnet + Viterbi ─▶ tab

Models are created once with `Models.load(device)` and passed in, so a server
can keep them resident (on ZeroGPU they must be created on "cuda" at startup).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from . import fret, score as scoring, stringnet

# Demucs htdemucs_6s source names, per instrument choice
STEMS = {"piano": "piano", "guitar": "guitar", "bass": "bass", "vocals": "vocals"}
MAX_SECONDS = 180


@dataclass
class Models:
    separator: object
    beats: object
    strings: stringnet.Net
    device: str

    @classmethod
    def load(cls, device: str = "cpu") -> "Models":
        from beat_this.inference import Audio2Beats
        from demucs.pretrained import get_model
        separator = get_model("htdemucs_6s").to(device).eval()
        beats = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
        return cls(separator, beats, stringnet.load(device), device)


@dataclass
class Result:
    musicxml: str               # path
    midi: str                   # path
    svg_pages: list[str]
    tab: str                    # ASCII tab ("" unless guitar)
    tab_path: str               # path ("" unless guitar)
    stem: str                   # path to the isolated stem that was transcribed
    info: scoring.ScoreInfo


def load_audio(path: str, sr: int, mono: bool = True, max_seconds: int = MAX_SECONDS) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=mono, duration=max_seconds)
    return y


def separate(models: Models, path: str, instrument: str) -> tuple[np.ndarray, int]:
    """Isolate one instrument from a mix. Returns (mono signal, sample rate)."""
    import torch
    from demucs.apply import apply_model
    m = models.separator
    y = load_audio(path, m.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.from_numpy(y).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(m, wav[None].to(models.device), device=models.device, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    stem = out[m.sources.index(STEMS[instrument])].mean(0).cpu().numpy()
    return stem, m.samplerate


def track_beats(models: Models, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    beats, downbeats = models.beats(y, sr)
    return np.asarray(beats), np.asarray(downbeats)


def detect_notes(stem_path: str) -> list[scoring.Note]:
    """basic-pitch note events (ONNX model, so it behaves the same on every OS)."""
    import basic_pitch
    from basic_pitch.inference import predict
    onnx = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    _, _, events = predict(stem_path, str(onnx))
    return [scoring.Note(float(s), float(e), int(p), int(round(a * 127)))
            for s, e, p, a, *_ in events]


def tab_for(models: Models, y: np.ndarray, sr: int, notes: list[scoring.Note]) -> tuple[str, list]:
    """Pick string + fret per note from the audio and hand effort, render ASCII tab."""
    kept = sorted((n for n in notes if n.velocity >= scoring.GHOST_VEL), key=lambda n: n.start)
    onsets = [(n.start, n.pitch) for n in kept]
    y22 = librosa.resample(y, orig_sr=sr, target_sr=stringnet.SR) if sr != stringnet.SR else y
    logp = stringnet.string_log_probs(models.strings, y22, onsets, models.device)
    frets = fret.assign_with_probs(onsets, logp) if onsets else []
    return fret.to_ascii_tab(onsets, frets), frets


def transcribe(models: Models, audio_path: str, instrument: str = "piano",
               isolate: bool = True, out_dir: str | None = None) -> Result:
    """Run the whole pipeline on one file."""
    if instrument not in STEMS:
        raise ValueError(f"instrument must be one of {sorted(STEMS)}")
    out = Path(out_dir or tempfile.mkdtemp(prefix="song2sheet-"))
    out.mkdir(parents=True, exist_ok=True)
    name = Path(audio_path).stem[:60] or "song"

    if isolate:
        y, sr = separate(models, audio_path, instrument)
    else:
        sr = 44100
        y = load_audio(audio_path, sr)
    stem_path = str(out / f"{name}-{instrument}.wav")
    sf.write(stem_path, y, sr)

    mix = load_audio(audio_path, 44100)        # beats come from the full mix: drums help
    beats, downbeats = track_beats(models, mix, 44100)
    notes = detect_notes(stem_path)
    sc, info = scoring.build(notes, beats, downbeats, instrument)

    xml_path = str(out / f"{name}.musicxml")
    sc.write("musicxml", fp=xml_path)
    midi_path = str(out / f"{name}.mid")
    _write_midi(notes, midi_path, instrument)

    tab, tab_path = "", ""
    if instrument == "guitar":
        tab, _ = tab_for(models, y, sr, notes)
        tab_path = str(out / f"{name}-tab.txt")
        Path(tab_path).write_text(tab)

    return Result(xml_path, midi_path, scoring.render_svg(xml_path), tab, tab_path, stem_path, info)


def _write_midi(notes: list[scoring.Note], path: str, instrument: str) -> None:
    program = {"piano": 0, "guitar": 24, "bass": 33, "vocals": 52}[instrument]
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        if n.velocity >= scoring.GHOST_VEL:
            inst.notes.append(pretty_midi.Note(velocity=min(n.velocity, 127), pitch=n.pitch,
                                               start=n.start, end=max(n.end, n.start + 0.05)))
    pm.instruments.append(inst)
    pm.write(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Transcribe a song to sheet music (and guitar tab).")
    ap.add_argument("audio")
    ap.add_argument("--instrument", default="piano", choices=sorted(STEMS))
    ap.add_argument("--no-isolate", action="store_true", help="the recording is the instrument alone")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    r = transcribe(Models.load(os.environ.get("DEVICE", dev)), a.audio, a.instrument,
                   not a.no_isolate, a.out)
    print(r.info)
    print("wrote", r.musicxml, r.midi, r.tab_path or "")
