"""Which string was this note played on? A small CNN that listens to the note.

Input: a pitch-aligned CQT patch (the note's fundamental always sits at the same
row, so the network sees harmonics, inharmonicity and brightness, which differ
by string) plus the pitch. Output: logits for the 6 strings, masked to strings
that can play that pitch.

Trained on GuitarSet players 00-02 (CC BY 4.0); on held-out players 03-05 it
picks the right string for 81.4% of comp notes and 72.2% of solo notes.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from . import fret

SR, HOP, BPS = 22050, 256, 3          # 11.6 ms frames, 3 CQT bins per semitone
LOW = 38                               # MIDI of first CQT bin
NBINS = (123 - LOW) * BPS              # top bin ~9.4 kHz, below Nyquist
PATCH_SEMIS = (-1, 35)                 # fundamental .. 8th harmonic
PATCH_FRAMES = (-2, 22)                # ~ -23 ms .. +255 ms around onset
ROWS = (PATCH_SEMIS[1] - PATCH_SEMIS[0]) * BPS
COLS = PATCH_FRAMES[1] - PATCH_FRAMES[0]
WEIGHTS = Path(__file__).parent / "weights" / "stringnet.pt"


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.conv = nn.Sequential(block(1, 32), block(32, 64), block(64, 96))
        self.pitch = nn.Embedding(50, 32)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96 * (ROWS // 8) + 32, 128),
                                  nn.ReLU(), nn.Linear(128, 6))

    def forward(self, x, pitch):
        h = self.conv(x[:, None]).mean(-1).flatten(1)  # average over time, keep harmonic rows
        return self.head(torch.cat([h, self.pitch((pitch - 38).clamp(0, 49))], 1))


def load(device: str = "cpu") -> Net:
    model = Net()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.to(device).eval()


def cqt(y: np.ndarray) -> np.ndarray:
    """Log-compressed CQT of a mono 22.05 kHz signal."""
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(LOW),
                           n_bins=NBINS, bins_per_octave=12 * BPS))
    return np.log1p(30 * C / (C.max() + 1e-9)).astype(np.float32)


def patch(C: np.ndarray, onset: float, pitch: int) -> np.ndarray:
    r0 = (pitch - LOW + PATCH_SEMIS[0]) * BPS
    c0 = int(round(onset * SR / HOP)) + PATCH_FRAMES[0]
    P = np.zeros((ROWS, COLS), np.float32)
    rs, re_ = max(r0, 0), min(r0 + ROWS, C.shape[0])
    cs, ce = max(c0, 0), min(c0 + COLS, C.shape[1])
    if rs < re_ and cs < ce:
        P[rs - r0:re_ - r0, cs - c0:ce - c0] = C[rs:re_, cs:ce]
    return P


def feasible(pitch: np.ndarray) -> np.ndarray:
    m = np.zeros((len(pitch), 6), bool)
    for s, o in enumerate(fret.TUNING):
        m[:, s] = (pitch - o >= 0) & (pitch - o <= fret.MAX_FRET)
    return m


def string_log_probs(model: Net, y: np.ndarray, notes: list[tuple[float, int]],
                     device: str = "cpu", batch: int = 512) -> np.ndarray:
    """log P(string | audio) for each (onset, pitch), shape (n_notes, 6)."""
    if not notes:
        return np.zeros((0, 6), np.float32)
    C = cqt(y)
    X = np.stack([patch(C, t, p) for t, p in notes])
    P = np.array([p for _, p in notes])
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(device)
            p = torch.from_numpy(P[i:i + batch].astype(np.int64)).to(device)
            out.append(model(x, p).float().cpu().numpy())
    L = np.concatenate(out)
    L[~feasible(P)] = -1e9
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))
