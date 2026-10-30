"""Offline tests: scoring, fretting and the string model, no audio models or network."""
from __future__ import annotations

import numpy as np
import pretty_midi
import pytest
import torch

from song2sheet import fret, score, stringnet
from song2sheet.pipeline import _write_midi
from song2sheet.score import Note


def _beats(bpm: float = 120, n: int = 33, per_bar: int = 4):
    beats = np.arange(n) * 60 / bpm
    return beats, beats[::per_bar]


def _quarters(pitches, bpm=120, start_beat=0):
    spb = 60 / bpm
    return [Note((start_beat + i) * spb, (start_beat + i + 1) * spb, p, 90) for i, p in enumerate(pitches)]


# ── score ─────────────────────────────────────────────────────────────────────

def test_steady_quarters_become_a_clean_score():
    beats, downs = _beats()
    sc, info = score.build(_quarters([60, 62, 64, 65, 67, 69, 71, 72]), beats, downs, "piano")
    assert info.bpm == 120 and info.meter == "4/4"
    assert info.grid == "16th" and info.off_grid_pct == 0
    assert info.kept_notes == 8 and info.measures >= 2
    assert [n.pitch.midi for n in sc.flatten().notes][:3] == [60, 62, 64]


def test_pickup_notes_before_the_first_downbeat_are_kept():
    beats, downs = _beats()
    spb = 0.5
    pickup = [Note(beats[4] - 0.2 * spb, beats[4] + 0.5 * spb, 64, 90)]   # just before bar 1
    sc, info = score.build(pickup + _quarters([60, 62, 64, 65], start_beat=4), beats, downs[1:], "piano")
    assert info.kept_notes == 5
    assert 64 in [n.pitch.midi for n in sc.flatten().notes][:2]


def test_waltz_meter_is_detected():
    beats, downs = _beats(per_bar=3)
    _, info = score.build(_quarters([67] * 12), beats, downs, "piano")
    assert info.meter == "3/4"


def test_piano_splits_at_middle_c_and_guitar_uses_one_staff():
    beats, downs = _beats()
    notes = _quarters([48, 72, 50, 74])
    piano, _ = score.build(notes, beats, downs, "piano")
    guitar, _ = score.build(notes, beats, downs, "guitar")
    assert len(piano.parts) == 2 and len(guitar.parts) == 1


def test_quiet_ghost_notes_are_dropped():
    beats, downs = _beats()
    notes = _quarters([60, 62, 64, 65]) + [Note(0.1, 0.2, 90, velocity=10)]
    _, info = score.build(notes, beats, downs, "piano")
    assert info.kept_notes == 4 and info.dropped_notes == 1


def test_key_is_inferred():
    beats, downs = _beats()
    g_major = [67, 71, 74, 66, 69, 72, 67, 71, 74, 79, 78, 76]
    _, info = score.build(_quarters(g_major), beats, downs, "piano")
    assert info.key.startswith("G major") or info.key.startswith("e minor")


def test_helpful_errors_for_unusable_input():
    beats, downs = _beats()
    with pytest.raises(ValueError, match="No notes"):
        score.build([], beats, downs, "piano")
    with pytest.raises(ValueError, match="Too few beats"):
        score.build(_quarters([60]), beats[:2], downs[:1], "piano")


def test_score_engraves_to_svg(tmp_path):
    beats, downs = _beats()
    sc, _ = score.build(_quarters([60, 64, 67, 72]), beats, downs, "piano")
    xml = tmp_path / "s.musicxml"
    sc.write("musicxml", fp=str(xml))
    pages = score.render_svg(str(xml))
    assert pages and pages[0].lstrip().startswith("<svg")


# ── fretting ──────────────────────────────────────────────────────────────────

def test_every_note_gets_a_playable_position():
    notes = [(i * 0.5, p) for i, p in enumerate([40, 45, 50, 55, 59, 64, 76])]
    for (_, pitch), sf in zip(notes, fret.assign(notes)):
        string, f = sf
        assert fret.TUNING[string] + f == pitch and 0 <= f <= fret.MAX_FRET


def test_chord_notes_land_on_distinct_strings():
    e_major = [(0.0, p) for p in (40, 47, 52, 56, 59, 64)]
    strings = [sf[0] for sf in fret.assign(e_major)]
    assert len(set(strings)) == 6


def test_audio_probabilities_can_override_hand_effort():
    notes = [(0.0, 64)]                 # E4: open high E, or 5th fret on B, ...
    logp = np.full((1, 6), -20.0)
    logp[0, 4] = 0.0                    # the audio is sure it was the B string
    assert fret.assign_with_probs(notes, logp, alpha=1.0) == [(4, 5)]


def test_out_of_range_notes_are_dropped_not_crashing():
    assert fret.assign([(0.0, 20)]) == [None]


def test_ascii_tab_has_six_strings():
    notes = [(0.0, 40), (0.5, 45)]
    tab = fret.to_ascii_tab(notes, fret.assign(notes))
    assert [line[0] for line in tab.splitlines()[:6]] == list("eBGDAE")


# ── string model ──────────────────────────────────────────────────────────────

def test_shipped_weights_load_and_give_valid_string_probs():
    model = stringnet.load("cpu")
    sr = stringnet.SR
    t = np.arange(sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 329.63 * t).astype(np.float32)  # E4 for one second
    logp = stringnet.string_log_probs(model, y, [(0.1, 64)])
    assert logp.shape == (1, 6)
    assert np.isclose(np.exp(logp).sum(), 1.0, atol=1e-4)
    assert logp[0, 0] < -100                  # E4 would be fret 24 on low E: not playable
    assert (logp[0, 1:] > -100).all()         # A (fret 19) up to high E (open) can play it


def test_feasible_mask_matches_fretboard():
    m = stringnet.feasible(np.array([40, 84]))
    assert m[0].tolist() == [True, False, False, False, False, False]
    assert m[1].tolist() == [False, False, False, False, False, True]


def test_patch_shape_is_fixed_even_at_edges():
    C = np.random.rand(stringnet.NBINS, 50).astype(np.float32)
    assert stringnet.patch(C, 0.0, 40).shape == (stringnet.ROWS, stringnet.COLS)
    assert stringnet.patch(C, 10.0, 90).shape == (stringnet.ROWS, stringnet.COLS)


# ── outputs ───────────────────────────────────────────────────────────────────

def test_midi_keeps_only_audible_notes(tmp_path):
    path = tmp_path / "x.mid"
    _write_midi(_quarters([60, 64]) + [Note(0, 0.1, 70, velocity=5)], str(path), "guitar")
    pm = pretty_midi.PrettyMIDI(str(path))
    assert [n.pitch for n in pm.instruments[0].notes] == [60, 64]
    assert pm.instruments[0].program == 24
