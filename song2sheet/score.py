"""Notes + beats -> a readable score: the step existing tools get wrong.

Note times are mapped onto the tracked beats (so tempo drift doesn't smear the
rhythm), snapped to one grid per song (16ths, or triplet 8ths when they fit
clearly better), and each note is cut where the next one in its staff starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from music21 import chord, clef, key as m21key, meter, note, stream, tempo as m21tempo

GHOST_VEL = 45      # drop very quiet notes: mostly overtones and bleed from other stems
PIANO_SPLIT = 60    # middle C: notes below go to the bass staff


@dataclass
class Note:
    start: float    # seconds
    end: float
    pitch: int      # MIDI
    velocity: int = 80


@dataclass
class ScoreInfo:
    bpm: int
    meter: str
    key: str
    grid: str
    measures: int
    kept_notes: int
    dropped_notes: int
    off_grid_pct: float
    notes: list = field(default_factory=list, repr=False)  # (beat position, beats, pitch)


def _beat_mapper(beats: np.ndarray):
    """seconds -> position in beats, following tempo drift through the tracked beats."""
    step = beats[1] - beats[0]
    bt = np.concatenate([[beats[0] - step * k for k in (4, 3, 2, 1)], beats])
    bi = np.arange(len(bt)) - 4.0

    def to_beats(t: float) -> float:
        if bt[0] <= t <= bt[-1]:
            return float(np.interp(t, bt, bi))
        if t < bt[0]:
            return float(bi[0] + (t - bt[0]) / (bt[1] - bt[0]))
        return float(bi[-1] + (t - bt[-1]) / (bt[-1] - bt[-2]))
    return to_beats


def beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> int:
    if len(downbeats) < 2:
        return 4
    per_bar = [int(np.sum((beats >= downbeats[i]) & (beats < downbeats[i + 1])))
               for i in range(len(downbeats) - 1)]
    per_bar = [n for n in per_bar if 2 <= n <= 7]
    return int(np.bincount(per_bar).argmax()) if per_bar else 4


def build(notes: list[Note], beats: np.ndarray, downbeats: np.ndarray,
          instrument: str = "piano") -> tuple[stream.Score, ScoreInfo]:
    """Return a music21 score and what was inferred about it."""
    beats = np.asarray(beats, float)
    downbeats = np.asarray(downbeats, float)
    if len(beats) < 4:
        raise ValueError("Too few beats were found to write a score; try a longer or clearer clip.")
    kept = [n for n in notes if n.velocity >= GHOST_VEL]
    if not kept:
        raise ValueError("No notes were detected; try a clip where the instrument is clearly audible.")

    bpm = 60 / float(np.median(np.diff(beats)))
    per_bar = beats_per_bar(beats, downbeats)
    to_beats = _beat_mapper(beats)

    grids = {"16th": 1 / 4, "triplet-8th": 1 / 3}

    def fit(g: float) -> float:
        return float(np.mean([abs(b / g - round(b / g)) * g for b in (to_beats(n.start) for n in kept)]))
    grid_name = min(grids, key=lambda k: fit(grids[k]) * (1.3 if k == "triplet-8th" else 1.0))
    g = grids[grid_name]

    def snap(b: float) -> tuple[float, float]:
        s = round(b / g) * g
        return s, abs(b - s)

    first_down = downbeats[0] if len(downbeats) else beats[0]
    origin = to_beats(first_down)   # first downbeat = bar 1
    events, resid = [], []
    for n in kept:
        if to_beats(n.start) - origin < -0.5:
            continue
        s, r = snap(to_beats(n.start) - origin)
        e, _ = snap(to_beats(n.end) - origin)
        events.append((s, max(e - s, g), n.pitch))
        resid.append(r)
    if not events:
        raise ValueError("No notes fall after the first downbeat.")

    split = PIANO_SPLIT if instrument == "piano" else 0
    score = stream.Score()
    for staff, sel in (("upper", lambda p: p >= split), ("lower", lambda p: p < split)):
        evs = [e for e in events if sel(e[2])]
        if not evs:
            continue
        part = stream.Part()
        part.insert(0, meter.TimeSignature(f"{per_bar}/4"))
        part.insert(0, m21tempo.MetronomeMark(number=round(bpm)))
        if instrument == "guitar":
            part.insert(0, clef.Treble8vbClef())
        elif instrument == "bass":
            part.insert(0, clef.BassClef())
        else:
            part.insert(0, clef.TrebleClef() if staff == "upper" else clef.BassClef())
        by_onset: dict[float, list[tuple[float, int]]] = {}
        for s, d, p in evs:
            by_onset.setdefault(s, []).append((d, p))
        onsets = sorted(by_onset)
        for i, s in enumerate(onsets):
            group = by_onset[s]
            d = max(x[0] for x in group)
            if i + 1 < len(onsets):  # a note ends where the next one in its staff starts
                d = min(d, onsets[i + 1] - s)
            pitches = sorted({x[1] for x in group})
            el = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            el.quarterLength = d
            part.insert(s, el)
        score.insert(0, part)

    k = score.analyze("key")
    for p in score.parts:
        p.insert(0, m21key.Key(k.tonic, k.mode))
    score = score.makeNotation()
    for p in score.parts:  # recompute accidentals against the key signature
        for n in p.recurse().notes:
            for pt in n.pitches:
                if pt.accidental is not None and pt.accidental.name == "natural":
                    pt.accidental = None
                if pt.accidental is not None:
                    pt.accidental.displayStatus = None
        p.makeAccidentals(inPlace=True, overrideStatus=True)

    resid_a = np.array(resid)
    info = ScoreInfo(
        bpm=round(bpm), meter=f"{per_bar}/4", key=str(k), grid=grid_name,
        measures=len(score.parts[0].getElementsByClass("Measure")),
        kept_notes=len(events), dropped_notes=len(notes) - len(events),
        off_grid_pct=round(float(np.mean(resid_a > 0.1) * 100), 1),
        notes=events,
    )
    return score, info


def render_svg(musicxml_path: str, page_width: int = 2100) -> list[str]:
    """Engrave a MusicXML file to one SVG string per page with Verovio."""
    import verovio
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": page_width, "pageHeight": 2970, "scale": 40,
                   "adjustPageHeight": True, "footer": "none", "header": "none"})
    tk.loadFile(musicxml_path)
    return [tk.renderToSVG(i) for i in range(1, tk.getPageCount() + 1)]
