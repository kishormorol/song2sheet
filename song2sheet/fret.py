"""Assign each note to a guitar string and fret so the passage is playable.

Notes are grouped into chords (same onset); each chord gets every valid
string assignment; Viterbi picks the sequence with the least hand effort.
"""
import itertools
import numpy as np

TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4, string 0 = low E
MAX_FRET = 20
CHORD_GAP = 0.05  # seconds: onsets closer than this are played together


def group_chords(notes, gap=CHORD_GAP):
    """notes: list of (onset, pitch). Returns list of lists of note indices."""
    order = sorted(range(len(notes)), key=lambda i: notes[i][0])
    groups, cur = [], []
    for i in order:
        if cur and notes[i][0] - notes[cur[0]][0] > gap:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    # a chord can use each string once: split groups bigger than 6
    out = []
    for g in groups:
        while len(g) > 6:
            out.append(g[:6]); g = g[6:]
        out.append(g)
    return out


def positions(pitch, tuning=TUNING):
    return [(s, pitch - o) for s, o in enumerate(tuning) if 0 <= pitch - o <= MAX_FRET]


def chord_options(pitches, tuning=TUNING, limit=40):
    """All ways to put these pitches on distinct strings, best `limit` by local cost."""
    per_note = [positions(p, tuning) for p in pitches]
    if any(not p for p in per_note):  # out of range: drop unplayable notes
        keep = [i for i, p in enumerate(per_note) if p]
        sub = chord_options([pitches[i] for i in keep], tuning, limit)
        return [tuple(dict(zip(keep, o)).get(i) for i in range(len(pitches))) for o in sub]
    opts = []
    for combo in itertools.product(*per_note):
        strings = [s for s, _ in combo]
        if len(set(strings)) == len(strings):
            opts.append(combo)
    opts.sort(key=local_cost)
    return opts[:limit] or [tuple(None for _ in pitches)]


def hand(combo):
    fretted = [f for sf in combo if sf for _, f in [sf] if f > 0]
    return (float(np.mean(fretted)) if fretted else None), fretted


W = dict(span=3.0, height=0.15, move=0.6, open_bonus=0.3, big_stretch=10.0)


def local_cost(combo, w=W):
    pos, fretted = hand(combo)
    if not fretted:
        return -w["open_bonus"] * len(combo)
    span = max(fretted) - min(fretted)
    c = w["span"] * max(0, span - 3) + w["big_stretch"] * max(0, span - 5)
    c += w["height"] * pos
    c -= w["open_bonus"] * sum(1 for sf in combo if sf and sf[1] == 0)
    return c


def move_cost(a, b, w=W):
    pa, _ = hand(a)
    pb, _ = hand(b)
    if pa is None or pb is None:
        return 0.0
    return w["move"] * abs(pa - pb)


def assign(notes, tuning=TUNING, w=W):
    """notes: list of (onset, pitch). Returns list of (string, fret) or None per note."""
    groups = group_chords(notes)
    options = [chord_options([notes[i][1] for i in g], tuning) for g in groups]
    cost = [np.array([local_cost(o, w) for o in options[0]])]
    back = []
    for t in range(1, len(groups)):
        prev, cur = options[t - 1], options[t]
        trans = np.array([[move_cost(p, c, w) for p in prev] for c in cur])  # cur x prev
        tot = trans + cost[-1][None, :]
        back.append(tot.argmin(1))
        cost.append(tot.min(1) + np.array([local_cost(c, w) for c in cur]))
    path = [int(cost[-1].argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    out = [None] * len(notes)
    for g, opts, k in zip(groups, options, path):
        for i, sf in zip(g, opts[k]):
            out[i] = sf
    return out


def assign_lowest_fret(notes, tuning=TUNING):
    """Baseline used by most MIDI->tab converters: every note at its lowest fret."""
    out = []
    for _, p in notes:
        pos = positions(p, tuning)
        out.append(min(pos, key=lambda sf: sf[1]) if pos else None)
    return out


def to_ascii_tab(notes, frets, per_line=32):
    """Very small ASCII tab renderer, one column per chord."""
    groups = group_chords(notes)
    names = ["E", "A", "D", "G", "B", "e"]
    cols = []
    for g in groups:
        col = ["-"] * 6
        for i in g:
            if frets[i]:
                s, f = frets[i]
                col[s] = str(f)
        width = max(len(c) for c in col)
        cols.append([c.ljust(width, "-") for c in col])
    lines = []
    for k in range(0, len(cols), per_line):
        chunk = cols[k:k + per_line]
        for s in reversed(range(6)):
            lines.append(names[s] + "|-" + "-".join(c[s] for c in chunk) + "-|")
        lines.append("")
    return "\n".join(lines)


# --- v3: fuse an audio string classifier into the Viterbi search ----------
def assign_with_probs(notes, logp, alpha=1.0, tuning=TUNING, w=W, limit=60):
    """logp: array (n_notes, 6) of log P(string | audio). alpha weighs audio vs hand effort."""
    groups = group_chords(notes)

    def opt_cost(g, combo):
        audio = sum(logp[i][sf[0]] for i, sf in zip(g, combo) if sf is not None)
        return local_cost(combo, w) - alpha * audio

    options, local = [], []
    for g in groups:
        per_note = [positions(notes[i][1], tuning) or [None] for i in g]
        opts = []
        for combo in itertools.product(*per_note):
            strings = [sf[0] for sf in combo if sf is not None]
            if len(set(strings)) == len(strings):
                opts.append(combo)
        if not opts:  # more simultaneous notes than strings allow: drop to single best per note
            opts = [tuple(max(p, key=lambda sf: logp[i][sf[0]]) if p != [None] else None
                          for i, p in zip(g, per_note))]
        opts.sort(key=lambda c: opt_cost(g, c))
        opts = opts[:limit]
        options.append(opts)
        local.append(np.array([opt_cost(g, c) for c in opts]))
    cost = local[0]
    back = []
    for t in range(1, len(groups)):
        trans = np.array([[move_cost(p, c, w) for p in options[t - 1]] for c in options[t]])
        tot = trans + cost[None, :]
        back.append(tot.argmin(1))
        cost = tot.min(1) + local[t]
    path = [int(cost.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    out = [None] * len(notes)
    for g, opts, k in zip(groups, options, path):
        for i, sf in zip(g, opts[k]):
            out[i] = sf
    return out
