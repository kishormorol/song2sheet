# 🎼 Song2Sheet

**Any song → sheet music you can read, a MIDI file, and guitar tab that's actually playable.**
Open source, runs locally, built entirely on open models.

**Try it:** [huggingface.co/spaces/kishormorol/song2sheet](https://huggingface.co/spaces/kishormorol/song2sheet)

Paid tools (Klangio, Songscription) do this behind a subscription. Open-source options
either stop at MIDI or handle solo piano only. Song2Sheet goes all the way to an engraved score
and, for guitar, chooses *which string* each note was played on by listening to it.

## How it works

```
audio ──Demucs──▶ instrument stem ──basic-pitch──▶ notes ──┐
   └──beat_this──▶ beats + downbeats ─────────────────────┴─▶ rhythm grid ─▶ MusicXML · SVG · MIDI
                              guitar: string CNN + Viterbi ─▶ tab
```

1. **Isolate** the instrument from a full mix with [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`).
2. **Detect notes** with [basic-pitch](https://github.com/spotify/basic-pitch).
3. **Find beats and bars** with [beat_this](https://github.com/CPJKU/beat_this). Beats come from the
   full mix, where the drums help.
4. **Write the rhythm**, the step most tools get wrong. Note times are mapped onto the tracked beats,
   so tempo drift doesn't smear the rhythm. They're snapped to one grid per song (16ths, or triplet
   8ths when those fit clearly better), each note is cut where the next one in its staff starts,
   and the key is inferred. [music21](https://github.com/cuthbertLab/music21) builds the score and
   [Verovio](https://www.verovio.org) engraves it.
5. **Guitar tab:** a small CNN looks at a pitch-aligned spectrogram patch of each note. Strings sound
   different (harmonics, inharmonicity, brightness), so it predicts which string was played. A Viterbi
   search then combines that with hand-movement cost to pick a playable fingering.

## How good is it?

| | Result |
|---|---|
| Solo piano | Notes are near-perfect; meter and key are usually right |
| Guitar tab, right string for a correct note | **81.4%** accompaniment / **72.2%** solo (GuitarSet held-out players 03–05) |
| …the usual "lowest fret" approach | 49.3% / 32.1% |
| Guitar end to end (audio → right note on the right string) | 57.1% accompaniment / 60.5% solo |

**Still rough:**
- rhythm in free-tempo (rubato) playing, where beat tracking locks onto the wrong pulse
- dense full-band mixes
- vocals

The guitar string model was trained on only 3 GuitarSet players. Adding the third added about
3 points, so more data should help.

## Run it yourself

```bash
pip install -r requirements.txt torch
python -m song2sheet.pipeline song.mp3 --instrument piano          # or guitar, bass, vocals
python -m song2sheet.pipeline solo_guitar.wav --instrument guitar --no-isolate
```

The output goes to `out/`: `.musicxml` (open it in MuseScore), `.mid`, and `-tab.txt` for guitar.
For the web demo, run `pip install -r requirements-dev.txt && python app.py`.

```bash
pytest tests        # offline: scoring, fretting, string model
```

## Credits and licences

Song2Sheet is MIT-licensed. It builds on Demucs (MIT), basic-pitch (Apache-2.0), beat_this (MIT),
music21 (BSD-3), and Verovio (LGPL-3.0). The string model in `song2sheet/weights/` was trained on
[GuitarSet](https://guitarset.weebly.com/) (CC BY 4.0; Xi et al., ISMIR 2018).

Only transcribe recordings you have the rights to use.
