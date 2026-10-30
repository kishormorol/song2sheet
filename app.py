"""Song2Sheet demo: any song -> sheet music, MIDI and guitar tab. Runs on ZeroGPU."""
# ZeroGPU needs `spaces` imported before torch or anything that touches CUDA
try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # running locally
    def gpu(fn=None, **_):
        return fn if fn else (lambda f: f)

import html
import os
import tempfile

import gradio as gr
import torch

from song2sheet.pipeline import Models, transcribe

if torch.cuda.is_available() or os.getenv("SPACES_ZERO_GPU"):
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "120"))
REPO = "https://github.com/kishormorol/song2sheet"

# every model is created on DEVICE up front: under ZeroGPU a model built later,
# or kept on the CPU, leaves the forked GPU workers unable to see the GPU
models = Models.load(DEVICE)

INSTRUMENTS = {"Piano": "piano", "Guitar": "guitar", "Bass": "bass", "Vocals (melody)": "vocals"}


@gpu(duration=150)
def run(audio_path, instrument, isolate):
    return transcribe(models, audio_path, instrument, isolate, tempfile.mkdtemp(prefix="s2s-"))


def convert(audio_path, instrument_label, isolate, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Upload a song or record a clip first.")
    import librosa
    if librosa.get_duration(path=audio_path) > MAX_SECONDS:
        gr.Info(f"Only the first {MAX_SECONDS} seconds are transcribed.")
    instrument = INSTRUMENTS[instrument_label]
    progress(0.1, desc="Separating, transcribing and engraving…")
    try:
        r = run(audio_path, instrument, isolate)
    except ValueError as exc:   # no notes, too few beats, …
        raise gr.Error(str(exc))
    i = r.info
    summary = (f"**{i.key}** · **{i.meter}** · ♩ = {i.bpm} · {i.measures} bars · "
               f"{i.kept_notes} notes · rhythm grid: {i.grid} · "
               f"{i.off_grid_pct}% of notes needed a big nudge to fit the grid")
    sheet = "".join(f'<div class="s2s-page">{svg}</div>' for svg in r.svg_pages)
    files = [r.musicxml, r.midi] + ([r.tab_path] if r.tab_path else [])
    tab_update = gr.update(value=r.tab, visible=bool(r.tab))
    return summary, f'<div class="s2s-sheet">{sheet}</div>', tab_update, r.stem, files


CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.s2s-hero { padding: 26px 2px 6px; }
.s2s-hero h1 { font-size: clamp(1.7rem, 3.4vw, 2.5rem); line-height: 1.15; margin: 0 0 10px;
  font-weight: 750; color: var(--body-text-color); }
.s2s-hero p { margin: 0; max-width: 760px; color: var(--body-text-color-subdued); font-size: 1.02rem; }
.s2s-hero a { color: var(--color-accent); }
.s2s-steps { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.s2s-steps span { font-size: .84rem; padding: 6px 11px; border-radius: 999px;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  color: var(--body-text-color-subdued); }
.s2s-sheet { max-height: 760px; overflow: auto; border-radius: 12px; background: #fff;
  border: 1px solid var(--border-color-primary); }
.s2s-page { padding: 8px 4px; }
.s2s-page svg { width: 100%; height: auto; display: block; }
#s2s-tab textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
  font-size: .85rem !important; line-height: 1.35 !important; white-space: pre !important; }
.s2s-note { font-size: .84rem; color: var(--body-text-color-subdued); }
"""

HERO = f"""
<div class="s2s-hero">
  <h1>🎼 Song2Sheet</h1>
  <p>Turn a recording into sheet music you can read, a MIDI file, and guitar tab that's actually
     playable. Open source and runs end to end on open models:
     <a href="{REPO}" target="_blank" rel="noopener">code on GitHub</a>.</p>
  <div class="s2s-steps">
    <span>1 · isolate the instrument (Demucs)</span>
    <span>2 · detect notes (basic-pitch)</span>
    <span>3 · find beats and bars (beat_this)</span>
    <span>4 · fit a rhythm grid and engrave (music21 + Verovio)</span>
    <span>5 · guitar: pick strings by ear (our string CNN)</span>
  </div>
</div>"""

NOTES = """
<p class="s2s-note"><b>What works well:</b> solo piano and steady, strummed or picked guitar.
<b>Still rough:</b> rhythm in free-tempo (rubato) playing, dense full-band mixes, and vocals.
Tab picks the string from the sound of each note; on held-out GuitarSet players it chooses the right
string 81% of the time for accompaniment and 72% for solos. Only upload audio you have the rights to.</p>"""

with gr.Blocks(title="Song2Sheet") as demo:
    gr.HTML(HERO, padding=False)
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            audio = gr.Audio(label="Song or clip", type="filepath", sources=["upload", "microphone"])
        with gr.Column(scale=1):
            instrument = gr.Radio(list(INSTRUMENTS), value="Piano", label="Transcribe")
            isolate = gr.Checkbox(value=True, label="Isolate it from a full mix",
                                  info="Untick for a recording of the instrument alone")
            go = gr.Button("Make sheet music", variant="primary")
    summary = gr.Markdown()
    sheet = gr.HTML(padding=False)
    tab = gr.Textbox(label="Guitar tab", lines=14, max_lines=40, visible=False,
                     elem_id="s2s-tab", buttons=["copy"])
    with gr.Row():
        stem = gr.Audio(label="What was transcribed (isolated stem)", interactive=False)
        files = gr.File(label="Downloads: MusicXML (open in MuseScore), MIDI, tab", file_count="multiple")
    gr.HTML(NOTES, padding=False)
    go.click(convert, [audio, instrument, isolate], [summary, sheet, tab, stem, files])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="violet"), css=CSS)
