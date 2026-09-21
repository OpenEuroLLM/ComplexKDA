# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Synthesize an original four-on-the-floor disco-techno loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import wave

import numpy as np

BASS_PATTERN = (
    (38, 1.00, 1.65), None, (45, 0.62, 0.55), (41, 0.78, 0.72),
    (33, 0.92, 1.20), None, (36, 0.58, 0.50), (38, 0.86, 0.78),
    (38, 0.95, 1.15), None, (41, 0.62, 0.48), (45, 0.82, 0.72),
    (48, 0.90, 0.90), (45, 0.62, 0.50), None, (36, 0.58, 0.52),
    (38, 1.00, 1.45), None, (50, 0.70, 0.55), (48, 0.75, 0.68),
    (45, 0.92, 1.10), None, (41, 0.62, 0.52), (43, 0.75, 0.70),
    (38, 0.98, 1.10), (36, 0.55, 0.45), None, (41, 0.74, 0.65),
    (45, 0.88, 0.82), (43, 0.62, 0.48), (41, 0.70, 0.55), (36, 0.56, 0.50),
)
PULSE_PATTERN = (None, 62, None, 69, 65, None, 67, None, None, 70, None, 65, 62, None, 67, None)


def midi_frequency(note: int) -> float:
    return 440 * 2 ** ((note - 69) / 12)


def pan_mono(samples: np.ndarray, pan: float) -> np.ndarray:
    angle = (pan + 1) * math.pi / 4
    return np.column_stack((samples * math.cos(angle), samples * math.sin(angle)))


def add_event(track: np.ndarray, event: np.ndarray, start: int) -> None:
    destination_start = max(start, 0)
    destination_stop = min(start + event.shape[0], track.shape[0])
    if destination_stop <= destination_start:
        return
    source_start = destination_start - start
    source_stop = source_start + destination_stop - destination_start
    track[destination_start:destination_stop] += event[source_start:source_stop]


def kick(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(round(0.46 * sample_rate)) / sample_rate
    phase = 2 * math.pi * (47 * time + 115 * (1 - np.exp(-32 * time)) / 32)
    envelope = (1 - np.exp(-260 * time)) * np.exp(-10.5 * time)
    body = np.sin(phase) + 0.13 * np.sin(2 * phase)
    click = rng.standard_normal(time.shape[0]) * np.exp(-95 * time)
    return 0.92 * body * envelope + 0.055 * click


def clap(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(round(0.32 * sample_rate)) / sample_rate
    noise = rng.standard_normal(time.shape[0])
    smooth = np.convolve(noise, np.full(19, 1 / 19), mode="same")
    noise -= smooth
    envelope = np.zeros_like(time)
    for offset, level in ((0.0, 1.0), (0.012, 0.72), (0.025, 0.55)):
        shifted = np.maximum(time - offset, 0)
        envelope += level * (time >= offset) * np.exp(-24 * shifted)
    tone = np.sin(2 * math.pi * 175 * time) * np.exp(-18 * time)
    return 0.24 * noise * envelope + melt(tone)


def melt(samples: np.ndarray) -> np.ndarray:
    return 0.09 * np.tanh(1.8 * samples)


def hat(sample_rate: int, rng: np.random.Generator, open_hat: bool) -> np.ndarray:
    duration = 0.28 if open_hat else 0.075
    time = np.arange(round(duration * sample_rate)) / sample_rate
    noise = rng.standard_normal(time.shape[0])
    smooth = np.convolve(noise, np.full(11, 1 / 11), mode="same")
    metallic = noise - smooth
    decay = 15 if open_hat else 58
    envelope = (1 - np.exp(-900 * time)) * np.exp(-decay * time)
    return 0.18 * metallic * envelope


def synth_note(frequency: float, duration: float, sample_rate: int, accent: float) -> np.ndarray:
    time = np.arange(round(duration * sample_rate)) / sample_rate
    phase = frequency * time
    saw = 2 * np.mod(phase, 1) - 1
    detuned = 2 * np.mod(phase * 1.004, 1) - 1
    fundamental = np.sin(2 * math.pi * phase)
    envelope = (1 - np.exp(-260 * time)) * np.exp(-11 * time)
    return accent * envelope * (0.46 * saw + 0.34 * detuned + 0.36 * fundamental)


def lowpass(signal: np.ndarray, cutoff: np.ndarray, sample_rate: int) -> np.ndarray:
    output = np.empty_like(signal)
    state = 0.0
    for index, sample in enumerate(signal):
        coefficient = 1 - math.exp(-2 * math.pi * cutoff[index] / sample_rate)
        state += coefficient * (sample - state)
        output[index] = state
    return output


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(audio.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def synthesize(sample_rate: int, bpm: float, bars: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    beat_seconds = 60 / bpm
    step_seconds = beat_seconds / 4
    total_seconds = bars * 4 * beat_seconds
    n_samples = round(total_seconds * sample_rate)
    drums = np.zeros((n_samples, 2), dtype=np.float64)

    kick_sound = kick(sample_rate, rng)
    for beat in range(bars * 4):
        add_event(drums, pan_mono(kick_sound, 0), round(beat * beat_seconds * sample_rate))

    for bar in range(bars):
        for beat in (1, 3):
            sound = clap(sample_rate, rng)
            start = round((bar * 4 + beat) * beat_seconds * sample_rate)
            add_event(drums, pan_mono(sound, -0.08 if beat == 1 else 0.08), start)

    for eighth in range(bars * 8):
        beat_position = eighth / 2
        is_offbeat = eighth % 2 == 1
        sound = hat(sample_rate, rng, open_hat=is_offbeat and eighth % 4 == 1)
        level = 0.62 if is_offbeat else 0.20
        pan = -0.32 if eighth % 4 < 2 else 0.32
        add_event(drums, pan_mono(level * sound, pan), round(beat_position * beat_seconds * sample_rate))

    bass_mono = np.zeros(n_samples, dtype=np.float64)
    pulse_left = np.zeros(n_samples, dtype=np.float64)
    pulse_right = np.zeros(n_samples, dtype=np.float64)
    n_steps = bars * 16
    for step in range(n_steps):
        swing_offset = 0.08 * step_seconds if step % 2 else 0
        start = round((step * step_seconds + swing_offset) * sample_rate)
        bass_event = BASS_PATTERN[step % len(BASS_PATTERN)]
        if bass_event is not None:
            note, accent, gate = bass_event
            bass_note = synth_note(midi_frequency(note), gate * step_seconds, sample_rate, accent)
            add_event(bass_mono, bass_note, start)

        pulse_note_number = PULSE_PATTERN[step % len(PULSE_PATTERN)]
        if pulse_note_number is not None:
            pulse_note = synth_note(midi_frequency(pulse_note_number), 0.55 * step_seconds, sample_rate, 0.28)
            target = pulse_left if step % 4 < 2 else pulse_right
            add_event(target, pulse_note, start)

    phase_in_step = np.mod(np.arange(n_samples) / sample_rate, step_seconds) / step_seconds
    cutoff = 230 + 1120 * np.exp(-4.8 * phase_in_step)
    bass_mono = lowpass(bass_mono, cutoff, sample_rate)
    bass_track = 0.82 * pan_mono(np.tanh(1.35 * bass_mono), 0)
    pulse_track = np.column_stack((0.10 * pulse_left, 0.10 * pulse_right))

    phase_in_beat = np.mod(np.arange(n_samples) / sample_rate, beat_seconds)
    ducking = 0.32 + 0.68 * (1 - np.exp(-8.5 * phase_in_beat))
    bass_track *= ducking[:, None]
    pulse_track *= ducking[:, None]
    tonal = bass_track + pulse_track
    mix = drums + tonal
    mix = np.tanh(1.22 * mix) / np.tanh(1.22)
    peak = float(np.max(np.abs(mix)))
    scale = 0.94 / max(peak, 0.94)
    mix *= scale
    drums *= scale
    bass_track *= scale
    tonal *= scale
    drums *= min(1.0, 0.98 / max(float(np.max(np.abs(drums))), 0.98))
    bass_track *= min(1.0, 0.98 / max(float(np.max(np.abs(bass_track))), 0.98))
    tonal *= min(1.0, 0.98 / max(float(np.max(np.abs(tonal))), 0.98))

    fade_samples = round(0.012 * sample_rate)
    fade = np.linspace(0, 1, fade_samples)
    for audio in (mix, drums, bass_track, tonal):
        audio[:fade_samples] *= fade[:, None]
        audio[-fade_samples:] *= fade[::-1, None]
    return {"mix": mix, "drums": drums, "bass": bass_track, "synths": tonal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="audio_toys/audio/four_on_floor")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.sample_rate < 8000 or args.bpm <= 0 or args.bars < 1:
        parser.error("sample-rate >= 8000, bpm > 0, and bars >= 1 are required")

    tracks = synthesize(args.sample_rate, args.bpm, args.bars, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    paths = {}
    for name, audio in tracks.items():
        path = os.path.join(args.output_dir, f"four_on_floor_{name}.wav")
        write_wav(path, audio, args.sample_rate)
        paths[name] = path
    metadata = {
        "bpm": args.bpm,
        "bars": args.bars,
        "sample_rate": args.sample_rate,
        "duration_seconds": args.bars * 4 * 60 / args.bpm,
        "seed": args.seed,
        "description": "Original four-on-the-floor loop with kick, clap, hats, and a swung syncopated bassline.",
        "files": paths,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
