"""
Module 2 -- Speech Recognition (Mic)

Detects the phrase "I need a hug" and emits `speech_hug_request`.

Uses openai-whisper on short rolling audio windows and fuzzy-matches the
transcript against the target phrase -- transcription is never going to be
exact, so an exact string match would miss real matches.

Standalone test (opens a live waveform + transcript visualization window):
    python module2_speech/main.py

Used by the Brain:
    from module2_speech.main import run_speech_module
    run_speech_module(on_event=my_callback, stop_event=threading.Event())
"""
from __future__ import annotations

import sys
import time
import queue
import threading
from pathlib import Path
from difflib import SequenceMatcher
from typing import Callable, Optional

import cv2
import numpy as np
import sounddevice as sd
import whisper

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.events import EventType, make_event

TARGET_PHRASE = "i need a hug"
SIMILARITY_THRESHOLD = 0.75      # 0-1, tune against false positives/negatives
WINDOW_SECONDS = 3.0             # how much audio to transcribe at a time
SAMPLE_RATE = 16000               # whisper wants 16kHz mono
COOLDOWN_AFTER_FIRE_SECONDS = 15.0

WINDOW_NAME = "Module 2 - Speech (q to quit)"
DISPLAY_SECONDS = 3.0             # how much waveform history to show at once
CANVAS_W, CANVAS_H = 640, 360

COLOR_NEUTRAL = (200, 200, 200)
COLOR_LISTENING = (255, 200, 0)     # buffering audio, waiting on a full window
COLOR_TRANSCRIBING = (0, 165, 255)  # whisper is running (window pauses briefly here)
COLOR_MATCH = (0, 255, 0)           # similarity above threshold
COLOR_FIRED = (0, 0, 255)           # just fired
COLOR_COOLDOWN = (128, 128, 128)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _draw_frame(display_buffer, state, transcript, sim, in_cooldown, cooldown_remaining):
    """Renders a canvas: rolling waveform + level meter + transcript + similarity bar."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # --- waveform ---
    mid_y = 140
    if len(display_buffer) > 1:
        xs = np.linspace(0, CANVAS_W, num=len(display_buffer), dtype=np.int32)
        ys = (mid_y - display_buffer * 120).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 200, 255), thickness=1)
    cv2.line(canvas, (0, mid_y), (CANVAS_W, mid_y), (60, 60, 60), 1)

    # --- input level meter ---
    level = float(np.abs(display_buffer).mean()) if len(display_buffer) else 0.0
    bar_w = int(min(1.0, level * 8) * (CANVAS_W - 20))
    cv2.rectangle(canvas, (10, 220), (CANVAS_W - 10, 240), (60, 60, 60), -1)
    cv2.rectangle(canvas, (10, 220), (10 + bar_w, 240), (0, 200, 255), -1)
    cv2.putText(canvas, "input level", (10, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEUTRAL, 1)

    # --- status banner ---
    if in_cooldown:
        banner, color = f"COOLDOWN ({cooldown_remaining:.0f}s left)", COLOR_COOLDOWN
    else:
        banner, color = {
            "listening": ("listening...", COLOR_LISTENING),
            "transcribing": ("transcribing...", COLOR_TRANSCRIBING),
            "match": ("MATCH! firing hug request", COLOR_FIRED),
        }.get(state, ("listening...", COLOR_LISTENING))
    cv2.putText(canvas, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- last transcript + similarity ---
    cv2.putText(canvas, f'target: "{TARGET_PHRASE}"', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEUTRAL, 1)
    cv2.putText(canvas, f'heard:  "{transcript}"', (10, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    sim_color = COLOR_MATCH if sim >= SIMILARITY_THRESHOLD else COLOR_NEUTRAL
    cv2.putText(canvas, f"similarity: {sim:.2f}  (threshold {SIMILARITY_THRESHOLD})", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sim_color, 1)

    bar_w2 = int(min(1.0, sim) * (CANVAS_W - 20))
    thresh_x = 10 + int(SIMILARITY_THRESHOLD * (CANVAS_W - 20))
    cv2.rectangle(canvas, (10, 330), (CANVAS_W - 10, 344), (60, 60, 60), -1)
    cv2.rectangle(canvas, (10, 330), (10 + bar_w2, 344), sim_color, -1)
    cv2.line(canvas, (thresh_x, 326), (thresh_x, 348), (255, 255, 255), 1)

    return canvas


def run_speech_module(
    on_event: Callable,
    stop_event: Optional[threading.Event] = None,
    model_name: str = "base",
    show_window: bool = True,
):
    """
    Runs the mic loop. Calls on_event(Event) whenever the target phrase is
    heard with high enough similarity. Blocks until stop_event is set.

    show_window=True (default) opens a live window with a rolling waveform,
    an input-level meter, the last transcript heard, and a similarity bar
    against the match threshold. Press 'q' or close the window to stop
    (this also flips stop_event, so it stops any brain threads too).

    Note: the window won't repaint during the actual whisper.transcribe()
    call, since that's a blocking call on this same thread -- normally
    under a second with the default "base" model. The banner flips to
    "transcribing..." right before it starts so a brief pause there reads
    as "processing," not "hung."
    """
    stop_event = stop_event or threading.Event()
    print(f"[module2_speech] loading whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[module2_speech] audio status: {status}")
        audio_q.put(indata.copy())

    last_fired_at = 0.0
    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    display_samples = int(DISPLAY_SECONDS * SAMPLE_RATE)
    buffer = np.zeros((0, 1), dtype=np.float32)

    last_transcript = ""
    last_sim = 0.0
    state = "listening"

    print("[module2_speech] mic loop started, listening for 'I need a hug'...")
    if show_window:
        cv2.namedWindow(WINDOW_NAME)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_callback):
            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    chunk = None

                if chunk is not None:
                    buffer = np.concatenate([buffer, chunk], axis=0)

                now = time.time()
                in_cooldown = (now - last_fired_at) < COOLDOWN_AFTER_FIRE_SECONDS
                cooldown_remaining = COOLDOWN_AFTER_FIRE_SECONDS - (now - last_fired_at)

                if len(buffer) >= window_samples:
                    window = buffer[:window_samples, 0]
                    buffer = buffer[window_samples:]  # non-overlapping windows -- simplest correct version

                    state = "transcribing"
                    if show_window:
                        display_slice = window[-display_samples:] if len(window) >= display_samples else window
                        canvas = _draw_frame(display_slice, state, last_transcript, last_sim, in_cooldown, cooldown_remaining)
                        cv2.imshow(WINDOW_NAME, canvas)
                        cv2.waitKey(1)  # flush the "transcribing..." frame before the blocking call below

                    result = model.transcribe(window, fp16=False, language="en")
                    transcript = result.get("text", "").strip().lower()
                    last_transcript = transcript or last_transcript
                    last_sim = _similarity(transcript, TARGET_PHRASE) if transcript else 0.0

                    if transcript and last_sim >= SIMILARITY_THRESHOLD and not in_cooldown:
                        state = "match"
                        event = make_event(
                            EventType.SPEECH_HUG_REQUEST,
                            source="module2_speech",
                            transcript=transcript,
                            similarity=last_sim,
                        )
                        print(f"[module2_speech] firing {event.type.value} (heard: '{transcript}', sim={last_sim:.2f})")
                        on_event(event)
                        last_fired_at = now
                    else:
                        state = "listening"

                if show_window:
                    display_slice = buffer[-display_samples:, 0] if len(buffer) else np.zeros(0, dtype=np.float32)
                    canvas = _draw_frame(display_slice, state, last_transcript, last_sim, in_cooldown, cooldown_remaining)
                    cv2.imshow(WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                    if key == ord("q") or closed:
                        print("[module2_speech] window closed / 'q' pressed, stopping")
                        stop_event.set()
    finally:
        if show_window:
            cv2.destroyWindow(WINDOW_NAME)
        print("[module2_speech] mic loop stopped")


if __name__ == "__main__":
    # Standalone test: prints events as they fire and shows the live waveform/transcript window.
    run_speech_module(on_event=lambda e: print("EVENT:", e.to_json()))