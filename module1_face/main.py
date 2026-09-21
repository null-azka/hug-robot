"""
Module 1 -- Facial Recognition (Camera)

Detects a sustained sad expression and emits `face_sad_detected`.

Standalone test:
    python module1_face/main.py

Used by the Brain:
    from module1_face.main import run_face_module
    run_face_module(on_event=my_callback, stop_event=threading.Event())
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
from fer.fer import FER

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.events import EventType, make_event

SAD_LABEL = "sad"
SAD_CONFIDENCE_THRESHOLD = 0.4       # how confident the classifier must be
SAD_PERSISTENCE_SECONDS = 1.5        # how long "sad" must persist before firing
COOLDOWN_AFTER_FIRE_SECONDS = 15.0   # don't refire immediately after emitting
HEARTBEAT_SECONDS = 2.0              # print a status line this often so the loop never looks "stuck"


WINDOW_NAME = "Module 1 - Face (q to quit)"

# BGR colors for the overlay
COLOR_NEUTRAL = (200, 200, 200)
COLOR_SAD_BUILDING = (0, 165, 255)   # orange -- sad, timer counting up
COLOR_SAD_FIRED = (0, 0, 255)        # red -- just fired
COLOR_COOLDOWN = (128, 128, 128)     # grey -- in cooldown, box drawn but muted


def _draw_annotations(frame, faces, sad_since, now, in_cooldown, cooldown_remaining):
    """Draws a box + label per detected face, plus a status banner at the top."""
    for face in faces:
        x, y, w, h = face["box"]
        emotions = face["emotions"]
        label = max(emotions, key=emotions.get)
        score = emotions[label]

        if in_cooldown:
            color = COLOR_COOLDOWN
        elif label == SAD_LABEL and score >= SAD_CONFIDENCE_THRESHOLD:
            color = COLOR_SAD_FIRED if sad_since and (now - sad_since) >= SAD_PERSISTENCE_SECONDS else COLOR_SAD_BUILDING
        else:
            color = COLOR_NEUTRAL

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, f"{label} {score:.2f}", (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # Progress bar under the face box showing sad-persistence countdown to firing.
        if sad_since is not None and not in_cooldown:
            progress = min(1.0, (now - sad_since) / SAD_PERSISTENCE_SECONDS)
            bar_w = int(w * progress)
            cv2.rectangle(frame, (x, y + h + 4), (x + w, y + h + 10), (60, 60, 60), -1)
            cv2.rectangle(frame, (x, y + h + 4), (x + bar_w, y + h + 10), color, -1)

    banner = "COOLDOWN" if in_cooldown else "watching for sustained sad expression"
    banner_color = COLOR_COOLDOWN if in_cooldown else COLOR_NEUTRAL
    if in_cooldown:
        banner += f" ({cooldown_remaining:.0f}s left)"
    cv2.putText(frame, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)
    return frame


def run_face_module(
    on_event: Callable,
    stop_event: Optional[threading.Event] = None,
    camera_index: int = 0,
    use_mtcnn: bool = False,
    show_window: bool = True,
):
    """
    Runs the camera loop. Calls on_event(Event) whenever a sustained sad
    face is detected. Blocks until stop_event is set (or forever if none given).

    use_mtcnn=False (default) uses FER's faster OpenCV Haar-cascade face
    detector -- accurate enough for this and much closer to real-time on
    CPU than mtcnn=True, which can take 0.5-2s per frame and make the
    loop look hung even though it's just slow.

    show_window=True (default) opens a live OpenCV window with the camera
    feed and a drawn box + emotion label per detected face, a countdown
    bar while a sad expression is building toward firing, and a cooldown
    banner. Press 'q' or close the window to stop (this also flips
    stop_event, so it stops any brain threads reading it too).
    """
    stop_event = stop_event or threading.Event()
    detector = FER(mtcnn=use_mtcnn)

    # On Windows, the default VideoCapture backend (MSMF) can hang
    # indefinitely on some webcams. CAP_DSHOW avoids that.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    sad_since: Optional[float] = None
    last_fired_at: float = 0.0
    last_heartbeat: float = 0.0
    frame_count = 0

    print("[module1_face] camera loop started, watching for sustained sad expression...")
    if show_window:
        cv2.namedWindow(WINDOW_NAME)

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print("[module1_face] frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            faces = detector.detect_emotions(frame)  # list of {"box": (x,y,w,h), "emotions": {...}}

            if faces:
                # Use the largest detected face (closest/most prominent person).
                primary = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                label = max(primary["emotions"], key=primary["emotions"].get)
                score = primary["emotions"][label]
            else:
                label, score = None, None

            is_sad_now = (
                label == SAD_LABEL
                and score is not None
                and score >= SAD_CONFIDENCE_THRESHOLD
            )

            now = time.time()
            in_cooldown = (now - last_fired_at) < COOLDOWN_AFTER_FIRE_SECONDS
            cooldown_remaining = COOLDOWN_AFTER_FIRE_SECONDS - (now - last_fired_at)

            # Heartbeat so you can always tell the loop is alive, and see
            # what it's currently reading even when nothing has fired.
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                fps = frame_count / HEARTBEAT_SECONDS
                print(f"[module1_face] alive -- ~{fps:.1f} fps, current read: {label} ({score})")
                frame_count = 0
                last_heartbeat = now

            if is_sad_now and not in_cooldown:
                if sad_since is None:
                    sad_since = now
                elif (now - sad_since) >= SAD_PERSISTENCE_SECONDS:
                    event = make_event(
                        EventType.FACE_SAD_DETECTED,
                        source="module1_face",
                        confidence=score,
                    )
                    print(f"[module1_face] firing {event.type.value} (confidence={score:.2f})")
                    on_event(event)
                    last_fired_at = now
                    sad_since = None  # require a fresh sustained window before firing again
            else:
                sad_since = None

            if show_window:
                annotated = _draw_annotations(frame, faces, sad_since, now, in_cooldown, cooldown_remaining)
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                if key == ord("q") or closed:
                    print("[module1_face] window closed / 'q' pressed, stopping")
                    stop_event.set()
    finally:
        cap.release()
        if show_window:
            cv2.destroyWindow(WINDOW_NAME)
        print("[module1_face] camera loop stopped")


if __name__ == "__main__":
    # Standalone test: prints events as they fire and shows the annotated camera window.
    run_face_module(on_event=lambda e: print("EVENT:", e.to_json()))