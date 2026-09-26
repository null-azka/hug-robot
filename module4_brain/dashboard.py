"""
Module 4 -- Brain dashboard (live trigger + state visualization).

Shows the state machine (Idle / Giving hug / Cooldown) with the current
state highlighted, a scrolling log of recent trigger events color-coded
by source, a cooldown progress bar, and an emergency-stop banner.

This is created and updated on the Brain's own main thread (inside
Brain.run()), not a background thread -- unlike Modules 1 and 2, the
Brain's main loop isn't itself doing blocking camera/mic I/O, so it's
safe and simple to redraw here directly.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque

import cv2
import numpy as np

WINDOW_NAME = "Module 4 - Brain (q to quit)"
CANVAS_W, CANVAS_H = 720, 420
LOG_MAX = 8

COLOR_NEUTRAL = (200, 200, 200)
COLOR_IDLE = (255, 200, 0)
COLOR_GIVING_HUG = (0, 0, 255)
COLOR_COOLDOWN = (150, 150, 150)
COLOR_ESTOP = (0, 0, 255)
COLOR_DIM = (70, 70, 70)

# BGR colors per trigger source, so the log reads at a glance which
# module fired -- matches the palette used in each module's own viz.
SOURCE_COLORS = {
    "module1_face": (255, 140, 0),      # orange-ish, matches Module 1's "sad" box color
    "module2_speech": (0, 200, 255),    # matches Module 2's waveform color
    "software_button": (0, 255, 0),     # green
}


class Dashboard:
    def __init__(self):
        self.log: Deque[str] = deque(maxlen=LOG_MAX)
        self.log_sources: Deque[str] = deque(maxlen=LOG_MAX)
        cv2.namedWindow(WINDOW_NAME)

    def log_event(self, source: str, event_type: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendleft(f"[{ts}] {event_type} <- {source}")
        self.log_sources.appendleft(source)

    def _render(self, state: str, cooldown_remaining: float, cooldown_total: float, estop: bool) -> np.ndarray:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # --- state machine: three boxes with arrows between them ---
        box_y, box_w, box_h = 40, 160, 60
        boxes = [("IDLE", "idle", 40), ("GIVING HUG", "giving_hug", 280), ("COOLDOWN", "cooldown", 520)]
        for label, key, x in boxes:
            active = (state == key)
            if estop:
                color = COLOR_ESTOP if active else COLOR_DIM
            elif key == "giving_hug":
                color = COLOR_GIVING_HUG
            elif key == "cooldown":
                color = COLOR_COOLDOWN
            else:
                color = COLOR_IDLE
            if active:
                cv2.rectangle(canvas, (x, box_y), (x + box_w, box_y + box_h), color, -1)
                text_color = (0, 0, 0)
            else:
                cv2.rectangle(canvas, (x, box_y), (x + box_w, box_y + box_h), COLOR_DIM, 2)
                text_color = COLOR_DIM
            cv2.putText(canvas, label, (x + 14, box_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        cv2.arrowedLine(canvas, (200, 70), (280, 70), COLOR_NEUTRAL, 2, tipLength=0.2)
        cv2.arrowedLine(canvas, (440, 70), (520, 70), COLOR_NEUTRAL, 2, tipLength=0.2)
        cv2.putText(canvas, "back to idle", (420, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEUTRAL, 1)
        cv2.arrowedLine(canvas, (600, 100), (600, 118), COLOR_NEUTRAL, 1, tipLength=0.35)
        cv2.line(canvas, (600, 118), (120, 118), COLOR_NEUTRAL, 1)
        cv2.arrowedLine(canvas, (120, 118), (120, 100), COLOR_NEUTRAL, 1, tipLength=0.35)

        # --- e-stop banner (overrides everything else when tripped) ---
        if estop:
            cv2.putText(canvas, "EMERGENCY STOP TRIPPED", (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ESTOP, 2)

        # --- cooldown progress bar ---
        if state == "cooldown" and cooldown_total > 0:
            progress = max(0.0, min(1.0, 1.0 - (cooldown_remaining / cooldown_total)))
            bar_w = int(progress * (CANVAS_W - 80))
            cv2.putText(canvas, f"cooldown: {cooldown_remaining:.1f}s left", (40, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEUTRAL, 1)
            cv2.rectangle(canvas, (40, 160), (CANVAS_W - 40, 178), (60, 60, 60), -1)
            cv2.rectangle(canvas, (40, 160), (40 + bar_w, 178), COLOR_COOLDOWN, -1)

        # --- recent trigger log ---
        cv2.putText(canvas, "recent triggers:", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NEUTRAL, 1)
        for i, (line, source) in enumerate(zip(self.log, self.log_sources)):
            y = 235 + i * 22
            color = SOURCE_COLORS.get(source, (255, 255, 255))
            cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if not self.log:
            cv2.putText(canvas, "(none yet)", (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1)

        # --- legend ---
        legend_y = CANVAS_H - 20
        legend_x = 40
        for source, color in SOURCE_COLORS.items():
            cv2.circle(canvas, (legend_x, legend_y - 4), 5, color, -1)
            cv2.putText(canvas, source, (legend_x + 12, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            legend_x += 190

        return canvas

    def update(self, state: str, cooldown_remaining: float, cooldown_total: float, estop: bool) -> bool:
        """Draws the current frame. Returns True if the user asked to quit ('q' or closed window)."""
        canvas = self._render(state, cooldown_remaining, cooldown_total, estop)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        return key == ord("q") or closed

    def close(self):
        cv2.destroyWindow(WINDOW_NAME)