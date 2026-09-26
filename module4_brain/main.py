"""
Module 4 -- Brain

State machine: Idle -> Giving hug -> Cooldown -> Idle.
Listens for events from Module 1 (camera thread), Module 2 (mic thread),
and the software button (localhost trigger, see software_button.py).
Calls Module 3's run_hug().

The emergency stop is still hardware (Arduino/Pico, serial) on purpose --
it's checked before every step and short-circuits straight to shutdown,
so it doesn't depend on the same software path everything else runs
through. The button itself no longer needs any hardware.

Run with:
    python module4_brain/main.py
"""
from __future__ import annotations

import sys
import time
import queue
import threading
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import serial  # pyserial -- used only for the hardware e-stop now

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.events import Event
from module1_face.main import run_face_module
from module2_speech.main import run_speech_module
from module3_hug_replay.hug_replay import run_hug
from module4_brain.software_button import run_software_button
from module4_brain.dashboard import Dashboard

ENABLE_ESTOP = False            # flip to True once you have the Arduino/Pico e-stop wired up
ESTOP_SERIAL_PORT = "COM8"      # update to match your Arduino/Pico's port
ESTOP_BAUD_RATE = 9600
COOLDOWN_SECONDS = 12.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("brain.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("brain")


class State(Enum):
    IDLE = "idle"
    GIVING_HUG = "giving_hug"
    COOLDOWN = "cooldown"


class Brain:
    def __init__(self, show_dashboard: bool = True):
        self.event_q: "queue.Queue[Event]" = queue.Queue()
        self.stop_event = threading.Event()
        self.estop_event = threading.Event()
        self.state = State.IDLE
        self.cooldown_until = 0.0
        self.show_dashboard = show_dashboard
        self.dashboard: Optional[Dashboard] = None

    # ---- event sources ----

    def _on_event(self, event: Event):
        self.event_q.put(event)
        if self.dashboard:
            self.dashboard.log_event(event.source, event.type.value)

    def _start_perception_threads(self):
        threading.Thread(
            target=run_face_module,
            # show_window=True: on Windows, OpenCV's HighGUI backend tolerates
            # one window per thread fine (the strict main-thread-only rule is
            # really a macOS/Cocoa restriction). If this ever gets flaky or
            # you move this to macOS/Linux, set this back to False and run
            # `python module1_face/main.py` standalone instead.
            kwargs=dict(on_event=self._on_event, stop_event=self.stop_event, show_window=True),
            daemon=True,
        ).start()
        threading.Thread(
            target=run_speech_module,
            # show_window=True: same Windows-specific reasoning as Module 1.
            kwargs=dict(on_event=self._on_event, stop_event=self.stop_event, show_window=True),
            daemon=True,
        ).start()
        threading.Thread(
            target=run_software_button,
            kwargs=dict(on_event=self._on_event, stop_event=self.stop_event),
            daemon=True,
        ).start()
        if ENABLE_ESTOP:
            threading.Thread(target=self._listen_for_estop, daemon=True).start()
        else:
            log.warning("ENABLE_ESTOP is False -- running with no e-stop. "
                        "Fine for now, but wire one up before running near anyone.")

    def _listen_for_estop(self):
        """Reads line-delimited serial messages from the Arduino/Pico.
        Firmware sends 'ESTOP' repeatedly while the hardware e-stop
        line is open. This is the only thing the Arduino/Pico does now."""
        try:
            ser = serial.Serial(ESTOP_SERIAL_PORT, ESTOP_BAUD_RATE, timeout=1)
        except serial.SerialException as e:
            log.error(f"could not open e-stop serial port {ESTOP_SERIAL_PORT}: {e}")
            return

        log.info(f"listening for e-stop on {ESTOP_SERIAL_PORT}")
        while not self.stop_event.is_set():
            line = ser.readline().decode(errors="ignore").strip()
            if line == "ESTOP":
                log.warning("hardware e-stop signal received")
                self.estop_event.set()

    # ---- state machine ----

    def run(self):
        self._start_perception_threads()
        if self.show_dashboard:
            self.dashboard = Dashboard()
        log.info(f"brain started, state={self.state.value}")
        try:
            while True:
                if self._update_dashboard_and_check_quit():
                    break

                if self.estop_event.is_set():
                    self._handle_estop()
                    continue

                if self.state == State.IDLE:
                    self._idle_step()
                elif self.state == State.GIVING_HUG:
                    self._giving_hug_step()
                elif self.state == State.COOLDOWN:
                    self._cooldown_step()
        except KeyboardInterrupt:
            log.info("shutting down (Ctrl+C)")
            self.stop_event.set()
        finally:
            if self.dashboard:
                self.dashboard.close()

    def _update_dashboard_and_check_quit(self) -> bool:
        if not self.dashboard:
            return False
        cooldown_remaining = max(0.0, self.cooldown_until - time.time()) if self.state == State.COOLDOWN else 0.0
        quit_requested = self.dashboard.update(self.state.value, cooldown_remaining, COOLDOWN_SECONDS, self.estop_event.is_set())
        if quit_requested:
            log.info("dashboard window closed / 'q' pressed, shutting down")
            self.stop_event.set()
        return quit_requested

    def _transition(self, new_state: State):
        log.info(f"state transition: {self.state.value} -> {new_state.value}")
        self.state = new_state

    def _idle_step(self):
        try:
            event = self.event_q.get(timeout=0.2)
        except queue.Empty:
            return
        log.info(f"trigger received: {event.type.value} from {event.source}")
        self._transition(State.GIVING_HUG)

    def _giving_hug_step(self):
        # Drain any events that arrived mid-hug so they don't pile up.
        while not self.event_q.empty():
            self.event_q.get_nowait()

        success = run_hug()
        if not success:
            log.warning("run_hug() did not complete cleanly")
        self.cooldown_until = time.time() + COOLDOWN_SECONDS
        self._transition(State.COOLDOWN)

    def _cooldown_step(self):
        while not self.event_q.empty():
            self.event_q.get_nowait()  # ignored during cooldown, per the plan

        if time.time() >= self.cooldown_until:
            self._transition(State.IDLE)
        else:
            time.sleep(0.1)

    def _handle_estop(self):
        # Safety-critical: this path intentionally does NOT go through
        # run_hug() or any of the normal state-machine logic above.
        log.critical("EMERGENCY STOP -- halting brain loop")
        self.stop_event.set()
        raise SystemExit("e-stop triggered, brain halted")


if __name__ == "__main__":
    Brain().run()