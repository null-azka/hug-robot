"""
Software button -- replaces the physical push button.

The Brain runs a tiny localhost TCP listener. Anything that connects and
sends the word TRIGGER fires a button_pressed event. "Pressing the button"
is just running trigger_button.py (or anything else that opens a socket
and sends that word) -- no wiring required.

Used by the Brain:
    from module4_brain.software_button import run_software_button
    run_software_button(on_event=my_callback, stop_event=threading.Event())
"""
from __future__ import annotations

import sys
import socket
import threading
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.events import EventType, make_event

HOST = "127.0.0.1"
PORT = 5005


def run_software_button(
    on_event: Callable,
    stop_event: Optional[threading.Event] = None,
    host: str = HOST,
    port: int = PORT,
):
    """
    Runs the localhost listener. Calls on_event(Event) whenever a client
    connects and sends "TRIGGER". Blocks until stop_event is set.
    """
    stop_event = stop_event or threading.Event()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)  # lets us check stop_event periodically instead of blocking forever

    print(f"[software_button] listening on {host}:{port} -- run trigger_button.py to fire a hug")
    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                data = conn.recv(1024).decode(errors="ignore").strip()
                if data == "TRIGGER":
                    event = make_event(EventType.BUTTON_PRESSED, source="software_button")
                    print(f"[software_button] firing {event.type.value}")
                    on_event(event)
    finally:
        server.close()
        print("[software_button] stopped")


if __name__ == "__main__":
    # Standalone test: just print events as they fire.
    run_software_button(on_event=lambda e: print("EVENT:", e.to_json()))