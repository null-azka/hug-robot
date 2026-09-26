"""
"Presses" the software button by sending a trigger to the running Brain.

Run this any time module4_brain/main.py is already running:
    python module4_brain\\trigger_button.py

This is deliberately just a script -- point a keyboard shortcut, a
scheduled task, a webpage button, or anything else that can run a
command at it, and that becomes your "button."
"""
import socket

from software_button import HOST, PORT  # same defaults the Brain listens on

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"TRIGGER")
    print("Sent hug trigger.")