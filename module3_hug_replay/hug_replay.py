"""
Module 3 -- Hug Replay (Dual Arm)

Replays the recorded hug trajectory on both follower arms in parallel.
No training, no policy inference -- just two `lerobot-replay` subprocesses
launched together and waited on.

Called by the Brain as:
    from module3_hug_replay.hug_replay import run_hug
    run_hug()

Standalone test (run with no person present first):
    python module3_hug_replay/hug_replay.py
"""
from __future__ import annotations

import subprocess
import time

# --- Configuration: update to match your actual ports/ids/dataset ---
DATASET_REPO_ID = "null-azka/record-hug-v5_20260919_192922"
DATASET_EPISODE = 1

ARM_LEFT = {"port": "COM5", "id": "follower-left"}
ARM_RIGHT = {"port": "COM7", "id": "follower-right"}  # set this once follower-right is calibrated


def _replay_cmd(arm: dict) -> list[str]:
    return [
        "lerobot-replay",
        "--robot.type=so101_follower",
        f"--robot.port={arm['port']}",
        f"--robot.id={arm['id']}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.episode={DATASET_EPISODE}",
    ]


def run_hug(timeout_seconds: float = 30.0) -> bool:
    """
    Launches replay on both arms as close together in time as possible,
    waits for both to finish, and returns True if both exited cleanly.
    """
    print("[module3_hug_replay] starting dual-arm replay...")
    proc_left = subprocess.Popen(_replay_cmd(ARM_LEFT))
    proc_right = subprocess.Popen(_replay_cmd(ARM_RIGHT))

    start = time.time()
    left_code = right_code = None
    while time.time() - start < timeout_seconds:
        left_code = proc_left.poll()
        right_code = proc_right.poll()
        if left_code is not None and right_code is not None:
            break
        time.sleep(0.05)
    else:
        print("[module3_hug_replay] timeout waiting for replay to finish -- killing both")
        proc_left.kill()
        proc_right.kill()
        return False

    ok = (left_code == 0) and (right_code == 0)
    if ok:
        print("[module3_hug_replay] both arms finished replay cleanly")
    else:
        print(f"[module3_hug_replay] replay exited with codes left={left_code} right={right_code}")
    return ok


if __name__ == "__main__":
    run_hug()
