"""Dummy observations for the ALOHA right-arm multi-task policy server.

Two formats, because the server speaks a different one depending on how it was started:

    server WITHOUT --use-sim-policy-wrapper  -> Gr00tPolicy          -> nested   (use dummy_nested)
    server WITH    --use-sim-policy-wrapper  -> Gr00tSimPolicyWrapper -> flat     (use dummy_flat)

Both shapes below were checked against the real validators
(Gr00tPolicy.check_observation / Gr00tSimPolicyWrapper.check_observation) on the
checkpoint-8000 weights. For a real robot, prefer the nested form and start the
server without the sim wrapper — the wrapper exists only to retro-fit the GR00T
sim environments.

Run standalone to fire one dummy request at a server:

    python examples/ALOHA/dummy_obs_right_arm.py --host 127.0.0.1 --port 5555
    python examples/ALOHA/dummy_obs_right_arm.py --port 5555 --sim-wrapper   # if server has the flag
"""

import numpy as np


H, W = 480, 640
DRAWER = "Use the right gripper to open the top drawer, pick up the red square block, place it inside the top drawer, and then close the drawer."
KITCHEN = "Use the right gripper to pick up the banana and place it into the pot. Then pick up the lid with the right gripper and place it on top of the pot to close it."


def dummy_nested(instruction: str = DRAWER, batch: int = 1) -> dict:
    """For a server started WITHOUT --use-sim-policy-wrapper (plain Gr00tPolicy)."""
    return {
        "video": {
            "front": np.zeros((batch, 1, H, W, 3), dtype=np.uint8),
            "wrist": np.zeros((batch, 1, H, W, 3), dtype=np.uint8),
        },
        "state": {
            "single_arm": np.zeros((batch, 1, 6), dtype=np.float32),
            "gripper": np.zeros((batch, 1, 1), dtype=np.float32),
        },
        # nested under "language"; each batch item is a LIST holding one string
        "language": {"annotation.human.task_description": [[instruction]] * batch},
    }


def dummy_flat(instruction: str = DRAWER, batch: int = 1) -> dict:
    """For a server started WITH --use-sim-policy-wrapper (Gr00tSimPolicyWrapper)."""
    return {
        "video.front": np.zeros((batch, 1, H, W, 3), dtype=np.uint8),
        "video.wrist": np.zeros((batch, 1, H, W, 3), dtype=np.uint8),
        "state.single_arm": np.zeros((batch, 1, 6), dtype=np.float32),
        "state.gripper": np.zeros((batch, 1, 1), dtype=np.float32),
        # flat, and each batch item is a bare STRING — not a nested list
        "annotation.human.task_description": [instruction] * batch,
    }


def main(host: str = "127.0.0.1", port: int = 5556, sim_wrapper: bool = False):
    from gr00t.policy.server_client import PolicyClient

    obs = dummy_flat() if sim_wrapper else dummy_nested()
    print(f"format: {'flat (sim wrapper)' if sim_wrapper else 'nested (Gr00tPolicy)'}")
    for k, v in obs.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                shown = np.asarray(vv).shape if isinstance(vv, np.ndarray) else type(vv).__name__
                print(f"  {k}.{kk}: {shown}")
        else:
            shown = v.shape if isinstance(v, np.ndarray) else v
            print(f"  {k}: {shown}")

    client = PolicyClient(host=host, port=port, timeout_ms=600_000)
    action, _ = client.get_action(obs)   # ONE positional dict — no ** unpacking
    print("\naction:")
    for k, v in action.items():
        print(f"  {k}: {np.asarray(v).shape}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
    