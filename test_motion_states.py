"""Preview Clive's emotional motion states in MuJoCo.

Usage:
    python test_motion_states.py happy
    python test_motion_states.py sleepy
    python test_motion_states.py --list

This is for tuning motion clips before wiring them into the live perception
pipeline. Each state drives the four current actuators:
    ankle_pitch_pos, knee_pitch_pos, hip_pitch_pos, head_spin_pos
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer


STATE_NAMES = (
    "idle",
    "sleepy",
    "concerned",
    "curious",
    "waving",
    "happy",
    "engaged",
)


def sin01(t, speed=1.0, phase=0.0):
    return 0.5 + 0.5 * math.sin(t * speed + phase)


def wave(t, speed=1.0, amp=1.0, phase=0.0):
    return amp * math.sin(t * speed + phase)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def state_targets(state, t):
    """Return actuator targets: ankle, knee, hip, head."""
    if state == "idle":
        return (
            wave(t, 1.0, 0.035),
            -0.10 - 0.025 * sin01(t, 1.2),
            wave(t, 0.8, 0.045),
            wave(t, 0.7, 0.08),
        )

    if state == "sleepy":
        return (
            -0.08 + wave(t, 0.45, 0.025),
            -0.55 - 0.05 * sin01(t, 0.55),
            0.48 + wave(t, 0.35, 0.08),
            wave(t, 0.35, 0.12),
        )

    if state == "concerned":
        return (
            0.08 + wave(t, 0.9, 0.025),
            -0.30 - 0.04 * sin01(t, 1.1),
            0.42 + wave(t, 0.6, 0.07),
            wave(t, 1.5, 0.16),
        )

    if state == "curious":
        look = 0.52 * math.tanh(math.sin(t * 0.55) * 2.0)
        return (
            wave(t, 0.8, 0.08),
            -0.16 - 0.03 * sin01(t, 1.0),
            wave(t, 0.55, 0.25),
            look + wave(t, 2.0, 0.06),
        )

    if state == "waving":
        bounce = sin01(t, 5.0)
        return (
            wave(t, 5.0, 0.16),
            -0.20 - 0.34 * bounce,
            wave(t, 3.3, 0.23),
            wave(t, 5.8, 0.58),
        )

    if state == "happy":
        bounce = sin01(t, 3.8)
        return (
            wave(t, 3.8, 0.13),
            -0.16 - 0.28 * bounce,
            wave(t, 2.9, 0.20),
            wave(t, 3.6, 0.42),
        )

    if state == "engaged":
        return (
            wave(t, 1.2, 0.05),
            -0.14 - 0.03 * sin01(t, 1.3),
            0.22 + wave(t, 0.9, 0.05),
            wave(t, 1.1, 0.14),
        )

    raise ValueError(f"Unknown state: {state}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview one Clive motion state.")
    parser.add_argument("state", nargs="?", choices=STATE_NAMES)
    parser.add_argument("--list", action="store_true", help="List available states and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\n".join(STATE_NAMES))
        return

    if args.state is None:
        valid = ", ".join(STATE_NAMES)
        raise SystemExit(f"Choose a state. Valid states: {valid}")

    model = mujoco.MjModel.from_xml_path("clive.xml")
    data = mujoco.MjData(model)

    actuator_ids = {
        "ankle": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_pitch_pos"),
        "knee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_pitch_pos"),
        "hip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_pitch_pos"),
        "head": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_spin_pos"),
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Testing {args.state}")
        t0 = time.time()

        while viewer.is_running():
            t = time.time() - t0
            ankle, knee, hip, head = state_targets(args.state, t)

            data.ctrl[actuator_ids["ankle"]] = clamp(ankle, -1.2, 1.2)
            data.ctrl[actuator_ids["knee"]] = clamp(knee, -1.5708, 0.0)
            data.ctrl[actuator_ids["hip"]] = clamp(hip, -1.5708, 1.5708)
            data.ctrl[actuator_ids["head"]] = clamp(head, -1.0, 1.0)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
