"""Isolated joint tester for Clive.

Usage:
    python test_joints.py              # cycle through every actuator
    python test_joints.py knee_pitch   # test only one joint
    python test_joints.py head_spin    # test limited head spin both directions
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer


JOINT_TESTS = [
    {"name": "ankle_pitch", "actuator": 0, "kind": "position", "range": (-1.2, 1.2)},
    {"name": "knee_pitch", "actuator": 1, "kind": "position", "range": (-1.5708, 0.0)},
    {"name": "hip_pitch", "actuator": 2, "kind": "position", "range": (-1.5708, 1.5708)},
    {"name": "head_spin", "actuator": 3, "kind": "position", "range": (-1.0, 1.0)},
]

ALIASES = {
    test["name"]: test for test in JOINT_TESTS
}
ALIASES.update({
    "ankle": ALIASES["ankle_pitch"],
    "left_ankle": ALIASES["ankle_pitch"],
    "right_ankle": ALIASES["ankle_pitch"],
    "left_ankle_pos": ALIASES["ankle_pitch"],
    "right_ankle_pos": ALIASES["ankle_pitch"],
    "ankle_pitch_pos": ALIASES["ankle_pitch"],
    "knee": ALIASES["knee_pitch"],
    "left_knee": ALIASES["knee_pitch"],
    "right_knee": ALIASES["knee_pitch"],
    "left_knee_pos": ALIASES["knee_pitch"],
    "right_knee_pos": ALIASES["knee_pitch"],
    "knee_pitch_pos": ALIASES["knee_pitch"],
    "hip": ALIASES["hip_pitch"],
    "left_hip": ALIASES["hip_pitch"],
    "right_hip": ALIASES["hip_pitch"],
    "hip_pitch_pos": ALIASES["hip_pitch"],
    "head_spin_vel": ALIASES["head_spin"],
    "head_spin_pos": ALIASES["head_spin"],
})


def target_for(test, local_t):
    lo, hi = test["range"]
    wave = np.sin(local_t * 1.2)

    if test["kind"] == "velocity":
        return hi * wave

    if hi == 0.0 and lo < 0.0:
        bend_amount = 0.5 * (1.0 - np.cos(local_t * 1.2))
        return lo * bend_amount

    midpoint = 0.5 * (lo + hi)
    amplitude = 0.5 * (hi - lo)
    return midpoint + amplitude * wave


def selected_tests():
    if len(sys.argv) == 1:
        return JOINT_TESTS, True

    name = sys.argv[1].strip()
    if name not in ALIASES:
        valid = ", ".join(test["name"] for test in JOINT_TESTS)
        raise SystemExit(f"Unknown joint '{name}'. Valid names: {valid}")

    return [ALIASES[name]], False


def main():
    model = mujoco.MjModel.from_xml_path("clive.xml")
    data = mujoco.MjData(model)
    tests, should_cycle = selected_tests()
    cycle_seconds = 4.0
    last_name = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            if should_cycle:
                index = int(t // cycle_seconds) % len(tests)
                local_t = t % cycle_seconds
            else:
                index = 0
                local_t = t

            active = tests[index]
            data.ctrl[:] = 0.0
            data.ctrl[active["actuator"]] = target_for(active, local_t)

            if active["name"] != last_name:
                print(f"Testing {active['name']}")
                last_name = active["name"]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
