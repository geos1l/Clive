import mujoco
import mujoco.viewer
import math
import time

model = mujoco.MjModel.from_xml_path("clive.xml")
data = mujoco.MjData(model)

joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]
actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(model.nu)
]

print("Joints:", joint_names)
print("Actuators:", actuator_names)
for i in range(model.njnt):
    lo, hi = model.jnt_range[i]
    print(f"  {joint_names[i]}: range [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg")

period = 120.0
last_idx = -1

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # run enough physics steps to match real-time (~60fps rendering)
        while time.time() - step_start < 1.0 / 60.0:
            sim_time = data.time
            idx = int(sim_time / period) % model.nu
            phase = (sim_time % period) / period

            if idx != last_idx:
                jnt_id = model.actuator_trnid[idx, 0]
                lo, hi = model.jnt_range[jnt_id]
                print(f"\nTesting: {actuator_names[idx]}")
                print(f"  Joint range: [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg")
                last_idx = idx

            data.ctrl[:] = 0

            jnt_id = model.actuator_trnid[idx, 0]
            lo, hi = model.jnt_range[jnt_id]

            if lo >= 0:
                data.ctrl[idx] = lo + (hi - lo) * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
            else:
                mid = (lo + hi) / 2
                half = (hi - lo) / 2
                data.ctrl[idx] = mid + half * math.sin(2 * math.pi * phase)

            mujoco.mj_step(model, data)

        viewer.sync()
