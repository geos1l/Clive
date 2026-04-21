import mujoco
import mujoco.viewer
import time

# Minimal XML (just a ball on a plane)
xml = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3"/>
      <geom type="plane" size="5 5 0.1"/>
      <body pos="0 0 1">
        <joint type="free"/>
        <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
      </body>
    </worldbody>
  </mujoco>
  """

# Launch model viewer
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(model.opt.timestep)