import threading 
from perception_state import PerceptionState
from perception_mediapipe import run_mediapipe
from perception_deepface import run_deepface
from state_machine import run_state_machine

state = PerceptionState()

deepface_thread = threading.Thread(target=run_deepface, args=(state,), daemon=True)
deepface_thread.start()

sm_thread = threading.Thread(target=run_state_machine, args=(state,), daemon=True)
sm_thread.start()

run_mediapipe(state)