import threading 
from perception_state import PerceptionState
from perception_mediapipe import run_mediapipe
from perception_deepface import run_deepface

state = PerceptionState()

deepface_thread = threading.Thread(target=run_deepface, args=(state,), daemon=True)
deepface_thread.start()

run_mediapipe(state)