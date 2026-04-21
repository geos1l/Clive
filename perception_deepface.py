import time
from deepface import DeepFace
from perception_state import PerceptionState

def run_deepface(state: PerceptionState, interval: float = 2.0):
    """Runs on its own thread. Grabs latest frame, classifies emotion."""
    while True:
        frame = None
        with state.lock:
            if state.latest_frame is not None:
                frame = state.latest_frame.copy()
        
        if frame is not None:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if results:
                    emotion = results[0]["dominant_emotion"]
                    with state.lock:
                        state.emotion = emotion
            except Exception:
                pass
        time.sleep(interval)