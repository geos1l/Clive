import time
from emotional_state import EmotionalState
from perception_state import PerceptionState

def run_state_machine(state: PerceptionState, tick_rate: float = 0.1):
    """Reads perception signals, decides emotional state. Runs on its own thread."""
    current = EmotionalState.IDLE
    debounce_time = 0.5
    last_change = time.time()
    no_face_since = time.time()

    while True:
        with state.lock:
            face = state.face_present
            gaze = state.gaze_on_clive
            emotion = state.emotion
            wave = state.wave_detected

        now = time.time()

        if face:
            no_face_since = now
        else:
            with state.lock:
                state.emotion = "neutral"

        candidate = _decide_state(face, gaze, emotion, wave, now, no_face_since)

        if candidate != current:
            if now - last_change > debounce_time:
                current = candidate
                last_change = now
        else:
            last_change = now

        with state.lock:
            state.emotional_state = current.value

        print(f"[state] {current.value}")
        time.sleep(tick_rate)

def _decide_state(face, gaze, emotion, wave, now, no_face_since):
    """Pure logic - perception signals in, emotional state out."""
    if not face:
        if now - no_face_since > 10.0:
            return EmotionalState.SLEEPY
        return EmotionalState.IDLE
    
    if emotion in ("sad", "fear"):
        return EmotionalState.CONCERNED
    
    if not gaze:
        return EmotionalState.CURIOUS
    
    # Face present and looking at Clive
    if wave:
        return EmotionalState.WAVING
    if emotion == "happy":
        return EmotionalState.HAPPY
    
    return EmotionalState.ENGAGED
